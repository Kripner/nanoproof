import argparse
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Protocol, Optional, runtime_checkable, Union, Callable

import torch
import torch.distributed as dist
from tqdm import tqdm
from leantree.repl_adapter.server import LeanClient

from nanoproof.common import compute_init, compute_cleanup, print0, autodetect_device_type, get_dist_info
from nanoproof.data import minif2f
from nanoproof.data import leanworkbook
from nanoproof.search import run_mcts, Config, Game, Node, Player
from nanoproof.inference import TacticModel, BlockingTacticModel
from nanoproof.cli import log


# Lean environment setup commands
LEAN_OPEN_SCOPED_COMMANDS = """
    open scoped Real
    open scoped Nat
    open scoped Topology
    open scoped Polynomial
"""


@runtime_checkable
class EvalProgressCallback(Protocol):
    """Protocol for receiving evaluation progress updates."""
    
    def on_start(self, dataset: str, total: int) -> None:
        """Called when evaluation starts."""
        ...
    
    def on_update(self, current: int, solved: int, errors: int) -> None:
        """Called after each theorem is evaluated with exact global counts."""
        ...


@dataclass
class EvalCounters:
    """Thread-safe counters for evaluation progress."""
    solved: int = 0
    errors: int = 0
    processed: int = 0
    
    def as_dict(self, total: int) -> dict:
        """Convert to final results dictionary."""
        success_rate = self.solved / total if total > 0 else 0.0
        error_rate = self.errors / total if total > 0 else 0.0
        return {
            "success_rate": success_rate,
            "solved": self.solved,
            "total": total,
            "errors": self.errors,
            "error_rate": error_rate,
        }


def evaluate_theorem(
    theorem: str,
    env,
    config: Config,
    model: Union[TacticModel, BlockingTacticModel],
) -> tuple[bool, bool]:
    """
    Evaluate a single theorem using MCTS.
    
    Returns:
        (success, is_error): success=True if proof found, is_error=True if theorem parsing failed
    """
    init_branch = env.proof_from_sorry(theorem)
    if not init_branch.is_success():
        return False, True  # Error case
    
    init_branch = init_branch.value
    game = Game(theorem, num_simulations=config.num_simulations)
    game.root = Node(
        action=None,
        prior=None,
        state=[init_branch],
        to_play=Player.OR,
        reward=None,
    )
    
    run_mcts(config, game, model)
    return game.root.is_solved, False


def aggregate_results(
    local_solved: int,
    local_errors: int, 
    local_total: int,
    device: torch.device,
    ddp: bool,
) -> dict:
    """Aggregate local results across DDP ranks and return final metrics."""
    local_metrics = torch.tensor(
        [local_solved, local_errors, local_total],
        dtype=torch.long, device=device
    )
    if ddp:
        dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)
    
    global_solved = local_metrics[0].item()
    global_errors = local_metrics[1].item()
    global_total = local_metrics[2].item()
    
    return EvalCounters(global_solved, global_errors, global_total).as_dict(global_total)


def broadcast_progress(
    progress: Optional[EvalProgressCallback],
    processed: int,
    solved: int,
    errors: int,
    device: torch.device,
    ddp: bool,
    ddp_rank: int,
    use_allreduce: bool = False,
):
    """
    Report progress update to callback.
    
    Args:
        use_allreduce: If True, aggregate stats across DDP ranks using all_reduce.
                       WARNING: Only use this when you can guarantee ALL ranks will
                       call this function the SAME number of times!
    """
    if progress is None:
        return
    
    if use_allreduce and ddp:
        local_stats = torch.tensor(
            [processed, solved, errors],
            dtype=torch.long, device=device
        )
        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)
        processed = local_stats[0].item()
        solved = local_stats[1].item()
        errors = local_stats[2].item()
    
    # Only rank 0 reports progress to avoid duplicate updates
    if ddp_rank == 0:
        progress.on_update(
            current=processed,
            solved=solved,
            errors=errors,
        )


@torch.inference_mode()
def eval_success_rate(
    tactic_model: Union[TacticModel, BlockingTacticModel],
    theorems=None,
    use_tqdm=False,
    progress: Optional[EvalProgressCallback] = None,
    num_actors: int = 1,
):
    """
    Evaluates the success rate of the model on the given theorems.
    Returns a dictionary with 'success_rate', 'solved', and 'total'.
    
    Args:
        tactic_model: The model to evaluate (TacticModel or BlockingTacticModel)
        theorems: List of theorems to evaluate
        use_tqdm: Whether to show tqdm progress bar
        progress: Optional callback object for progress updates
        num_actors: Number of parallel actors for evaluation. If >1, uses parallel evaluation.
                    When num_actors > 1 and tactic_model is a TacticModel, it will be wrapped
                    in a BlockingTacticModel automatically.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    theorems_subset = theorems[ddp_rank::ddp_world_size]
    
    if progress is not None and ddp_rank == 0:
        progress.on_start("", len(theorems))
    
    if num_actors > 1:
        return _eval_parallel(tactic_model, theorems, theorems_subset, progress, num_actors)
    else:
        return _eval_sequential(tactic_model, theorems_subset, use_tqdm, progress)


def _eval_sequential(
    tactic_model: Union[TacticModel, BlockingTacticModel],
    theorems_subset: list[str],
    use_tqdm: bool,
    progress: Optional[EvalProgressCallback],
) -> dict:
    """Sequential (single-threaded) evaluation."""
    ddp, ddp_rank, _, _ = get_dist_info()
    config = Config()
    device = tactic_model.network.get_device()
    
    counters = EvalCounters()
    client = LeanClient(config.server_address, config.server_port)
    
    process = client.get_process()
    if process is None:
        raise RuntimeError(f"Failed to acquire Lean process from {config.server_address}:{config.server_port}")
    
    with process as env:
        env.send_command(LEAN_OPEN_SCOPED_COMMANDS)
        
        iterator = enumerate(theorems_subset)
        if use_tqdm:
            iterator = tqdm(iterator, total=len(theorems_subset), desc=f"Rank {ddp_rank}", position=ddp_rank)
        
        for idx, theorem in iterator:
            solved, is_error = evaluate_theorem(theorem, env, config, tactic_model)
            
            if is_error:
                counters.errors += 1
                print0(f"Error on theorem: {theorem[:80]}...")
            elif solved:
                counters.solved += 1
            counters.processed += 1
            
            # In sequential mode, all ranks process their subset in lockstep, so we can safely use all_reduce
            broadcast_progress(progress, counters.processed, counters.solved, counters.errors, device, ddp, ddp_rank, use_allreduce=True)
    
    return aggregate_results(counters.solved, counters.errors, len(theorems_subset), device, ddp)


def _eval_parallel(
    tactic_model: Union[TacticModel, BlockingTacticModel],
    theorems: list[str],
    theorems_subset: list[str],
    progress: Optional[EvalProgressCallback],
    num_actors: int,
) -> dict:
    """Parallel evaluation using multiple threads with work-stealing."""
    ddp, ddp_rank, _, _ = get_dist_info()
    config = Config()
    
    # Wrap in BlockingTacticModel if necessary
    if isinstance(tactic_model, TacticModel):
        model = BlockingTacticModel(tactic_model, batch_size=num_actors, timeout_seconds=0.1)
        owns_model = True
    else:
        model = tactic_model
        owns_model = False
    
    device = model.network.get_device()
    
    # Thread-safe state - use separate lists to avoid any aliasing issues
    lock = threading.Lock()
    results = []  # List of (solved: bool, is_error: bool) tuples
    next_idx = [0]
    stop_flag = threading.Event()
    done_flag = threading.Event()  # Set when all workers are done
    
    # Thread-safe counters for live progress tracking
    counters = {"processed": 0, "solved": 0, "errors": 0}
    
    def actor_loop(actor_id: int):
        """Single actor thread that evaluates theorems until done."""
        asyncio.set_event_loop(asyncio.new_event_loop())
        local_results = []  # Collect results locally first
        
        log(f"[Eval Actor {actor_id}] Connecting to {config.server_address}:{config.server_port}")
        
        try:
            client = LeanClient(config.server_address, config.server_port)
        except Exception as e:
            log(f"[Eval Actor {actor_id}] Failed to connect: {e}")
            return
        
        try:
            process = client.get_process()
            if process is None:
                log(f"[Eval Actor {actor_id}] Failed to acquire Lean process (server overloaded?)")
                return
            
            with process as env:
                env.send_command(LEAN_OPEN_SCOPED_COMMANDS)
                log(f"[Eval Actor {actor_id}] Ready")
                
                while not stop_flag.is_set():
                    # Get next theorem (work stealing)
                    with lock:
                        if next_idx[0] >= len(theorems_subset):
                            break
                        my_idx = next_idx[0]
                        next_idx[0] += 1
                    
                    theorem = theorems_subset[my_idx]
                    solved, is_error = evaluate_theorem(theorem, env, config, model)
                    local_results.append((solved, is_error))
                    
                    # Update live counters for progress tracking
                    with lock:
                        counters["processed"] += 1
                        if solved and not is_error:
                            counters["solved"] += 1
                        if is_error:
                            counters["errors"] += 1
                    
                    if is_error:
                        log(f"[Eval Actor {actor_id}] Error: {theorem[:50]}...")
                
                log(f"[Eval Actor {actor_id}] Done, processed {len(local_results)} theorems")
        except Exception as e:
            log(f"[Eval Actor {actor_id}] Error: {e}")
        
        # Merge local results into global results under lock
        with lock:
            results.extend(local_results)
    
    # Run actors in parallel
    with ThreadPoolExecutor(max_workers=num_actors) as executor:
        futures = [executor.submit(actor_loop, i) for i in range(num_actors)]
        
        # Poll for progress while workers are running
        # WARNING: Do NOT use DDP collectives here! Each rank finishes at different times,
        # so the number of loop iterations is non-deterministic. Using all_reduce would
        # cause ranks to get out of sync and timeout.
        all_done = False
        while not all_done:
            # Check if all futures are done
            all_done = all(f.done() for f in futures)
            
            # Report local progress only (no DDP aggregation in the polling loop)
            if progress is not None and ddp_rank == 0:
                with lock:
                    current_processed = counters["processed"]
                    current_solved = counters["solved"]
                    current_errors = counters["errors"]
                # Report local stats only - final aggregation happens after all ranks finish
                progress.on_update(
                    current=current_processed,
                    solved=current_solved,
                    errors=current_errors,
                )
            
            if not all_done:
                time.sleep(0.5)  # Poll every 500ms
        
        # Collect any exceptions
        for i, future in enumerate(futures):
            try:
                future.result(timeout=1.0)  # Should be instant since already done
            except FuturesTimeoutError:
                log(f"[Eval] Actor {i} timed out")
            except Exception as e:
                log(f"[Eval] Actor {i} exception: {e}")
    
    if owns_model:
        model.shutdown()
    
    # Count results from the collected tuples
    solved_count = sum(1 for solved, is_error in results if solved and not is_error)
    error_count = sum(1 for solved, is_error in results if is_error)
    processed_count = len(results)
    
    log(f"[Eval] Parallel evaluation complete: {solved_count} solved, {error_count} errors, {processed_count} processed out of {len(theorems_subset)} theorems")
    
    # Synchronize all ranks before final aggregation
    if ddp:
        dist.barrier()
    
    # Final progress update with DDP aggregation (now safe since all ranks are synchronized)
    if progress is not None:
        broadcast_progress(progress, processed_count, solved_count, error_count, device, ddp, ddp_rank, use_allreduce=True)
    
    return aggregate_results(solved_count, error_count, processed_count, device, ddp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-theorems", type=int, default=50, help="Max theorems to evaluate")
    parser.add_argument("--num-actors", type=int, default=4, help="Number of parallel actors (1 for sequential)")
    args = parser.parse_args()

    device_type = autodetect_device_type()
    compute_init(device_type)

    tactic_model = TacticModel.create()
    minif2f_theorems = minif2f.list_theorems(split="Valid")[:args.max_theorems]
    leanworkbook_theorems = leanworkbook.list_theorems(split="val")[:args.max_theorems]

    def print_results(results, name):
        print0("-" * 80)
        print0(f"Evaluation results for {name}")
        print0(f"Success rate: {results['success_rate']:.4%}")
        print0(f"Solved: {results['solved']}/{results['total']}")
        print0(f"Errors: {results['errors']}/{results['total']}")
        print0(f"Error rate: {results['error_rate']:.4%}")
        print0("-" * 80)

    print0(f"Using {args.num_actors} parallel actor(s)")
    
    leanworkbook_results = eval_success_rate(
        tactic_model, leanworkbook_theorems, use_tqdm=True, num_actors=args.num_actors
    )
    print_results(leanworkbook_results, "LeanWorkBook")

    minif2f_results = eval_success_rate(
        tactic_model, minif2f_theorems, use_tqdm=True, num_actors=args.num_actors
    )
    print_results(minif2f_results, "MiniF2F")

    compute_cleanup()


if __name__ == "__main__":
    main()
