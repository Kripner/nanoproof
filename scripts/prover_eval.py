import argparse
import asyncio
import atexit
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Protocol, Optional, runtime_checkable, Union, Callable

import torch
import torch.distributed as dist
from tqdm import tqdm
from leantree.repl_adapter.server import LeanClient

from nanoproof.common import compute_init, compute_cleanup, print0, autodetect_device_type, get_dist_info, theorem_to_example, linearize_proof, construct_proof_source, Player, active_barrier_master, active_barrier_wait
from nanoproof.data import minif2f
from nanoproof.data import leanworkbook
from nanoproof.search import run_mcts, Config, Game, Node, prune_redundant_nodes, verify_node, compute_value_target
from nanoproof.inference import TacticModel, BlockingTacticModel
from nanoproof.checkpoints import load_model, load_rl_model, get_rl_checkpoint_dir, get_pretrained_checkpoint_dir, resolve_step
from nanoproof.cli import log

# TODO: during verification, maybe set 'set_option maxHeartbeats 0\nset_option maxRecDepth 100000'

@dataclass
class TheoremResult:
    """Result of evaluating a single theorem."""
    theorem: str
    is_solved: bool
    error: Optional[str]
    proof_tree: Optional[dict]  # Serialized (simplified) Node tree, None if not solved
    unsimplified_proof_tree: Optional[dict]  # Serialized (unsimplified) Node tree, None if not solved
    linearized_proof: Optional[str]  # Human-readable proof source, None if not solved
    num_iterations: int  # Number of MCTS iterations run


@dataclass
class CheckpointInfo:
    """Information about the loaded checkpoint, used for saving eval results."""
    checkpoint_dir: str
    step: int
    seed: int = 0
    
    def get_eval_path(self, dataset_name: str) -> str:
        """Get the path where eval results should be saved for a dataset."""
        seed_suffix = f"-{self.seed}" if self.seed != 0 else ""
        return os.path.join(self.checkpoint_dir, f"eval_{self.step:06d}_{dataset_name}{seed_suffix}.jsonl")


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


# TODO: use ProverWorker._play_game
# TODO: debug: it seems that sometimes, errors on proof_from_sorry are not reported or printed
def evaluate_theorem(
    theorem: str,
    env,
    config: Config,
    model: Union[TacticModel, BlockingTacticModel],
) -> TheoremResult:
    """
    Evaluate a single theorem using MCTS.
    
    Returns:
        TheoremResult with is_solved, error, proof_tree (serialized), and num_iterations.
    """
    init_branch = env.proof_from_sorry(theorem_to_example(theorem))
    if not init_branch.is_success():
        print(f"Error starting theorem: {theorem[:500]}{'...' if len(theorem) > 500 else ''}: {init_branch.error}")
        # Error case - return max iterations since we couldn't even start
        return TheoremResult(
            theorem=theorem,
            is_solved=False,
            error=init_branch.error or "Could not start proof due to unknown error",
            proof_tree=None,
            num_iterations=config.num_simulations,
        )
    
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
    
    is_solved = game.root.is_solved
    proof_tree = None
    unsimplified_proof_tree = None
    linearized_proof = None
    
    if is_solved:
        verify_node(game.root)
        
        unsimplified_proof_tree = game.unsimplified_root.serialize()
        prune_redundant_nodes(game.root)
        compute_value_target(game.root)

        verify_node(game.root)

        # Verify the linearized proof compiles correctly
        tactics = linearize_proof(game.root)
        proof_source = construct_proof_source(theorem, tactics)
        if not env.is_valid_source(proof_source):
            log(f"!! Linearized proof source verification failed:\n\"\"\"\n{proof_source}\n\"\"\"\n", component=f"ProverEval")
            is_solved = False
        else:
            proof_tree = game.root.serialize()
            linearized_proof = proof_source
    
    return TheoremResult(
        theorem=theorem,
        is_solved=is_solved,
        error=None,
        proof_tree=proof_tree,
        unsimplified_proof_tree=unsimplified_proof_tree,
        linearized_proof=linearized_proof,
        num_iterations=game.num_iterations,
    )


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
    num_simulations: int = 50,
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
        num_simulations: Number of MCTS simulations per theorem
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    theorems_subset = theorems[ddp_rank::ddp_world_size]
    
    if progress is not None and ddp_rank == 0:
        progress.on_start("", len(theorems))
    
    if num_actors > 1:
        return _eval_parallel(tactic_model, theorems_subset, progress, num_actors, num_simulations)
    else:
        return _eval_sequential(tactic_model, theorems_subset, use_tqdm, progress, num_simulations)


def _eval_sequential(
    tactic_model: Union[TacticModel, BlockingTacticModel],
    theorems_subset: list[str],
    use_tqdm: bool,
    progress: Optional[EvalProgressCallback],
    num_simulations: int = 50,
) -> dict:
    """Sequential (single-threaded) evaluation."""
    ddp, ddp_rank, _, _ = get_dist_info()
    config = Config(num_simulations=num_simulations)
    device = tactic_model.network.get_device()
    
    counters = EvalCounters()
    detailed_results: list[TheoremResult] = []
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
            result = evaluate_theorem(theorem, env, config, tactic_model)
            detailed_results.append(result)
            
            if result.error is not None:
                counters.errors += 1
                ellipsis = "..." if len(theorem) > 500 else ""
                log(f"Error on theorem: {theorem[:500]}{ellipsis}: {result.error}")
            elif result.is_solved:
                counters.solved += 1
            counters.processed += 1
            
            # In sequential mode, all ranks process their subset in lockstep, so we can safely use all_reduce
            broadcast_progress(progress, counters.processed, counters.solved, counters.errors, device, ddp, ddp_rank, use_allreduce=True)
    
    results = aggregate_results(counters.solved, counters.errors, len(theorems_subset), device, ddp)
    results["detailed_results"] = detailed_results
    return results


def _eval_parallel(
    tactic_model: Union[TacticModel, BlockingTacticModel],
    theorems_subset: list[str],
    progress: Optional[EvalProgressCallback],
    num_actors: int,
    num_simulations: int = 50,
) -> dict:
    """Parallel evaluation using multiple threads with work-stealing."""
    ddp, ddp_rank, _, _ = get_dist_info()
    config = Config(num_simulations=num_simulations)
    
    # Wrap in BlockingTacticModel if necessary
    if isinstance(tactic_model, TacticModel):
        model = BlockingTacticModel(tactic_model, max_batch_tokens=8000, timeout_seconds=0.1)
        owns_model = True
    else:
        model = tactic_model
        owns_model = False
    
    device = model.network.get_device()
    
    # Thread-safe state for work-stealing and progress tracking
    lock = threading.Lock()
    next_idx = [0]
    counters = {"processed": 0, "solved": 0, "errors": 0}
    detailed_results: list[TheoremResult] = []
    
    def actor_loop(actor_id: int):
        """Single actor thread that evaluates theorems until done."""
        # LeanClient may use asyncio internally, so each thread needs its own event loop
        asyncio.set_event_loop(asyncio.new_event_loop())
        local_processed = 0
        
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
                
                while True:
                    # Get next theorem (work stealing)
                    with lock:
                        if next_idx[0] >= len(theorems_subset):
                            break
                        my_idx = next_idx[0]
                        next_idx[0] += 1
                    
                    theorem = theorems_subset[my_idx]
                    result = evaluate_theorem(theorem, env, config, model)
                    local_processed += 1
                    
                    # Update counters and store result
                    with lock:
                        counters["processed"] += 1
                        detailed_results.append(result)
                        if result.is_solved and result.error is None:
                            counters["solved"] += 1
                        if result.error is not None:
                            counters["errors"] += 1
                    
                    if result.error is not None:
                        ellipsis = "..." if len(theorem) > 500 else ""
                        log(f"[Eval Actor {actor_id}] Error: {theorem[:500]}{ellipsis}: {result.error}")
                
                log(f"[Eval Actor {actor_id}] Done, processed {local_processed} theorems")
        except Exception as e:
            log(f"[Eval Actor {actor_id}] Error: {e}")
            with lock:
                counters["errors"] += 1
    
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
    
    # Extract final counts from thread-safe counters
    solved_count = counters["solved"]
    error_count = counters["errors"]
    processed_count = counters["processed"]
    
    log(f"[Eval] Parallel evaluation complete: {solved_count} solved, {error_count} errors, {processed_count} processed out of {len(theorems_subset)} theorems")
    
    # Synchronize all ranks before final aggregation
    if ddp:
        dist.barrier()
    
    # Final progress update with DDP aggregation (now safe since all ranks are synchronized)
    if progress is not None:
        broadcast_progress(progress, processed_count, solved_count, error_count, device, ddp, ddp_rank, use_allreduce=True)
    
    # Use len(theorems_subset) as total to be consistent with _eval_sequential
    results = aggregate_results(solved_count, error_count, len(theorems_subset), device, ddp)
    results["detailed_results"] = detailed_results
    return results


def _write_eval_results_jsonl(jsonl_path: str, results: dict, prepend_entries: list[dict] = None):
    """Write evaluation results to a JSONL file (internal helper).
    
    Args:
        prepend_entries: Optional list of already-converted dict entries to write first
                        (used by --continue to preserve successful entries from previous run)
    """
    detailed_results = results.get("detailed_results", [])
    
    with open(jsonl_path, "w") as f:
        # Write prepended entries first (already in dict format)
        if prepend_entries:
            for entry in prepend_entries:
                f.write(json.dumps(entry) + "\n")
        
        for item in detailed_results:
            # Handle both dict (from distributed) and TheoremResult (from local) formats
            if hasattr(item, "theorem"):
                # It's a TheoremResult dataclass
                entry = {
                    "theorem": item.theorem,
                    "proof": item.proof_tree,
                    "unsimplified_proof": item.unsimplified_proof_tree,
                    "linearized_proof": item.linearized_proof,
                    "num_iterations": item.num_iterations,
                    "error": item.error,
                }
            else:
                # It's a dict (from distributed_eval)
                entry = {
                    "theorem": item["theorem"],
                    "proof": item["proof_tree"],
                    "unsimplified_proof": item.get("unsimplified_proof_tree"),
                    "linearized_proof": item.get("linearized_proof"),
                    "num_iterations": item["num_iterations"],
                    "error": item.get("error"),
                }
            f.write(json.dumps(entry) + "\n")
    
    total_count = len(detailed_results) + (len(prepend_entries) if prepend_entries else 0)
    log(f"Saved {total_count} eval results to {jsonl_path}", component="Eval")


def save_eval_results(checkpoint_info: CheckpointInfo, dataset_name: str, results: dict, prepend_entries: list[dict] = None):
    """
    Save evaluation results to a JSONL file alongside the checkpoint.
    
    For standalone evaluation scripts. Results are saved as:
        {checkpoint_dir}/eval_{step:06d}_{dataset}.jsonl
    
    Args:
        checkpoint_info: Information about the checkpoint (directory and step)
        dataset_name: Name of the dataset (e.g., "minif2f", "leanworkbook")
        results: Dict containing 'detailed_results' with evaluation details
        prepend_entries: Optional entries to write before new results (for --continue mode)
    """
    jsonl_path = checkpoint_info.get_eval_path(dataset_name)
    _write_eval_results_jsonl(jsonl_path, results, prepend_entries=prepend_entries)


def save_eval_results_to_run_dir(output_dir: str, step: int, dataset_name: str, results: dict):
    """
    Save evaluation results to a JSONL file in the RL run's eval directory.
    
    For the RL training loop. Results are saved as:
        {output_dir}/evals/{step}/{dataset}.jsonl
    
    Args:
        output_dir: The output directory for the RL run
        step: Current training step
        dataset_name: Name of the dataset (e.g., "minif2f", "leanworkbook")
        results: Dict containing 'detailed_results' with evaluation details
    """
    eval_dir = os.path.join(output_dir, "evals", str(step))
    os.makedirs(eval_dir, exist_ok=True)
    jsonl_path = os.path.join(eval_dir, f"{dataset_name}.jsonl")
    _write_eval_results_jsonl(jsonl_path, results)


def load_existing_eval_results(jsonl_path: str) -> tuple[list[dict], list[dict]]:
    """
    Load existing evaluation results from a JSONL file.
    
    Returns:
        Tuple of (successful_entries, error_entries) where:
        - successful_entries: entries with no error (proof may or may not exist)
        - error_entries: entries with non-null error field
    """
    successful = []
    errors = []
    
    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("error") is not None:
                errors.append(entry)
            else:
                successful.append(entry)
    
    return successful, errors


def load_model_for_eval(
    device: torch.device,
    rl_run: Optional[str] = None,
    source: str = "sft",
    model_tag: Optional[str] = None,
    step: Optional[int] = None,
    seed: int = 0,
) -> tuple[TacticModel, CheckpointInfo]:
    """
    Load a model for evaluation.
    
    Supports two modes:
    1. RL checkpoint: --rl-run <run_name> [--step <step>]
    2. Pretrained checkpoint: --source <sft|base> --model-tag <tag> [--step <step>]
    
    Args:
        device: Device to load the model on
        rl_run: Name of RL run to load from (e.g., "25-01-15_10-30-my-run")
        source: Source for pretrained checkpoints ("sft" or "base")
        model_tag: Model tag for pretrained checkpoints
        step: Checkpoint step (optional, defaults to latest)
        seed: Random seed for tactic generation (default: 0)
    
    Returns:
        Tuple of (TacticModel, CheckpointInfo)
    """
    from nanoproof.engine import Engine
    
    if rl_run:
        # Load from RL checkpoint
        checkpoint_dir = get_rl_checkpoint_dir(rl_run)
        resolved_step = resolve_step(checkpoint_dir, step)
        print0(f"Loading RL checkpoint: run={rl_run}, step={resolved_step}")
        model, tokenizer, meta_data = load_rl_model(rl_run, device, phase="eval", step=resolved_step)
    else:
        # Load from pretrained checkpoint
        if model_tag is None:
            raise ValueError("--model-tag is required when not using --rl-run")
        checkpoint_dir = get_pretrained_checkpoint_dir(source, model_tag)
        resolved_step = resolve_step(checkpoint_dir, step)
        print0(f"Loading pretrained checkpoint: source={source}, model_tag={model_tag}, step={resolved_step}")
        model, tokenizer, meta_data = load_model(source, device, phase="eval", model_tag=model_tag, step=resolved_step)
    
    engine = Engine(model, tokenizer)
    tactic_model = TacticModel(model, tokenizer, engine, num_samples=6, seed=seed)
    checkpoint_info = CheckpointInfo(checkpoint_dir=checkpoint_dir, step=resolved_step, seed=seed)
    
    return tactic_model, checkpoint_info


@dataclass
class DistributedEvalInfra:
    """Infrastructure for distributed evaluation (one per process)."""
    inference_model: BlockingTacticModel
    is_master: bool
    
    def shutdown(self):
        """Shutdown the inference model (coordinator shutdown handled separately)."""
        self.inference_model.shutdown()


def setup_distributed_eval_infra(
    tactic_model: TacticModel,
    infra_config,
) -> DistributedEvalInfra:
    """
    Set up distributed evaluation infrastructure on ALL DDP ranks.
    
    Each rank starts an inference server. Rank 0 also starts the coordinator.
    This should be called once before evaluating any datasets.
    
    Args:
        tactic_model: The TacticModel to use for inference
        infra_config: InfraConfig from infra.toml
    
    Returns:
        DistributedEvalInfra handle for cleanup
    """
    from nanoproof.inference import start_inference_server
    from nanoproof.rl_server import start_coordinator, shutdown_coordinator
    
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    master_process = ddp_rank == 0
    coordinator_port = infra_config.rl_server_port
    
    # Create BlockingTacticModel for the inference server
    inference_model = BlockingTacticModel(
        inner_model=tactic_model,
        timeout_seconds=0.2,
        max_batch_tokens=8000
    )
    
    # Each rank starts its own inference server on a unique port
    inference_port = coordinator_port + 1 + ddp_rank
    inference_server_thread = start_inference_server(inference_model, inference_port)
    log(f"Started inference server on port {inference_port}", component=f"Rank{ddp_rank}")
    
    # Sync to ensure all inference servers are up before coordinator starts
    if ddp:
        dist.barrier()
    
    # Master starts the coordinator with all inference endpoints
    if master_process:
        inference_endpoints = [f"http://127.0.0.1:{coordinator_port + 1 + r}" for r in range(ddp_world_size)]
        print0(f"Starting coordinator on port {coordinator_port} with {len(inference_endpoints)} inference endpoints...")
        coordinator_thread, inference_router = start_coordinator(
            coordinator_port, inference_endpoints, startup_timeout=30.0
        )
    
    # Sync again so all ranks wait for coordinator
    if ddp:
        dist.barrier()
    
    # Register cleanup with atexit
    def cleanup():
        if master_process:
            print0("Shutting down distributed infrastructure...")
            shutdown_coordinator()
        inference_model.shutdown()
    atexit.register(cleanup)
    
    return DistributedEvalInfra(inference_model=inference_model, is_master=master_process)


def run_distributed_eval_on_dataset(
    theorems: list[str],
    dataset_name: str,
    no_progress_timeout: float = 600.0,
) -> dict:
    """
    Run distributed evaluation on a single dataset.
    
    Must be called after setup_distributed_eval_infra(). Only master rank
    coordinates the actual evaluation; other ranks just return empty dict
    (their inference servers handle requests in background threads).
    
    Args:
        theorems: List of theorem strings to evaluate
        dataset_name: Name of the dataset (for logging)
        no_progress_timeout: Maximum time to wait without progress (seconds)
    
    Returns:
        Dict with 'success_rate', 'solved', 'total', 'errors', 'detailed_results'
        (only meaningful on rank 0; other ranks return empty dict)
    """
    from nanoproof.rl_server import distributed_eval
    
    ddp, ddp_rank, _, _ = get_dist_info()
    master_process = ddp_rank == 0
    
    # Only master runs the evaluation coordination
    if master_process:
        print0(f"Running distributed evaluation on {len(theorems)} theorems from {dataset_name}")
        results = distributed_eval(theorems, dataset_name=dataset_name, no_progress_timeout=no_progress_timeout)
    else:
        # Other ranks keep their inference server running and wait
        results = {}
    
    return results


def get_checkpoint_info_early(
    rl_run: Optional[str],
    source: str,
    model_tag: Optional[str],
    step: Optional[int],
    seed: int = 0,
) -> CheckpointInfo:
    """
    Get checkpoint info without loading the model (for early existence checks).
    """
    if rl_run:
        checkpoint_dir = get_rl_checkpoint_dir(rl_run)
    else:
        if model_tag is None:
            raise ValueError("--model-tag is required when not using --rl-run")
        checkpoint_dir = get_pretrained_checkpoint_dir(source, model_tag)
    
    resolved_step = resolve_step(checkpoint_dir, step)
    return CheckpointInfo(checkpoint_dir=checkpoint_dir, step=resolved_step, seed=seed)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a prover model on theorem proving benchmarks")
    
    # Model loading options (mutually exclusive groups)
    model_group = parser.add_argument_group("Model selection")
    model_group.add_argument("--rl-run", type=str, default=None, help="Load from RL checkpoint (run name, e.g., '25-01-15_10-30-my-run')")
    model_group.add_argument("--source", type=str, default="sft", choices=["sft", "base"], help="Source for pretrained checkpoints (default: sft)")
    model_group.add_argument("--model-tag", type=str, default="d26", help="Model tag for pretrained checkpoints (e.g., 'd26')")
    model_group.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    
    # Evaluation options
    eval_group = parser.add_argument_group("Evaluation settings")
    eval_group.add_argument("--datasets", type=str, default="minif2f", help="Comma-separated list of datasets to evaluate (example: minif2f,leanworkbook)")
    eval_group.add_argument("--split", type=str, default="valid", choices=["valid", "test"], help="Dataset split to evaluate (default: valid)")
    eval_group.add_argument("--max-theorems", type=int, default=None, help="Max theorems to evaluate per dataset")
    eval_group.add_argument("--num-actors", type=int, default=4, help="Number of parallel actors for local mode (default: 4)")
    eval_group.add_argument("--num-simulations", type=int, default=512, help="Number of MCTS simulations per theorem (default: 50)")
    eval_group.add_argument("--seed", type=int, default=0, help="Random seed for tactic generation (default: 0)")
    
    # Distributed mode
    dist_group = parser.add_argument_group("Distributed mode")
    dist_group.add_argument("--infra-file", type=str, default="", help="Path to infra.toml for distributed mode (empty = local mode)")
    dist_group.add_argument("--no-progress-timeout", type=float, default=6000.0, help="Timeout in seconds if no progress is made (default: 600)")
    
    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--force", action="store_true", help="Overwrite existing eval results")
    output_group.add_argument("--continue", dest="continue_eval", action="store_true", help="Load existing results and retry only theorems that failed with errors")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.rl_run and not args.model_tag:
        parser.error("Either --rl-run or --model-tag must be specified")
    
    if args.force and args.continue_eval:
        parser.error("--force and --continue are mutually exclusive")
    
    # Parse datasets
    datasets = [d.strip().lower() for d in args.datasets.split(",")]
    valid_datasets = {"minif2f", "leanworkbook"}
    for d in datasets:
        if d not in valid_datasets:
            parser.error(f"Unknown dataset: {d}. Valid options: {valid_datasets}")
    
    # Validate split for leanworkbook
    if "leanworkbook" in datasets and args.split == "test":
        raise ValueError("leanworkbook does not have a test split. Use --split=valid instead.")
    
    # Warn about test split evaluation
    if args.split == "test":
        print0("WARNING: Evaluating on TEST split. Results will be saved with '-test' suffix.")
    
    # Suffix for saved files based on split
    split_suffix = "-test" if args.split == "test" else ""
    
    # Check distributed prover mode (using external prover servers)
    distributed = bool(args.infra_file)
    if distributed:
        if not os.path.exists(args.infra_file):
            print0(f"Error: Infrastructure file {args.infra_file} does not exist")
            sys.exit(1)
        from nanoproof.infra import load_infra_config
        infra_config = load_infra_config(args.infra_file)
    else:
        infra_config = None
    
    # Initialize compute (must happen before any DDP operations)
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    
    # Get checkpoint info early to check for existing results before loading model
    checkpoint_info = get_checkpoint_info_early(
        rl_run=args.rl_run,
        source=args.source,
        model_tag=args.model_tag,
        step=args.step,
        seed=args.seed,
    )
    
    # Check if eval results already exist (only on rank 0, then broadcast decision)
    # In --continue mode, we load existing results and retry only error theorems
    should_exit = False
    continue_data = {}  # dataset_name -> (successful_entries, error_theorems)
    
    if master_process:
        existing_results = []
        for dataset_name in datasets:
            eval_path = checkpoint_info.get_eval_path(dataset_name + split_suffix)
            if os.path.exists(eval_path):
                existing_results.append((dataset_name, eval_path))
        
        if args.continue_eval:
            # --continue mode: require existing results
            if not existing_results:
                print0("Error: --continue requires existing evaluation results, but none found.")
                print0(f"Expected files in: {checkpoint_info.checkpoint_dir}")
                should_exit = True
            else:
                # Load existing results and identify error theorems
                for dataset_name, eval_path in existing_results:
                    successful, errors = load_existing_eval_results(eval_path)
                    if not errors:
                        print0(f"No error entries found in {eval_path}, skipping {dataset_name}")
                        # Remove from datasets to process
                        continue_data[dataset_name] = (successful, [])
                    else:
                        error_theorems = [e["theorem"] for e in errors]
                        continue_data[dataset_name] = (successful, error_theorems)
                        print0(f"Found {len(errors)} error entries to retry in {dataset_name}")
        elif existing_results and not args.force:
            print0("Evaluation results already exist:")
            for _, path in existing_results:
                print0(f"  {path}")
            print0("\nUse --force to overwrite existing results, or --continue to retry errors.")
            should_exit = True
    
    # Broadcast exit decision to all ranks
    if ddp:
        exit_tensor = torch.tensor([1 if should_exit else 0], device=device)
        dist.broadcast(exit_tensor, src=0)
        should_exit = exit_tensor.item() == 1
    
    if should_exit:
        compute_cleanup()
        sys.exit(1)
    
    if master_process:
        if distributed:
            print0(f"Distributed prover mode enabled with {args.infra_file}")
            # Warn if num_simulations is changed from default - must be set in prover_server.py
            if args.num_simulations != 512:
                print0(f"WARNING: --num-simulations={args.num_simulations} is ignored in distributed mode. "
                       f"Set --num-simulations on prover_server.py instead.")
        else:
            print0("Local mode enabled")
    
    # Load model
    tactic_model, checkpoint_info = load_model_for_eval(
        device=device,
        rl_run=args.rl_run,
        source=args.source,
        model_tag=args.model_tag,
        step=args.step,
        seed=args.seed,
    )
    
    print0(f"Results will be saved to: {checkpoint_info.checkpoint_dir}")
    
    # Load theorems for each dataset
    dataset_theorems = {}
    if args.continue_eval:
        # In continue mode, only retry error theorems
        for dataset_name, (successful, error_theorems) in continue_data.items():
            if error_theorems:
                dataset_theorems[dataset_name] = error_theorems
    else:
        # Normal mode: load full datasets
        if "minif2f" in datasets:
            minif2f_split = "Valid" if args.split == "valid" else "Test"
            dataset_theorems["minif2f"] = minif2f.list_theorems(split=minif2f_split)
        if "leanworkbook" in datasets:
            # leanworkbook only has "val" split (validated above that split != "test")
            dataset_theorems["leanworkbook"] = leanworkbook.list_theorems(split="val")

        # Apply max_theorems limit
        if args.max_theorems:
            for name in dataset_theorems:
                dataset_theorems[name] = dataset_theorems[name][:args.max_theorems]
    
    def print_results(results, name):
        print0("-" * 80)
        print0(f"Evaluation results for {name}")
        print0(f"Success rate: {results['success_rate']:.4%}")
        print0(f"Solved: {results['solved']}/{results['total']}")
        print0(f"Errors: {results['errors']}/{results['total']}")
        if 'error_rate' in results:
            print0(f"Error rate: {results['error_rate']:.4%}")
        if results.get('timed_out'):
            print0("WARNING: Evaluation timed out!")
        if results.get('invalid'):
            print0("WARNING: Some results were invalid!")
        
        # Print success rates at different simulation thresholds
        detailed = results.get('detailed_results', [])
        if detailed:
            total = len(detailed)
            thresholds = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
            print0("Success rate by simulation budget:")
            rates = []
            for t in thresholds:
                # Count theorems solved with <= t simulations
                solved_at_t = sum(
                    1 for item in detailed
                    if (hasattr(item, 'is_solved') and item.is_solved and item.num_iterations <= t) or
                       (isinstance(item, dict) and item.get('proof_tree') is not None and item.get('num_iterations', 0) <= t)
                )
                rate = solved_at_t / total if total > 0 else 0.0
                rates.append(f"{t:>3}: {rate:.2%}")
            print0("  " + "  |  ".join(rates))
        
        print0("-" * 80)
    
    print0(f"Evaluating with {args.num_simulations} MCTS simulations per theorem")
    print0(f"Using random seed: {args.seed}")
    if not distributed:
        print0(f"Using {args.num_actors} parallel actor(s)")
    
    # Set up distributed infrastructure once (before evaluating any datasets)
    if distributed:
        print0(f"Setting up distributed infrastructure with {ddp_world_size} GPU(s)...")
        eval_infra = setup_distributed_eval_infra(tactic_model, infra_config)
    
    # Run evaluation for each dataset
    all_results = {}
    for dataset_name, theorems in dataset_theorems.items():
        print0(f"\nEvaluating on {len(theorems)} theorems from {dataset_name}")
        
        if distributed:
            # Distributed prover mode: all ranks participate in inference
            results = run_distributed_eval_on_dataset(
                theorems, dataset_name, no_progress_timeout=args.no_progress_timeout
            )
            # Use active barrier (polling) instead of dist.barrier() to avoid NCCL timeout
            # during long evaluations
            if ddp:
                if master_process:
                    active_barrier_master(f"eval_done_{dataset_name}")
                else:
                    active_barrier_wait(f"eval_done_{dataset_name}")
        else:
            # Local mode: each rank evaluates a subset of theorems
            results = eval_success_rate(
                tactic_model, theorems, use_tqdm=True,
                num_actors=args.num_actors, num_simulations=args.num_simulations
            )
        
        if master_process and results is not None:
            all_results[dataset_name] = results
            print_results(results, dataset_name)
            
            if results.get('timed_out') or results.get('invalid'):
                print0(f"WARNING: Evaluation timed out or invalid for {dataset_name}")
            if results.get('errors', 0) > 0:
                print0(f"WARNING: Evaluation failed for {dataset_name} due to {results['errors']} system errors")

            # In continue mode, prepend the original successful entries
            prepend = continue_data[dataset_name][0] if args.continue_eval else None
            save_eval_results(checkpoint_info, dataset_name + split_suffix, results, prepend_entries=prepend)
    
    compute_cleanup()
    
    return all_results


if __name__ == "__main__":
    main()
