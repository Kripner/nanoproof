"""
Standalone prover evaluation entry point.

Loads a checkpoint and runs evaluation on one or more theorem-proving benchmarks
through the unified ``Prover`` abstraction:

- ``--lean-server host:port``  → ``LocalProver`` (in-process MCTS actors).
- ``--infra-file infra.toml``  → ``DistributedProver`` (coordinator + remote provers).

Result files are saved next to the checkpoint as
``eval_{step:06d}_{dataset}{seed?}{-test?}.jsonl``.
"""

import argparse
import atexit
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from nanoproof.checkpoints import load_model, parse_checkpoint_path
from nanoproof.cli import log
from nanoproof.common import autodetect_device_type, compute_cleanup, compute_init, get_dist_info, print0
from nanoproof.data.bench import minif2f
from nanoproof.data.rl import leanworkbook
from nanoproof.engine import Engine
from nanoproof.inference import BlockingTacticModel, TacticModel, start_inference_server
from nanoproof.prover import DistributedProver, LocalProver, Prover
from nanoproof.search import SearchConfig

# TODO: during verification, maybe set 'set_option maxHeartbeats 0\nset_option maxRecDepth 100000'


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


# -----------------------------------------------------------------------------
# Result persistence
# -----------------------------------------------------------------------------

def _write_eval_results_jsonl(jsonl_path: str, results: dict, prepend_entries: list[dict] = None):
    """Write evaluation results to a JSONL file (internal helper).

    Args:
        prepend_entries: Optional list of already-converted dict entries to write first
                        (used by --continue to preserve successful entries from previous run)
    """
    detailed_results = results.get("detailed_results", [])

    if not detailed_results and not prepend_entries:
        log(f"Skipping write of empty eval results to {jsonl_path}", component="Eval")
        return

    with open(jsonl_path, "w") as f:
        # Write prepended entries first (already in dict format)
        if prepend_entries:
            for entry in prepend_entries:
                f.write(json.dumps(entry) + "\n")

        for item in detailed_results:
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
    """
    jsonl_path = checkpoint_info.get_eval_path(dataset_name)
    _write_eval_results_jsonl(jsonl_path, results, prepend_entries=prepend_entries)


def save_eval_results_to_run_dir(output_dir: str, step: int, dataset_name: str, results: dict):
    """
    Save evaluation results to a JSONL file in the RL run's eval directory.

    For the RL training loop. Results are saved as:
        {output_dir}/evals/{step}/{dataset}.jsonl
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


# -----------------------------------------------------------------------------
# Model + checkpoint loading
# -----------------------------------------------------------------------------

def load_model_for_eval(
    device: torch.device,
    model_path: str,
    seed: int = 0,
) -> tuple[TacticModel, CheckpointInfo]:
    """Load a model for evaluation."""
    checkpoint_dir, step = parse_checkpoint_path(model_path)
    print0(f"Loading checkpoint: {checkpoint_dir}, step={step}")
    model, tokenizer, _ = load_model(model_path, device, phase="eval")

    engine = Engine(model, tokenizer)
    tactic_model = TacticModel(model, tokenizer, engine, num_samples=6, seed=seed)
    checkpoint_info = CheckpointInfo(checkpoint_dir=checkpoint_dir, step=step, seed=seed)

    return tactic_model, checkpoint_info


def get_checkpoint_info_early(model_path: str, seed: int = 0) -> CheckpointInfo:
    """Get checkpoint info without loading the model (for early existence checks)."""
    checkpoint_dir, step = parse_checkpoint_path(model_path)
    return CheckpointInfo(checkpoint_dir=checkpoint_dir, step=step, seed=seed)


# -----------------------------------------------------------------------------
# Prover construction
# -----------------------------------------------------------------------------

def _build_prover(
    inner_tactic_model: TacticModel,
    distributed: bool,
    infra_config,
    lean_server_str: Optional[str],
    num_actors: int,
    num_simulations: int,
) -> Prover:
    """Wrap the inner ``TacticModel`` in a ``BlockingTacticModel`` and build the
    appropriate ``Prover`` for the chosen mode.

    In distributed mode this also starts the rank-local inference server and,
    on the master, the coordinator.
    """
    blocking_model = BlockingTacticModel(
        inner_model=inner_tactic_model,
        timeout_seconds=0.2,
        max_batch_tokens=8000,
    )

    if distributed:
        from nanoproof.rl_server import start_coordinator

        ddp, ddp_rank, _, ddp_world_size = get_dist_info()
        master_process = ddp_rank == 0
        coordinator_port = infra_config.rl_server_port

        # Each rank starts its own inference server on a unique port
        inference_port = coordinator_port + 1 + ddp_rank
        start_inference_server(blocking_model, inference_port)
        log(f"Started inference server on port {inference_port}", component=f"Rank{ddp_rank}")

        if ddp:
            dist.barrier()

        if master_process:
            inference_endpoints = [f"http://127.0.0.1:{coordinator_port + 1 + r}" for r in range(ddp_world_size)]
            print0(f"Starting coordinator on port {coordinator_port} with {len(inference_endpoints)} inference endpoints...")
            start_coordinator(coordinator_port, inference_endpoints, startup_timeout=30.0)

        if ddp:
            dist.barrier()

        return DistributedProver(inference_model=blocking_model, poll_interval=3.0)

    # Local mode
    from nanoproof.infra import parse_lean_server
    local_lean = parse_lean_server(lean_server_str)
    config = SearchConfig(num_simulations=num_simulations, num_actors=num_actors)
    return LocalProver(
        config=config,
        tactic_model=blocking_model,
        lean_address=local_lean.address,
        lean_port=local_lean.port,
        num_simulations_eval=num_simulations,
    )


# -----------------------------------------------------------------------------
# Result printing
# -----------------------------------------------------------------------------

def print_results(results, name):
    print0("-" * 80)
    print0(f"Evaluation results for {name}")
    print0(f"Success rate: {results['success_rate']:.4%}")
    print0(f"Solved: {results['solved']}/{results['total']}")
    print0(f"Errors: {results['errors']}/{results['total']}")
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
            solved_at_t = sum(
                1 for item in detailed
                if item.get('proof_tree') is not None and item.get('num_iterations', 0) <= t
            )
            rate = solved_at_t / total if total > 0 else 0.0
            rates.append(f"{t:>3}: {rate:.2%}")
        print0("  " + "  |  ".join(rates))

    print0("-" * 80)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a prover model on theorem proving benchmarks")

    parser.add_argument("--model-path", type=str, required=True, help="path to model_NNNNNN.pt to evaluate (relative to models/ or absolute, e.g., 'sft/14-45-26_06-04-26_run/model_005000.pt')")
    parser.add_argument("--datasets", type=str, default="minif2f", help="Comma-separated list of datasets to evaluate (example: minif2f,leanworkbook)")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"], help="Dataset split to evaluate (default: valid)")
    parser.add_argument("--max-theorems", type=int, default=None, help="Max theorems to evaluate per dataset")
    parser.add_argument("--num-actors", type=int, default=4, help="Number of parallel actors for local mode (default: 4)")
    parser.add_argument("--num-simulations", type=int, default=512, help="Number of MCTS simulations per theorem (default: 512)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for tactic generation (default: 0)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing eval results")
    parser.add_argument("--continue", dest="continue_eval", action="store_true", help="Load existing results and retry only theorems that failed with errors")

    # Exactly one of --infra-file (distributed mode) or --lean-server (local mode) must be given.
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--infra-file", type=str, default=None, help="Path to infra.toml for distributed prover mode")
    mode_group.add_argument("--lean-server", type=str, default=None, help="Lean server host:port for local mode (e.g., '10.10.25.31:8000')")

    args = parser.parse_args()

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
    distributed = args.infra_file is not None
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
    checkpoint_info = get_checkpoint_info_early(model_path=args.model_path, seed=args.seed)

    # Check if eval results already exist (only on rank 0, then broadcast decision)
    # In --continue mode, we load existing results and retry only error theorems
    should_exit = False
    continue_data = {}  # dataset_name -> (successful_entries, error_theorems)

    if master_process:
        existing_results = []
        for dataset_name in datasets:
            eval_path = checkpoint_info.get_eval_path(dataset_name + split_suffix)
            if os.path.exists(eval_path):
                if os.path.getsize(eval_path) == 0:
                    # Leftover from a previous crashed run — silently ignore.
                    os.remove(eval_path)
                else:
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
    inner_tactic_model, checkpoint_info = load_model_for_eval(
        device=device,
        model_path=args.model_path,
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
            dataset_theorems["minif2f"] = minif2f.list_theorems(split=args.split)
        if "leanworkbook" in datasets:
            # leanworkbook only has a "valid" split (validated above that split != "test")
            dataset_theorems["leanworkbook"] = leanworkbook.list_theorems(split="valid")

        # Apply max_theorems limit
        if args.max_theorems:
            for name in dataset_theorems:
                dataset_theorems[name] = dataset_theorems[name][:args.max_theorems]

    print0(f"Evaluating with {args.num_simulations} MCTS simulations per theorem")
    print0(f"Using random seed: {args.seed}")
    if not distributed:
        print0(f"Using {args.num_actors} parallel actor(s)")

    # Construct the prover (also brings up coordinator + inference servers in
    # distributed mode).
    prover = _build_prover(
        inner_tactic_model=inner_tactic_model,
        distributed=distributed,
        infra_config=infra_config,
        lean_server_str=args.lean_server,
        num_actors=args.num_actors,
        num_simulations=args.num_simulations,
    )
    atexit.register(prover.shutdown)

    # Run evaluation for each dataset
    all_results = {}
    for dataset_name, theorems in dataset_theorems.items():
        print0(f"\nEvaluating on {len(theorems)} theorems from {dataset_name}")
        results = prover.evaluate(theorems, dataset_name=dataset_name)

        if master_process and results:
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
