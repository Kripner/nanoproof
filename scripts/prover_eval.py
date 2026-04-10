"""
Standalone prover evaluation.

Loads a checkpoint, queries Lean servers for available processes, runs MCTS
evaluation on one or more theorem-proving benchmarks using the same ``Prover``
class as the RL training loop, and saves results next to the checkpoint.

Usage:
    python scripts/prover_eval.py \\
        --model-path sft/.../model_005000.pt \\
        --lean-servers 10.10.25.33:8000 \\
        --datasets minif2f,leanworkbook
"""

import argparse
import atexit
import json
import os
import sys
from dataclasses import dataclass

import torch

from nanoproof.checkpoints import load_model, parse_checkpoint_path
from nanoproof.cli import log
from nanoproof.common import autodetect_device_type, compute_cleanup, compute_init, print0
from nanoproof.data.bench import minif2f
from nanoproof.data.rl import leanworkbook
from nanoproof.engine import Engine
from nanoproof.inference import BlockingTacticModel, InferenceBalancer, TacticModel, start_inference_server
from nanoproof.prover import Prover, assign_lean_servers, query_lean_servers
from nanoproof.search import SearchConfig

# TODO: during verification, maybe set 'set_option maxHeartbeats 0\nset_option maxRecDepth 100000'


@dataclass
class CheckpointInfo:
    """Information about the loaded checkpoint, used for saving eval results."""
    checkpoint_dir: str
    step: int
    seed: int = 0

    def get_eval_path(self, dataset_name: str) -> str:
        seed_suffix = f"-{self.seed}" if self.seed != 0 else ""
        return os.path.join(self.checkpoint_dir, f"eval_{self.step:06d}_{dataset_name}{seed_suffix}.jsonl")


# -----------------------------------------------------------------------------
# Result persistence
# -----------------------------------------------------------------------------

def _write_eval_results_jsonl(jsonl_path: str, results: dict, prepend_entries: list[dict] = None):
    """Write evaluation results to a JSONL file."""
    detailed_results = results.get("detailed_results", [])

    if not detailed_results and not prepend_entries:
        log(f"Skipping write of empty eval results to {jsonl_path}", component="Eval")
        return

    with open(jsonl_path, "w") as f:
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
    """Save evaluation results alongside the checkpoint."""
    jsonl_path = checkpoint_info.get_eval_path(dataset_name)
    _write_eval_results_jsonl(jsonl_path, results, prepend_entries=prepend_entries)


def save_eval_results_to_run_dir(output_dir: str, step: int, dataset_name: str, results: dict):
    """Save evaluation results in the RL run's eval directory."""
    eval_dir = os.path.join(output_dir, "evals", str(step))
    os.makedirs(eval_dir, exist_ok=True)
    jsonl_path = os.path.join(eval_dir, f"{dataset_name}.jsonl")
    _write_eval_results_jsonl(jsonl_path, results)


def load_existing_eval_results(jsonl_path: str) -> tuple[list[dict], list[dict]]:
    """Load existing results. Returns (successful_entries, error_entries)."""
    successful, errors = [], []
    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            (errors if entry.get("error") is not None else successful).append(entry)
    return successful, errors


# -----------------------------------------------------------------------------
# Result printing
# -----------------------------------------------------------------------------

def print_results(results, name):
    print0("-" * 80)
    print0(f"Evaluation results for {name}")
    print0(f"Success rate: {results['success_rate']:.4%}")
    print0(f"Solved: {results['solved']}/{results['total']}")
    print0(f"Errors: {results['errors']}/{results['total']}")

    detailed = results.get('detailed_results', [])
    if detailed:
        total = len(detailed)
        thresholds = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        rates = []
        for t in thresholds:
            solved_at_t = sum(
                1 for item in detailed
                if item.get('proof_tree') is not None and item.get('num_iterations', 0) <= t
            )
            rate = solved_at_t / total if total > 0 else 0.0
            rates.append(f"{t:>3}: {rate:.2%}")
        print0("Success rate by simulation budget:")
        print0("  " + "  |  ".join(rates))

    print0("-" * 80)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a prover model on theorem proving benchmarks")

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--lean-servers", type=str, required=True,
                        help="comma-separated Lean server addresses (e.g., '10.10.25.33:8000')")
    parser.add_argument("--datasets", type=str, default="minif2f",
                        help="comma-separated datasets (minif2f, leanworkbook)")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--max-theorems", type=int, default=None)
    parser.add_argument("--num-simulations", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite existing results")
    parser.add_argument("--continue", dest="continue_eval", action="store_true",
                        help="retry only theorems that failed with errors")
    parser.add_argument("--inference-server-port", type=int, default=5000)
    args = parser.parse_args()

    if args.force and args.continue_eval:
        parser.error("--force and --continue are mutually exclusive")

    datasets = [d.strip().lower() for d in args.datasets.split(",")]
    valid_datasets = {"minif2f", "leanworkbook"}
    for d in datasets:
        if d not in valid_datasets:
            parser.error(f"Unknown dataset: {d}. Valid: {valid_datasets}")

    if "leanworkbook" in datasets and args.split == "test":
        raise ValueError("leanworkbook does not have a test split")

    split_suffix = "-test" if args.split == "test" else ""

    # Init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0

    # Check for existing results early (before loading model)
    checkpoint_dir, step = parse_checkpoint_path(args.model_path)
    checkpoint_info = CheckpointInfo(checkpoint_dir=checkpoint_dir, step=step, seed=args.seed)

    should_exit = False
    continue_data = {}

    if master_process:
        existing_results = []
        for dataset_name in datasets:
            eval_path = checkpoint_info.get_eval_path(dataset_name + split_suffix)
            if os.path.exists(eval_path):
                if os.path.getsize(eval_path) == 0:
                    os.remove(eval_path)
                else:
                    existing_results.append((dataset_name, eval_path))

        if args.continue_eval:
            if not existing_results:
                print0("Error: --continue requires existing results")
                should_exit = True
            else:
                for dataset_name, eval_path in existing_results:
                    successful, errors = load_existing_eval_results(eval_path)
                    error_theorems = [e["theorem"] for e in errors]
                    continue_data[dataset_name] = (successful, error_theorems)
                    if error_theorems:
                        print0(f"Found {len(errors)} error entries to retry in {dataset_name}")
        elif existing_results and not args.force:
            print0("Evaluation results already exist:")
            for _, path in existing_results:
                print0(f"  {path}")
            print0("\nUse --force to overwrite, or --continue to retry errors.")
            should_exit = True

    if ddp:
        exit_tensor = torch.tensor([1 if should_exit else 0], device=device)
        torch.distributed.broadcast(exit_tensor, src=0)
        should_exit = exit_tensor.item() == 1

    if should_exit:
        compute_cleanup()
        sys.exit(1)

    # Load model
    print0(f"Loading checkpoint: {checkpoint_dir}, step={step}")
    model, tokenizer, _ = load_model(args.model_path, device, phase="eval")
    engine = Engine(model, tokenizer)
    inner_tactic_model = TacticModel(model, tokenizer, engine, num_samples=6, seed=args.seed)
    tactic_model = BlockingTacticModel(inner_model=inner_tactic_model, timeout_seconds=0.2, max_batch_tokens=8000)

    # Query Lean servers
    lean_server_addrs = [s.strip() for s in args.lean_servers.split(",")]
    lean_server_info = query_lean_servers(lean_server_addrs)
    lean_assignments = assign_lean_servers(lean_server_info)
    total_lean_procs = len(lean_assignments)
    print0(f"Lean servers: {lean_server_addrs} → {total_lean_procs} processes")

    # Build inference balancer + prover (same pattern as rl.py)
    if not master_process:
        inference_port = args.inference_server_port + 1 + ddp_rank
        start_inference_server(tactic_model, inference_port)

    prover: Prover | None = None
    if master_process:
        remote_endpoints = [
            f"127.0.0.1:{args.inference_server_port + 1 + r}"
            for r in range(1, ddp_world_size)
        ]
        balancer = InferenceBalancer(tactic_model, remote_endpoints)
        config = SearchConfig(num_simulations=args.num_simulations)
        prover = Prover(
            config=config,
            tactic_model=balancer,
            lean_servers=lean_assignments,
            num_simulations_eval=args.num_simulations,
        )

    if ddp:
        torch.distributed.barrier()

    atexit.register(lambda: prover.shutdown() if prover else None)

    # Load theorems
    dataset_theorems = {}
    if args.continue_eval:
        for dataset_name, (successful, error_theorems) in continue_data.items():
            if error_theorems:
                dataset_theorems[dataset_name] = error_theorems
    else:
        if "minif2f" in datasets:
            dataset_theorems["minif2f"] = minif2f.list_theorems(split=args.split)
        if "leanworkbook" in datasets:
            dataset_theorems["leanworkbook"] = leanworkbook.list_theorems(split="valid")
        if args.max_theorems:
            for name in dataset_theorems:
                dataset_theorems[name] = dataset_theorems[name][:args.max_theorems]

    print0(f"Evaluating with {args.num_simulations} MCTS simulations")

    # Run evaluation
    all_results = {}
    for dataset_name, theorems in dataset_theorems.items():
        print0(f"\nEvaluating on {len(theorems)} theorems from {dataset_name}")

        if master_process:
            results = prover.evaluate(theorems, dataset_name=dataset_name)
            all_results[dataset_name] = results
            print_results(results, dataset_name)

            prepend = continue_data.get(dataset_name, (None, None))[0] if args.continue_eval else None
            save_eval_results(checkpoint_info, dataset_name + split_suffix, results, prepend_entries=prepend)

    compute_cleanup()
    return all_results


if __name__ == "__main__":
    main()
