"""
Standalone prover evaluation.

Uses the same Prover + InferenceBalancer as the RL training loop.

Usage:
    python scripts/prover_eval.py \\
        --model-path sft/.../model_005000.pt \\
        --lean-servers 10.10.25.33:8000 \\
        --datasets minif2f,leanworkbook
"""

import argparse
import atexit
import os
import sys

import torch
import torch.distributed as dist

from nanoproof.checkpoints import (
    CheckpointInfo,
    load_existing_eval_results,
    parse_checkpoint_path,
    save_eval_results,
)
from nanoproof.common import active_barrier_master, active_barrier_wait, autodetect_device_type, compute_cleanup, compute_init, print0
from nanoproof.data.bench import minif2f
from nanoproof.data.rl import leanworkbook
from nanoproof.inference import BlockingTacticModel, TacticModel
from nanoproof.prover import ProverWorker
from nanoproof.inference import setup_distributed_inference

# TODO: during verification, maybe set 'set_option maxHeartbeats 0\nset_option maxRecDepth 100000'


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a prover model on theorem proving benchmarks")

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--lean-servers", type=str, nargs="+", required=True,
                        help="Lean server addresses (e.g., 10.10.25.33:8000 10.10.25.34); port defaults to 8000")
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
    ddp, ddp_rank, _, _, device = compute_init(device_type)
    master_process = ddp_rank == 0

    # Check for existing results early (before loading model)
    checkpoint_info = CheckpointInfo(*parse_checkpoint_path(args.model_path), seed=args.seed)

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
        dist.broadcast(exit_tensor, src=0)
        should_exit = exit_tensor.item() == 1

    if should_exit:
        compute_cleanup()
        sys.exit(1)

    # Load model + set up inference
    print0(f"Loading checkpoint: {checkpoint_info.checkpoint_dir}, step={checkpoint_info.step}")
    inner_tactic_model = TacticModel.create(num_samples=6, model_path=args.model_path)
    tactic_model = BlockingTacticModel(inner_model=inner_tactic_model, timeout_seconds=0.2, max_batch_tokens=8000)

    balancer = setup_distributed_inference(tactic_model, args.inference_server_port)
    if balancer:
        prover = ProverWorker(balancer, args.lean_servers)
    else:
        prover = None

    if ddp:
        dist.barrier()

    atexit.register(lambda: tactic_model.shutdown())

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

    # Evaluate (rank 0 only; worker ranks serve inference via their daemon threads)
    all_results = {}
    if master_process:
        for dataset_name, theorems in dataset_theorems.items():
            print0(f"\nEvaluating on {len(theorems)} theorems from {dataset_name}")
            results = prover.evaluate(theorems, dataset_name=dataset_name, num_simulations=args.num_simulations)
            all_results[dataset_name] = results
            print_results(results, dataset_name)

            prepend = continue_data.get(dataset_name, (None, None))[0] if args.continue_eval else None
            save_eval_results(checkpoint_info, dataset_name + split_suffix, results, prepend_entries=prepend)

        active_barrier_master("prover_eval_done")
    else:
        active_barrier_wait("prover_eval_done")

    compute_cleanup()
    return all_results


if __name__ == "__main__":
    main()
