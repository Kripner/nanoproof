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
import logging
import os
import sys
import threading
import time

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist

from nanoproof.checkpoints import (
    CheckpointInfo,
    load_existing_eval_results,
    parse_checkpoint_path,
    save_eval_results,
)
from nanoproof.common import active_barrier_master, active_barrier_wait, autodetect_device_type, broadcast_value, compute_cleanup, compute_init, print0
from nanoproof.data.bench import minif2f
from nanoproof.data.rl import leanworkbook
from nanoproof.inference import BlockingTacticModel, TacticModel, compute_max_batch_prompt_tokens
from nanoproof.prover import ProverWorker
from nanoproof.inference import setup_distributed_inference

# TODO: during verification, maybe set 'set_option maxHeartbeats 0\nset_option maxRecDepth 100000'


def print_results(results, name, num_simulations):
    print0("-" * 80)
    print0(f"Evaluation results for {name}")
    print0(f"Success rate: {results['success_rate']:.4%}")
    print0(f"Solved: {results['solved']}/{results['total']}")
    print0(f"Errors: {results['errors']}/{results['total']}")

    detailed = results.get('detailed_results', [])
    if detailed:
        total = len(detailed)
        thresholds = [t for t in [8, 16, 32, 64, 128, 256, 512, 1024, 2048] if t <= num_simulations]
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
    parser.add_argument("--num-sampled-tactics", type=int, default=6)
    parser.add_argument("--batch-time-limit", type=float, default=0.5)
    parser.add_argument("--batch-max-gen-samples", type=int, default=None,
                        help="max generation samples per batch (default: num_actors * num_sampled_tactics)")
    parser.add_argument("--batch-max-prompt-tokens", type=int, default=None,
                        help="max estimated prompt tokens per batch (default: auto from VRAM)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite existing results")
    parser.add_argument("--continue", dest="continue_eval", action="store_true",
                        help="retry only theorems that failed with errors")
    parser.add_argument("--inference-server-port", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true", help="enable debug logging for inference and proving")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("nanoproof").setLevel(logging.DEBUG)

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
    inner_tactic_model = TacticModel.create(num_samples=args.num_sampled_tactics, model_path=args.model_path)
    # Defer max_gen_samples default until we know num_actors
    tactic_model = BlockingTacticModel(inner_model=inner_tactic_model, timeout_seconds=args.batch_time_limit, max_gen_samples=None)

    balancer = setup_distributed_inference(tactic_model, args.inference_server_port)
    if balancer:
        prover = ProverWorker(balancer, args.lean_servers)
        max_gen_samples = args.batch_max_gen_samples or prover.num_actors * args.num_sampled_tactics
        tactic_model.max_gen_samples = max_gen_samples
        print0(f"Batch max gen samples: {max_gen_samples} ({prover.num_actors} actors * {args.num_sampled_tactics} samples)")
    else:
        prover = None

    # Prompt token limit for inference batches (prevents OOM on long prompts)
    max_prompt_tokens = args.batch_max_prompt_tokens
    if max_prompt_tokens is None:
        max_prompt_tokens = compute_max_batch_prompt_tokens(inner_tactic_model.network.config, args.num_sampled_tactics, device)
        print0(f"Batch max prompt tokens: {max_prompt_tokens} (auto from {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GiB VRAM, {torch.cuda.memory_allocated(device) / 1024**3:.1f} GiB used)")
    else:
        print0(f"Batch max prompt tokens: {max_prompt_tokens} (manual)")
    tactic_model.max_batch_prompt_tokens = max_prompt_tokens

    # Broadcast from master to worker ranks so their Flask servers can batch correctly.
    if ddp:
        tactic_model.max_gen_samples = broadcast_value(tactic_model.max_gen_samples)
        tactic_model.max_batch_prompt_tokens = broadcast_value(tactic_model.max_batch_prompt_tokens)

    if ddp:
        active_barrier_master("inference_ready") if master_process else active_barrier_wait("inference_ready")

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
        eval_start = time.monotonic()
        for dataset_name, theorems in dataset_theorems.items():
            print0(f"\nEvaluating on {len(theorems)} theorems from {dataset_name}")
            total = len(theorems)
            latest = [0, 0, 0, 0]  # started, finished, solved, errors
            printed = list(latest)
            lock = threading.Lock()
            done = threading.Event()

            def progress_callback(started, finished, solved, errors):
                with lock:
                    latest[:] = [started, finished, solved, errors]

            def printer_loop():
                while not done.wait(timeout=1.0):
                    with lock:
                        snap = list(latest)
                    if snap != printed:
                        printed[:] = snap
                        s, f, ok, err = snap
                        print0(f"  started={s}/{total}  finished={f}/{total}  solved={ok}  errors={err}")

            printer = threading.Thread(target=printer_loop, daemon=True)
            printer.start()

            dataset_start = time.monotonic()
            results = prover.evaluate(theorems, dataset_name=dataset_name, num_simulations=args.num_simulations,
                                      progress_callback=progress_callback)
            dataset_elapsed = time.monotonic() - dataset_start
            done.set()
            printer.join()
            all_results[dataset_name] = results
            print_results(results, dataset_name, args.num_simulations)
            print0(f"Time for {dataset_name}: {dataset_elapsed:.1f}s")

            prepend = continue_data.get(dataset_name, (None, None))[0] if args.continue_eval else None
            save_eval_results(checkpoint_info, dataset_name + split_suffix, results, prepend_entries=prepend)

        total_elapsed = time.monotonic() - eval_start
        print0(f"\nTotal evaluation time: {total_elapsed:.1f}s")

        active_barrier_master("prover_eval_done")
    else:
        active_barrier_wait("prover_eval_done")

    compute_cleanup()
    return all_results


if __name__ == "__main__":
    main()
