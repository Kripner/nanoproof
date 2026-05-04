"""
Standalone prover evaluation.

Uses the same Prover + InferenceBalancer as the RL training loop.

Usage:
    python scripts/prover_eval.py \\
        --model-path sft/.../model_005000.pt \\
        --lean-servers 10.10.25.33:8000 \\
        --datasets minif2f,proofnet

Multiple --model-path values are evaluated sequentially, sharing the
distributed compute init and dataset loading; results are saved after
each (model, dataset) pair.
"""

import argparse
import atexit
import collections
import gc
import logging
import math
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
from nanoproof.common import (
    active_barrier,
    add_dataclass_args,
    autodetect_device_type,
    broadcast_value,
    compute_cleanup,
    compute_init,
    dataclass_from_args,
    enable_memory_profiling,
    print0,
)
from nanoproof.data.bench import minif2f, proofnet
from nanoproof.data.bench.common import BenchTheorem
from nanoproof.data.check_init import read_lean_version, resolve_lean_project
from nanoproof.data.rl import leanworkbook
from nanoproof.inference import (
    BlockingTacticModel,
    TacticModel,
    compute_max_batch_prompt_tokens,
)
from nanoproof.prover import ProverWorker
from nanoproof.search import SearchConfig
from nanoproof.inference import setup_distributed_inference

# TODO: during verification, maybe set 'set_option maxHeartbeats 0\nset_option maxRecDepth 100000'


def merge_continue_results(results: dict, prepend_entries: list[dict] | None) -> dict:
    """Fold previously-successful (--continue) entries back into the retry results.

    The live evaluator only saw the retried (formerly errored) theorems, so
    its summary reflects just that subset. To produce a summary over the full
    benchmark we materialize the prepended successes into the same shape that
    ``compute_success_rate_by_simulations`` and ``print_results`` consume —
    notably normalizing the on-disk ``proof`` key to the live ``proof_tree``
    key so the threshold breakdown picks them up.

    Returns the original ``results`` unchanged when there are no prepend
    entries (non-continue runs and continue runs with no past successes).
    """
    if not prepend_entries:
        return results
    detailed = list(results.get("detailed_results", []))
    for e in prepend_entries:
        detailed.append(
            {
                "proof_tree": e.get("proof"),
                "num_iterations": e.get("num_iterations", 0),
                "error": e.get("error"),
            }
        )
    total = len(detailed)
    solved = sum(
        1 for r in detailed if r.get("proof_tree") is not None and not r.get("error")
    )
    errors = sum(1 for r in detailed if r.get("error"))
    return {
        "success_rate": solved / total if total > 0 else 0.0,
        "solved": solved,
        "total": total,
        "errors": errors,
        "detailed_results": detailed,
    }


def compute_success_rate_by_simulations(results, num_simulations):
    """Return ``{threshold: success_rate}`` for thresholds <= num_simulations.

    A theorem counts as solved at threshold ``t`` iff it has a proof tree and
    was solved within ``t`` iterations.
    """
    detailed = results.get("detailed_results", [])
    if not detailed:
        return {}
    total = len(detailed)
    thresholds = [
        t for t in [8, 16, 32, 64, 128, 256, 512, 1024, 2048] if t <= num_simulations
    ]
    breakdown = {}
    for t in thresholds:
        solved_at_t = sum(
            1
            for item in detailed
            if item.get("proof_tree") is not None
            and item.get("num_iterations", 0) <= t
        )
        breakdown[t] = solved_at_t / total if total > 0 else 0.0
    return breakdown


def binary_search_order(n: int) -> list[int]:
    """Indices [0, n) emitted as middle, then midpoints of remaining halves.

    Used to order checkpoint sweeps so an interrupted run still gives even
    step coverage — first the middle checkpoint, then the 1/4 and 3/4
    points, then 1/8, 3/8, 5/8, 7/8, etc.
    """
    if n <= 0:
        return []
    result = []
    queue = collections.deque([(0, n - 1)])
    while queue:
        lo, hi = queue.popleft()
        mid = (lo + hi) // 2
        result.append(mid)
        if lo <= mid - 1:
            queue.append((lo, mid - 1))
        if mid + 1 <= hi:
            queue.append((mid + 1, hi))
    return result


def resolve_run_dir_models(run_dir: str) -> list[str]:
    """List ``model_*.pt`` checkpoints in ``run_dir``, ordered for sweep.

    Order is: last checkpoint first (most-trained, the headline number),
    then the first (untrained baseline), then the interior checkpoints in
    binary-search order (middle, quarters, eighths, ...). An interrupted
    sweep still pins both endpoints and progressively fills the interior.
    """
    if not os.path.isdir(run_dir):
        raise ValueError(f"--run-dir {run_dir!r} is not a directory")
    by_step = []
    for name in os.listdir(run_dir):
        full = os.path.join(run_dir, name)
        if not os.path.isfile(full):
            continue
        try:
            _, step = parse_checkpoint_path(full)
        except ValueError:
            continue
        by_step.append((step, full))
    if not by_step:
        raise ValueError(f"No model_*.pt checkpoints found in {run_dir}")
    by_step.sort()
    sorted_paths = [p for _, p in by_step]
    n = len(sorted_paths)
    if n <= 2:
        return list(reversed(sorted_paths))
    interior = [sorted_paths[i + 1] for i in binary_search_order(n - 2)]
    return [sorted_paths[-1], sorted_paths[0], *interior]


def print_results(results, name, breakdown):
    print0("-" * 80)
    print0(f"Evaluation results for {name}")
    print0(f"Success rate: {results['success_rate']:.4%}")
    print0(f"Solved: {results['solved']}/{results['total']}")
    print0(f"Errors: {results['errors']}/{results['total']}")

    if breakdown:
        rates = [f"{t:>3}: {rate:.2%}" for t, rate in breakdown.items()]
        print0("Success rate by simulation budget:")
        print0("  " + "  |  ".join(rates))

    print0("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a prover model on theorem proving benchmarks",
        allow_abbrev=False,
    )

    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument(
        "--model-path",
        type=str,
        nargs="+",
        help="one or more model_NNNNNN.pt files (absolute or relative to "
        "$NANOPROOF_HOME/models/). Multiple paths are evaluated in order.",
    )
    model_source.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="run directory containing model_*.pt checkpoints; expanded to "
        "all checkpoints in binary-search order (middle, 1/4, 3/4, 1/8, ...) "
        "so an interrupted sweep still has even step coverage.",
    )
    parser.add_argument(
        "--lean-project",
        type=str,
        default=None,
        help="Path to the Lean project directory (contains lean-toolchain). The Lean version is read from this file and used to select per-dataset whitelists. Falls back to $LEAN_PROJECT_PATH if unset.",
    )
    parser.add_argument(
        "--lean-servers",
        type=str,
        nargs="+",
        required=True,
        help="Lean server addresses (e.g., 10.10.25.33:8000 10.10.25.34); port defaults to 8000",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="minif2f",
        help="comma-separated datasets (minif2f, leanworkbook, proofnet)",
    )
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--max-theorems", type=int, default=None)
    parser.add_argument("--num-simulations", type=int, default=512)
    parser.add_argument("--num-sampled-tactics", type=int, default=6)
    parser.add_argument(
        "--first-token-occurrences-cap",
        type=lambda s: None if s.lower() == "none" else int(s),
        default=2,
        help="cap on how many sampled tactics may share the same first token "
        "(per state). Pass 'none' to disable the cap.",
    )
    parser.add_argument(
        "--max-gen-tokens",
        type=int,
        default=24,
        help="hard cap on tokens generated per tactic sample.",
    )
    parser.add_argument("--batch-time-limit", type=float, default=0.5)
    parser.add_argument(
        "--batch-max-gen-samples",
        type=int,
        default=None,
        help="max generation samples per batch (default: num_actors * num_sampled_tactics)",
    )
    parser.add_argument(
        "--batch-max-prompt-tokens",
        type=int,
        default=None,
        help="max estimated prompt tokens per batch (default: auto from VRAM)",
    )
    parser.add_argument(
        "--memory-profile",
        type=str,
        default=None,
        help="if set, record CUDA memory history and dump snapshot to this dir on first OOM",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="extra suffix appended to the eval output directory name, e.g. "
        "'_cap2' produces eval_<step>_<dataset>_cap2/. Useful to keep variant "
        "runs from colliding with the baseline directory.",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing results"
    )
    parser.add_argument(
        "--continue",
        dest="continue_eval",
        action="store_true",
        help="retry only theorems that failed with errors",
    )
    parser.add_argument("--inference-server-port", type=int, default=5000)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="enable debug logging for inference and proving",
    )
    parser.add_argument(
        "--disable-solvers",
        action="store_true",
        help="Filter {grind, lia, grobner, aesop} from model generation; "
        "during eval, `grind` is still tried at every node expansion via "
        "synthetic injection.",
    )
    # Search hyperparameters. Eval theorems are harder to verify than
    # collection theorems, so we override verify_timeout to 30s here.
    add_dataclass_args(
        parser, SearchConfig, prefix="search_", overrides={"verify_timeout": 30000}
    )
    args = parser.parse_args()

    if args.run_dir is not None:
        args.model_path = resolve_run_dir_models(args.run_dir)
        print0(f"Resolved {len(args.model_path)} checkpoints from {args.run_dir}:")
        for p in args.model_path:
            print0(f"  {p}")

    args_dict = vars(args).copy()

    if args.verbose:
        logging.getLogger("nanoproof").setLevel(logging.DEBUG)

    search_config = dataclass_from_args(SearchConfig, args, prefix="search_")

    if args.force and args.continue_eval:
        parser.error("--force and --continue are mutually exclusive")

    datasets = [d.strip().lower() for d in args.datasets.split(",")]
    valid_datasets = {"minif2f", "leanworkbook", "proofnet"}
    for d in datasets:
        if d not in valid_datasets:
            parser.error(f"Unknown dataset: {d}. Valid: {valid_datasets}")

    if "leanworkbook" in datasets and args.split == "test":
        raise ValueError("leanworkbook does not have a test split")

    split_suffix = "-test" if args.split == "test" else ""
    output_suffix = args.output_suffix

    # Init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, _, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0

    # Enable memory profiling before any model load so weight allocations are captured.
    if args.memory_profile:
        enable_memory_profiling(args.memory_profile)

    # Resolve lean project + load full dataset theorem lists once; reused
    # across all model evaluations (continue mode rebuilds these per-model
    # from the existing results files instead).
    full_dataset_theorems = {}
    if not args.continue_eval:
        args.lean_project = resolve_lean_project(args.lean_project)
        lean_version = read_lean_version(args.lean_project)
        print0(
            f"Lean version: {lean_version} (from {args.lean_project}/lean-toolchain)"
        )
        if "minif2f" in datasets:
            full_dataset_theorems["minif2f"] = minif2f.list_theorems(split=args.split)
        if "leanworkbook" in datasets:
            full_dataset_theorems["leanworkbook"] = leanworkbook.list_theorems(
                split="valid", lean_version=lean_version
            )
        if "proofnet" in datasets:
            full_dataset_theorems["proofnet"] = proofnet.list_theorems(
                split=args.split
            )
        if args.max_theorems:
            for name in full_dataset_theorems:
                full_dataset_theorems[name] = full_dataset_theorems[name][
                    : args.max_theorems
                ]

    print0(f"Evaluating with {args.num_simulations} MCTS simulations")

    # Mutable holder so a single atexit handler can shut down whichever
    # model's resources are live when the process exits.
    current = {"tactic_model": None, "prover": None}

    def _cleanup_current():
        # Shutdown inference first so sample_tactic waiters unblock before
        # prover.close() tries to join actor threads.
        tm = current["tactic_model"]
        pr = current["prover"]
        if tm is not None:
            tm.shutdown()
            # Flask daemon threads from start_inference_server still hold a
            # closure on tm. Drop the inner TacticModel reference so the GPU
            # network/Engine become collectible once the surrounding scope
            # releases its locals; otherwise the next model loads on top.
            tm.inner_model = None
        if pr is not None:
            pr.close()
        current["tactic_model"] = None
        current["prover"] = None

    atexit.register(_cleanup_current)

    all_results_by_model = {}
    n_models = len(args.model_path)
    any_evaluated = False

    for model_idx, model_path in enumerate(args.model_path):
        # Per-iteration unique tag so distributed-store barrier counters
        # from previous iterations don't get reused.
        tag = f"m{model_idx}"

        # Each iteration's Flask servers bind fresh ports — daemon threads
        # from prior iterations stay running but receive no traffic.
        port_base = args.inference_server_port + model_idx * ddp_world_size

        if n_models > 1:
            print0(
                f"\n{'=' * 80}\n"
                f"Model {model_idx + 1}/{n_models}: {model_path}\n"
                f"{'=' * 80}"
            )

        # Per-model copy so saved results attribute to this checkpoint, not
        # the full --model-path list.
        model_args_dict = dict(args_dict, model_path=model_path)

        # Check for existing results early (before loading the model)
        checkpoint_info = CheckpointInfo(
            *parse_checkpoint_path(model_path), seed=args.seed
        )

        should_skip = False
        continue_data = {}

        if master_process:
            existing_results = []
            for dataset_name in datasets:
                eval_dir = checkpoint_info.get_eval_dir(
                    dataset_name + split_suffix + output_suffix
                )
                theorems_path = os.path.join(eval_dir, "theorems.jsonl")
                if os.path.exists(theorems_path):
                    if os.path.getsize(theorems_path) == 0:
                        os.remove(theorems_path)
                    else:
                        existing_results.append(
                            (dataset_name, eval_dir, theorems_path)
                        )

            if args.continue_eval:
                if not existing_results:
                    print0(
                        f"Skipping {model_path}: --continue requires existing results"
                    )
                    should_skip = True
                else:
                    for dataset_name, _, theorems_path in existing_results:
                        successful, errors = load_existing_eval_results(theorems_path)
                        error_theorems = [
                            BenchTheorem(
                                source=e["theorem"],
                                dataset=e["dataset"],
                                id=e["id"],
                            )
                            for e in errors
                        ]
                        continue_data[dataset_name] = (successful, error_theorems)
                        if error_theorems:
                            print0(
                                f"Found {len(errors)} error entries to retry in {dataset_name}"
                            )
            elif existing_results and not args.force:
                # In --run-dir sweep mode, treat existing-with-errors as an
                # implicit --continue: retry just the errored theorems and
                # merge with the prior successes. Skip only if all clean.
                if args.run_dir is not None:
                    for dataset_name, _, theorems_path in existing_results:
                        successful, errors = load_existing_eval_results(theorems_path)
                        if not errors:
                            continue
                        error_theorems = [
                            BenchTheorem(
                                source=e["theorem"],
                                dataset=e["dataset"],
                                id=e["id"],
                            )
                            for e in errors
                        ]
                        continue_data[dataset_name] = (successful, error_theorems)
                        print0(
                            f"Found {len(errors)} error entries to retry in {dataset_name}"
                        )
                if continue_data:
                    print0(f"Retrying errored entries in {model_path}")
                else:
                    print0("Evaluation results already exist:")
                    for _, eval_dir, _ in existing_results:
                        print0(f"  {eval_dir}")
                    print0(
                        f"Skipping {model_path}. Use --force to overwrite, or --continue to retry errors."
                    )
                    should_skip = True

        if ddp:
            skip_tensor = torch.tensor([1 if should_skip else 0], device=device)
            dist.broadcast(skip_tensor, src=0)
            should_skip = skip_tensor.item() == 1

        if should_skip:
            # All ranks must hit the per-iteration barriers in lockstep,
            # otherwise the next model's barriers see leftover counts.
            active_barrier(f"inference_ready_{tag}")
            active_barrier(f"prover_eval_done_{tag}", timeout=None)
            continue

        any_evaluated = True

        # Load model + set up inference
        print0(
            f"Loading checkpoint: {checkpoint_info.checkpoint_dir}, step={checkpoint_info.step}"
        )
        inner_tactic_model = TacticModel.create(
            num_samples=args.num_sampled_tactics,
            model_path=model_path,
            first_token_occurrences_cap=args.first_token_occurrences_cap,
            max_gen_tokens=args.max_gen_tokens,
            disable_solvers=args.disable_solvers,
        )
        # Defer max_gen_samples default until we know num_actors
        tactic_model = BlockingTacticModel(
            inner_model=inner_tactic_model,
            timeout_seconds=args.batch_time_limit,
            max_gen_samples=None,
        )
        current["tactic_model"] = tactic_model

        balancer = setup_distributed_inference(tactic_model, port_base)
        if balancer:
            prover = ProverWorker(balancer, args.lean_servers)
            # Per-GPU capacity: the busy-aware balancer funnels actors onto one
            # GPU until it flips busy, so sizing aggregate would starve the rest.
            max_gen_samples = args.batch_max_gen_samples or math.ceil(
                prover.num_actors * args.num_sampled_tactics / ddp_world_size
            )
            tactic_model.max_gen_samples = max_gen_samples
            print0(
                f"Batch max gen samples: {max_gen_samples} ({prover.num_actors} actors * {args.num_sampled_tactics} samples / {ddp_world_size} ranks)"
            )
        else:
            prover = None
        current["prover"] = prover

        # Prompt token limit for inference batches (prevents OOM on long prompts)
        max_prompt_tokens = args.batch_max_prompt_tokens
        if max_prompt_tokens is None:
            max_prompt_tokens = compute_max_batch_prompt_tokens(
                inner_tactic_model.network.config, args.num_sampled_tactics, device
            )
            print0(
                f"Batch max prompt tokens: {max_prompt_tokens} (auto from {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GiB VRAM, {torch.cuda.memory_allocated(device) / 1024**3:.1f} GiB used)"
            )
        else:
            print0(f"Batch max prompt tokens: {max_prompt_tokens} (manual)")
        tactic_model.max_batch_prompt_tokens = max_prompt_tokens

        # Broadcast from master to worker ranks so their Flask servers can batch correctly.
        if ddp:
            tactic_model.max_gen_samples = broadcast_value(tactic_model.max_gen_samples)
            tactic_model.max_batch_prompt_tokens = broadcast_value(
                tactic_model.max_batch_prompt_tokens
            )

        active_barrier(f"inference_ready_{tag}")

        # Build per-model dataset list. Continue mode (explicit --continue or
        # auto-continue from --run-dir) filters to errored theorems only;
        # continue_data is populated only on master, so workers fall through
        # to full_dataset_theorems but never iterate it.
        if args.continue_eval or continue_data:
            dataset_theorems = {}
            for dataset_name, (_, error_theorems) in continue_data.items():
                if error_theorems:
                    dataset_theorems[dataset_name] = error_theorems
        else:
            dataset_theorems = full_dataset_theorems

        # Evaluate (rank 0 only; worker ranks serve inference via daemon threads)
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
                            print0(
                                f"  started={s}/{total}  finished={f}/{total}  solved={ok}  errors={err}"
                            )

                printer = threading.Thread(target=printer_loop, daemon=True)
                printer.start()

                dataset_start = time.monotonic()
                results = prover.evaluate(
                    theorems,
                    dataset_name=dataset_name,
                    num_simulations=args.num_simulations,
                    search_config=search_config,
                    progress_callback=progress_callback,
                    disable_solvers=args.disable_solvers,
                )
                dataset_elapsed = time.monotonic() - dataset_start
                done.set()
                printer.join()

                prepend = (
                    continue_data.get(dataset_name, (None, None))[0]
                    if (args.continue_eval or continue_data)
                    else None
                )
                # Fold previously-successful entries into the metrics so the
                # printed numbers and summary.toml reflect the full benchmark,
                # not just the retry batch.
                merged = merge_continue_results(results, prepend)
                all_results[dataset_name] = merged
                breakdown = compute_success_rate_by_simulations(
                    merged, args.num_simulations
                )
                print_results(merged, dataset_name, breakdown)
                print0(f"Time for {dataset_name}: {dataset_elapsed:.1f}s")

                summary = {
                    "dataset": dataset_name,
                    "split": args.split,
                    "num_simulations": args.num_simulations,
                    "total": merged["total"],
                    "solved": merged["solved"],
                    "errors": merged["errors"],
                    "success_rate": merged["success_rate"],
                    "elapsed_seconds": dataset_elapsed,
                    "success_rate_by_simulations": breakdown,
                }
                # Pass the live (retry-only) results to the JSONL writer; it
                # prepends the saved successes itself, so passing `merged`
                # would duplicate those entries.
                save_eval_results(
                    checkpoint_info,
                    dataset_name + split_suffix + output_suffix,
                    results,
                    summary,
                    model_args_dict,
                    prepend_entries=prepend,
                )

            total_elapsed = time.monotonic() - eval_start
            print0(f"\nTotal evaluation time for {model_path}: {total_elapsed:.1f}s")

        all_results_by_model[model_path] = all_results

        # Workers are blocked here for the entire master-side evaluation,
        # which can take many minutes; no timeout (use SIGUSR1 to debug).
        active_barrier(f"prover_eval_done_{tag}", timeout=None)

        # Tear down this model's inference + actors before loading the next.
        _cleanup_current()
        del inner_tactic_model, tactic_model, balancer, prover
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    compute_cleanup()
    if not any_evaluated:
        sys.exit(1)
    return all_results_by_model


if __name__ == "__main__":
    main()
