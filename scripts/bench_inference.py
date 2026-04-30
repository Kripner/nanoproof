"""
Benchmark single-GPU tactic-generation throughput of the nanoproof Engine.

Cycles through states from a generated_tactics.jsonl file, tokenizes them as
tactic prompts (matching nanoproof.inference.TacticModel), and repeatedly
calls engine.generate_batch with a fixed batch size.

Usage:
    python scripts/bench_inference.py \\
        --input path/to/generated_tactics.jsonl \\
        --model-path sft/.../model_005000.pt
"""

import argparse
import json
import os
import time

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch

from nanoproof.common import autodetect_device_type, compute_cleanup, compute_init
from nanoproof.inference import TacticModel


def load_states(path: str) -> list[tuple[str, list[str]]]:
    """Load (state_str, recorded_tactics) rows from a generated_tactics.jsonl.

    The recorded tactics are kept so the bench can derive a realistic
    per-batch decode length from the empirical output distribution.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            state = obj["state"].replace("\\n", "\n")
            tactics = [t["tactic"] for t in obj.get("tactics", [])]
            rows.append((state, tactics))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark tactic-generation throughput on one GPU",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a generated_tactics.jsonl file",
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Static batch size (number of prompts per batch)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="Number of tactic samples generated per prompt",
    )
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--benchmark-seconds", type=float, default=30.0)
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=512,
        help="Truncate prompts longer than this many tokens (passed to TacticModel)",
    )
    parser.add_argument(
        "--max-gen-tokens",
        type=int,
        default=24,
        help="Hard cap on tokens generated per sample. Also acts as the per-batch ceiling when --gen-tokens is unset.",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=None,
        help="If set, fixed gen-tokens for every batch (sets min=max). If unset, derive per-batch max from the input file's recorded tactic lengths, capped at --max-gen-tokens.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device_type = autodetect_device_type()
    assert device_type == "cuda", "benchmark requires CUDA"
    _, _, _, _, device = compute_init(device_type)

    print(f"Loading states from {args.input}")
    raw_rows = load_states(args.input)
    print(f"Loaded {len(raw_rows)} states")
    assert len(raw_rows) > 0, "no states found"

    print(f"Loading model from {args.model_path}")
    model = TacticModel.create(
        num_samples=args.num_samples,
        model_path=args.model_path,
        seed=args.seed,
        max_prompt_len=args.max_prompt_len,
    )

    # Pre-tokenize every state and pre-compute its per-state max output length
    # from the recorded tactics (+1 for the EOS step the engine generates).
    # Capped at --max-gen-tokens so the bench mirrors production's hard ceiling.
    prompts = []
    per_prompt_max_gen = []
    for state, tactics in raw_rows:
        tokens = model.prepare_tactic_prompt(state)
        if tactics:
            empirical_max = max(len(model.tokenizer(t)) for t in tactics) + 1
        else:
            empirical_max = 1
        prompts.append(tokens)
        per_prompt_max_gen.append(min(empirical_max, args.max_gen_tokens))
    print(f"Usable prompts: {len(prompts)} / {len(raw_rows)}")
    assert len(prompts) >= args.batch_size, (
        f"not enough usable prompts ({len(prompts)}) for batch size {args.batch_size}"
    )

    prompt_lens = [len(p) for p in prompts]
    print(
        f"Prompt length: min={min(prompt_lens)}, max={max(prompt_lens)}, "
        f"mean={sum(prompt_lens) / len(prompt_lens):.1f}"
    )
    print(
        f"Per-prompt gen tokens (capped at {args.max_gen_tokens}): "
        f"min={min(per_prompt_max_gen)}, max={max(per_prompt_max_gen)}, "
        f"mean={sum(per_prompt_max_gen) / len(per_prompt_max_gen):.1f}"
    )

    def run_one_batch(batch_prompts, batch_gen_tokens, seed):
        results, masks = model.engine.generate_batch(
            batch_prompts,
            num_samples=args.num_samples,
            min_tokens=batch_gen_tokens,
            max_tokens=batch_gen_tokens,
            temperature=args.temperature,
            seed=seed,
        )
        # results[prompt_idx][sample_idx] is the prompt + generated tokens (minus eos/bos).
        generated = 0
        for i, p in enumerate(batch_prompts):
            for s in range(args.num_samples):
                generated += len(results[i][s]) - len(p)
        return generated

    def run_phase(duration_s, label, cursor, seed_base):
        n_batches = 0
        n_generated = 0
        n_samples = 0
        latencies = []
        phase_start = time.perf_counter()
        while True:
            now = time.perf_counter()
            if now - phase_start >= duration_s:
                break
            batch = []
            batch_max_gens = []
            for _ in range(args.batch_size):
                idx = cursor % len(prompts)
                batch.append(prompts[idx])
                batch_max_gens.append(per_prompt_max_gen[idx])
                cursor += 1
            if args.gen_tokens is not None:
                batch_gen_tokens = args.gen_tokens
            else:
                batch_gen_tokens = max(batch_max_gens)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            gen = run_one_batch(batch, batch_gen_tokens, seed=seed_base + n_batches)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
            n_batches += 1
            n_generated += gen
            n_samples += args.batch_size * args.num_samples
        elapsed = time.perf_counter() - phase_start
        print(
            f"[{label}] {elapsed:.2f}s elapsed, {n_batches} batches, "
            f"{n_samples} samples, {n_generated} generated tokens"
        )
        return cursor, n_batches, n_samples, n_generated, latencies, elapsed

    if args.gen_tokens is not None:
        gen_tokens_desc = f"fixed={args.gen_tokens}"
    else:
        gen_tokens_desc = f"per-batch-max (cap={args.max_gen_tokens})"
    print(
        f"\nBatch size: {args.batch_size}, num_samples: {args.num_samples}, "
        f"max_prompt_len: {args.max_prompt_len}, gen_tokens: {gen_tokens_desc}, "
        f"temperature: {args.temperature}"
    )
    print(f"Warmup: {args.warmup_seconds}s, benchmark: {args.benchmark_seconds}s\n")

    cursor = 0
    cursor, *_ = run_phase(args.warmup_seconds, "warmup", cursor, seed_base=10**6)
    cursor, n_batches, n_samples, n_generated, latencies, elapsed = run_phase(
        args.benchmark_seconds, "bench", cursor, seed_base=args.seed
    )

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    mean_lat = sum(latencies) / len(latencies)

    print("\n== results ==")
    print(f"batches/sec : {n_batches / elapsed:.3f}")
    print(f"samples/sec : {n_samples / elapsed:.2f}")
    print(f"gen tok/sec : {n_generated / elapsed:.1f}")
    print(f"tok/sample  : {n_generated / n_samples:.2f}")
    print(
        f"batch latency: mean={mean_lat * 1000:.1f}ms  "
        f"p50={p50 * 1000:.1f}ms  p95={p95 * 1000:.1f}ms"
    )

    compute_cleanup()


if __name__ == "__main__":
    main()
