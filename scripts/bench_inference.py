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


def load_states(path: str) -> list[str]:
    states = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            state = obj["state"].replace("\\n", "\n")
            states.append(state)
    return states


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark tactic-generation throughput on one GPU",
        allow_abbrev=False,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to a generated_tactics.jsonl file")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Static batch size (number of prompts per batch)")
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--benchmark-seconds", type=float, default=30.0)
    parser.add_argument("--gen-tokens", type=int, default=8,
                        help="Tokens to generate per sample (sets both min_tokens and max_tokens so every batch generates the same amount)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device_type = autodetect_device_type()
    assert device_type == "cuda", "benchmark requires CUDA"
    _, _, _, _, device = compute_init(device_type)

    print(f"Loading states from {args.input}")
    raw_states = load_states(args.input)
    print(f"Loaded {len(raw_states)} states")
    assert len(raw_states) > 0, "no states found"

    print(f"Loading model from {args.model_path}")
    model = TacticModel.create(num_samples=1, model_path=args.model_path, seed=args.seed)

    # Pre-tokenize every state once; drop ones that exceed the prompt budget.
    prompts = []
    for state in raw_states:
        tokens, too_long = model.prepare_tactic_prompt(state)
        if too_long:
            continue
        prompts.append(tokens)
    print(f"Usable prompts: {len(prompts)} / {len(raw_states)}")
    assert len(prompts) >= args.batch_size, \
        f"not enough usable prompts ({len(prompts)}) for batch size {args.batch_size}"

    prompt_lens = [len(p) for p in prompts]
    print(f"Prompt length: min={min(prompt_lens)}, max={max(prompt_lens)}, "
          f"mean={sum(prompt_lens) / len(prompt_lens):.1f}")

    def run_one_batch(batch_prompts, seed):
        results, masks = model.engine.generate_batch(
            batch_prompts,
            num_samples=1,
            min_tokens=args.gen_tokens,
            max_tokens=args.gen_tokens,
            temperature=args.temperature,
            seed=seed,
        )
        # results[prompt_idx][sample_idx] is the prompt + generated tokens (minus eos/bos).
        generated = 0
        for i, p in enumerate(batch_prompts):
            generated += len(results[i][0]) - len(p)
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
            for _ in range(args.batch_size):
                batch.append(prompts[cursor % len(prompts)])
                cursor += 1
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            gen = run_one_batch(batch, seed=seed_base + n_batches)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
            n_batches += 1
            n_generated += gen
            n_samples += args.batch_size
        elapsed = time.perf_counter() - phase_start
        print(f"[{label}] {elapsed:.2f}s elapsed, {n_batches} batches, "
              f"{n_samples} samples, {n_generated} generated tokens")
        return cursor, n_batches, n_samples, n_generated, latencies, elapsed

    print(f"\nBatch size: {args.batch_size}, gen_tokens: {args.gen_tokens}, "
          f"temperature: {args.temperature}")
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
    print(f"batch latency: mean={mean_lat * 1000:.1f}ms  "
          f"p50={p50 * 1000:.1f}ms  p95={p95 * 1000:.1f}ms")

    compute_cleanup()


if __name__ == "__main__":
    main()
