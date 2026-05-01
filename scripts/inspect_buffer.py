#!/usr/bin/env python3
"""Inspect RL replay-buffer artifacts (theorems.jsonl shards).

A run dir produced by `nanoproof rl ...` contains per-step shards at
``step_<N>/theorems.jsonl``. Each line is a serialized TheoremAttempt:

    {"dataset": str,
     "id": str,
     "theorem": str,                # source declaration
     "num_simulations": int,        # MCTS budget allocated by matchmaker
     "num_iterations": int,         # iterations actually run (0 on error)
     "outcome": "proven"|"unproven"|"error",
     "error": str|None,
     "full_tree": dict|None,        # pre-prune game tree (proven only)
     "simplified_tree": dict|None,  # post-prune game tree (proven only)
     "transitions": [[ctx, tac, value], ...],   # only on proven attempts
     "proof_size": int|None}        # number of tactics in linearized proof

`value` is the regression target written to the buffer: 0 at terminal
nodes, propagated up as -1+max at OR nodes and min at AND nodes. So for
a transition picked from a solved proof, value == -(remaining tactics
including this one).

Subcommands:
  stats     -- aggregate counts, value/proof-size distributions, top tactics
  view      -- browse individual transitions (state/tactic/value)
  attempts  -- list theorem-level outcomes (proven/unproven/error)
"""

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict


STEP_PREFIX = "step_"
THEOREMS_FILENAME = "theorems.jsonl"


def list_shards(path):
    """Return [(step_or_None, jsonl_path), ...] sorted by step.

    Accepts either a run directory (containing step_*/theorems.jsonl) or
    a single .jsonl file (returned with step=None).
    """
    if os.path.isfile(path):
        return [(None, path)]
    pattern = os.path.join(path, f"{STEP_PREFIX}*", THEOREMS_FILENAME)
    shards = []
    for shard_path in glob.glob(pattern):
        name = os.path.basename(os.path.dirname(shard_path))
        try:
            step = int(name[len(STEP_PREFIX):])
        except ValueError:
            continue
        shards.append((step, shard_path))
    if not shards:
        sys.exit(f"No shards found under {path!r} (looked for {pattern})")
    shards.sort(key=lambda s: s[0])
    return shards


def step_in_filter(shard_step, only_step, step_range):
    if shard_step is None:
        return True
    if only_step is not None and shard_step != only_step:
        return False
    if step_range is not None:
        lo, hi = step_range
        if shard_step < lo or shard_step > hi:
            return False
    return True


def iter_attempts(path, *, step=None, steps=None, dataset=None, outcome=None):
    for shard_step, shard_path in list_shards(path):
        if not step_in_filter(shard_step, step, steps):
            continue
        with open(shard_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a = json.loads(line)
                if dataset is not None and a.get("dataset") != dataset:
                    continue
                if outcome is not None and a.get("outcome") != outcome:
                    continue
                yield shard_step, a


def iter_transitions(path, **filters):
    for shard_step, a in iter_attempts(path, **filters):
        for t in a.get("transitions") or []:
            yield shard_step, a, (t[0], t[1], t[2])


def transition_to_theorem(context, tactic, value):
    """Render a (state, tactic, value) row as a faux Lean example block."""
    binders, goal = [], ""
    for raw in context.strip().split("\n"):
        line = raw.strip()
        if not line or line.startswith("case"):
            continue
        if line.startswith("⊢"):
            goal = line[1:].strip()
        else:
            binders.append(f"({line})")
    head = "example"
    if binders:
        head += " " + " ".join(binders)
    head += f" : {goal}"
    remaining_after = -int(value) - 1
    suffix = (
        f"\n  sorry  -- {remaining_after} more tactic(s) until QED"
        if remaining_after > 0
        else ""
    )
    return f"{head} := by\n  {tactic}{suffix}"


def fmt_value(v):
    iv = int(v)
    return str(iv) if iv == v else f"{v:.2f}"


def cmd_stats(args):
    by_step = defaultdict(Counter)
    by_dataset = defaultdict(Counter)
    state_lens, tactic_lens, values, proof_sizes = [], [], [], []
    tactic_heads = Counter()
    n_attempts = 0
    n_transitions = 0
    error_counts = Counter()

    for shard_step, a in iter_attempts(
        args.path, step=args.step, steps=args.steps, dataset=args.dataset
    ):
        n_attempts += 1
        outcome = a.get("outcome", "?")
        by_step[shard_step][outcome] += 1
        by_step[shard_step]["__total__"] += 1
        by_dataset[a.get("dataset", "?")][outcome] += 1
        by_dataset[a.get("dataset", "?")]["__total__"] += 1
        if outcome == "error" and a.get("error"):
            error_counts[a["error"].splitlines()[0][:80]] += 1
        if a.get("proof_size") is not None:
            proof_sizes.append(a["proof_size"])
        for ctx, tac, val in a.get("transitions") or []:
            n_transitions += 1
            state_lens.append(len(ctx))
            tactic_lens.append(len(tac))
            values.append(val)
            head = tac.split()[0] if tac.strip() else "<empty>"
            tactic_heads[head] += 1

    if n_attempts == 0:
        print("No matching attempts.")
        return

    print(f"Path: {args.path}")
    if args.step is not None:
        print(f"Step: {args.step}")
    elif args.steps is not None:
        print(f"Steps: {args.steps[0]}..{args.steps[1]}")
    if args.dataset:
        print(f"Dataset: {args.dataset}")
    print()

    outcomes_total = Counter()
    for s in by_step.values():
        for k, v in s.items():
            if k != "__total__":
                outcomes_total[k] += v
    print(f"Attempts: {n_attempts:,}")
    for o in ("proven", "unproven", "error"):
        c = outcomes_total.get(o, 0)
        pct = 100 * c / n_attempts if n_attempts else 0
        print(f"  {o:>8}: {c:>7,}  ({pct:5.1f}%)")
    print(f"Transitions: {n_transitions:,}")
    if n_attempts:
        print(f"  per attempt (all):    {n_transitions / n_attempts:.2f}")
    if outcomes_total.get("proven", 0):
        print(f"  per proven attempt:   {n_transitions / outcomes_total['proven']:.2f}")
    print()

    if proof_sizes:
        ps_counts = Counter(proof_sizes)
        print("Proof size (proven attempts):")
        for size in sorted(ps_counts):
            c = ps_counts[size]
            pct = 100 * c / len(proof_sizes)
            print(f"  size={size:>3}: {c:>6,}  ({pct:5.1f}%)")
        print(f"  mean: {sum(proof_sizes) / len(proof_sizes):.2f}")
        print()

    if values:
        v_counts = Counter(int(v) for v in values)
        print("Value-target distribution (-N = N tactics until QED, including this one):")
        for val in sorted(v_counts):
            c = v_counts[val]
            pct = 100 * c / len(values)
            print(f"  {val:>4}: {c:>6,}  ({pct:5.1f}%)")
        print(f"  mean: {sum(values) / len(values):.4f}")
        print()

    if state_lens:
        print(
            "Lengths (chars): "
            f"state mean={sum(state_lens) / len(state_lens):.0f} "
            f"max={max(state_lens)}; "
            f"tactic mean={sum(tactic_lens) / len(tactic_lens):.0f} "
            f"max={max(tactic_lens)}"
        )
        print()

    if any(s is not None for s in by_step) and len(by_step) > 1:
        print("Per-step:")
        print(f"  {'step':>5}  {'total':>6}  {'proven':>6}  {'unproven':>8}  {'error':>5}  {'proven%':>7}")
        for s in sorted(by_step):
            row = by_step[s]
            tot = row["__total__"]
            prov = row.get("proven", 0)
            pct = 100 * prov / tot if tot else 0
            label = "?" if s is None else str(s)
            print(
                f"  {label:>5}  {tot:>6,}  {prov:>6,}  "
                f"{row.get('unproven', 0):>8,}  {row.get('error', 0):>5,}  {pct:>6.1f}%"
            )
        print()

    if len(by_dataset) > 1:
        print("Per-dataset:")
        for d in sorted(by_dataset):
            row = by_dataset[d]
            tot = row["__total__"]
            prov = row.get("proven", 0)
            pct = 100 * prov / tot if tot else 0
            print(
                f"  {d}: total={tot:,} proven={prov:,} "
                f"unproven={row.get('unproven', 0):,} "
                f"error={row.get('error', 0):,} "
                f"({pct:.1f}% proven)"
            )
        print()

    if tactic_heads:
        top = tactic_heads.most_common(args.top_tactics)
        print(f"Top {len(top)} tactic heads (first whitespace-separated token, in transitions):")
        for tac, c in top:
            pct = 100 * c / n_transitions
            print(f"  {c:>6,}  ({pct:5.1f}%)  {tac}")
        print()

    if error_counts:
        top = error_counts.most_common(args.top_errors)
        print(f"Top {len(top)} error messages (first line, truncated):")
        for msg, c in top:
            print(f"  {c:>5,}  {msg}")


def cmd_view(args):
    items = list(
        iter_transitions(
            args.path, step=args.step, steps=args.steps, dataset=args.dataset
        )
    )
    if not items:
        print("No matching transitions.")
        return
    if args.random:
        rng = random.Random(args.seed)
        items = rng.sample(items, min(args.count, len(items)))
    else:
        items = items[args.offset : args.offset + args.count]

    for i, (shard_step, a, (ctx, tac, val)) in enumerate(items):
        header = (
            f"[{i}] step={shard_step} "
            f"dataset={a.get('dataset')} id={a.get('id')} "
            f"value={fmt_value(val)} "
            f"proof_size={a.get('proof_size')}"
        )
        print("=" * len(header))
        print(header)
        print("=" * len(header))
        if args.theorem:
            print(transition_to_theorem(ctx, tac, val))
        else:
            print("State:")
            print(ctx)
            print(f"Tactic: {tac}")
        print()


def cmd_attempts(args):
    rows = list(
        iter_attempts(
            args.path,
            step=args.step,
            steps=args.steps,
            dataset=args.dataset,
            outcome=args.outcome,
        )
    )
    if args.id is not None:
        rows = [(s, a) for s, a in rows if a.get("id") == args.id]
    if not rows:
        print("No matching attempts.")
        return
    if args.random:
        rng = random.Random(args.seed)
        rows = rng.sample(rows, min(args.count, len(rows)))
    else:
        rows = rows[args.offset : args.offset + args.count]

    for shard_step, a in rows:
        n_trans = len(a.get("transitions") or [])
        print(
            f"step={shard_step} dataset={a.get('dataset')} id={a.get('id')} "
            f"outcome={a.get('outcome')} proof_size={a.get('proof_size')} "
            f"sims={a.get('num_simulations')} iters={a.get('num_iterations')} "
            f"transitions={n_trans}"
        )
        if a.get("error"):
            print(f"  error: {a['error'].splitlines()[0]}")
        if args.theorem:
            print(f"  source: {a.get('theorem', '').strip()}")


def parse_steps(s):
    if ":" not in s:
        raise argparse.ArgumentTypeError("expected LO:HI (inclusive)")
    lo, hi = s.split(":", 1)
    return int(lo), int(hi)


def add_filter_args(p):
    p.add_argument("--step", type=int, help="restrict to a single step")
    p.add_argument(
        "--steps",
        type=parse_steps,
        metavar="LO:HI",
        help="restrict to inclusive step range",
    )
    p.add_argument(
        "--dataset",
        help="restrict to one dataset (leanworkbook, numinamath, deepseek_prover, ...)",
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        help="run directory (contains step_*/theorems.jsonl) or a single theorems.jsonl",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_stats = sub.add_parser("stats", help="aggregate statistics across attempts")
    add_filter_args(p_stats)
    p_stats.add_argument("--top-tactics", type=int, default=15)
    p_stats.add_argument("--top-errors", type=int, default=10)
    p_stats.set_defaults(func=cmd_stats)

    p_view = sub.add_parser("view", help="browse individual transitions")
    add_filter_args(p_view)
    p_view.add_argument("-n", "--count", type=int, default=5)
    p_view.add_argument("-o", "--offset", type=int, default=0)
    p_view.add_argument("-r", "--random", action="store_true")
    p_view.add_argument("--seed", type=int, default=0)
    p_view.add_argument(
        "-t",
        "--theorem",
        action="store_true",
        help="render each transition as a Lean example block",
    )
    p_view.set_defaults(func=cmd_view)

    p_atts = sub.add_parser("attempts", help="list theorem-level outcomes")
    add_filter_args(p_atts)
    p_atts.add_argument("-n", "--count", type=int, default=20)
    p_atts.add_argument("-o", "--offset", type=int, default=0)
    p_atts.add_argument("-r", "--random", action="store_true")
    p_atts.add_argument("--seed", type=int, default=0)
    p_atts.add_argument("--outcome", choices=["proven", "unproven", "error"])
    p_atts.add_argument("--id", help="match a specific theorem id")
    p_atts.add_argument(
        "-t",
        "--theorem",
        action="store_true",
        help="also print the theorem source declaration",
    )
    p_atts.set_defaults(func=cmd_attempts)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
