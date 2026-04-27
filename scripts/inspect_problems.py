#!/usr/bin/env python3
"""Inspect every prove attempt the RL loop made on a single theorem.

Walks ``<run_dir>/step_NNNNN/theorems.jsonl`` in step order, filters to
attempts matching ``(dataset, id)``, and prints one line per attempt with
the simulation budget, outcome, error (if any), and the running matchmaker
weight after each attempt is counted in. For successful attempts, also
prints the linearized proof source (same format used by inspect_proofs.py).
"""

import argparse
import json

from nanoproof.common import construct_proof_source, linearize_proof
from nanoproof.experience_collection import (
    MatchmakerConfig,
    TheoremStats,
    list_step_shards,
)
from nanoproof.search import Node


def _format_proof(theorem_source: str, full_tree: dict | None) -> str | None:
    if full_tree is None:
        return None
    node = Node.deserialize(full_tree)
    tactics = linearize_proof(node)
    return construct_proof_source(theorem_source, tactics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show every attempt at one theorem across an RL run.",
        allow_abbrev=False,
    )
    parser.add_argument("run_dir", help="path to the RL run directory")
    parser.add_argument("--dataset", required=True, help="dataset name")
    parser.add_argument("--id", required=True, help="theorem id within the dataset")
    parser.add_argument(
        "--show-proofs",
        action="store_true",
        help="print the full linearized proof source for proven attempts",
    )
    args = parser.parse_args()

    # Use the live run's matchmaker config when available so the weights
    # printed here match what training actually used.
    config = MatchmakerConfig()
    args_path = f"{args.run_dir}/args.json"
    try:
        with open(args_path, "r") as f:
            run_args = json.load(f)
    except FileNotFoundError:
        run_args = {}
    for field in (
        "trust_count",
        "trust_count_proved",
        "weight_interesting",
        "weight_undecided",
        "weight_fully_proved",
        "base_simulations",
        "failure_multiplier",
        "cap_simulations",
    ):
        key = f"mm_{field}"
        if key in run_args:
            config = type(config)(**{**config.__dict__, field: run_args[key]})

    stats = TheoremStats()
    found = 0
    print(f"# {args.dataset}/{args.id}  (run: {args.run_dir})")
    print(
        f"# {'step':>6}  {'outcome':<9} {'sims':>6} {'iters':>6} {'trans':>6}  "
        f"{'weight_after':>14}  details"
    )
    for step, shard_path in list_step_shards(args.run_dir):
        with open(shard_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("dataset") != args.dataset or obj.get("id") != args.id:
                    continue
                stats.update(obj["outcome"])
                weight = stats.weight(config)
                num_sims = obj.get("num_simulations", 0)
                iters = obj.get("num_iterations", 0)
                trans = len(obj.get("transitions", []))
                detail = ""
                if obj["outcome"] == "error":
                    err = (obj.get("error") or "").splitlines()
                    detail = err[0] if err else ""
                print(
                    f"  {step:>6d}  {obj['outcome']:<9} {num_sims:>6d} {iters:>6d} "
                    f"{trans:>6d}  {weight:>14.6e}  {detail}"
                )
                if args.show_proofs and obj["outcome"] == "proven":
                    proof = _format_proof(obj["theorem"], obj.get("full_tree"))
                    if proof is not None:
                        print("    --- proof ---")
                        for proof_line in proof.splitlines():
                            print(f"    {proof_line}")
                        print("    -------------")
                found += 1

    if found == 0:
        print(f"# (no attempts found for {args.dataset}/{args.id})")
    else:
        print(f"# {found} attempt(s); current weight: {stats.weight(config):.6e}")


if __name__ == "__main__":
    main()
