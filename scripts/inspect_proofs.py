#!/usr/bin/env python3
"""Inspect proofs found during evaluation."""

import argparse
import json
import random
from pathlib import Path

from leantree.repl_adapter.server import LeanClient

from nanoproof.common import (
    linearize_proof,
    format_linearized_proof,
    construct_proof_source,
)
from nanoproof.data.bench import minif2f, proofnet
from nanoproof.search import Node, revive_tree_states
from nanoproof.experience_collection import prune_redundant_nodes, compute_value_target

# Datasets that the RL training loop evaluates each step (see rl.py).
EVAL_DATASETS = ["minif2f", "proofnet"]

# Imports needed when writing exported proofs as standalone .lean files.
# (Not sent to the REPL - the server is launched with --imports Mathlib
# already, and per-theorem headers below cover opens + aux defs.)
_EXPORT_IMPORTS = """
import Mathlib
import FormalConjecturesForMathlib.Analysis.SpecialFunctions.NthRoot
import FormalConjectures.Util.Answer
"""


_DECL_KEYWORDS = ("theorem ", "lemma ", "example ", "example:", "def ")


def _proof_preview(item: dict, max_len: int = 80) -> str:
    """Linearized tactics joined with the literal ``\\n``, capped to
    ``max_len`` chars. Empty for unproven records."""
    proof_dict = item.get("proof")
    if proof_dict is None:
        return ""
    node = Node.deserialize(proof_dict)
    tactics = linearize_proof(node)
    text = "\\n".join(tactics)
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def _theorem_summary(item: dict) -> str:
    """Single-line label for a record. Prefers the stored ``name``; else
    extracts the declaration line from the source (skipping the preamble)."""
    name = item.get("name")
    if name:
        return name
    source = item.get("theorem", "(no theorem)")
    for line in source.split("\n"):
        stripped = line.lstrip()
        if any(stripped.startswith(kw) for kw in _DECL_KEYWORDS):
            return stripped
    return source.replace("\n", " ").strip()


def _record_header(item: dict) -> str:
    """Return the per-record REPL header, or "" when the record has none.

    Current eval JSONL records embed all opens / aux defs inside the
    ``theorem`` source itself (BenchTheorem.header was folded into
    .source), so this returns "" for those. The lookup remains for
    backwards compat with older JSONL files that stored a header
    separately.
    """
    return item.get("header") or ""


def load_proofs(path: str) -> list[dict]:
    """Load evaluation results from JSONL file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_transitions(
    proof: dict, transitions: list[tuple[str, str, str]] | None = None
) -> list[tuple[str, str, str]]:
    """Extract all transitions from a proof tree.

    Returns a list of (parent_id, tactic, child_id) tuples.
    """
    if transitions is None:
        transitions = []

    parent_id = proof.get("id", "?")
    children = proof.get("children")

    if children:
        for tactic, child in children.items():
            child_id = child.get("id", "?")
            transitions.append((parent_id, tactic, child_id))
            extract_transitions(child, transitions)

    return transitions


def format_transitions(proof: dict | None) -> str:
    """Format transitions as node_id --- tactic ---> node_id."""
    if proof is None:
        return "(not proven)"

    transitions = extract_transitions(proof)
    if not transitions:
        return "(no transitions)"

    lines = []
    for parent_id, tactic, child_id in transitions:
        # Shorten UUIDs for readability (first 8 chars)
        short_parent = parent_id[:8] if len(parent_id) > 8 else parent_id
        short_child = child_id[:8] if len(child_id) > 8 else child_id
        lines.append(f"{short_parent} --- {tactic} ---> {short_child}")

    return "\n".join(lines)


def format_proof_tree(proof: dict | None) -> str:
    """Format a proof tree dict for display using Node's pretty print."""
    if proof is None:
        return "(not proven)"

    node = Node.deserialize(proof)
    return node.pp_tree()


def cmd_view(args):
    """View proofs from the evaluation results."""
    proofs = load_proofs(args.path)

    if len(proofs) == 0:
        print("No proofs found.")
        return

    # Filter by proven/unproven if specified
    if args.proven:
        proofs = [p for p in proofs if p.get("proof") is not None]
    elif args.unproven:
        proofs = [p for p in proofs if p.get("proof") is None]

    if len(proofs) == 0:
        print("No proofs matching criteria.")
        return

    if args.random:
        selected = random.sample(proofs, min(args.count, len(proofs)))
    else:
        start = args.offset
        end = min(start + args.count, len(proofs))
        selected = proofs[start:end]

    for i, item in enumerate(selected):
        idx = f"[{args.offset + i}]" if not args.random else ""
        print(f"{'=' * 60}")
        print(f"Proof {i + 1} {idx}")
        print(f"{'=' * 60}")

        # Print theorem
        print(f"Theorem:")
        print(f"  {item.get('theorem', '(no theorem)')}")
        print()

        # Print linearized proof
        proof = item.get("proof")
        print(f"Linearized proof:")
        if proof is None:
            print("  (not proven)")
        else:
            node = Node.deserialize(proof)
            tactics = linearize_proof(node)
            formatted_linearized = format_linearized_proof(tactics)
            for line in formatted_linearized.split("\n"):
                print(f"  {line}")
        print()

        # Print transitions and proof tree

        print(f"Transitions:")
        if proof is None:
            print("  (not proven)")
        else:
            formatted_transitions = format_transitions(proof)
            for line in formatted_transitions.split("\n"):
                print(f"  {line}")
        print()

        print(f"Proof tree:")
        if proof is None:
            print("  (not proven)")
        else:
            formatted = format_proof_tree(proof)
            for line in formatted.split("\n"):
                print(f"  {line}")
        print()

        # Print iterations
        num_iterations = item.get("num_iterations", "(unknown)")
        print(f"MCTS iterations: {num_iterations}")
        print()


def cmd_stats(args):
    """Print statistics about the evaluation results."""
    proofs = load_proofs(args.path)

    if not proofs:
        print("No proofs found.")
        return

    proven = [p for p in proofs if p.get("proof") is not None]
    errors = [p for p in proofs if p.get("error") is not None]

    print(f"Path: {args.path}")
    print(f"Total: {len(proofs):,}")
    print(f"Proven: {len(proven):,} ({100 * len(proven) / len(proofs):.1f}%)")
    print(f"Errors: {len(errors):,}")

    if not proven:
        return

    proof_lengths = []  # tactics per proven proof
    tactic_lengths = []  # chars per tactic across all proven proofs
    for p in proven:
        node = Node.deserialize(p["proof"])
        tactics = linearize_proof(node)
        proof_lengths.append(len(tactics))
        tactic_lengths.extend(len(t) for t in tactics)

    def _print_stats(label: str, values: list[int]) -> None:
        avg = sum(values) / len(values)
        print(f"{label}: avg={avg:.1f} min={min(values)} max={max(values)}")

    print()
    _print_stats("Proof length (tactics)", proof_lengths)
    _print_stats("Tactic length (chars)", tactic_lengths)


def cmd_list(args):
    """List theorems with their proof status."""
    proofs = load_proofs(args.path)

    if len(proofs) == 0:
        print("No proofs found.")
        return

    # Filter by proven/unproven if specified
    if args.proven:
        proofs = [p for p in proofs if p.get("proof") is not None]
    elif args.unproven:
        proofs = [p for p in proofs if p.get("proof") is None]

    if len(proofs) == 0:
        print("No proofs matching criteria.")
        return

    start = args.offset
    end = min(start + args.count, len(proofs))
    selected = proofs[start:end]

    for i, item in enumerate(selected):
        summary = _theorem_summary(item)
        if len(summary) > 80:
            summary = summary[:77] + "..."

        status = "✓" if item.get("proof") is not None else "✗"
        iterations = item.get("num_iterations", "?")
        preview = _proof_preview(item)

        line = f"[{start + i:4d}] {status} (iter={iterations:>4}) {summary}"
        if preview:
            line += f"  |  {preview}"
        print(line)


def cmd_simplify(args):
    """Simplify proof trees by pruning redundant nodes."""
    print(f"Loading proofs from {args.path}...")
    proofs = load_proofs(args.path)

    if len(proofs) == 0:
        print("No proofs found.")
        return

    # Filter to only solved theorems
    proofs = [p for p in proofs if p.get("proof") is not None]

    if len(proofs) == 0:
        print("No solved theorems found.")
        return

    if args.random:
        selected = random.sample(proofs, min(args.count, len(proofs)))
    else:
        start = args.offset
        end = min(start + args.count, len(proofs))
        selected = proofs[start:end]

    # Connect to Lean server
    print(f"Connecting to Lean server {args.server}:{args.port}...")
    client = LeanClient(args.server, args.port)
    process = client.get_process()

    if process is None:
        print(f"Failed to acquire Lean process from {args.server}:{args.port}")
        return

    with process as env:
        for i, item in enumerate(selected):
            # Per-theorem header (opens + aux defs). Pool rollback resets env
            # to the post-Mathlib checkpoint on release, but within a single
            # process session these accumulate. That's fine for inspection -
            # name clashes would only matter if headers conflict across
            # theorems in the same file, which doesn't happen in practice.
            env.send_command(_record_header(item))

            idx = f"[{args.offset + i}]" if not args.random else ""
            print(f"{'=' * 60}")
            print(f"Proof {i + 1} {idx}")
            print(f"{'=' * 60}")

            # Print theorem
            theorem = item.get("theorem", "(no theorem)")
            print(f"Theorem:")
            print(f"  {theorem}")
            print()

            # Load and deserialize proof tree
            proof_dict = item.get("proof")
            node = Node.deserialize(proof_dict)

            # Print tree before pruning
            print(f"Proof tree (before pruning):")
            formatted = node.pp_tree()
            for line in formatted.split("\n"):
                print(f"  {line}")
            print()

            # Capture linearized proof before pruning
            tactics_before = linearize_proof(node)

            # Revive tree states
            revive_tree_states(node, theorem, env)

            # Prune redundant nodes
            pruned_count = prune_redundant_nodes(node)
            if pruned_count > 0:
                compute_value_target(node)

            # Print tree after pruning
            print(f"Proof tree (after pruning and recomputing value target):")
            formatted = node.pp_tree()
            for line in formatted.split("\n"):
                print(f"  {line}")
            print()

            print(f"Pruned nodes: {pruned_count}")

            # Print linearized proofs if pruning occurred
            if pruned_count > 0:
                tactics_after = linearize_proof(node)

                print()
                print(
                    f"Linearized proof (before pruning, {len(tactics_before)} tactics):"
                )
                for line in format_linearized_proof(tactics_before).split("\n"):
                    print(f"  {line}")

                print()
                print(
                    f"Linearized proof (after pruning, {len(tactics_after)} tactics):"
                )
                for line in format_linearized_proof(tactics_after).split("\n"):
                    print(f"  {line}")

            print()


def cmd_check(args):
    """Check that an eval JSONL file contains exactly the benchmark theorems."""
    bench_loaders = {
        "minif2f": lambda: minif2f.list_theorems(split="valid"),
        "proofnet": lambda: proofnet.list_theorems(split="valid"),
    }

    path = Path(args.path)
    stem = path.stem.lower()
    dataset = next((name for name in bench_loaders if name in stem), None)
    if dataset is None:
        print(f"Cannot determine dataset from filename: {path.name}")
        print(f"Expected one of: {', '.join(bench_loaders)}")
        return

    bench_theorems = bench_loaders[dataset]()
    proofs = load_proofs(str(path))

    print(f"Dataset: {dataset} (split=valid)")
    print(f"Expected: {len(bench_theorems)} theorems")
    print(f"In file:  {len(proofs)} records")

    expected = {t.source for t in bench_theorems}
    file_sources = [p.get("theorem", "") for p in proofs]
    file_set = set(file_sources)

    duplicates = len(file_sources) - len(file_set)
    missing = expected - file_set
    extra = file_set - expected

    if duplicates == 0 and not missing and not extra:
        print("OK: file contains exactly the benchmark theorems.")
        return

    print()
    print("MISMATCH:")
    if duplicates:
        print(f"  Duplicate records: {duplicates}")
    if missing:
        print(f"  Missing from file: {len(missing)}")
        for s in sorted(missing)[:5]:
            print(f"    - {_one_line_preview(s)}")
        if len(missing) > 5:
            print(f"    ... ({len(missing) - 5} more)")
    if extra:
        print(f"  Extra in file (not in benchmark): {len(extra)}")
        for s in sorted(extra)[:5]:
            print(f"    - {_one_line_preview(s)}")
        if len(extra) > 5:
            print(f"    ... ({len(extra) - 5} more)")


def _one_line_preview(source: str, max_len: int = 120) -> str:
    """Compact one-line representation of a Lean source for diff output."""
    for line in source.split("\n"):
        stripped = line.lstrip()
        if any(stripped.startswith(kw) for kw in _DECL_KEYWORDS):
            return stripped[:max_len] + ("..." if len(stripped) > max_len else "")
    flat = " ".join(source.split())
    return flat[:max_len] + ("..." if len(flat) > max_len else "")


def cmd_gather_lean(args):
    """Gather Lean theorems with proofs from evaluation steps."""
    run_dir = Path(args.run_dir)
    evals_dir = run_dir / "evals"

    if not evals_dir.exists():
        print(f"Error: evals directory not found at {evals_dir}")
        return

    step_dirs = sorted(
        (d for d in evals_dir.iterdir() if d.is_dir()),
        key=lambda d: int(d.name),
    )

    if not step_dirs:
        print(f"No step directories found in {evals_dir}")
        return

    # Create output directory
    output_base = Path(args.output_dir) / run_dir.name
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(step_dirs)} evaluation steps: {[d.name for d in step_dirs]}")
    print(f"Output directory: {output_base}")
    print()

    for step_dir in step_dirs:
        for dataset in EVAL_DATASETS:
            jsonl_path = step_dir / f"{dataset}.jsonl"
            if not jsonl_path.exists():
                continue

            proofs = load_proofs(str(jsonl_path))

            # Collect theorems, each wrapped in its own `section ... end` with
            # its stored header so opens/defs don't leak between theorems.
            lean_blocks = []
            proven_count = 0

            for item in proofs:
                theorem = item.get("theorem", "").strip()
                proof_dict = item.get("proof")

                if proof_dict is not None:
                    proven_count += 1
                    # Prefer the cached source written by the prover; fall
                    # back to reconstructing it for older JSONL records.
                    cached = item.get("linearized_proof")
                    if cached:
                        theorem = cached
                    else:
                        node = Node.deserialize(proof_dict)
                        tactics = linearize_proof(node)
                        theorem = construct_proof_source(theorem, tactics)

                header = _record_header(item)
                if header:
                    lean_blocks.append(f"section\n{header}\n\n{theorem}\n\nend")
                else:
                    lean_blocks.append(f"section\n\n{theorem}\n\nend")

            output_path = output_base / f"{step_dir.name}-{dataset}.lean"
            with open(output_path, "w") as f:
                f.write(_EXPORT_IMPORTS.strip() + "\n\n\n" + "\n\n".join(lean_blocks))

            print(
                f"Step {step_dir.name} [{dataset}]: {proven_count}/{len(lean_blocks)} proven -> {output_path}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Inspect proofs found during evaluation.",
        allow_abbrev=False,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # View subcommand
    view_parser = subparsers.add_parser("view", help="View proofs in detail")
    view_parser.add_argument("path", help="Path to the evaluation results JSONL file")
    view_parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=5,
        help="Number of proofs to print (default: 5)",
    )
    view_parser.add_argument(
        "--random",
        "-r",
        action="store_true",
        help="Select random proofs instead of sequential",
    )
    view_parser.add_argument(
        "--offset",
        "-o",
        type=int,
        default=0,
        help="Offset to start from when not random (default: 0)",
    )
    view_parser.add_argument(
        "--proven", "-p", action="store_true", help="Only show proven theorems"
    )
    view_parser.add_argument(
        "--unproven", "-u", action="store_true", help="Only show unproven theorems"
    )
    view_parser.set_defaults(func=cmd_view)

    # Stats subcommand
    stats_parser = subparsers.add_parser("stats", help="Print statistics")
    stats_parser.add_argument("path", help="Path to the evaluation results JSONL file")
    stats_parser.set_defaults(func=cmd_stats)

    # Check subcommand
    check_parser = subparsers.add_parser(
        "check",
        help="Verify the eval JSONL contains exactly the benchmark theorems",
    )
    check_parser.add_argument("path", help="Path to the evaluation results JSONL file")
    check_parser.set_defaults(func=cmd_check)

    # List subcommand
    list_parser = subparsers.add_parser("list", help="List theorems with status")
    list_parser.add_argument("path", help="Path to the evaluation results JSONL file")
    list_parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=20,
        help="Number of theorems to list (default: 20)",
    )
    list_parser.add_argument(
        "--offset", "-o", type=int, default=0, help="Offset to start from (default: 0)"
    )
    list_parser.add_argument(
        "--proven", "-p", action="store_true", help="Only show proven theorems"
    )
    list_parser.add_argument(
        "--unproven", "-u", action="store_true", help="Only show unproven theorems"
    )
    list_parser.set_defaults(func=cmd_list)

    # Simplify subcommand
    simplify_parser = subparsers.add_parser(
        "simplify", help="Simplify proof trees by pruning redundant nodes"
    )
    simplify_parser.add_argument(
        "path", help="Path to the evaluation results JSONL file"
    )
    simplify_parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=1,
        help="Number of proofs to simplify (default: 1)",
    )
    simplify_parser.add_argument(
        "--offset",
        "-o",
        type=int,
        default=0,
        help="Offset to start from when not random (default: 0)",
    )
    simplify_parser.add_argument(
        "--random",
        "-r",
        action="store_true",
        help="Select random proofs instead of sequential",
    )
    simplify_parser.add_argument(
        "--server",
        "-s",
        type=str,
        default="10.10.24.32",
        help="Lean server address (default: 10.10.24.32)",
    )
    simplify_parser.add_argument(
        "--port", "-p", type=int, default=8000, help="Lean server port (default: 8000)"
    )
    simplify_parser.set_defaults(func=cmd_simplify)

    # Gather Lean subcommand
    gather_parser = subparsers.add_parser(
        "gather_lean", help="Gather Lean theorems with proofs from evaluation steps"
    )
    gather_parser.add_argument(
        "run_dir", help="Path to the run's output directory (containing 'evals' subdir)"
    )
    gather_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=".",
        help="Output directory for Lean files (default: current directory)",
    )
    gather_parser.set_defaults(func=cmd_gather_lean)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
