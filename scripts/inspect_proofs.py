#!/usr/bin/env python3
"""Inspect proofs found during evaluation."""

import argparse
import json
import random
from collections import Counter

from leantree.repl_adapter.server import LeanClient

from nanoproof.common import linearize_proof, format_linearized_proof
from nanoproof.search import Node, revive_tree_states, prune_redundant_nodes, compute_value_target

MINIF2F_IMPORTS = """
import Mathlib
import FormalConjectures.ForMathlib.Analysis.SpecialFunctions.NthRoot
import FormalConjectures.Util.Answer
"""

# Lean environment setup commands
LEAN_OPEN_SCOPED_COMMANDS = """
open scoped Real
open scoped Nat
open scoped Topology
open scoped Polynomial
"""


def load_proofs(path: str) -> list[dict]:
    """Load evaluation results from JSONL file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_transitions(proof: dict, transitions: list[tuple[str, str, str]] | None = None) -> list[tuple[str, str, str]]:
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
    
    if len(proofs) == 0:
        print("No proofs found.")
        return
    
    # Count proven/unproven
    proven = [p for p in proofs if p.get("proof") is not None]
    unproven = [p for p in proofs if p.get("proof") is None]
    
    # Iteration statistics
    iterations = [p.get("num_iterations", 0) for p in proofs]
    proven_iterations = [p.get("num_iterations", 0) for p in proven]
    unproven_iterations = [p.get("num_iterations", 0) for p in unproven]
    
    print(f"Evaluation Results Statistics")
    print(f"{'=' * 60}")
    print(f"Path: {args.path}")
    print(f"Total theorems: {len(proofs):,}")
    print()
    
    print(f"Success Rate:")
    success_rate = len(proven) / len(proofs) if proofs else 0
    print(f"  Proven: {len(proven):,} ({100 * success_rate:.1f}%)")
    print(f"  Unproven: {len(unproven):,} ({100 * (1 - success_rate):.1f}%)")
    print()
    
    print(f"MCTS Iterations:")
    if iterations:
        avg_iter = sum(iterations) / len(iterations)
        print(f"  Overall average: {avg_iter:.1f}")
        print(f"  Min: {min(iterations)}")
        print(f"  Max: {max(iterations)}")
    if proven_iterations:
        avg_proven = sum(proven_iterations) / len(proven_iterations)
        print(f"  Average for proven: {avg_proven:.1f}")
    if unproven_iterations:
        avg_unproven = sum(unproven_iterations) / len(unproven_iterations)
        print(f"  Average for unproven: {avg_unproven:.1f}")
    print()
    
    # Iteration distribution
    if iterations:
        print(f"Iteration Distribution:")
        # Bucket iterations into ranges
        buckets = Counter()
        for it in iterations:
            if it <= 10:
                buckets["1-10"] += 1
            elif it <= 50:
                buckets["11-50"] += 1
            elif it <= 100:
                buckets["51-100"] += 1
            elif it <= 500:
                buckets["101-500"] += 1
            else:
                buckets["500+"] += 1
        
        for bucket in ["1-10", "11-50", "51-100", "101-500", "500+"]:
            count = buckets.get(bucket, 0)
            pct = 100 * count / len(iterations)
            print(f"  {bucket:>10}: {count:,} ({pct:.1f}%)")


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
        theorem = item.get("theorem", "(no theorem)")
        # Truncate long theorems
        if len(theorem) > 80:
            theorem = theorem[:77] + "..."
        
        status = "✓" if item.get("proof") is not None else "✗"
        iterations = item.get("num_iterations", "?")
        
        print(f"[{start + i:4d}] {status} (iter={iterations:>4}) {theorem}")


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
        env.send_command(LEAN_OPEN_SCOPED_COMMANDS)
        
        for i, item in enumerate(selected):
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
                print(f"Linearized proof (before pruning, {len(tactics_before)} tactics):")
                for line in format_linearized_proof(tactics_before).split("\n"):
                    print(f"  {line}")
                
                print()
                print(f"Linearized proof (after pruning, {len(tactics_after)} tactics):")
                for line in format_linearized_proof(tactics_after).split("\n"):
                    print(f"  {line}")
            
            print()


def cmd_gather_lean(args):
    """Gather Lean theorems with proofs from evaluation steps."""
    from pathlib import Path
    
    run_dir = Path(args.run_dir)
    evals_dir = run_dir / "evals"
    
    if not evals_dir.exists():
        print(f"Error: evals directory not found at {evals_dir}")
        return
    
    # Parse step subdirs as ints and sort
    steps = []
    for subdir in evals_dir.iterdir():
        if subdir.is_dir():
            step = int(subdir.name)
            steps.append(step)
    
    steps.sort()
    
    if not steps:
        print(f"No step directories found in {evals_dir}")
        return
    
    # Create output directory
    output_base = Path(args.output_dir) / run_dir.name
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(steps)} evaluation steps: {steps}")
    print(f"Output directory: {output_base}")
    print()
    
    for step in steps:
        step_dir = evals_dir / str(step)
        minif2f_path = step_dir / "minif2f.jsonl"
        
        if not minif2f_path.exists():
            print(f"Warning: {minif2f_path} not found, skipping step {step}")
            continue
        
        proofs = load_proofs(str(minif2f_path))
        
        # Collect theorems
        lean_content = []
        proven_count = 0
        
        for item in proofs:
            theorem = item.get("theorem", "").strip()
            proof_dict = item.get("proof")
            
            if proof_dict is not None:
                # Deserialize and linearize the proof
                node = Node.deserialize(proof_dict)
                tactics = linearize_proof(node)
                
                assert theorem.endswith("sorry"), "theorem should end with 'sorry'"
                proven_count += 1
                # Replace "sorry" with the proof
                theorem_body = theorem[:-len("sorry")]  # Remove "sorry"
                if len(tactics) == 1:
                    theorem = theorem_body + tactics[0]
                else:
                    # Multi-line proof with indentation
                    proof_lines = "\n  ".join(tactics)
                    theorem = theorem_body + proof_lines
            
            lean_content.append(theorem)
        
        # Write the Lean file
        output_path = output_base / f"{step}-minif2f.lean"
        with open(output_path, "w") as f:
            f.write(MINIF2F_IMPORTS.strip() + "\n\n" + LEAN_OPEN_SCOPED_COMMANDS.strip() + "\n\n\n" + "\n\n".join(lean_content))
        
        print(f"Step {step}: {proven_count}/{len(lean_content)} proven -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect proofs found during evaluation."
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # View subcommand
    view_parser = subparsers.add_parser("view", help="View proofs in detail")
    view_parser.add_argument("path", help="Path to the evaluation results JSONL file")
    view_parser.add_argument(
        "--count", "-n", type=int, default=5,
        help="Number of proofs to print (default: 5)"
    )
    view_parser.add_argument(
        "--random", "-r", action="store_true",
        help="Select random proofs instead of sequential"
    )
    view_parser.add_argument(
        "--offset", "-o", type=int, default=0,
        help="Offset to start from when not random (default: 0)"
    )
    view_parser.add_argument(
        "--proven", "-p", action="store_true",
        help="Only show proven theorems"
    )
    view_parser.add_argument(
        "--unproven", "-u", action="store_true",
        help="Only show unproven theorems"
    )
    view_parser.set_defaults(func=cmd_view)
    
    # Stats subcommand
    stats_parser = subparsers.add_parser("stats", help="Print statistics")
    stats_parser.add_argument("path", help="Path to the evaluation results JSONL file")
    stats_parser.set_defaults(func=cmd_stats)
    
    # List subcommand
    list_parser = subparsers.add_parser("list", help="List theorems with status")
    list_parser.add_argument("path", help="Path to the evaluation results JSONL file")
    list_parser.add_argument(
        "--count", "-n", type=int, default=20,
        help="Number of theorems to list (default: 20)"
    )
    list_parser.add_argument(
        "--offset", "-o", type=int, default=0,
        help="Offset to start from (default: 0)"
    )
    list_parser.add_argument(
        "--proven", "-p", action="store_true",
        help="Only show proven theorems"
    )
    list_parser.add_argument(
        "--unproven", "-u", action="store_true",
        help="Only show unproven theorems"
    )
    list_parser.set_defaults(func=cmd_list)
    
    # Simplify subcommand
    simplify_parser = subparsers.add_parser("simplify", help="Simplify proof trees by pruning redundant nodes")
    simplify_parser.add_argument("path", help="Path to the evaluation results JSONL file")
    simplify_parser.add_argument(
        "--count", "-n", type=int, default=1,
        help="Number of proofs to simplify (default: 1)"
    )
    simplify_parser.add_argument(
        "--offset", "-o", type=int, default=0,
        help="Offset to start from when not random (default: 0)"
    )
    simplify_parser.add_argument(
        "--random", "-r", action="store_true",
        help="Select random proofs instead of sequential"
    )
    simplify_parser.add_argument(
        "--server", "-s", type=str, default="10.10.24.32",
        help="Lean server address (default: 10.10.24.32)"
    )
    simplify_parser.add_argument(
        "--port", "-p", type=int, default=8000,
        help="Lean server port (default: 8000)"
    )
    simplify_parser.set_defaults(func=cmd_simplify)
    
    # Gather Lean subcommand
    gather_parser = subparsers.add_parser(
        "gather_lean",
        help="Gather Lean theorems with proofs from evaluation steps"
    )
    gather_parser.add_argument(
        "run_dir",
        help="Path to the run's output directory (containing 'evals' subdir)"
    )
    gather_parser.add_argument(
        "--output-dir", "-o", type=str, default=".",
        help="Output directory for Lean files (default: current directory)"
    )
    gather_parser.set_defaults(func=cmd_gather_lean)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
