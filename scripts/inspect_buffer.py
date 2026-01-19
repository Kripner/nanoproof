#!/usr/bin/env python3
"""Inspect a saved replay buffer file."""

import argparse
import json
import random
from collections import Counter


def load_buffer(path: str) -> list[tuple[str, str, float]]:
    """Load replay buffer from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def cmd_view(args):
    """View transitions from the buffer."""
    buffer = load_buffer(args.path)
    
    if len(buffer) == 0:
        print("Buffer is empty.")
        return
    
    if args.random:
        transitions = random.sample(buffer, min(args.count, len(buffer)))
    else:
        start = args.offset
        end = min(start + args.count, len(buffer))
        transitions = buffer[start:end]
    
    for i, (state, tactic, value) in enumerate(transitions):
        idx = f"[{args.offset + i}]" if not args.random else ""
        print(f"{'=' * 60}")
        print(f"Transition {i + 1} {idx}")
        print(f"{'=' * 60}")
        print(f"State:")
        print(f"{state}")
        print(f"Tactic: {tactic}")
        print(f"Value: {value}")
        print()


def cmd_stats(args):
    """Print statistics about the buffer."""
    buffer = load_buffer(args.path)
    
    if len(buffer) == 0:
        print("Buffer is empty.")
        return
    
    # Extract components
    states = [t[0] for t in buffer]
    tactics = [t[1] for t in buffer]
    values = [t[2] for t in buffer]
    
    # Value statistics
    avg_value = sum(values) / len(values)
    value_counts = Counter(values)
    
    # Length statistics
    avg_state_len = sum(len(s) for s in states) / len(states)
    avg_tactic_len = sum(len(t) for t in tactics) / len(tactics)
    
    print(f"Replay Buffer Statistics")
    print(f"{'=' * 60}")
    print(f"Path: {args.path}")
    print(f"Total transitions: {len(buffer):,}")
    print()
    
    print(f"Value Statistics:")
    print(f"  Average value: {avg_value:.4f}")
    print(f"  Value distribution:")
    for value, count in sorted(value_counts.items()):
        pct = 100 * count / len(buffer)
        print(f"    {value:6.2f}: {count:,} ({pct:.1f}%)")
    print()
    
    print(f"Length Statistics:")
    print(f"  Average state length: {avg_state_len:.1f} chars")
    print(f"  Average tactic length: {avg_tactic_len:.1f} chars")
    print(f"  Min state length: {min(len(s) for s in states)} chars")
    print(f"  Max state length: {max(len(s) for s in states)} chars")
    print(f"  Min tactic length: {min(len(t) for t in tactics)} chars")
    print(f"  Max tactic length: {max(len(t) for t in tactics)} chars")
    print()
    
    # Top tactics
    tactic_counts = Counter(tactics)
    print(f"Top 10 Most Common Tactics:")
    for tactic, count in tactic_counts.most_common(10):
        pct = 100 * count / len(buffer)
        # Truncate long tactics for display
        display = tactic if len(tactic) <= 50 else tactic[:47] + "..."
        print(f"  {count:6,} ({pct:5.1f}%): {display}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a saved replay buffer file."
    )
    parser.add_argument("path", help="Path to the replay buffer JSON file")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # View subcommand
    view_parser = subparsers.add_parser("view", help="View transitions from the buffer")
    view_parser.add_argument(
        "--count", "-n", type=int, default=5,
        help="Number of transitions to print (default: 5)"
    )
    view_parser.add_argument(
        "--random", "-r", action="store_true",
        help="Select random transitions instead of sequential"
    )
    view_parser.add_argument(
        "--offset", "-o", type=int, default=0,
        help="Offset to start from when not random (default: 0)"
    )
    view_parser.set_defaults(func=cmd_view)
    
    # Stats subcommand
    stats_parser = subparsers.add_parser("stats", help="Print buffer statistics")
    stats_parser.set_defaults(func=cmd_stats)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
