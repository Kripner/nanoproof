"""
Analyze a PyTorch CUDA memory snapshot to identify what's holding memory at
the time of dump (typically OOM).

Usage: python scripts/analyze_memory_snapshot.py path/to/memory_snapshot_rank0.pickle

The snapshot is a dict produced by torch.cuda.memory._dump_snapshot(). It contains:
  - segments: list of reserved memory segments (allocator pool state)
  - device_traces: per-device timelines of alloc/free/segment events
Each block within a segment has allocated_size and (optionally) a frames stack
trace. Free blocks are cached-but-unallocated memory; allocated blocks are live
tensors.
"""

import argparse
import pickle
import sys
from collections import defaultdict


def fmt_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def frames_to_key(frames, depth=6):
    """Collapse a stack trace into a short multi-line key for grouping."""
    if not frames:
        return "<no stack>"
    lines = []
    for f in frames[:depth]:
        filename = f.get("filename", "?")
        # Trim long paths, keep last 2 components
        parts = filename.replace("\\", "/").split("/")
        filename = "/".join(parts[-2:])
        line = f.get("line", "?")
        name = f.get("name", "?")
        lines.append(f"  {filename}:{line} {name}")
    return "\n".join(lines)


def analyze(path):
    with open(path, "rb") as f:
        snap = pickle.load(f)

    segments = snap.get("segments", [])

    # Collect all blocks from all segments
    live_blocks = []  # list of (size, frames)
    free_blocks = []
    for seg in segments:
        for block in seg.get("blocks", []):
            state = block.get("state", "")
            size = block.get("size", 0) or block.get("requested_size", 0)
            frames = block.get("frames") or []
            if state == "active_allocated":
                live_blocks.append((size, frames))
            elif state.startswith("inactive") or state == "active_pending_free":
                free_blocks.append((size, frames))

    total_live = sum(s for s, _ in live_blocks)
    total_free = sum(s for s, _ in free_blocks)
    total_reserved = sum(seg.get("total_size", 0) for seg in segments)

    print(f"Snapshot: {path}")
    print(f"  Segments (reserved by allocator): {len(segments)}, total {fmt_bytes(total_reserved)}")
    print(f"  Live blocks: {len(live_blocks)}, total {fmt_bytes(total_live)}")
    print(f"  Free-in-pool blocks: {len(free_blocks)}, total {fmt_bytes(total_free)}")
    print()

    # Group live blocks by stack trace
    groups = defaultdict(lambda: {"count": 0, "size": 0, "sizes": []})
    for size, frames in live_blocks:
        key = frames_to_key(frames)
        groups[key]["count"] += 1
        groups[key]["size"] += size
        groups[key]["sizes"].append(size)

    # Sort by total bytes
    ranked = sorted(groups.items(), key=lambda kv: -kv[1]["size"])

    print("=" * 100)
    print(f"LIVE ALLOCATIONS grouped by stack trace (top frames), ordered by total bytes")
    print(f"(Each group = distinct allocation site. Total = what's keeping memory alive right now.)")
    print("=" * 100)
    for i, (key, info) in enumerate(ranked[:25]):
        pct = 100 * info["size"] / total_live if total_live else 0
        sizes = sorted(info["sizes"], reverse=True)
        sample = ", ".join(fmt_bytes(s) for s in sizes[:5])
        if len(sizes) > 5:
            sample += f", ... ({len(sizes) - 5} more)"
        print(f"\n#{i+1}  {fmt_bytes(info['size'])} ({pct:.1f}%)  in {info['count']} allocations")
        print(f"    sizes: {sample}")
        print(f"    stack:")
        print(key)

    # Largest individual live allocations
    print()
    print("=" * 100)
    print("TOP 10 INDIVIDUAL LIVE ALLOCATIONS")
    print("=" * 100)
    live_blocks.sort(key=lambda x: -x[0])
    for i, (size, frames) in enumerate(live_blocks[:10]):
        print(f"\n#{i+1}  {fmt_bytes(size)}")
        print(frames_to_key(frames, depth=8))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    analyze(args.path)
