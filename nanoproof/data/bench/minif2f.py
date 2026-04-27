"""miniF2F benchmark dataset.

Public interface:
- ``download_dataset()`` - fetch the .lean files from the GitHub repo.
- ``list_theorems(split)`` - return the parsed theorems for the requested
  split (``"valid"`` or ``"test"``), each wrapped as a ``BenchTheorem``.

CLI: see ``python -m nanoproof.data.bench.minif2f --help``.
"""

import argparse
import os
import re
from collections import Counter
from pathlib import Path

from nanoproof.common import get_base_dir
from nanoproof.data.bench.common import BenchTheorem, MINIF2F_PREAMBLE
from nanoproof.data.check_init import add_check_init_args, run_check_init_cli
from nanoproof.data.rl.common import download_file

DATA_DIR = os.path.join(get_base_dir(), "data", "minif2f")
BASE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/miniF2F/refs/heads/main/MiniF2F/"
)

DATASET_NAME = "minif2f"

# The split name -> source filename mapping. Both files contain ``sorry``-stub
# theorems, except for two test entries that ship with proofs (patched below).
_SPLIT_FILES = {"valid": "Valid.lean", "test": "Test.lean"}

# Theorem names live in the source as ``theorem <name>``; we recover them as
# the per-theorem id.
_THEOREM_NAME_RE = re.compile(r"\btheorem\s+(\S+)")


def download_dataset() -> None:
    """Download Valid.lean and Test.lean from the upstream GitHub repo."""
    for filename in _SPLIT_FILES.values():
        dest = os.path.join(DATA_DIR, filename)
        download_file(BASE_URL + filename, dest, desc=filename)


def _split_file(split: str) -> str:
    assert split in _SPLIT_FILES, (
        f"Invalid split: {split!r}. Must be one of {sorted(_SPLIT_FILES)}"
    )
    return os.path.join(DATA_DIR, _SPLIT_FILES[split])


def list_theorems(split: str) -> list[BenchTheorem]:
    file_path = Path(_split_file(split))
    if not file_path.exists():
        raise FileNotFoundError(
            f"miniF2F file not found at {file_path}. Run with `download` first."
        )

    text = file_path.read_text()
    if split == "test":
        # Two upstream test theorems ship with proofs filled in instead of `sorry`.
        # Theorem mathd_numbertheory_66 also receives missing `by` keyword.
        text = text.replace(
            "\ntheorem mathd_numbertheory_66 : 194 % 11 = 7 :=\n  rfl\n",
            "\ntheorem mathd_numbertheory_66 : 194 % 11 = 7 := by\n  sorry\n",
        )
        text = text.replace(
            "\ntheorem mathd_algebra_302 : (Complex.I / 2) ^ 2 = -(1 / 4) := by\n  norm_num [div_pow]\n",
            "\ntheorem mathd_algebra_302 : (Complex.I / 2) ^ 2 = -(1 / 4) := by\n  sorry\n",
        )

    sources: list[str] = []
    theorem_lines: list[str] = []
    in_theorem = False
    for line in text.split("\n"):
        if line.lstrip().startswith("theorem"):
            assert not in_theorem, "minif2f: overlapping theorems"
            in_theorem = True
            theorem_lines.append(line)
        elif line.lstrip().startswith("sorry"):
            assert in_theorem, "minif2f: sorry without theorem"
            in_theorem = False
            theorem_lines.append(line)
            sources.append("\n".join(theorem_lines))
            theorem_lines = []
        elif in_theorem:
            theorem_lines.append(line)

    assert all(s.count("sorry") == 1 for s in sources), (
        "Found a theorem with no or multiple `sorry`."
    )
    expected_count = 256 if split == "valid" else 244
    assert len(sources) == expected_count, (
        f"minif2f: expected {expected_count} theorems, got {len(sources)}"
    )

    ids: list[str] = []
    for s in sources:
        m = _THEOREM_NAME_RE.search(s)
        assert m is not None, f"minif2f: could not extract id from: {s[:120]!r}"
        ids.append(m.group(1))
    duplicates = [tid for tid, n in Counter(ids).items() if n > 1]
    assert not duplicates, f"minif2f/{split}: duplicate theorem ids: {duplicates}"

    return [
        BenchTheorem(source=MINIF2F_PREAMBLE + s, dataset=DATASET_NAME, id=tid)
        for s, tid in zip(sources, ids)
    ]


# -----------------------------------------------------------------------------
# CLI: download / show / stats / check-init


def _main():
    parser = argparse.ArgumentParser(
        description="miniF2F benchmark dataset", allow_abbrev=False
    )
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("download", help="Download Valid.lean and Test.lean from GitHub")
    show = sub.add_parser("show", help="Print the first N theorems from a split")
    show.add_argument("--split", choices=list(_SPLIT_FILES), default="valid")
    show.add_argument("--n", type=int, default=5)
    sub.add_parser("stats", help="Print theorem counts per split")
    check = sub.add_parser(
        "check-init",
        help="Try to initialize each theorem's proof in a Lean REPL and report failures "
        "(benchmarks do not get whitelists - failures are warnings)",
    )
    check.add_argument("--split", choices=list(_SPLIT_FILES), default="valid")
    add_check_init_args(check, default_jobs=1)
    args = parser.parse_args()

    if args.action == "download":
        download_dataset()
    elif args.action == "show":
        for thm in list_theorems(args.split)[: args.n]:
            print(thm.source)
            print("-" * 80)
    elif args.action == "stats":
        for split in _SPLIT_FILES:
            print(f"{split}: {len(list_theorems(split))} theorems")
    elif args.action == "check-init":
        run_check_init_cli(
            theorems=list_theorems(args.split),
            dataset_file=_split_file(args.split),
            lean_server=args.lean_server,
            lean_project=args.lean_project,
            num_workers=args.jobs,
            limit=args.limit,
            verbose=args.verbose,
            save=False,
        )


if __name__ == "__main__":
    _main()
