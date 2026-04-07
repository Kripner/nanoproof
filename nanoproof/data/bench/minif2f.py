"""miniF2F benchmark dataset.

Public interface:
- ``download_dataset()`` — fetch the .lean files from the GitHub repo.
- ``list_theorems(split)`` — return the parsed theorems for the requested
  split (``"valid"`` or ``"test"``). The dataset has no train split.
- ``get_imports()`` — the canonical Lean preamble used by miniF2F problems.

CLI: see ``python -m nanoproof.data.bench.minif2f --help``.
"""

import argparse
import os
from pathlib import Path

from nanoproof.common import get_base_dir
from nanoproof.data.rl.common import download_hf_file

DATA_DIR = os.path.join(get_base_dir(), "data", "minif2f")
BASE_URL = "https://raw.githubusercontent.com/google-deepmind/miniF2F/refs/heads/main/MiniF2F/"

# The split name → source filename mapping. Both files contain ``sorry``-stub
# theorems, except for two test entries that ship with proofs (patched below).
_SPLIT_FILES = {"valid": "Valid.lean", "test": "Test.lean"}


def download_dataset() -> None:
    """Download Valid.lean, Test.lean and ProblemImports.lean from the upstream GitHub repo."""
    for filename in [*_SPLIT_FILES.values(), "ProblemImports.lean"]:
        dest = os.path.join(DATA_DIR, filename)
        download_hf_file(BASE_URL + filename, dest, desc=filename)


def list_theorems(split: str) -> list[str]:
    assert split in _SPLIT_FILES, f"Invalid split: {split!r}. Must be one of {sorted(_SPLIT_FILES)}"
    file_path = Path(DATA_DIR) / _SPLIT_FILES[split]
    if not file_path.exists():
        raise FileNotFoundError(
            f"miniF2F file not found at {file_path}. Run with `download` first."
        )

    text = file_path.read_text()
    if split == "test":
        # Two upstream test theorems ship with proofs filled in instead of `sorry`.
        # TODO: submit a PR upstream so we can drop this patch.
        text = text.replace(
            "\ntheorem mathd_numbertheory_66 : 194 % 11 = 7 :=\n  rfl\n",
            "\ntheorem mathd_numbertheory_66 : 194 % 11 = 7 :=\n  sorry\n",
        )
        text = text.replace(
            "\ntheorem mathd_algebra_302 : (Complex.I / 2) ^ 2 = -(1 / 4) := by\n  norm_num [div_pow]\n",
            "\ntheorem mathd_algebra_302 : (Complex.I / 2) ^ 2 = -(1 / 4) := by\n  sorry\n",
        )

    theorems = []
    theorem_lines = []
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
            theorems.append("\n".join(theorem_lines))
            theorem_lines = []
        elif in_theorem:
            theorem_lines.append(line)

    assert all(t.count("sorry") == 1 for t in theorems), "Found a theorem with no or multiple `sorry`."
    expected_count = 256 if split == "valid" else 244
    assert len(theorems) == expected_count, f"minif2f: expected {expected_count} theorems, got {len(theorems)}"
    return theorems


def get_imports() -> str:
    """Return the Lean preamble (imports + open scoped namespaces) used by miniF2F problems."""
    file_path = Path(DATA_DIR) / "ProblemImports.lean"
    return file_path.read_text() + """
open scoped Real
open scoped Nat
open scoped Topology
open scoped Polynomial"""


# -----------------------------------------------------------------------------
# CLI: download / show / stats

def _main():
    parser = argparse.ArgumentParser(description="miniF2F benchmark dataset")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("download", help="Download Valid.lean, Test.lean, ProblemImports.lean from GitHub")
    show = sub.add_parser("show", help="Print the first N theorems from a split")
    show.add_argument("--split", choices=list(_SPLIT_FILES), default="valid")
    show.add_argument("--n", type=int, default=5)
    sub.add_parser("stats", help="Print theorem counts per split")
    args = parser.parse_args()

    if args.action == "download":
        download_dataset()
    elif args.action == "show":
        for thm in list_theorems(args.split)[:args.n]:
            print(thm)
            print("-" * 80)
    elif args.action == "stats":
        for split in _SPLIT_FILES:
            print(f"{split}: {len(list_theorems(split))} theorems")


if __name__ == "__main__":
    _main()
