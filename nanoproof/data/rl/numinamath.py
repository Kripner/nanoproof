"""NuminaMath-LEAN dataset.

Public interface:
- ``download_dataset()`` - fetch the source parquet from HuggingFace.
- ``list_theorems(split)`` - return the parsed formal statements for the
  requested split (``"train"`` or ``"valid"``). The dataset has no test split.
  Each entry is a Lean source string ending in ``sorry``.

CLI: see ``python -m nanoproof.data.rl.numinamath --help``.
"""

import argparse
import os
import re

import pyarrow.parquet as pq

from nanoproof.common import get_base_dir
from nanoproof.data.rl.common import download_hf_file, shuffle_train_valid_split

DATA_DIR = os.path.join(get_base_dir(), "data", "numinamath")
PARQUET_PATH = os.path.join(DATA_DIR, "numinamath.parquet")
HF_URL = "https://huggingface.co/datasets/AI-MO/NuminaMath-LEAN/resolve/main/data/train-00000-of-00001.parquet"


def _process_statement(statement: str) -> str | None:
    """Strip leading directive lines and append a `sorry` placeholder.
    Returns None if the statement doesn't end with `:=`, `:= by`, or
    `:=` followed by whitespace and ``by``.
    """
    statement = statement.strip()

    # Drop leading import / set_option / #check lines
    lines = statement.split("\n")
    while lines and (lines[0].startswith("import ") or lines[0].startswith("set_option ") or lines[0].startswith("#check ")):
        lines.pop(0)
    statement = "\n".join(lines).rstrip()

    if re.search(r":=\s*by\s*$", statement):
        return statement + " sorry"
    if statement.endswith(":="):
        return statement + " by sorry"
    return None


def download_dataset() -> None:
    download_hf_file(HF_URL, PARQUET_PATH, desc="numinamath.parquet")


def list_theorems(split: str) -> list[str]:
    assert split in ("train", "valid"), f"Invalid split: {split!r}"
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(
            f"NuminaMath-LEAN dataset not found at {PARQUET_PATH}. Run with `download` first."
        )

    table = pq.read_table(PARQUET_PATH)
    raw_statements = table.column("formal_statement").to_pylist()

    theorems = []
    skipped = 0
    skipped_example = None
    for stmt in raw_statements:
        processed = _process_statement(stmt)
        if processed is None:
            skipped += 1
            if skipped_example is None:
                skipped_example = stmt
            continue
        theorems.append(processed)

    if skipped > 0:
        print(f"Skipped {skipped} statements that don't end with `:= by` or `:=`")
        if skipped_example is not None:
            print(f"Example skipped statement:\n{skipped_example}")

    return shuffle_train_valid_split(theorems, valid_size=500, seed=0)[split]


# -----------------------------------------------------------------------------
# CLI: download / show / stats

def _main():
    parser = argparse.ArgumentParser(description="NuminaMath-LEAN dataset")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("download", help="Download the source parquet from HuggingFace")
    show = sub.add_parser("show", help="Print the first N theorems from a split")
    show.add_argument("--split", choices=["train", "valid"], default="train")
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
        for split in ("train", "valid"):
            print(f"{split}: {len(list_theorems(split))} theorems")


if __name__ == "__main__":
    _main()
