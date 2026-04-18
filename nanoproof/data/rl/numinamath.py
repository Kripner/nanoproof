"""NuminaMath-LEAN dataset.

Public interface:
- ``download_dataset()`` - fetch the source parquet from HuggingFace.
- ``list_theorems(split, lean_version=None)`` - return the parsed formal
  statements for the requested split (``"train"`` or ``"valid"``). The dataset
  has no test split. Each entry is a Lean source string ending in ``sorry``.
  If ``lean_version`` is given, filter via the matching on-disk whitelist.

CLI: see ``python -m nanoproof.data.rl.numinamath --help``.
"""

import argparse
import os
import re

import pyarrow.parquet as pq

from nanoproof.common import get_base_dir
from nanoproof.data.bench.common import BenchTheorem
from nanoproof.data.check_init import (
    add_check_init_args,
    filter_by_whitelist,
    run_check_init_cli,
    whitelist_path,
)
from nanoproof.data.rl.common import download_hf_file, shuffle_train_valid_split

DATA_DIR = os.path.join(get_base_dir(), "data", "numinamath")
PARQUET_PATH = os.path.join(DATA_DIR, "numinamath.parquet")
HF_URL = "https://huggingface.co/datasets/AI-MO/NuminaMath-LEAN/resolve/main/data/train-00000-of-00001.parquet"

# Lines starting with these prefixes are stripped from the raw
# ``formal_statement``. Imports are server-side; ``set_option`` values
# (heartbeats, recursion depth) are server defaults and per-problem
# overrides would be unsafe.
_STRIP_PREFIXES = ("import ", "set_option ", "#check ")


def _process_statement(statement: str) -> str | None:
    """Strip import / set_option / #check lines and append a ``sorry``
    placeholder. Returns None if the statement doesn't end with ``:=``,
    ``:= by``, or ``:=`` followed by whitespace and ``by``.

    ``open`` lines are kept - they are per-theorem preamble that stays in
    ``source`` alongside the theorem declaration.
    """
    statement = statement.strip()

    lines = statement.split("\n")
    kept = []
    for line in lines:
        stripped = line.lstrip()
        if any(stripped.startswith(p) for p in _STRIP_PREFIXES):
            continue
        kept.append(line)
    statement = "\n".join(kept).strip()

    if re.search(r":=\s*by\s*$", statement):
        result = statement + " sorry"
    elif statement.endswith(":="):
        result = statement + " by sorry"
    else:
        return None

    # Skip statements that already contain sorry (e.g. auxiliary lemmas
    # with sorry proofs). The prover expects exactly one sorry -- the one
    # we just appended for the main theorem.
    if result.count("sorry") > 1:
        return None
    return result


def download_dataset() -> None:
    download_hf_file(HF_URL, PARQUET_PATH, desc="numinamath.parquet")


def _load_sources() -> list[str]:
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

    if skipped > 0 and int(os.environ.get("RANK", 0)) == 0:
        print(f"Skipped {skipped} statements that could not be parsed (no `:=`, or multiple `sorry`)")
        if skipped_example is not None:
            print(f"Example skipped statement:\n{skipped_example}")
    return theorems


def list_theorems(split: str, lean_version: str | None = None) -> list[BenchTheorem]:
    assert split in ("train", "valid"), f"Invalid split: {split!r}"
    sources = _load_sources()
    split_sources = shuffle_train_valid_split(sources, valid_size=500, seed=0)[split]
    theorems = [BenchTheorem(source=s) for s in split_sources]

    if lean_version is not None:
        theorems = filter_by_whitelist(
            theorems,
            whitelist_path(PARQUET_PATH, lean_version),
            dataset_name=f"numinamath/{split}",
        )
    return theorems


def _all_theorems() -> list[BenchTheorem]:
    """Every theorem across both splits, for whitelist generation."""
    return [BenchTheorem(source=s) for s in _load_sources()]


# -----------------------------------------------------------------------------
# CLI: download / show / stats / check-init

def _main():
    parser = argparse.ArgumentParser(description="NuminaMath-LEAN dataset", allow_abbrev=False)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("download", help="Download the source parquet from HuggingFace")
    show = sub.add_parser("show", help="Print the first N theorems from a split")
    show.add_argument("--split", choices=["train", "valid"], default="train")
    show.add_argument("--n", type=int, default=5)
    sub.add_parser("stats", help="Print theorem counts per split")
    check = sub.add_parser(
        "check-init",
        help="Try to initialize each theorem's proof in a Lean REPL and write a whitelist",
    )
    add_check_init_args(check, default_jobs=0)
    args = parser.parse_args()

    if args.action == "download":
        download_dataset()
    elif args.action == "show":
        for thm in list_theorems(args.split)[:args.n]:
            print(thm.source)
            print("-" * 80)
    elif args.action == "stats":
        for split in ("train", "valid"):
            print(f"{split}: {len(list_theorems(split))} theorems")
    elif args.action == "check-init":
        run_check_init_cli(
            theorems=_all_theorems(),
            dataset_file=PARQUET_PATH,
            lean_server=args.lean_server,
            lean_project=args.lean_project,
            num_workers=args.jobs,
            limit=args.limit,
            verbose=args.verbose,
            save=True,
        )


if __name__ == "__main__":
    _main()
