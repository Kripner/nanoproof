"""Lean-Workbook dataset (only theorems with a known InternLM-Prover proof).

Public interface:
- ``download_dataset()`` - fetch the source JSON from HuggingFace.
- ``list_theorems(split, lean_version=None)`` - return the parsed formal
  statements for the requested split (``"train"`` or ``"valid"``). If
  ``lean_version`` is given, filter via the matching on-disk whitelist.

CLI: see ``python -m nanoproof.data.rl.leanworkbook --help``.
"""

import argparse
import json
import os

from nanoproof.common import get_base_dir
from nanoproof.data.bench.common import BenchTheorem, LEANWORKBOOK_PREAMBLE
from nanoproof.data.check_init import (
    add_check_init_args,
    filter_by_whitelist,
    run_check_init_cli,
    whitelist_path,
)
from nanoproof.data.rl.common import (
    download_file,
    download_whitelists,
    shuffle_train_valid_split,
)

DATA_DIR = os.path.join(get_base_dir(), "data", "leanworkbook")
JSON_PATH = os.path.join(DATA_DIR, "lean_workbook.json")
HF_URL = "https://huggingface.co/datasets/internlm/Lean-Workbook/resolve/main/lean_workbook.json"


def download_dataset() -> None:
    download_file(HF_URL, JSON_PATH, desc="lean_workbook.json")
    download_whitelists(JSON_PATH)


def _load_sources() -> list[str]:
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(
            f"Lean-Workbook dataset not found at {JSON_PATH}. Run with `download` first."
        )
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    # Keep only entries that InternLM Prover successfully proved (we don't
    # use the proof itself, but proven theorems are higher quality).
    return [item["formal_statement"] for item in data if item["proof"]]


def list_theorems(split: str, lean_version: str | None = None) -> list[BenchTheorem]:
    assert split in ("train", "valid"), f"Invalid split: {split!r}"
    sources = _load_sources()
    split_sources = shuffle_train_valid_split(sources, valid_size=500, seed=0)[split]
    theorems = [BenchTheorem(source=LEANWORKBOOK_PREAMBLE + s) for s in split_sources]

    if lean_version is not None:
        theorems = filter_by_whitelist(
            theorems,
            whitelist_path(JSON_PATH, lean_version),
            dataset_name=f"leanworkbook/{split}",
        )
    return theorems


def _all_theorems() -> list[BenchTheorem]:
    """Every theorem across both splits, for whitelist generation."""
    return [BenchTheorem(source=LEANWORKBOOK_PREAMBLE + s) for s in _load_sources()]


# -----------------------------------------------------------------------------
# CLI: download / show / stats / check-init


def _main():
    parser = argparse.ArgumentParser(
        description="Lean-Workbook dataset", allow_abbrev=False
    )
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("download", help="Download the source JSON from HuggingFace")
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
        for thm in list_theorems(args.split)[: args.n]:
            print(thm.source)
            print("-" * 80)
    elif args.action == "stats":
        for split in ("train", "valid"):
            print(f"{split}: {len(list_theorems(split))} theorems")
    elif args.action == "check-init":
        run_check_init_cli(
            theorems=_all_theorems(),
            dataset_file=JSON_PATH,
            lean_server=args.lean_server,
            lean_project=args.lean_project,
            num_workers=args.jobs,
            limit=args.limit,
            verbose=args.verbose,
            save=True,
        )


if __name__ == "__main__":
    _main()
