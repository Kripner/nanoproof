"""Lean-Workbook dataset (only theorems with a known InternLM-Prover proof).

Public interface:
- ``download_dataset()`` — fetch the source JSON from HuggingFace.
- ``list_theorems(split)`` — return the parsed formal statements for the
  requested split (``"train"`` or ``"valid"``). The dataset has no test split.

CLI: see ``python -m nanoproof.data.rl.leanworkbook --help``.
"""

import argparse
import json
import os

from nanoproof.common import get_base_dir
from nanoproof.data.rl.common import download_hf_file, shuffle_train_valid_split

DATA_DIR = os.path.join(get_base_dir(), "data", "leanworkbook")
JSON_PATH = os.path.join(DATA_DIR, "lean_workbook.json")
HF_URL = "https://huggingface.co/datasets/internlm/Lean-Workbook/resolve/main/lean_workbook.json"


def download_dataset() -> None:
    download_hf_file(HF_URL, JSON_PATH, desc="lean_workbook.json")


def list_theorems(split: str) -> list[str]:
    assert split in ("train", "valid"), f"Invalid split: {split!r}"
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(
            f"Lean-Workbook dataset not found at {JSON_PATH}. Run with `download` first."
        )
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    # Keep only entries that InternLM Prover successfully proved (we don't
    # use the proof itself, but proven theorems are higher quality).
    theorems = [item["formal_statement"] for item in data if item["proof"]]
    return shuffle_train_valid_split(theorems, valid_size=500, seed=0)[split]


# -----------------------------------------------------------------------------
# CLI: download / show / stats

def _main():
    parser = argparse.ArgumentParser(description="Lean-Workbook dataset")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("download", help="Download the source JSON from HuggingFace")
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
