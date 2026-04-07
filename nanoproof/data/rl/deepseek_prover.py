"""DeepSeek-Prover-V1 dataset.

Public interface:
- ``download_dataset()`` — fetch the source JSONL from HuggingFace.
- ``list_theorems(split)`` — return the parsed theorem statements for the
  requested split (``"train"`` or ``"valid"``). Each entry is a Lean source
  string ending in ``:= by sorry``, ready to feed to ``proof_from_sorry``.

CLI: see ``python -m nanoproof.data.rl.deepseek_prover --help``.

Note: rows in the source dataset contain a ``formal_statement`` that often
includes a complete proof body. We disregard those proofs and keep only the
statement portion (everything before the first ``:=``), then append
``:= by sorry`` so the prover can attempt its own proof.
"""

import argparse
import json
import os

from nanoproof.common import get_base_dir
from nanoproof.data.rl.common import download_hf_file, shuffle_train_valid_split

DATA_DIR = os.path.join(get_base_dir(), "data", "deepseek_prover")
JSONL_PATH = os.path.join(DATA_DIR, "deepseek_prover.jsonl")
HF_URL = "https://huggingface.co/datasets/deepseek-ai/DeepSeek-Prover-V1/resolve/main/dataset.jsonl"


def _statement_only(formal_statement: str) -> str | None:
    """Strip any proof body from a DeepSeek formal_statement and return just
    the statement, terminated with ``:= by sorry``. Returns ``None`` if the
    input has no ``:=`` and therefore can't be parsed as a Lean declaration.
    """
    text = formal_statement.strip()
    prefix, sep, _ = text.partition(":=")
    if not sep or not prefix.strip():
        return None
    return prefix.rstrip() + " := by sorry"


def download_dataset() -> None:
    download_hf_file(HF_URL, JSONL_PATH, desc="deepseek_prover.jsonl")


def list_theorems(split: str) -> list[str]:
    assert split in ("train", "valid"), f"Invalid split: {split!r}"
    if not os.path.exists(JSONL_PATH):
        raise FileNotFoundError(
            f"DeepSeek-Prover-V1 dataset not found at {JSONL_PATH}. Run with `download` first."
        )

    with open(JSONL_PATH, "r") as f:
        rows = [json.loads(line) for line in f]

    theorems = []
    skipped = 0
    skipped_example = None
    for row in rows:
        raw = row.get("formal_statement")
        if raw is None:
            skipped += 1
            continue
        processed = _statement_only(raw)
        if processed is None:
            skipped += 1
            if skipped_example is None:
                skipped_example = raw
            continue
        theorems.append(processed)

    if skipped > 0:
        print(f"Skipped {skipped} statements that could not be parsed (no `:=`)")
        if skipped_example is not None:
            print(f"Example skipped statement:\n{skipped_example}")

    return shuffle_train_valid_split(theorems, valid_size=500, seed=0)[split]


# -----------------------------------------------------------------------------
# CLI: download / show / stats

def _main():
    parser = argparse.ArgumentParser(description="DeepSeek-Prover-V1 dataset")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("download", help="Download the source JSONL from HuggingFace")
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
