"""DeepSeek-Prover-V1 dataset.

Public interface:
- ``download_dataset()`` - fetch the source JSONL from HuggingFace.
- ``list_theorems(split, lean_version=None)`` - return the parsed theorem
  statements for the requested split (``"train"`` or ``"valid"``). Each
  entry is a Lean source string ending in ``sorry``, ready to feed to
  ``proof_from_sorry``. If ``lean_version`` is given, filter via the
  matching on-disk whitelist.

CLI: see ``python -m nanoproof.data.rl.deepseek_prover --help``.

Note on disregarding proofs: in this dataset, each row's ``formal_statement``
field contains *only* the theorem header (always ending in ``:= by``); the
actual proof tactics live in a separate ``formal_proof`` field that we never
read. So "use only the statements" is automatically satisfied - we just take
``formal_statement`` and append ``sorry`` so the prover can attempt its own
proof.
"""

import argparse
import json
import os

from nanoproof.common import get_base_dir
from nanoproof.data.bench.common import BenchTheorem, DEEPSEEK_PROVER_PREAMBLE
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

DATA_DIR = os.path.join(get_base_dir(), "data", "deepseek_prover")
JSONL_PATH = os.path.join(DATA_DIR, "deepseek_prover.jsonl")
HF_URL = "https://huggingface.co/datasets/deepseek-ai/DeepSeek-Prover-V1/resolve/main/dataset.jsonl"


def _statement_only(formal_statement: str) -> str | None:
    """Append a ``sorry`` placeholder to a DeepSeek formal_statement so it can
    be fed to ``proof_from_sorry``. Returns ``None`` if the statement doesn't
    end cleanly with ``:= by`` or ``:=``.

    Conservative on purpose: we do NOT try to strip a proof body via partition
    on ``:=``, because theorem headers can contain ``let x := ...`` bindings
    and a naive split would truncate the statement. The current dataset never
    has a proof body inside ``formal_statement`` anyway.
    """
    text = formal_statement.strip()
    if text.endswith(":= by"):
        return text + " sorry"
    if text.endswith(":="):
        return text + " by sorry"
    return None


def download_dataset() -> None:
    download_file(HF_URL, JSONL_PATH, desc="deepseek_prover.jsonl")
    download_whitelists(JSONL_PATH)


def _load_sources() -> list[str]:
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

    if skipped > 0 and int(os.environ.get("RANK", 0)) == 0:
        print(f"Skipped {skipped} statements that could not be parsed (no `:=`)")
        if skipped_example is not None:
            print(f"Example skipped statement:\n{skipped_example}")
    return theorems


def list_theorems(split: str, lean_version: str | None = None) -> list[BenchTheorem]:
    assert split in ("train", "valid"), f"Invalid split: {split!r}"
    sources = _load_sources()
    split_sources = shuffle_train_valid_split(sources, valid_size=500, seed=0)[split]
    theorems = [
        BenchTheorem(source=DEEPSEEK_PROVER_PREAMBLE + s) for s in split_sources
    ]

    if lean_version is not None:
        theorems = filter_by_whitelist(
            theorems,
            whitelist_path(JSONL_PATH, lean_version),
            dataset_name=f"deepseek_prover/{split}",
        )
    return theorems


def _all_theorems() -> list[BenchTheorem]:
    """Every theorem across both splits, for whitelist generation."""
    return [BenchTheorem(source=DEEPSEEK_PROVER_PREAMBLE + s) for s in _load_sources()]


# -----------------------------------------------------------------------------
# CLI: download / show / stats / check-init


def _main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-Prover-V1 dataset", allow_abbrev=False
    )
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("download", help="Download the source JSONL from HuggingFace")
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
            dataset_file=JSONL_PATH,
            lean_server=args.lean_server,
            lean_project=args.lean_project,
            num_workers=args.jobs,
            limit=args.limit,
            verbose=args.verbose,
            save=True,
        )


if __name__ == "__main__":
    _main()
