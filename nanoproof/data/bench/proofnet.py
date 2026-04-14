"""ProofNet benchmark dataset.

Public interface:
- ``download_dataset()`` - fetch the .jsonl file.
- ``list_theorems(split)`` - return the parsed theorems for the requested
  split (``"valid"`` or ``"test"``) as ``BenchTheorem``. Each carries its own
  ``header`` (opens + any auxiliary ``def``s) taken from the upstream record;
  the dataset uses 25 distinct headers across 371 theorems, so a single
  shared preamble is not workable.

CLI: see ``python -m nanoproof.data.bench.proofnet --help``.
"""

import argparse
import json
import os
from pathlib import Path

from nanoproof.common import get_base_dir
from nanoproof.data.bench.common import BenchTheorem
from nanoproof.data.check_init import add_check_init_args, run_check_init_cli
from nanoproof.data.rl.common import download_hf_file

DATA_DIR = os.path.join(get_base_dir(), "data", "proofnet")
SOURCE_URL = "https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Prover-V1.5/refs/heads/main/datasets/proofnet.jsonl"
FILENAME = "proofnet.jsonl"
FILE_PATH = os.path.join(DATA_DIR, FILENAME)

_SPLITS = ("valid", "test")

_IMPORT_PREFIX = "import Mathlib"


def download_dataset() -> None:
    """Download proofnet.jsonl from the DeepSeek-Prover-V1.5 GitHub repo."""
    download_hf_file(SOURCE_URL, FILE_PATH, desc=FILENAME)


def _load_records() -> list[dict]:
    file_path = Path(FILE_PATH)
    if not file_path.exists():
        raise FileNotFoundError(
            f"ProofNet file not found at {file_path}. Run with `download` first."
        )
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _strip_import_mathlib(header: str) -> str:
    """Remove the `import Mathlib` line from a raw proofnet header.

    Imports are applied server-side once per process. The remaining body
    (opens, open scoped, and any auxiliary defs) becomes BenchTheorem.header.
    """
    lines = header.split("\n")
    out = []
    for line in lines:
        if line.strip() == _IMPORT_PREFIX:
            continue
        out.append(line)
    return "\n".join(out).strip()


def list_theorems(split: str) -> list[BenchTheorem]:
    assert split in _SPLITS, f"Invalid split: {split!r}. Must be one of {list(_SPLITS)}"
    records = [r for r in _load_records() if r["split"] == split]
    theorems: list[BenchTheorem] = []
    for r in records:
        # ``formal_statement`` ends with ``:=`` (no body); append a ``sorry``
        # stub so it matches the minif2f shape and feeds ``proof_from_sorry``.
        source = r["formal_statement"].rstrip() + "\n  sorry"
        header = _strip_import_mathlib(r["header"])
        theorems.append(BenchTheorem(source=source, header=header, name=r.get("name")))
    assert all(t.source.count("sorry") == 1 for t in theorems), "Found a theorem with no or multiple `sorry`."
    return theorems


# -----------------------------------------------------------------------------
# CLI: download / show / stats / check-init

def _main():
    parser = argparse.ArgumentParser(description="ProofNet benchmark dataset")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("download", help=f"Download {FILENAME} from GitHub")
    show = sub.add_parser("show", help="Print the first N theorems from a split")
    show.add_argument("--split", choices=list(_SPLITS), default="valid")
    show.add_argument("--n", type=int, default=5)
    sub.add_parser("stats", help="Print theorem counts per split")
    check = sub.add_parser(
        "check-init",
        help="Try to initialize each theorem's proof in a Lean REPL and report failures "
        "(benchmarks do not get whitelists - failures are warnings)",
    )
    check.add_argument("--split", choices=list(_SPLITS), default="valid")
    add_check_init_args(check, default_jobs=1)
    args = parser.parse_args()

    if args.action == "download":
        download_dataset()
    elif args.action == "show":
        for thm in list_theorems(args.split)[:args.n]:
            print(f"# {thm.name}")
            print(f"# header:\n{thm.header}")
            print(thm.source)
            print("-" * 80)
    elif args.action == "stats":
        for split in _SPLITS:
            print(f"{split}: {len(list_theorems(split))} theorems")
    elif args.action == "check-init":
        run_check_init_cli(
            theorems=list_theorems(args.split),
            dataset_file=FILE_PATH,
            lean_server=args.lean_server,
            lean_project=args.lean_project,
            num_workers=args.jobs,
            limit=args.limit,
            verbose=args.verbose,
            save=False,
        )


if __name__ == "__main__":
    _main()
