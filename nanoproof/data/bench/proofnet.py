"""ProofNet benchmark dataset.

Public interface:
- ``download_dataset()`` - fetch the .jsonl file.
- ``list_theorems(split)`` - return the parsed theorems for the requested
  split (``"valid"`` or ``"test"``) as ``BenchTheorem``. Each theorem's
  ``source`` includes per-theorem opens / auxiliary ``def``s from the
  upstream record (25 distinct preambles across 371 theorems).

CLI: see ``python -m nanoproof.data.bench.proofnet --help``.
"""

import argparse
import json
import os
from pathlib import Path

from nanoproof.common import get_base_dir
from nanoproof.data.bench.common import BenchTheorem
from nanoproof.data.check_init import add_check_init_args, run_check_init_cli
from nanoproof.data.rl.common import download_file

DATA_DIR = os.path.join(get_base_dir(), "data", "proofnet")
SOURCE_URL = "https://raw.githubusercontent.com/Kripner/ProofNet/refs/heads/main/data/proofnet.jsonl"
FILENAME = "proofnet.jsonl"
FILE_PATH = os.path.join(DATA_DIR, FILENAME)

_SPLITS = ("valid", "test")


def download_dataset() -> None:
    """Download proofnet.jsonl from the ProofNet GitHub repo."""
    download_file(SOURCE_URL, FILE_PATH, desc=FILENAME)


def _load_records() -> list[dict]:
    file_path = Path(FILE_PATH)
    if not file_path.exists():
        raise FileNotFoundError(
            f"ProofNet file not found at {file_path}. Run with `download` first."
        )
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _strip_imports(header: str) -> str:
    """Remove ``import`` lines from a raw proofnet header.

    Imports are applied server-side once per process. The remaining body
    (opens, open scoped, and any auxiliary defs) becomes part of the source.
    """
    lines = header.split("\n")
    out = [line for line in lines if not line.strip().startswith("import ")]
    return "\n".join(out).strip()


def list_theorems(split: str) -> list[BenchTheorem]:
    assert split in _SPLITS, f"Invalid split: {split!r}. Must be one of {list(_SPLITS)}"
    records = [r for r in _load_records() if r["split"] == split]
    theorems: list[BenchTheorem] = []
    for r in records:
        # Upstream ``formal_statement`` ends with ``:=`` (term-mode placeholder,
        # no body). Emit tactic-mode ``:= by\n  sorry`` so the shape matches
        # miniF2F exactly and we don't rely on leantree's ``:= sorry`` regex
        # rewrite in ``_eliminate_sorry_without_by``.
        stmt = r["formal_statement"].rstrip()
        if stmt.endswith(":= by"):
            source = stmt + "\n  sorry"
        elif stmt.endswith(":="):
            source = stmt + " by\n  sorry"
        else:
            raise ValueError(
                f"unexpected suffix in formal_statement for {r.get('name')!r}: {stmt!r}"
            )
        preamble = _strip_imports(r["header"])
        if preamble:
            source = preamble + "\n\n" + source
        theorems.append(BenchTheorem(source=source, name=r.get("name")))
    assert all(t.source.count("sorry") == 1 for t in theorems), (
        "Found a theorem with no or multiple `sorry`."
    )
    return theorems


# -----------------------------------------------------------------------------
# CLI: download / show / stats / check-init


def _main():
    parser = argparse.ArgumentParser(
        description="ProofNet benchmark dataset", allow_abbrev=False
    )
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("download", help=f"Download {FILENAME} from GitHub")
    show = sub.add_parser("show", help="Print the first N theorems from a split")
    show.add_argument("--split", choices=list(_SPLITS), default="valid")
    show.add_argument("--n", type=int, default=5)
    sub.add_parser("stats", help="Print theorem counts per split")
    export = sub.add_parser("export", help="Export theorems to a standalone .lean file")
    export.add_argument("--split", choices=list(_SPLITS), default="valid")
    export.add_argument("output", help="Path to the output .lean file")
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
        for thm in list_theorems(args.split)[: args.n]:
            print(f"# {thm.name}")
            print(thm.source)
            print("-" * 80)
    elif args.action == "stats":
        for split in _SPLITS:
            print(f"{split}: {len(list_theorems(split))} theorems")
    elif args.action == "export":
        theorems = list_theorems(args.split)
        with open(args.output, "w") as f:
            f.write("import Mathlib\n")
            for thm in theorems:
                f.write("\n" + thm.source + "\n")
        print(f"Exported {len(theorems)} theorems to {args.output}")
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
