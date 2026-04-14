"""miniF2F benchmark dataset.

Public interface:
- ``download_dataset()`` - fetch the .lean files from the GitHub repo.
- ``list_theorems(split)`` - return the parsed theorems for the requested
  split (``"valid"`` or ``"test"``), each wrapped as a ``BenchTheorem`` with
  the shared ``MINIF2F_HEADER``. The dataset has no train split.

CLI: see ``python -m nanoproof.data.bench.minif2f --help``.
"""

import argparse
import os
from pathlib import Path

from leantree.repl_adapter.server import LeanClient

from nanoproof.common import get_base_dir, theorem_to_example
from nanoproof.data.bench.common import BenchTheorem, MINIF2F_HEADER
from nanoproof.data.rl.common import download_hf_file

DATA_DIR = os.path.join(get_base_dir(), "data", "minif2f")
BASE_URL = "https://raw.githubusercontent.com/google-deepmind/miniF2F/refs/heads/main/MiniF2F/"

# The split name -> source filename mapping. Both files contain ``sorry``-stub
# theorems, except for two test entries that ship with proofs (patched below).
_SPLIT_FILES = {"valid": "Valid.lean", "test": "Test.lean"}


def download_dataset() -> None:
    """Download Valid.lean and Test.lean from the upstream GitHub repo."""
    for filename in _SPLIT_FILES.values():
        dest = os.path.join(DATA_DIR, filename)
        download_hf_file(BASE_URL + filename, dest, desc=filename)


def list_theorems(split: str) -> list[BenchTheorem]:
    assert split in _SPLIT_FILES, f"Invalid split: {split!r}. Must be one of {sorted(_SPLIT_FILES)}"
    file_path = Path(DATA_DIR) / _SPLIT_FILES[split]
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

    assert all(s.count("sorry") == 1 for s in sources), "Found a theorem with no or multiple `sorry`."
    expected_count = 256 if split == "valid" else 244
    assert len(sources) == expected_count, f"minif2f: expected {expected_count} theorems, got {len(sources)}"
    return [BenchTheorem(source=s, header=MINIF2F_HEADER) for s in sources]


def _check_init(split: str, lean_server: str, limit: int | None) -> None:
    """Try ``proof_from_sorry`` on each theorem and print the ones that fail.

    Mirrors the setup used by ``Prover.prove`` (sends ``theorem.header`` and
    runs ``theorem_to_example(theorem.source)``) so failures reproduce what
    the RL / prover_eval loops hit with "FAILED: Could not initialize proof".
    """
    theorems = list_theorems(split)
    if limit is not None:
        theorems = theorems[:limit]

    if ":" in lean_server:
        host, port_str = lean_server.rsplit(":", 1)
        port = int(port_str)
    else:
        host, port = lean_server, 8000

    print(f"Connecting to Lean server {host}:{port}...")
    client = LeanClient(host, port)
    process = client.get_process()
    if process is None:
        print(f"Failed to acquire Lean process from {host}:{port}")
        return

    num_failed = 0
    with process as env:
        for i, theorem in enumerate(theorems):
            env.send_command(theorem.header)
            example = theorem_to_example(theorem.source)
            try:
                init_branch = env.proof_from_sorry(example)
            except Exception as e:
                num_failed += 1
                print(f"[{i}] EXCEPTION: {e}\n--- theorem ---\n{theorem.source}\n")
                continue
            if not init_branch.is_success():
                num_failed += 1
                err = init_branch.error if hasattr(init_branch, 'error') else 'unknown error'
                print(f"[{i}] FAILED: {err}\n--- theorem ---\n{theorem.source}\n")
            else:
                print(f"[{i}] ok")

    print(f"\nDone: {num_failed}/{len(theorems)} theorems failed to initialize.")


# -----------------------------------------------------------------------------
# CLI: download / show / stats / check-init

def _main():
    parser = argparse.ArgumentParser(description="miniF2F benchmark dataset")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("download", help="Download Valid.lean and Test.lean from GitHub")
    show = sub.add_parser("show", help="Print the first N theorems from a split")
    show.add_argument("--split", choices=list(_SPLIT_FILES), default="valid")
    show.add_argument("--n", type=int, default=5)
    sub.add_parser("stats", help="Print theorem counts per split")
    check = sub.add_parser(
        "check-init",
        help="Try to initialize each theorem's proof in a Lean REPL and report failures",
    )
    check.add_argument("--split", choices=list(_SPLIT_FILES), default="valid")
    check.add_argument("--lean-server", type=str, required=True,
                       help="Lean server address (e.g., 10.10.25.33:8000); port defaults to 8000")
    check.add_argument("--limit", type=int, default=None, help="Only check the first N theorems")
    args = parser.parse_args()

    if args.action == "download":
        download_dataset()
    elif args.action == "show":
        for thm in list_theorems(args.split)[:args.n]:
            print(thm.source)
            print("-" * 80)
    elif args.action == "stats":
        for split in _SPLIT_FILES:
            print(f"{split}: {len(list_theorems(split))} theorems")
    elif args.action == "check-init":
        _check_init(args.split, args.lean_server, args.limit)


if __name__ == "__main__":
    _main()
