import os
import argparse
from pathlib import Path

import requests

from nanoproof.common import get_base_dir

base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "data", "minif2f")

BASE_URL = "https://raw.githubusercontent.com/google-deepmind/miniF2F/refs/heads/main/MiniF2F/"


def list_theorems(split):
    assert split in ["Valid", "Test"]
    file_path = Path(DATA_DIR) / f"{split}.lean"

    theorems = []
    theorem_lines = []
    in_theorem = False
    text = file_path.read_text()
    if split == "Test":
        # These are mistakes in the test file - a proof is already filled in. TODO: submit a pull request
        text = text.replace("""
theorem mathd_numbertheory_66 : 194 % 11 = 7 :=
  rfl
""",
"""
theorem mathd_numbertheory_66 : 194 % 11 = 7 :=
  sorry
""")
        text = text.replace("""
theorem mathd_algebra_302 : (Complex.I / 2) ^ 2 = -(1 / 4) := by
  norm_num [div_pow]
""",
"""
theorem mathd_algebra_302 : (Complex.I / 2) ^ 2 = -(1 / 4) := by
  sorry
""")
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
    expected_count = 256 if split == "Valid" else 244
    assert len(theorems) == expected_count, f"minif2f: expected {expected_count} theorems, got {len(theorems)}"
    return theorems


def get_imports():
    file_path = Path(DATA_DIR) / "ProblemImports.lean"
    return file_path.read_text() + """
open scoped Real
open scoped Nat
open scoped Topology
open scoped Polynomial"""
        
def download_dataset():
    """Download the miniF2F dataset from GitHub."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for filename in ["Valid.lean", "Test.lean", "ProblemImports.lean"]:
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(file_path):
            print(f"File already exists, skipping: {file_path}")
            continue
        
        url = BASE_URL + filename
        print(f"Downloading {filename} from {url}...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        
        print(f"Successfully downloaded {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")
    download_parser = subparsers.add_parser("download")
    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("--split", choices=["Valid", "Test"], default="Valid")
    args = parser.parse_args()

    if args.action == "download":
        download_dataset()
    elif args.action == "show":
        for theorem in list_theorems(args.split):
            print(theorem)
            print("\n-----------------\n")
    else:
        raise ValueError(f"Unknown action {args.action}")