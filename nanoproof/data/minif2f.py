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
    blocks = file_path.read_text().split("\n\n")
    theorems = []
    for block in blocks:
        lines = block.split("\n")
        theorem_line_idx = next((i for i, line in enumerate(lines) if line.startswith("theorem")), None)
        if theorem_line_idx is None:
            continue
        theorem = "\n".join([line.rstrip() for line in lines[theorem_line_idx:]])
        theorems.append(theorem.strip())
    assert all("sorry" in t for t in theorems), "Found a theorem with no `sorry`."
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
        raise f"Unknown action {args.action}"