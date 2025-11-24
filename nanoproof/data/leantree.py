import json
import os
import argparse
from itertools import islice
import time

import numpy as np
import requests

from tqdm import tqdm
import leantree

from nanoproof.common import get_base_dir
from nanoproof.tokenizer import get_tokenizer

base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "data", "leantree")

HF_URL = "https://huggingface.co/datasets/ufal/leantree/resolve/main/leantree_mathlib.jsonl"


def iter_data(split, eval_fraction=0.1):
    assert split in ["train", "val"]
    mathlib_file = os.path.join(DATA_DIR, "leantree_mathlib.jsonl")
    with open(mathlib_file, "r") as f:
        lines = f.readlines()
    eval_size = int(len(lines) * eval_fraction)
    lines = lines[:-eval_size] if split == "train" else lines[-eval_size:]

    for line in lines:
        lean_file = leantree.LeanFile.deserialize(json.loads(line))
        for thm in lean_file.theorems:
            if isinstance(thm, leantree.StoredError):
                continue
            for by_block in thm.by_blocks:
                if isinstance(by_block.tree, leantree.StoredError):
                    continue
                for node in by_block.tree.get_nodes():
                    yield {
                        "state": str(node.state),
                        "tactic": str(node.tactic.tactic)
                    }


def download_dataset():
    """Download the leantree dataset from HuggingFace."""
    jsonl_path = os.path.join(DATA_DIR, "leantree_mathlib.jsonl")

    # skip if already downloaded
    if os.path.exists(jsonl_path):
        print(f"Dataset already downloaded at {jsonl_path}")
        return

    try:
        print(f"Downloading leantree dataset from HuggingFace...")
        response = requests.get(HF_URL, stream=True, timeout=60)
        response.raise_for_status()

        temp_path = jsonl_path + ".tmp"
        total_size = int(response.headers.get("content-length", 0))
        with open(temp_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading leantree_mathlib.jsonl") as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        os.rename(temp_path, jsonl_path)
        print(f"Successfully downloaded {jsonl_path}")
    except (requests.RequestException, IOError):
        # Clean up any partial files
        for path in [jsonl_path + ".tmp", jsonl_path]:
            if os.path.exists(path):
                print(f"Cleaning up {path}")
                os.remove(path)
        raise

def main():
    parser = argparse.ArgumentParser(description="Download LeanTree dataset from HuggingFace.")
    subparsers = parser.add_subparsers(dest="action")

    download_parser = subparsers.add_parser("download")

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("--split", choices=["train", "val"], default="train")

    stats_parser = subparsers.add_parser("stats")

    args = parser.parse_args()

    if args.action == "download":
        os.makedirs(DATA_DIR, exist_ok=True)
        download_dataset()
    elif args.action == "show":
        for d in islice(iter_data(split=args.split), 10):
            print(d["state"])
            print("\n->\n")
            print(d["tactic"])
            print("\n-----------------\n")
    elif args.action == "stats":
        tokenizer = get_tokenizer()
        for split in ["train", "val"]:
            print(f"Calculating {split=}...")
            lens = {"state": [], "tactic": []}
            start_time = time.time()
            for d in iter_data(split=split):
                state = tokenizer.encode(d["state"])
                tactic = tokenizer.encode("[TACTIC]" + d["tactic"])
                lens["state"].append(len(state))
                lens["tactic"].append(len(tactic))
            end_time = time.time()
            print(f"time: {end_time - start_time:.2f}s")
            print(f"total: {len(lens['state'])}")
            for prop, max_len in [("state", 1536), ("tactic", 512)]:
                print(f"stats for {prop}:")
                print(f"  min: {np.min(lens[prop])}")
                print(f"  max: {np.max(lens[prop])}")
                print(f"  mean: {np.mean(lens[prop]):.2f}")
                print(f"  median: {np.median(lens[prop])}")
                print(f"  std: {np.std(lens[prop]):.2f}")
                print(f"  p90: {np.percentile(lens[prop], 90):.2f}")
                print(f"  p95: {np.percentile(lens[prop], 95):.2f}")
                print(f"  p99: {np.percentile(lens[prop], 99):.2f}")
                at_most_max = np.sum(np.array(lens[prop]) <= max_len)
                print(f"  <= {max_len}: {at_most_max / len(lens[prop]):%}% ({at_most_max}/{len(lens[prop])})")
            print()

if __name__ == "__main__":
    main()
