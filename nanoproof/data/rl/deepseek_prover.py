import json
import os
import random

import requests
from tqdm import tqdm

from nanoproof.common import get_base_dir

base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "data", "deepseek_prover")

# https://huggingface.co/datasets/deepseek-ai/DeepSeek-Prover-V1
HF_URL = "https://huggingface.co/datasets/deepseek-ai/DeepSeek-Prover-V1/resolve/main/dataset.jsonl"


def download_dataset():
    """Download the DeepSeek-Prover-V1 dataset from HuggingFace."""
    os.makedirs(DATA_DIR, exist_ok=True)
    jsonl_path = os.path.join(DATA_DIR, "deepseek_prover.jsonl")

    # skip if already downloaded
    if os.path.exists(jsonl_path):
        print(f"Dataset already downloaded at {jsonl_path}")
        return

    try:
        print(f"Downloading DeepSeek-Prover-V1 dataset from HuggingFace...")
        response = requests.get(HF_URL, stream=True, timeout=60)
        response.raise_for_status()

        temp_path = jsonl_path + ".tmp"
        total_size = int(response.headers.get("content-length", 0))
        with open(temp_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading deepseek_prover.jsonl") as pbar:
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


def _process_statement(statement: str) -> str | None:
    """Process a formal statement: strip, validate ends with 'by', append sorry."""
    statement = statement.strip()
    
    # Check that the statement ends with "by" (with or without := before it)
    if not statement.endswith("by"):
        return None
    
    # Append " sorry"
    return statement + " sorry"


def list_theorems(split: str):
    assert split in ["train", "val"], f"Invalid split: {split}. Must be 'train' or 'val'."
    
    jsonl_path = os.path.join(DATA_DIR, "deepseek_prover.jsonl")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"DeepSeek-Prover-V1 dataset not found at {jsonl_path}. Download it first.")
    
    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]
    # Filter out rows without formal_statement
    raw_statements = [item["formal_statement"] for item in data if "formal_statement" in item]
    
    # Process statements
    theorems = []
    skipped = 0
    skipped_example = None
    for stmt in raw_statements:
        processed = _process_statement(stmt)
        if processed is not None:
            theorems.append(processed)
        else:
            skipped += 1
            if skipped_example is None:
                skipped_example = stmt
    
    if skipped > 0:
        print(f"Warning: Skipped {skipped} statements that don't end with 'by'")
        print(f"Example of filtered statement:\n{skipped_example}")
    
    # shuffle with fixed seed and split into train/val
    random.Random(0).shuffle(theorems)
    
    if split == "val":
        return theorems[-500:]
    else:  # train
        return theorems[:-500]


if __name__ == "__main__":
    download_dataset()
    
    # Debug: print sample of the data to see what keys are available
    jsonl_path = os.path.join(DATA_DIR, "deepseek_prover.jsonl")
    with open(jsonl_path, "r") as f:
        sample = json.loads(f.readline())
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Sample item: {sample}")
    print()
    
    train_theorems = list_theorems(split="train")
    val_theorems = list_theorems(split="val")
    print(f"Retrieved {len(train_theorems)} train theorems")
    if train_theorems:
        print(train_theorems[0])
    print()
    print(f"Retrieved {len(val_theorems)} val theorems")
    if val_theorems:
        print(val_theorems[0])
