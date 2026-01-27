import os
import random

import pyarrow.parquet as pq
import requests
from tqdm import tqdm

from nanoproof.common import get_base_dir

base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "data", "numinamath")

# https://huggingface.co/datasets/AI-MO/NuminaMath-LEAN
HF_URL = "https://huggingface.co/datasets/AI-MO/NuminaMath-LEAN/resolve/main/data/train-00000-of-00001.parquet"


def download_dataset():
    """Download the NuminaMath-LEAN dataset from HuggingFace."""
    os.makedirs(DATA_DIR, exist_ok=True)
    parquet_path = os.path.join(DATA_DIR, "numinamath.parquet")

    # skip if already downloaded
    if os.path.exists(parquet_path):
        print(f"Dataset already downloaded at {parquet_path}")
        return

    try:
        print(f"Downloading NuminaMath-LEAN dataset from HuggingFace...")
        response = requests.get(HF_URL, stream=True, timeout=60)
        response.raise_for_status()

        temp_path = parquet_path + ".tmp"
        total_size = int(response.headers.get("content-length", 0))
        with open(temp_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading numinamath.parquet") as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        os.rename(temp_path, parquet_path)
        print(f"Successfully downloaded {parquet_path}")
    except (requests.RequestException, IOError):
        # Clean up any partial files
        for path in [parquet_path + ".tmp", parquet_path]:
            if os.path.exists(path):
                print(f"Cleaning up {path}")
                os.remove(path)
        raise


def _process_statement(statement: str) -> str | None:
    """Process a formal statement: strip, remove import/set_option lines, validate, append sorry."""
    statement = statement.strip()
    
    # Remove initial lines that start with "import " or "set_option "
    lines = statement.split('\n')
    while lines and (lines[0].startswith("import ") or lines[0].startswith("set_option ")):
        lines.pop(0)
    statement = '\n'.join(lines)
    
    # Check that the statement ends with ":= by" or ":="
    if statement.endswith(":= by"):
        return statement + " sorry"
    elif statement.endswith(":="):
        return statement + " by sorry"
    else:
        return None


def list_theorems(split: str):
    assert split in ["train", "val"], f"Invalid split: {split}. Must be 'train' or 'val'."
    
    parquet_path = os.path.join(DATA_DIR, "numinamath.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"NuminaMath-LEAN dataset not found at {parquet_path}. Download it first.")
    
    table = pq.read_table(parquet_path)
    raw_statements = table.column("formal_statement").to_pylist()
    
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
        print(f"Warning: Skipped {skipped} statements that don't end with ':= by' or ':='")
        print(f"Example of filtered statement:\n{skipped_example}")
    
    # shuffle with fixed seed and split into train/val
    random.Random(0).shuffle(theorems)
    
    if split == "val":
        return theorems[-500:]
    else:  # train
        return theorems[:-500]


if __name__ == "__main__":
    download_dataset()
    train_theorems = list_theorems(split="train")
    val_theorems = list_theorems(split="val")
    print(f"Retrieved {len(train_theorems)} train theorems")
    print(train_theorems[0])
    print()
    print(f"Retrieved {len(val_theorems)} val theorems")
    print(val_theorems[0])
