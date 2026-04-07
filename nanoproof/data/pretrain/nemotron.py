# Source: https://github.com/karpathy/nanochat/blob/master/nanochat/dataset.py

"""
The base/pretraining dataset is loaded from HuggingFace datasets.
This file contains utilities for:
- loading the dataset from HuggingFace
- iterating over the dataset and yielding documents from it
- limiting the number of files/shard used via CLI

For details of how the dataset was prepared, see the HuggingFace dataset page.
"""

import os
import argparse
import requests
import time
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import get_token
from tqdm import tqdm

from nanoproof.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

DATASET_NAME = "nvidia/Nemotron-CC-Math-v1"
BASE_URL = "https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1/resolve/main"

base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "data", "nemotron")

MAX_SHARD = 45 # the last datashard is part_000045.parquet
index_to_filename = lambda index: f"4plus/part_{index:06d}.parquet" # format of the filenames

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(os.path.join(data_dir, "4plus"))
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, "4plus", f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        row_group_indices = list(range(start, pf.num_row_groups, step))
        for rg_idx in row_group_indices:
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

def process_file(filepath):
    """
    Loads a parquet file, changes group size to 1024, drops all columns except 'text',
    and overwrites the file.
    """
    try:
        # Read the table
        table = pq.read_table(filepath)
        
        # Select only 'text' column if others exist
        if 'text' in table.column_names:
            table = table.select(['text'])
        else:
            print(f"Warning: 'text' column not found in {filepath}")
            
        # Write to a temporary file with new row group size
        temp_path = filepath + ".rechunk"
        pq.write_table(table, temp_path, row_group_size=1024)
        
        # Overwrite the original file
        os.rename(temp_path, filepath)
    except Exception as e:
        print(f"Failed to process {filepath}: {e}")
        # Clean up temp file if it was created
        if os.path.exists(filepath + ".rechunk"):
            try:
                os.remove(filepath + ".rechunk")
            except:
                pass

def download_single_file(index):
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}... to {filepath}")

    # Get Hugging Face token for authentication
    token = get_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30, headers=headers)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            Path(temp_path).parent.mkdir(parents=True, exist_ok=True)
            total_size = int(response.headers.get('content-length', 0))
            with open(temp_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=os.path.basename(filename), leave=False) as pbar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            # Move temp file to final location
            os.rename(temp_path, filepath)
            # Process the file immediately after download
            process_file(filepath)
            print(f"Successfully downloaded and processed {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage Nemotron-CC-Math-v1 dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_download = subparsers.add_parser("download", help="Download dataset shards")
    parser_download.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1), -1 = disable")
    parser_download.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")

    parser_process = subparsers.add_parser("process", help="Process all downloaded parquet files (re-chunk to 1024 group size)")
    
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.command == "download":
        num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
        ids_to_download = list(range(num))
        print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
        print(f"Target directory: {DATA_DIR}")
        print()
        with Pool(processes=args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(download_single_file, ids_to_download),
                total=len(ids_to_download),
                desc="Downloading shards"
            ))

        successful = sum(1 for success in results if success)
        print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")

    elif args.command == "process":
        parquet_paths = list_parquet_files()
        print(f"Found {len(parquet_paths)} parquet files to process in {DATA_DIR}...")
        for filepath in tqdm(parquet_paths, desc="Processing files"):
            process_file(filepath)
        print("Done processing all files.")
