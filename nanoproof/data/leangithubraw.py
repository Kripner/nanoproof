import os
import argparse
import subprocess
import shutil
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from huggingface_hub import HfApi, create_repo, upload_folder, snapshot_download
import requests
import time
from pathlib import Path
from collections import deque
from itertools import islice
import random
import torch

from nanoproof.common import get_base_dir, get_dist_info
from nanoproof.tokenizer import get_tokenizer

# 142.61 MB of text in total

# Not available anymore:
# - https://github.com/pthomas505/FOL.git
# - https://github.com/brown-cs22/CS22-Lean-2024.git
# Excluded:
# - https://github.com/mortarsanjaya/IMOSLLean4.git (contains IMO problems)

# TODO: maybe look at/exclude repos that have thousands of emojis?

URLS_FILE = os.path.join(os.path.dirname(__file__), "leangithub_urls.txt")
BASE_DIR = get_base_dir()
DATA_DIR = os.path.join(BASE_DIR, "data", "leangithubraw")

def build_dataset():
    """
    Builds the dataset by cloning repos listed in leangithub_urls.txt and reading .lean files.
    """
    output_dir = DATA_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(URLS_FILE):
        raise FileNotFoundError(f"URLs file not found at {URLS_FILE}")
        
    with open(URLS_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    print(f"Found {len(urls)} repositories to process.")
    
    repos_dir = os.path.join(DATA_DIR, "repos")
    os.makedirs(repos_dir, exist_ok=True)
    print(f"Cloning repositories into: {repos_dir}")
    
    total_chars = 0
    total_bytes = 0
    total_files = 0
    parquet_files = []

    pbar = tqdm(urls, desc="Processing repositories")
    for repo_url in pbar:
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_parquet_path = os.path.join(output_dir, f"repo_{repo_name}.parquet")
        if os.path.exists(repo_parquet_path):
            parquet_files.append(repo_parquet_path)
            continue
        
        repo_path = os.path.join(repos_dir, repo_name)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, repo_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=dict(os.environ, GIT_TERMINAL_PROMPT="0"),
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone {repo_url}, skipping: {e}")
            continue
        
        # get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        commit_hash = result.stdout.strip()
        
        repo_data = []
        base_url = repo_url[:-4] if repo_url.endswith('.git') else repo_url
        for root, _, files in os.walk(repo_path):
            for file in files:
                if not file.endswith(".lean"):
                    continue
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_path)
                
                with open(file_path, "rb") as f:
                    content_bytes = f.read()
                text = content_bytes.decode("utf-8")
                full_url = f"{base_url}/blob/{commit_hash}/{rel_path}"
                repo_data.append({
                    "text": text,
                    "url": full_url,
                    "commit": commit_hash,
                })
                
                total_bytes += len(content_bytes)
                total_chars += len(text)
                total_files += 1

                mb = total_bytes / (1024 * 1024)
                pbar.set_postfix_str(f"{repo_name}, total: {mb:.2f} MB, {total_files} files")

        if repo_data:
            df_repo = pd.DataFrame(repo_data)
            table = pa.Table.from_pandas(df_repo)
            pq.write_table(table, repo_parquet_path)
            parquet_files.append(repo_parquet_path)
            
        shutil.rmtree(repo_path)

    if not parquet_files:
        print("No data collected.")
        return

    print("Processed now:")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Total bytes: {total_bytes:,} ({total_bytes / 1024 / 1024:.2f} MB)")
    print(f"  Total files: {total_files:,}")

    print(f"Combining data from {len(parquet_files)} repositories...")
    tables = []
    for pf in tqdm(parquet_files):
        tables.append(pq.read_table(pf))
    combined_table = pa.concat_tables(tables)
    
    # sanity check: count characters
    # total_chars_pre = combined_table.to_pandas()["text"].str.len().sum()
    # print(f"Total characters before shuffle: {total_chars_pre:,}")

    # shuffle rows so that the number of characters in groups are not too different
    print("Shuffling rows ...")
    indices = torch.randperm(len(combined_table)).tolist()
    combined_table = combined_table.take(indices)

    # total_chars_post = combined_table.to_pandas()["text"].str.len().sum()
    # print(f"Total characters after shuffle: {total_chars_post:,}")

    combined_output_file = os.path.join(output_dir, "leangithubraw.parquet")
    # 1024 group size for efficient loading during training
    pq.write_table(combined_table, combined_output_file, row_group_size=1024)
    print(f"Dataset saved to: {combined_output_file}")

def publish_dataset(repo_id):
    """Uploads the dataset to Hugging Face Hub."""
    data_dir = DATA_DIR
        
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist. Build the dataset first.")
        return

    print(f"Uploading {data_dir} to {repo_id}...")
    api = HfApi()
    
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=data_dir,
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=".",
            ignore_patterns=["*.lock", "*.tmp"]
        )
        print(f"Successfully uploaded to https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Error uploading dataset: {e}")

def download_dataset(repo_id):
    """Downloads the dataset from Hugging Face Hub."""
    output_dir = DATA_DIR
        
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading dataset from {repo_id} to {output_dir}...")
    try:
        # Using snapshot_download is easier and more robust than manual requests for a folder
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.lock", "*.tmp"]
        )
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

def show_dataset(split="train", B=4, T=512, offset=0, num_batches=10):
    """Show the first N batches from the dataset."""
    print(f"Loading dataset (split={split})...")
    tokenizer = get_tokenizer()
    
    try:
        dataloader = iter_data(B=B, T=T, split=split, device="cpu")
        for batch_idx, batch in enumerate(islice(dataloader, offset, offset + num_batches)):
            if len(batch) == 4:
                inputs, targets, approx_progress, last_step = batch
            else:
                inputs, targets = batch
                approx_progress = "N/A"
                last_step = "N/A"

            print(f"\nBatch {batch_idx}:")
            print(f"  Inputs shape: {inputs.shape}")
            print(f"  Targets shape: {targets.shape}")
            print(f"  Approx progress: {approx_progress}")
            print(f"  Last step: {last_step}")
            
            for i in range(min(2, inputs.size(0))):  # Show first 2 samples per batch
                print(f"\n  Sample {i}:")
                print(f"    Input tokens: {inputs[i][:50].tolist()}...")  # First 50 tokens
                print(f"    Input text (first 200 chars):")
                decoded = tokenizer.decode(inputs[i].tolist())
                print(f"      {decoded[:200]}...")
                
                print(f"    Target tokens: {targets[i][:50].tolist()}...")  # First 50 tokens
                print(f"    Target text (first 200 chars):")
                decoded_target = tokenizer.decode(targets[i].tolist())
                print(f"      {decoded_target[:200]}...")
            
            print("-" * 100)
    except StopIteration:
        print("Dataset exhausted before reaching requested number of batches.")
    except Exception as e:
        print(f"Error showing dataset: {e}")
        raise

def show_whole_dataset(split="train", B=32, T=768):
    print(f"Iterating through dataset (split={split}, B={B}, T={T})...")
    
    dataloader = iter_data(B=B, T=T, split=split)
    batch_count = 0
    first_last_step = None
    
    for i, batch in enumerate(dataloader):
        if split == "train":
            inputs, targets, approx_progress, last_step = batch
            print(f"Batch {i:05d}: inputs.shape={inputs.shape}, targets.shape={targets.shape}, approx_progress={approx_progress}, last_step={last_step}")
        else:
            inputs, targets = batch
            print(f"Batch {i:05d}: inputs.shape={inputs.shape}, targets.shape={targets.shape}")
        
        batch_count += 1
        
        if last_step and first_last_step is None:
            print(f"\nTotal batches: {batch_count}")
            first_last_step = i
        if first_last_step is not None and i - first_last_step >= 100:
            break
    

def iter_data(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda"):
    """
    Create batches for unsupervised training from the leangithubraw parquet file.
    
    Args:
        B: Batch size
        T: Sequence length (tokens per sample)
        split: "train" or "val"
        tokenizer_threads: Number of threads for tokenization (unused, kept for compatibility)
        tokenizer_batch_size: Batch size for tokenization
        device: Device to move tensors to ("cuda" or "cpu")
    
    Yields:
        inputs: Tensor of shape (B, T) with input token IDs
        targets: Tensor of shape (B, T) with target token IDs
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    assert bos_token is not None

    parquet_path = os.path.join(DATA_DIR, "leangithubraw.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at {parquet_path}. Build it or download it first.")
    pf = pq.ParquetFile(parquet_path)

    def document_batches():
        # last 4 are validation, rest is train
        num_groups = pf.num_row_groups
        assert num_groups > 10
        group_indices = list(range(num_groups - 4) if split == "train" else range(num_groups - 4, num_groups))

        random.Random(0).shuffle(group_indices)

        group_indices = group_indices[ddp_rank::ddp_world_size]

        time.sleep(random.random())
        group_sizes = [pf.metadata.row_group(idx).num_rows for idx in group_indices]
        print(f"{ddp_rank=} {ddp_world_size=} {len(group_indices)=} {group_indices=} {group_sizes=}", flush=True)

        last_step = False
        while True:
            for i in range(len(group_indices)):
                group = pf.read_row_group(group_indices[i])
                samples = group.column("text").to_pylist()
                # batches for tokenizer
                for offset in range(0, len(samples), tokenizer_batch_size):
                    last_step = last_step or (i == len(group_indices) - 1 and offset + tokenizer_batch_size >= len(samples))
                    approx_progress = (i + offset / len(samples)) / len(group_indices) if not last_step else 1.0
                    yield samples[offset:offset+tokenizer_batch_size], approx_progress, last_step
            print(f"Warning: Rank {ddp_rank} will loop again on Lean-Github-Raw ({split=}).", flush=True)

    batches = document_batches()
    
    needed_tokens = B * T + 1  # +1 because we also need the target at the last token
    token_buffer = deque()  # we stream tokens on the right and pop from the left

    last_step = False
    approx_progress = 0.0
    while True:
        # accumulate enough tokens for one iteration before yielding
        while len(token_buffer) < needed_tokens:
            try:
                doc_batch, approx_progress, last_step = next(batches)
            except StopIteration:
                break
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        
        if len(token_buffer) < needed_tokens:
            break  # drop last
        
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)
        
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        
        # reshape to 2D and move to device async
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)

        if split == "train":
            yield inputs, targets, approx_progress, last_step
        else:
            yield inputs, targets

def iter_texts_batched(split, url_whitelist=None):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_path = os.path.join(DATA_DIR, "leangithubraw.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at {parquet_path}. Build it or download it first.")
    
    pf = pq.ParquetFile(parquet_path)
    
    # last two groups are validation, rest is train
    num_groups = pf.num_row_groups
    assert num_groups > 10
    if split == "train":
        group_indices = range(num_groups - 2)
    else:
        group_indices = range(num_groups - 2, num_groups)

    for rg_idx in group_indices:
        rg = pf.read_row_group(rg_idx)
        texts = rg.column("text").to_pylist()
        
        if url_whitelist is not None:
            urls = rg.column("url").to_pylist()
            filtered_texts = []
            for text, url in zip(texts, urls):
                if any(url.startswith(prefix) for prefix in url_whitelist):
                    filtered_texts.append(text)
            texts = filtered_texts
        
        yield texts

def dataset_stats():
    parquet_path = os.path.join(DATA_DIR, "leangithubraw.parquet")
    if not os.path.exists(parquet_path):
        print(f"Dataset not found at {parquet_path}. Build or download it first.")
        return

    print(f"Loading dataset from {parquet_path}...")
    try:
        table = pq.read_table(parquet_path)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    texts = table.column("text").to_pylist()
    num_samples = len(texts)
    
    print(f"Calculating statistics for {num_samples} samples...")
    
    total_chars = 0
    total_bytes = 0
    total_tokens = 0
    
    tokenizer = get_tokenizer()
    
    batch_size = 1000
    for i in tqdm(range(0, num_samples, batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        for text in batch_texts:
            total_chars += len(text)
            total_bytes += len(text.encode("utf-8"))
            
        encoded_batch = tokenizer.encode(batch_texts)
        for ids in encoded_batch:
            total_tokens += len(ids)
            
    print("\nDataset Stats:")
    print(f"{'Samples (Files):':<20} {num_samples:,}")
    print(f"{'Tokens:':<20} {total_tokens:,}")
    print(f"{'Characters:':<20} {total_chars:,}")
    print(f"{'Bytes:':<20} {total_bytes:,} ({total_bytes / 1024 / 1024:.2f} MB)")

def main():
    parser = argparse.ArgumentParser(description="Manage Lean GitHub Raw Dataset")
    subparsers = parser.add_subparsers(dest="action", required=True)
    
    # Build
    build_parser = subparsers.add_parser("build", help="Build the dataset from source URLs")
    
    # Publish
    publish_parser = subparsers.add_parser("publish", help="Upload dataset to Hugging Face")
    publish_parser.add_argument("--repo_id", default="Kripi/Lean-Github-Raw", help="Hugging Face dataset repository ID (e.g. username/dataset)")
    
    # Download
    download_parser = subparsers.add_parser("download", help="Download dataset from Hugging Face")
    download_parser.add_argument("--repo_id", default="Kripi/Lean-Github-Raw", help="Hugging Face dataset repository ID (e.g. username/dataset)")
    
    # Show
    show_parser = subparsers.add_parser("show", help="Show the first N batches from the dataset")
    show_parser.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split to show")
    show_parser.add_argument("--B", type=int, default=4, help="Batch size")
    show_parser.add_argument("--T", type=int, default=512, help="Sequence length")
    show_parser.add_argument("--offset", type=int, default=0)
    show_parser.add_argument("--num-batches", type=int, default=10, help="Number of batches to show")

    # Show whole
    show_whole_parser = subparsers.add_parser("show_whole", help="Show the whole dataset")
    show_whole_parser.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split to show")
    show_whole_parser.add_argument("--B", type=int, default=32, help="Batch size")
    show_whole_parser.add_argument("--T", type=int, default=768, help="Sequence length")

    # Stats
    subparsers.add_parser("stats", help="Display dataset statistics (tokens, chars, bytes, samples)")
    
    args = parser.parse_args()
    
    if args.action == "build":
        build_dataset()
    elif args.action == "publish":
        publish_dataset(args.repo_id)
    elif args.action == "download":
        download_dataset(args.repo_id)
    elif args.action == "show":
        show_dataset(split=args.split, B=args.B, T=args.T, offset=args.offset, num_batches=args.num_batches)
    elif args.action == "show_whole":
        show_whole_dataset(split=args.split, B=args.B, T=args.T)
    elif args.action == "stats":
        dataset_stats()

if __name__ == "__main__":
    main()

