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

from nanoproof.common import get_base_dir

# Not available anymore:
# - https://github.com/pthomas505/FOL.git
# - https://github.com/brown-cs22/CS22-Lean-2024.git
# Excluded:
# - https://github.com/mortarsanjaya/IMOSLLean4.git (contains IMO problems)

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

    print(f"Collected data from {len(parquet_files)} repositories.")
    print(f"Total characters: {total_chars:,}")
    print(f"Total bytes: {total_bytes:,} ({total_bytes / 1024 / 1024:.2f} MB)")
    print(f"Total files: {total_files:,}")

    print("Combining parquet files...")
    
    tables = []
    for pf in tqdm(parquet_files):
        tables.append(pq.read_table(pf))
    combined_table = pa.concat_tables(tables)
    combined_output_file = os.path.join(output_dir, "leangithubraw.parquet")
    pq.write_table(combined_table, combined_output_file)
    
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

def main():
    parser = argparse.ArgumentParser(description="Manage Lean GitHub Raw Dataset")
    subparsers = parser.add_subparsers(dest="action", required=True)
    
    # Build
    build_parser = subparsers.add_parser("build", help="Build the dataset from source URLs")
    
    # Publish
    publish_parser = subparsers.add_parser("publish", help="Upload dataset to Hugging Face")
    publish_parser.add_argument("repo_id", default="Kripi/Lean-Github-Raw", help="Hugging Face dataset repository ID (e.g. username/dataset)")
    
    # Download
    download_parser = subparsers.add_parser("download", help="Download dataset from Hugging Face")
    download_parser.add_argument("repo_id", default="Kripi/Lean-Github-Raw", help="Hugging Face dataset repository ID (e.g. username/dataset)")
    
    args = parser.parse_args()
    
    if args.action == "build":
        build_dataset()
    elif args.action == "publish":
        publish_dataset(args.repo_id)
    elif args.action == "download":
        download_dataset(args.repo_id)

if __name__ == "__main__":
    main()

