"""
Utilities for building, publishing, and downloading the raw Lean GitHub dataset.
"""

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

# Constants
URLS_FILE = os.path.join(os.path.dirname(__file__), "leangithub_urls.txt")
BASE_DIR = get_base_dir()
DATA_DIR = os.path.join(BASE_DIR, "data", "leangithubraw")

def list_parquet_files(data_dir=None):
    """Returns a list of all parquet files in the data directory."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    if not os.path.exists(data_dir):
        return []
    return sorted(glob.glob(os.path.join(data_dir, "*.parquet")))

def build_dataset(output_dir=None):
    """
    Builds the dataset by cloning repos listed in leangithub_urls.txt
    and extracting content from .lean files.
    """
    if output_dir is None:
        output_dir = DATA_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read URLs
    if not os.path.exists(URLS_FILE):
        raise FileNotFoundError(f"URLs file not found at {URLS_FILE}")
        
    with open(URLS_FILE, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    print(f"Found {len(urls)} repositories to process.")
    
    # Use a local directory for cloning
    repos_dir = os.path.join(DATA_DIR, "repos")
    os.makedirs(repos_dir, exist_ok=True)
    print(f"Cloning repositories into: {repos_dir}")
    
    total_chars = 0
    total_bytes = 0
    parquet_files = []

    for i, repo_url in enumerate(tqdm(urls, desc="Processing repositories")):
        repo_data = []
        try:
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            repo_path = os.path.join(repos_dir, repo_name)
            
            # Clone the repository
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path)
            
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, repo_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            commit_hash = result.stdout.strip()
            
            # Base URL for file links (strip .git if present)
            base_url = repo_url[:-4] if repo_url.endswith('.git') else repo_url
            
            # Walk through files
            for root, _, files in os.walk(repo_path):
                for file in files:
                    if file.endswith(".lean"):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, repo_path)
                        
                        try:
                            # Read bytes first to count bytes
                            with open(file_path, 'rb') as f:
                                content_bytes = f.read()
                                
                            text = content_bytes.decode('utf-8')
                            
                            # Update stats
                            total_bytes += len(content_bytes)
                            total_chars += len(text)

                            # Construct full URL
                            full_url = f"{base_url}/blob/{commit_hash}/{rel_path}"
                            
                            repo_data.append({
                                "text": text,
                                "url": full_url,
                                "commit": commit_hash
                            })
                        except UnicodeDecodeError:
                            # print(f"Skipping binary or non-utf8 file: {rel_path}")
                            pass
                        except Exception as e:
                            print(f"Error reading {rel_path}: {e}")
            
            # Save individual parquet file if data exists
            if repo_data:
                df_repo = pd.DataFrame(repo_data)
                repo_parquet_path = os.path.join(output_dir, f"part_{i:06d}_{repo_name}.parquet")
                table = pa.Table.from_pandas(df_repo)
                pq.write_table(table, repo_parquet_path)
                parquet_files.append(repo_parquet_path)
                
            # Cleanup repo
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path)
                            
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone or process {repo_url}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {repo_url}: {e}")

    if not parquet_files:
        print("No data collected.")
        return

    print(f"Collected data from {len(parquet_files)} repositories.")
    print(f"Total characters: {total_chars:,}")
    print(f"Total bytes: {total_bytes:,} ({total_bytes / 1024 / 1024:.2f} MB)")
    print("Combining parquet files...")
    
    # Combine all parquet files
    combined_output_file = os.path.join(output_dir, "leangithub_raw_000000.parquet")
    
    # Read all tables and concat
    tables = []
    for pf in parquet_files:
        tables.append(pq.read_table(pf))
        
    combined_table = pa.concat_tables(tables)
    pq.write_table(combined_table, combined_output_file)
    
    print(f"Final dataset saved to {combined_output_file}")

def publish_dataset(repo_id, data_dir=None):
    """Uploads the dataset to Hugging Face Hub."""
    if data_dir is None:
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

def download_dataset(repo_id, output_dir=None):
    """Downloads the dataset from Hugging Face Hub."""
    if output_dir is None:
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
    build_parser.add_argument("--output-dir", default=None, help="Directory to save parquet files")
    
    # Publish
    publish_parser = subparsers.add_parser("publish", help="Upload dataset to Hugging Face")
    publish_parser.add_argument("repo_id", help="Hugging Face dataset repository ID (e.g. username/dataset)")
    publish_parser.add_argument("--data-dir", default=None, help="Directory containing parquet files")
    
    # Download
    download_parser = subparsers.add_parser("download", help="Download dataset from Hugging Face")
    download_parser.add_argument("repo_id", help="Hugging Face dataset repository ID (e.g. username/dataset)")
    download_parser.add_argument("--output-dir", default=None, help="Directory to save downloaded files")
    
    args = parser.parse_args()
    
    if args.action == "build":
        build_dataset(args.output_dir)
    elif args.action == "publish":
        publish_dataset(args.repo_id, args.data_dir)
    elif args.action == "download":
        download_dataset(args.repo_id, args.output_dir)

if __name__ == "__main__":
    main()

