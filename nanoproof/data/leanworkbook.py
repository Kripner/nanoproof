import os
import json
import requests

from tqdm import tqdm

from nanoproof.common import get_base_dir

base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "data", "leanworkbook")

HF_URL = "https://huggingface.co/datasets/internlm/Lean-Workbook/resolve/main/lean_workbook.json"

# gather de-duplicated formal_statement from:
# https://huggingface.co/datasets/internlm/Lean-Workbook

def download_dataset():
    """Download the Lean-Workbook dataset from HuggingFace."""
    json_path = os.path.join(DATA_DIR, "lean_workbook.json")

    # skip if already downloaded
    if os.path.exists(json_path):
        print(f"Dataset already downloaded at {json_path}")
        return

    try:
        print(f"Downloading Lean-Workbook dataset from HuggingFace...")
        response = requests.get(HF_URL, stream=True, timeout=60)
        response.raise_for_status()

        temp_path = json_path + ".tmp"
        total_size = int(response.headers.get("content-length", 0))
        with open(temp_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading lean_workbook.json") as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        os.rename(temp_path, json_path)
        print(f"Successfully downloaded {json_path}")
    except (requests.RequestException, IOError):
        # Clean up any partial files
        for path in [json_path + ".tmp", json_path]:
            if os.path.exists(path):
                print(f"Cleaning up {path}")
                os.remove(path)
        raise

def list_theorems():
    json_path = os.path.join(DATA_DIR, "lean_workbook.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Lean-Workbook dataset not found at {json_path}. Download it first.")
    with open(json_path, "r") as f:
        data = json.load(f)
    # Select theorems that have been proven by InternLM Prover
    return [item["formal_statement"] for item in data if item["proof"]]


if __name__ == "__main__":
    download_dataset()
    theorems = list_theorems()
    print(f"Retrieved {len(theorems)} theorems")
    print(theorems[0])