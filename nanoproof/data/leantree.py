import os
import argparse
import requests
import zipfile
from pathlib import Path

from tqdm import tqdm

from nanoproof.common import get_base_dir

base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "data", "leantree")

KAGGLE_API_URL = "https://www.kaggle.com/api/v1/datasets/download/leantree/leantree"


def download_dataset():
    """Download the leantree dataset from Kaggle."""
    zip_path = os.path.join(DATA_DIR, "leantree.zip")

    # skip if already extracted
    extracted_marker = os.path.join(DATA_DIR, ".extracted")
    if os.path.exists(extracted_marker):
        print(f"Dataset already extracted at {DATA_DIR}")
        return True

    try:
        print(f"Downloading leantree dataset from Kaggle...")
        response = requests.get(KAGGLE_API_URL, stream=True, timeout=60)
        response.raise_for_status()

        temp_path = zip_path + ".tmp"
        total_size = int(response.headers.get("content-length", 0))
        with open(temp_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024,
                      desc="Downloading leantree.zip") as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        os.rename(temp_path, zip_path)
        print(f"Successfully downloaded {zip_path}")

        print(f"Extracting {zip_path} to {DATA_DIR}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member in zip_ref.infolist():
                zip_ref.extract(member, DATA_DIR)
        os.remove(zip_path)

        # create a marker file to indicate extraction is complete
        with open(extracted_marker, "w") as f:
            f.write("extracted\n")
        print(f"Successfully extracted dataset to {DATA_DIR}")
    except (requests.RequestException, IOError, zipfile.BadZipFile):
        # Clean up any partial files
        for path in [zip_path + ".tmp", zip_path]:
            if os.path.exists(path):
                os.remove(path)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LeanTree dataset from Kaggle.")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    download_dataset()
