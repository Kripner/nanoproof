"""Shared helpers for the RL theorem datasets (leanworkbook, numinamath,
deepseek_prover). Each of those modules is structured the same way: download
a single file from HuggingFace, parse out formal statements, then shuffle and
split deterministically into train/valid. The helpers here cover the parts
that would otherwise be duplicated three times.
"""

import os
import random
from typing import Iterable

import requests
from tqdm import tqdm

# Lean versions for which we ship pre-computed whitelists in the nanoproof repo
# (see data/whitelists/). Each RL dataset's download_dataset() fetches all of
# these alongside the source data so users don't need to regenerate them.
WHITELIST_VERSIONS = ("v4.27.0",)
WHITELIST_BASE_URL = (
    "https://raw.githubusercontent.com/Kripner/nanoproof/refs/heads/main/data/whitelists"
)


def download_file(url: str, dest_path: str, desc: str | None = None) -> None:
    """Download a single HTTP(S) URL with a tqdm progress bar. Streams to
    ``dest_path + ".tmp"`` and atomically renames on success; cleans up
    partial files on failure. No-op if ``dest_path`` already exists.
    """
    if os.path.exists(dest_path):
        print(f"Already downloaded: {dest_path}")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".tmp"
    desc = desc or os.path.basename(dest_path)

    try:
        print(f"Downloading {url} -> {dest_path}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with open(temp_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        os.rename(temp_path, dest_path)
        print(f"Successfully downloaded {dest_path}")
    except (requests.RequestException, IOError):
        for path in [temp_path, dest_path]:
            if os.path.exists(path):
                print(f"Cleaning up {path}")
                os.remove(path)
        raise


def download_whitelists(data_file_path: str) -> None:
    """Download every shipped whitelist version for ``data_file_path``.

    Each whitelist lives next to the data file as
    ``{data_file_path}.whitelist.{version}.json`` -- the same layout that
    ``nanoproof.data.check_init.whitelist_path`` expects at load time.
    """
    filename = os.path.basename(data_file_path)
    for version in WHITELIST_VERSIONS:
        wl_name = f"{filename}.whitelist.{version}.json"
        url = f"{WHITELIST_BASE_URL}/{wl_name}"
        dest = f"{data_file_path}.whitelist.{version}.json"
        download_file(url, dest, desc=wl_name)


def shuffle_train_valid_split(
    items: Iterable, valid_size: int = 500, seed: int = 0
) -> dict[str, list]:
    """Shuffle ``items`` deterministically (fixed ``seed``) and split into
    a train/valid dict. The last ``valid_size`` items become validation; the
    rest become training. Does not mutate the input.

    If ``len(items) <= valid_size``, all items go to validation and train is
    empty - the caller decides whether that's acceptable.
    """
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    if len(items) <= valid_size:
        return {"train": [], "valid": items}
    return {"train": items[:-valid_size], "valid": items[-valid_size:]}
