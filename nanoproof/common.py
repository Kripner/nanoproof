"""
Common utilities for nanochat.
"""

import os
import re
import logging
import math
import urllib.request
import gc
from collections import Counter
from filelock import FileLock

import torch
import torch.distributed as dist
import numpy as np

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == 'INFO':
            # Highlight numbers and percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOPROOF_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOPROOF_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanoproof")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read() # bytes

        # Write to local file
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
                                                                                   ██████ 
                                                                                  ███░░███
 ████████    ██████   ████████    ██████  ████████  ████████   ██████   ██████   ░███ ░░░ 
░░███░░███  ░░░░░███ ░░███░░███  ███░░███░░███░░███░░███░░███ ███░░███ ███░░███ ███████   
 ░███ ░███   ███████  ░███ ░███ ░███ ░███ ░███ ░███ ░███ ░░░ ░███ ░███░███ ░███░░░███░    
 ░███ ░███  ███░░███  ░███ ░███ ░███ ░███ ░███ ░███ ░███     ░███ ░███░███ ░███  ░███     
 ████ █████░░████████ ████ █████░░██████  ░███████  █████    ░░██████ ░░██████   █████    
░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░███░░░  ░░░░░      ░░░░░░   ░░░░░░   ░░░░░     
                                          ░███                                            
                                          █████                                           
                                         ░░░░░                                            
    """
    print0(banner)

def is_ddp():
    # TODO is there a proper way
    return int(os.environ.get('RANK', -1)) != -1

def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def autodetect_device_type():
    # prefer to use CUDA if available, otherwise use MPS, otherwise fallback on CPU
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type

def compute_init(device_type="cuda"): # cuda|cpu|mps
    """Basic initialization that we keep doing over and over, so make common."""

    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"

    # Reproducibility
    # Note that we set the global seeds here, but most of the code uses explicit rng objects.
    # The only place where global rng might be used is nn.Module initialization of the model weights.
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)

    # Precision
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high") # uses tf32 instead of fp32 for matmuls

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type) # mps|cpu

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp():
        dist.destroy_process_group()

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass

def format_distribution(bins: list[float], hist_height: int = 10, bin_labels: list[str] = None) -> str:
    bar_char = '❚'  # Heavy vertical bar character.

    num_bins = len(bins)
    max_bin = max(bins)
    result = ""

    if max_bin == 0:
        max_bin = 1  # To avoid division by zero; all bars will be zero height.

    scaled_bins = [(bin_value / max_bin) * hist_height for bin_value in bins]
    # Round up to ensure visibility of non-zero bins.
    bar_heights = [math.ceil(height) for height in scaled_bins]

    # Determine y-axis labels (from HIST_HEIGHT down to 1)
    for row in range(hist_height, 0, -1):
        label_value = (row / hist_height) * max_bin
        label = f"{label_value:>3.1f} |"
        row_str = label
        for height in bar_heights:
            if height >= row:
                row_str += f" {bar_char} "
            else:
                row_str += " " * 3
        result += row_str + "\n"

    x_axis = "    +" + "---" * num_bins
    result += x_axis + "\n"

    # x-axis labels.
    if not bin_labels:
        bin_labels = [f"{i}" for i in range(num_bins)]
    label_str = "     "
    for label in bin_labels:
        assert len(label) <= 2
        if len(label) == 1:
            label_str += f" {label} "
        else:
            label_str += f"{label} "
    result += label_str + "\n"
    return result

def deep_shape(obj, seen=None, level=0, pretty=False):
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return "<circular reference>"
    seen.add(id(obj))

    def join_parts(parts):
        if pretty:
            return "\n" + "  " * level + (",\n" + "  " * level).join(parts) + "\n" + "  " * (level - 1)
        return ", ".join(parts)

    if isinstance(obj, tuple):
        return "(" + join_parts([deep_shape(o, seen, level + 1, pretty) for o in obj]) + ")"
    if isinstance(obj, list):
        if all(isinstance(o, (int, float, str, bool, type(None))) for o in obj):
            type_counts = Counter(type(o).__name__ for o in obj)
            return f"[{', '.join(f'{k}-{v}' for k, v in type_counts.items())}]"
        return "[" + join_parts([deep_shape(o, seen, level + 1, pretty) for o in obj]) + "]"
    if isinstance(obj, dict):
        return "{" + join_parts([str(k) + ": " + deep_shape(v, seen, level + 1, pretty) for k, v in obj.items()]) + "}"
    if isinstance(obj, np.ndarray):
        return "np-" + str(obj.shape)
    if isinstance(obj, torch.Tensor):
        return "pt-" + str(tuple(obj.shape))
    if isinstance(obj, str):
        return "str-" + str(len(obj))
    return str(obj)


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def strict_zip(a: list, b: list):
    if len(a) != len(b):
        raise Exception(f"List sizes differ ({len(a)} != {len(b)}).")
    return zip(a, b)
