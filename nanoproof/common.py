"""
Common utilities for nanoproof.
"""

import os
import json
import enum
import time
import re
import logging
import math
import urllib.request
import gc
import faulthandler
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
from filelock import FileLock
from typing import Callable, Generic, TypeVar, Self

import torch
import torch.distributed as dist
import numpy as np
import wandb
import goodseed

# The dtype used for compute (matmuls, activations). Master weights stay fp32 for optimizer precision.
# Linear layers cast their weights to this dtype in forward, replacing torch.amp.autocast.
# Override with NANOPROOF_DTYPE env var: "bfloat16", "float16", "float32"
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
def _detect_compute_dtype():
    env = os.environ.get("NANOPROOF_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via NANOPROOF_DTYPE={env}"
    if torch.cuda.is_available():
        # bf16 requires SM 80+ (Ampere: A100, A10, etc.)
        # Older GPUs like V100 (SM 70) and T4 (SM 75) only have fp16 tensor cores
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        # fp16 training requires GradScaler (not yet implemented), so fall back to fp32.
        # Users can still force fp16 via NANOPROOF_DTYPE=float16 if they know what they're doing.
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"
COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()

# -----------------------------------------------------------------------------
# Global config: hardcoded constants that we want centralized but not exposed
# as CLI flags. The values here are coupled to the tokenizer (num_value_bins
# must match the number of <|bin_XX|> special tokens) and to the data layout
# (state_max_len + tactic_max_len define the natural max sequence length used
# by training and the dataloader's hard cutoff).

@dataclass(frozen=True)
class GlobalConfig:
    state_max_len: int = 640      # max state length (tokens) accepted by the dataloader
    tactic_max_len: int = 128     # max tactic length (tokens)
    num_value_bins: int = 64      # value head bin count; must match tokenizer special tokens

    @property
    def max_seq_len(self) -> int:
        return self.state_max_len + self.tactic_max_len  # 768

GLOBAL_CONFIG = GlobalConfig()


def get_lr_multiplier(progress: float, args) -> float:
    """Linear warmup → flat → linear warmdown to ``final_lr_frac``.

    ``progress`` is in [0, 1]. ``args`` must expose ``warmup_ratio``,
    ``warmdown_ratio`` and ``final_lr_frac`` (typically argparse Namespace).
    """
    if args.warmup_ratio > 0 and progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    if args.warmdown_ratio == 0 or progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
    return (1 - decay) + decay * args.final_lr_frac


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
    base = os.environ.get("NANOPROOF_HOME") or os.path.join(os.path.expanduser("~"), ".nanoproof")
    os.makedirs(base, exist_ok=True)
    return base

def create_run_dirs(stage: str, run: str, args_dict: dict | None = None):
    """Create log and model directories for a training run.

    Must be called after compute_init(). Only the master process creates
    directories; other ranks receive the paths via broadcast.

    Args:
        stage: one of "pretrain", "midtrain", "sft", "rl"
        run: the --run name (used in the directory name)
        args_dict: if provided, dumped as args.json in the log directory

    Returns:
        (log_dir, model_dir) - absolute paths
    """
    ddp = is_ddp_initialized()
    master = int(os.environ.get("RANK", 0)) == 0

    if master:
        base_dir = get_base_dir()
        timestamp = datetime.now().strftime("%H-%M-%S_%d-%m-%y")
        run_dirname = f"{timestamp}_{run}"
        log_dir = os.path.join(base_dir, "logs", stage, run_dirname)
        model_dir = os.path.join(base_dir, "models", stage, run_dirname)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        if args_dict is not None:
            with open(os.path.join(log_dir, "args.json"), "w") as f:
                json.dump(args_dict, f, indent=2)
        logger.info(f"Log directory: {log_dir}")
        logger.info(f"Model directory: {model_dir}")
    else:
        log_dir = None
        model_dir = None

    if ddp:
        paths = [log_dir, model_dir]
        dist.broadcast_object_list(paths, src=0)
        log_dir, model_dir = paths

    return log_dir, model_dir


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

def is_ddp_requested() -> bool:
    """True if launched by torchrun (env present), even before init."""
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def is_ddp_initialized() -> bool:
    """True if torch.distributed is available and the process group is initialized."""
    return dist.is_available() and dist.is_initialized()

# Legacy alias
is_ddp = is_ddp_requested

def get_dist_info():
    if is_ddp_requested():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def broadcast_value(value, src=0):
    """Broadcast a single scalar value from src rank to all other ranks."""
    assert isinstance(value, (int, float, str, bool)) or value is None, f"Expected scalar value, got {type(value)}"
    buf = [value]
    dist.broadcast_object_list(buf, src=src)
    assert buf[0] is not None, "Broadcast received None - src rank likely didn't set the value"
    return buf[0]

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
    if is_ddp_initialized():
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# CUDA memory profiling
#
# enable_memory_profiling() starts torch's memory history recorder. The first
# CUDA OOM that calls maybe_dump_memory_snapshot() writes a pickle that can be
# loaded into https://pytorch.org/memory_viz to inspect every live tensor's
# stack trace.
# -----------------------------------------------------------------------------

_memory_profile_path: str | None = None
_memory_profile_dumped: bool = False


def enable_memory_profiling(output_dir: str) -> None:
    """Start recording CUDA memory history; first OOM dump goes to output_dir.

    Call once at startup, before any significant GPU allocation.
    """
    global _memory_profile_path
    os.makedirs(output_dir, exist_ok=True)
    _, rank, _, _ = get_dist_info()
    _memory_profile_path = os.path.join(output_dir, f"memory_snapshot_rank{rank}.pickle")
    # max_entries=100k is plenty (a full run typically generates 10-30k allocations).
    # stacks="python" captures Python frames; "all" adds C++ but slows things down.
    torch.cuda.memory._record_memory_history(max_entries=100_000, stacks="python")
    logger.info(f"Memory profiling enabled; snapshot will be written to {_memory_profile_path} on first OOM")


def maybe_dump_memory_snapshot(context: str) -> None:
    """Dump a CUDA memory snapshot on the first OOM; no-op otherwise."""
    global _memory_profile_dumped
    if _memory_profile_path is None or _memory_profile_dumped:
        return
    _memory_profile_dumped = True
    try:
        torch.cuda.memory._dump_snapshot(_memory_profile_path)
        logger.info(f"Memory snapshot dumped to {_memory_profile_path} ({context})")
    except Exception as e:
        logger.warning(f"Failed to dump memory snapshot: {e}")


# hardcoded BF16 peak flops for various GPUs
# inspired by torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

    # Table order matters: more specific patterns first.
    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        # NVIDIA Hopper
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        # NVIDIA Ampere data center
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        # NVIDIA Ada data center
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        # AMD CDNA accelerators
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        # Consumer RTX
        (["5090"], 209.5e12),
        (["4090"], 165.2e12),
        (["3090"], 71e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')

class MetricsLogger:
    """Logs metrics to wandb, goodseed, or both."""

    def __init__(self, loggers, project, name, config, log_dir=None, save_code=False):
        self._wandb_run = None
        self._goodseed_run = None

        if "wandb" in loggers:
            kwargs = dict(project=project, name=name, config=config)
            if log_dir:
                kwargs["dir"] = log_dir
            if save_code:
                kwargs["save_code"] = True
            self._wandb_run = wandb.init(**kwargs)

        if "goodseed" in loggers:
            self._goodseed_run = goodseed.Run(name=f"{project}/{name}", tags=[project])
            self._goodseed_run.log_configs(config)

    def log(self, metrics, **kwargs):
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, **kwargs)
        if self._goodseed_run is not None:
            step = metrics.get("step")
            # Filter to numeric/string scalars (excludes wandb-specific objects like confusion matrices)
            safe = {k: v for k, v in metrics.items() if isinstance(v, (int, float, bool, str))}
            if safe:
                self._goodseed_run.log_metrics(safe, step=step)

    def finish(self):
        if self._wandb_run is not None:
            self._wandb_run.finish()
        if self._goodseed_run is not None:
            self._goodseed_run.close()


def add_logging_args(parser):
    """Add --run and --loggers arguments to an argparse parser."""
    parser.add_argument("--run", type=str, default="dummy",
                        help="Run name ('dummy' disables logging)")
    parser.add_argument("--loggers", nargs="*", default=["wandb", "goodseed"],
                        help="Logging backends to use (default: wandb goodseed)")


def create_metrics_logger(project, args, master_process, config, log_dir=None, save_code=False):
    """Create a MetricsLogger. Returns no-op logger if run=='dummy' or not master."""
    if args.run == "dummy" or not master_process:
        return MetricsLogger(loggers=[], project=project, name=args.run, config=config)
    return MetricsLogger(
        loggers=args.loggers, project=project, name=args.run,
        config=config, log_dir=log_dir, save_code=save_code,
    )

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


SomeValue = TypeVar('SomeValue')

class ValueOrError(Generic[SomeValue]):
    def __init__(self, value: SomeValue | None, error: str | None):
        assert (value is None) != (error is None)
        self._value = value
        self._error = error

    @classmethod
    def from_success(cls, value: SomeValue) -> Self:
        return cls(value, None)

    @classmethod
    def from_error(cls, error: str) -> Self:
        return cls(None, error)

    def is_success(self) -> bool:
        return self._value is not None

    @property
    def value(self) -> SomeValue:
        assert self.is_success()
        return self._value

    @property
    def error(self) -> str:
        assert not self.is_success()
        return self._error


TypeNode = TypeVar('TypeNode')
def pretty_print_tree(
        root: TypeNode,
        get_children: Callable[[TypeNode], list[TypeNode]],
        node_to_str: Callable[[TypeNode], str],
        edge_to_str: Callable[[TypeNode], str | None] | None = None,
        max_label_len=55,
        max_edge_label_len=None,
) -> str:
    def trimmed_edge_to_str(e: TypeNode) -> str | None:
        if edge_to_str is None:
            return None
        s = edge_to_str(e)
        if max_edge_label_len is None:
            return s
        if s is None:
            return s
        if len(s) > max_edge_label_len:
            dots = "..."
            return s[:max_edge_label_len - len(dots)] + dots
        return s

    from PrettyPrint import PrettyPrintTree
    pt = PrettyPrintTree(
        get_children=get_children,
        get_val=node_to_str,
        get_label=trimmed_edge_to_str,
        return_instead_of_print=True,
        # border=True,
        trim=max_label_len,
    )
    return pt(root)

class SimpleTimer:
    def __init__(self):
        self.times = {}
        self.start_times = {}

    def start(self, section: str):
        self.start_times[section] = time.perf_counter()

    def end(self, section: str):
        if section not in self.start_times:
            return
        elapsed = time.perf_counter() - self.start_times.pop(section)
        self.times[section] = self.times.get(section, 0.0) + elapsed

    def get_times(self) -> dict[str, float]:
        return self.times

    def log_times(self):
        if not self.times:
            return
        total = sum(self.times.values())
        print0("Timer results:")
        max_len = max(len(k) for k in self.times)
        for k, v in sorted(self.times.items(), key=lambda x: x[1], reverse=True):
            pct = (v / total * 100) if total > 0 else 0
            print0(f"  {k:<{max_len}} : {v:.4f}s ({pct:.1f}%)")

    def gather(self) -> Self:
        """Gather data from all ranks and return a new SimpleTimer with the aggregated (summed) times."""
        if not (dist.is_available() and dist.is_initialized()):
            new_timer = SimpleTimer()
            new_timer.times = self.times.copy()
            return new_timer
            
        print0("Gathering timer data from all ranks...")
        world_size = dist.get_world_size()
        local_times = self.times
        all_times_list = [None for _ in range(world_size)]
        dist.all_gather_object(all_times_list, local_times)
        
        aggregated_times = {}
        for rank_times in all_times_list:
            if rank_times is None: continue
            for k, v in rank_times.items():
                aggregated_times[k] = aggregated_times.get(k, 0.0) + v
        
        new_timer = SimpleTimer()
        new_timer.times = aggregated_times
        return new_timer

class DummyTimer(SimpleTimer):
    def start(self, section: str): pass
    def end(self, section: str): pass
    def get_times(self) -> dict[str, float]: return {}
    def log_times(self): pass
    def gather(self) -> Self: return DummyTimer()


# ---------------------------------------------------------------------------
# Timeline instrumentation
# ---------------------------------------------------------------------------

@dataclass
class TimelineEvent:
    """A single timed event in a prover's timeline."""
    type: str     # "llm" or "lean"
    start: float  # absolute time.time()
    end: float

    def to_dict(self) -> dict:
        return {"type": self.type, "start": self.start, "end": self.end}


class TimelineRecorder:
    """Records timeline events for a single prove() call.

    Create one per proof attempt, pass it through run_mcts / expand_node,
    then flush the collected events to the monitor.
    """

    def __init__(self):
        self.events: list[TimelineEvent] = []

    @contextmanager
    def record(self, event_type: str):
        start = time.time()
        yield
        end = time.time()
        self.events.append(TimelineEvent(event_type, start, end))


def active_barrier(key: str, timeout: float | None = 300.0, poll_interval: float = 0.5) -> None:
    """Rank-symmetric barrier over the distributed store.

    Does not use NCCL, so it never triggers the NCCL watchdog and leaves the
    Python thread in a sleep loop (Flask handler threads on worker ranks stay
    responsive). On timeout, dumps all-thread tracebacks on this rank and
    raises, turning silent rank desyncs into diagnosable failures.

    Pass `timeout=None` to wait indefinitely, for phases whose duration is
    genuinely unbounded (e.g. prover evaluation). Use a finite timeout for
    transitions that are expected to be quick.

    No-op when DDP is not active.
    """
    ddp, rank, _, world_size = get_dist_info()
    if not ddp:
        return
    store = dist.distributed_c10d._get_default_store()
    counter_key = f"active_barrier/{key}/count"
    store.add(counter_key, 1)
    deadline = None if timeout is None else time.time() + timeout
    while True:
        count = int(store.get(counter_key))
        if count >= world_size:
            return
        if deadline is not None and time.time() > deadline:
            faulthandler.dump_traceback()
            raise TimeoutError(
                f"active_barrier({key}) timed out on rank {rank}: {count}/{world_size} ranks arrived"
            )
        time.sleep(poll_interval)


class Player(enum.Enum):
    OR = 1
    AND = 2

def linearize_proof(node: "Node") -> list[str]:
    """Linearize a solved proof tree into a sequence of tactics using DFS.
    
    Traverses the AND/OR tree and collects all tactics from the solved path.
    Returns a list of tactic strings in order of application.
    """
    assert node.is_solved
    tactics = []

    def dfs(n: "Node"):
        assert n.is_solved

        if n.to_play == Player.OR:
            if n.is_terminal:
                return
            assert len(n.state) == 1, f"linearize_proof: Expected 1 branch at OR node, got {len(n.state)}"
            assert n.children, f"linearize_proof: No children at OR node"
            solved_actions = [a for a in n.children if n.children[a].is_solved]
            assert solved_actions, f"linearize_proof: No solved actions at OR node"
            action = min(solved_actions, key=lambda a: len(a))

            tactics.append(action)
            dfs(n.children[action])
        elif n.to_play == Player.AND:
            assert not n.is_terminal, f"linearize_proof: AND node is terminal: {n}"
            for action, child in n.children.items():
                dfs(child)
        else:
            raise ValueError(f"Unknown to_play: {n.to_play}")
    
    dfs(node)
    return tactics


def format_linearized_proof(tactics: list[str]) -> str:
    """Format a linearized proof as a list of tactics, one per line."""
    if not tactics:
        return "(no tactics)"
    
    lines = []
    for tactic in tactics:
        lines.append(f"{tactic}")
    return "\n".join(lines)


def _ensure_block_arrow(tactic: str) -> str:
    """Ensure that block-opening tactics like ``case`` and ``next`` end with ``=>``.

    The Lean REPL accepts these without ``=>`` in step-by-step mode, but Lean's
    file parser requires the arrow.  However, ``case tag := expr`` is a
    term-mode proof assignment and must be left unchanged.
    """
    stripped = tactic.rstrip()
    # Already has =>, or is a term-mode proof (case tag := expr).
    if stripped.endswith("=>") or ":=" in stripped:
        return tactic
    # Tactic-block openers that need => appended.
    if stripped.startswith("case ") or stripped == "next" or stripped.startswith("next "):
        return stripped + " =>"
    return tactic

def construct_proof_source(theorem: str, tactics: list[str]) -> str:
    """Construct the full Lean source by replacing 'sorry' in the theorem with the proof tactics.

    Args:
        theorem: The theorem statement ending with 'sorry'
        tactics: List of tactics from linearize_proof

    Returns:
        The complete Lean source with the proof filled in
    """
    assert len(tactics) > 0, f"construct_proof_source: No tactics provided"
    assert theorem.strip().endswith("sorry"), f"construct_proof_source: Theorem should end with 'sorry': {theorem}"

    # Remove "sorry" from the end
    theorem_body = theorem.rstrip()[:-len("sorry")].rstrip()

    # Multi-line proof with indentation
    proof_lines = "\n".join(f"  {_ensure_block_arrow(tactic.strip())}" for tactic in tactics)
    return theorem_body + "\n" + proof_lines


_THEOREM_NAME_RE = re.compile(
    r'(^|\n)(\s*(?:noncomputable\s+|private\s+|protected\s+)*)(?:theorem|def|lemma)\s+\S+',
)

def theorem_to_example(source: str) -> str:
    """Convert a Lean theorem/def/lemma statement to an example statement.

    Finds ``theorem <name>`` (or ``def``/``lemma``, possibly spanning
    whitespace / newlines between the keyword and the name) and replaces it
    with ``example``, preserving leading whitespace and everything after the
    name.

    Works on the raw source string rather than splitting by lines, so it
    handles multi-line declarations where the name sits alone on one line
    and the ``:`` / body continue on the next.
    """
    sorry_count = source.count("sorry")
    if sorry_count > 1:
        raise ValueError(
            f"Expected at most one 'sorry' but found {sorry_count} in: {source[:200]!r}"
        )
    matches = list(_THEOREM_NAME_RE.finditer(source))
    if not matches:
        raise ValueError(f"No 'theorem/def/lemma <name>' found in: {source[:200]!r}")
    # Use the *last* match - the actual theorem/exercise is always the final
    # declaration; earlier matches are auxiliary defs in the preamble.
    m = matches[-1]
    # m.group(0) is e.g. "\n  theorem lean_workbook_50099"
    # We want to replace "theorem <name>" part with "example", keeping
    # the leading newline + whitespace (groups 1 and 2).
    replacement = m.group(1) + m.group(2) + "example"
    return source[:m.start()] + replacement + source[m.end():]