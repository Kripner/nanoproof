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
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
from filelock import FileLock
from typing import Callable, Generic, TypeVar, Self

import torch
import torch.distributed as dist
import numpy as np

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
        (log_dir, model_dir) – absolute paths
    """
    import torch.distributed as dist

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


def active_barrier_master(key: str) -> None:
    """
    Signal completion to non-master ranks via the distributed store.
    
    This is used instead of dist.barrier() when non-master ranks need to
    actively poll (e.g., because they're running inference servers that
    need the Python thread to remain unblocked).
    
    The master process calls this after completing its work.
    Non-master processes should call active_barrier_wait() with the same key.
    """
    store = dist.distributed_c10d._get_default_store()
    store.set(key, "1")


def active_barrier_wait(key: str, poll_interval: float = 1.0) -> None:
    """
    Wait for master to signal completion via the distributed store.
    
    This actively polls instead of blocking, allowing the Python thread
    to remain responsive (e.g., for inference server requests).
    
    Args:
        key: The key that master will set when done
        poll_interval: How often to poll in seconds (default: 1.0)
    """
    store = dist.distributed_c10d._get_default_store()
    while True:
        if store.check([key]):
            break
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
    proof_lines = "\n".join(f"  {tactic.strip()}" for tactic in tactics)
    return theorem_body + "\n" + proof_lines


def theorem_to_example(source: str) -> str:
    """Convert a Lean theorem statement to an example statement.
    
    Finds the first line starting with 'theorem' and replaces 'theorem <name>' 
    with 'example', preserving everything else (including lines before and after).
    
    Args:
        source: A Lean source containing a theorem statement, e.g.:
            "import Mathlib\ntheorem foo (a b : ℂ) : a + b = b + a := by"
            
    Returns:
        The same source with 'theorem <name>' replaced by 'example', e.g.:
            "import Mathlib\nexample (a b : ℂ) : a + b = b + a := by"
            
    Raises:
        ValueError: If no line starts with 'theorem'
    """
    lines = source.split('\n')
    theorem_prefix = "theorem "
    
    # Find the first line that starts with "theorem "
    theorem_line_idx = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(theorem_prefix):
            theorem_line_idx = i
            break
    
    if theorem_line_idx is None:
        raise ValueError(f"No line starts with 'theorem ': {source[:100]!r}")
    
    theorem_line = lines[theorem_line_idx]
    leading_whitespace = theorem_line[:len(theorem_line) - len(theorem_line.lstrip())]
    stripped_line = theorem_line.lstrip()
    
    # Find the end of the theorem name (first whitespace after "theorem ")
    rest_after_theorem = stripped_line[len(theorem_prefix):]
    
    # The theorem name is the first token - find where it ends
    name_end = None
    for i, char in enumerate(rest_after_theorem):
        if char in (' ', '\t', '\n', '(', ':'):
            name_end = i
            break
    
    # If no delimiter found, the theorem name extends to end of line
    if name_end is None:
        name_end = len(rest_after_theorem)
    if name_end == 0:
        raise ValueError(f"Theorem name is empty in: {stripped_line[:80]!r}")
    
    # Extract the part after the theorem name
    after_name = rest_after_theorem[name_end:]
    
    # Reconstruct the line with "example" instead of "theorem <name>"
    new_line = leading_whitespace + "example" + after_name
    lines[theorem_line_idx] = new_line
    
    return '\n'.join(lines)