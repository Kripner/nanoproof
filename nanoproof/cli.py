"""
Web-based monitoring for the RL training loop.

Provides a real-time web dashboard showing:
- Training stats (loss, step, etc.)
- Prover server status with thread-level indicators
- GPU utilization and memory
- Inference wait times
- Log streams (per-component and merged)

The monitor runs a Flask server in a background thread. The web app polls
for state updates every second.
"""

import gzip
import json
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from statistics import median
from typing import Literal, TextIO, Any
from queue import Queue

from flask import Flask, jsonify, Response, send_from_directory, request

from nanoproof.common import TimelineEvent

# -----------------------------------------------------------------------------
# Logging utilities
# -----------------------------------------------------------------------------

_log_lock = threading.Lock()
_log_file: TextIO | None = None
_tactics_file: TextIO | None = None
_errors_file: TextIO | None = None
_is_master_process: bool = True  # Default to True for non-DDP usage
_ddp_rank: int = 0  # DDP rank for logging

# Dedup state for log_actionable_error: maps (component, error_first_line) ->
# (last_write_monotonic, suppressed_count). Without this, a dead Lean server
# can fill errors.jsonl with hundreds of thousands of identical entries.
_error_dedup: dict[tuple[str, str], tuple[float, int]] = {}
_error_dedup_window_seconds: float = 60.0

# In-memory tactics buffer for distributed mode (provers collect and report)
_tactics_buffer: deque = deque(maxlen=1000)
_tactics_buffer_lock = threading.Lock()

# In-memory log buffers for web streaming
_log_buffers: dict[str, deque] = {}
_log_buffers_lock = threading.Lock()
MAX_LOG_BUFFER_SIZE = 2000


def _get_log_buffer(component: str) -> deque:
    """Get or create a log buffer for a component."""
    with _log_buffers_lock:
        if component not in _log_buffers:
            _log_buffers[component] = deque(maxlen=MAX_LOG_BUFFER_SIZE)
        return _log_buffers[component]


def _add_to_log_buffer(component: str, entry: dict):
    """Add a log entry to the component's buffer and the merged buffer."""
    buffer = _get_log_buffer(component)
    buffer.append(entry)
    # Also add to merged buffer
    merged = _get_log_buffer("_merged")
    merged.append(entry)


def set_ddp_info(is_master: bool, rank: int = 0):
    """
    Set DDP info for logging.
    
    Args:
        is_master: Whether this process is the master (rank 0)
        rank: The DDP rank of this process
    """
    global _is_master_process, _ddp_rank
    _is_master_process = is_master
    _ddp_rank = rank


def configure_logging(output_dir: str | None):
    """
    Configure logging to write to files in the output directory.

    Args:
        output_dir: Directory where log files will be written.
                   If None, logging goes to console only.
    """
    global _log_file, _tactics_file, _errors_file

    with _log_lock:
        # Close any existing files
        if _log_file is not None:
            _log_file.close()
            _log_file = None
        if _tactics_file is not None:
            _tactics_file.close()
            _tactics_file = None
        if _errors_file is not None:
            _errors_file.close()
            _errors_file = None

        if output_dir is not None:
            _log_file = open(os.path.join(output_dir, "logs.txt"), "a")
            _tactics_file = open(os.path.join(output_dir, "tactics.txt"), "a")
            _errors_file = open(os.path.join(output_dir, "errors.jsonl"), "a")


def log(msg: str, component: str | None = None, actor_id: int | None = None):
    """
    Thread-safe logging with optional component/actor prefix.
    Writes to log file, console, and in-memory buffer for web streaming.
    
    Args:
        msg: The message to log
        component: Component name (e.g., "TacticModel", "Collection")
        actor_id: Actor thread ID if applicable
    """
    if actor_id is not None:
        prefix = f"[Actor {actor_id}]"
        log_component = f"actor_{actor_id}"
    elif component is not None:
        prefix = f"[{component}]"
        log_component = component
    else:
        prefix = ""
        log_component = "main"

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    with _log_lock:
        line = f"[{timestamp}] {prefix} {msg}" if prefix else f"[{timestamp}] {msg}"
        
        # Write to file if configured
        if _log_file is not None:
            _log_file.write(line + "\n")
            _log_file.flush()
        
        # Always print to console
        print(line, flush=True)
        
        # Add to in-memory buffer for web streaming
        entry = {
            "timestamp": timestamp,
            "component": log_component,
            "message": msg,
            "level": "info",
        }
        _add_to_log_buffer(log_component, entry)


def log0(msg: str, component: str | None = None, actor_id: int | None = None):
    """
    Log only if this is the master process (rank 0).
    
    Use this for informational messages that should only appear once in DDP.
    For errors or per-rank diagnostics, use regular log().
    """
    if _is_master_process:
        log(msg, component=component, actor_id=actor_id)


def log_error(msg: str, exception: Exception | None = None, component: str | None = None, actor_id: int | None = None):
    """Log an error with optional exception details."""
    if exception is not None:
        error_detail = f"{type(exception).__name__}: {exception}"
        full_msg = f"ERROR: {msg} - {error_detail}"
    else:
        full_msg = f"ERROR: {msg}"

    log(full_msg, component=component, actor_id=actor_id)

    if exception is not None:
        with _log_lock:
            if _log_file is not None:
                traceback.print_exc(file=_log_file)
                _log_file.flush()
            traceback.print_exc()


def log_actionable_error(component: str, error: str, **extra):
    """Append a structured error to errors.jsonl in the run directory.

    Use this for errors that may need human attention (OOM, repeated actor
    failures, etc.) -- not for routine per-theorem failures.

    Identical errors (same component + first line) are deduplicated within
    a 60s window so a stuck/dead Lean server cannot fill the file with
    hundreds of thousands of duplicate entries; the next emitted entry
    carries `suppressed_since_last` with the count of skipped duplicates.
    """
    with _log_lock:
        if _errors_file is None:
            return
        key = (component, error.split("\n", 1)[0][:200])
        now = time.monotonic()
        last_time, suppressed = _error_dedup.get(key, (0.0, 0))
        if last_time and (now - last_time) < _error_dedup_window_seconds:
            _error_dedup[key] = (last_time, suppressed + 1)
            return
        _error_dedup[key] = (now, 0)
        entry = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "rank": _ddp_rank,
            "component": component,
            "error": error,
            **extra,
        }
        if suppressed > 0:
            entry["suppressed_since_last"] = suppressed
        _errors_file.write(json.dumps(entry) + "\n")
        _errors_file.flush()


def log_tactic(state: str, tactic: str, status: str):
    """Log a generated tactic to the tactics file and in-memory buffer.
    
    Args:
        state: The proof state (goal)
        tactic: The tactic that was attempted
        status: One of "success", "error", or "cycle"
    """
    state_oneline = state.replace("\n", "\\n")
    
    with _log_lock:
        if _tactics_file is not None:
            _tactics_file.write(f"{status}\t{state_oneline}\t{tactic}\n")
            _tactics_file.flush()
    
    # Also add to in-memory buffer (for distributed mode collection)
    with _tactics_buffer_lock:
        _tactics_buffer.append({
            "status": status,
            "state": state_oneline,
            "tactic": tactic,
        })


def get_and_clear_tactics_buffer() -> list[dict]:
    """Get all tactics from the buffer and clear it.
    
    Used by prover servers to collect tactics and report them to the coordinator.
    
    Returns:
        List of {"status": str, "state": str, "tactic": str} dicts.
    """
    with _tactics_buffer_lock:
        tactics = list(_tactics_buffer)
        _tactics_buffer.clear()
        return tactics


# -----------------------------------------------------------------------------
# Data structures for state
# -----------------------------------------------------------------------------

@dataclass
class LocalActorStatus:
    """Status of a local actor thread."""
    id: int
    state: Literal["idle", "running", "error"] = "idle"
    games_played: int = 0
    games_solved: int = 0
    current_theorem: str = ""
    last_update: float = field(default_factory=time.time)


@dataclass
class GPUStatus:
    """Status of a GPU."""
    id: int
    name: str = "Unknown"
    utilization: float = 0.0  # 0-100
    memory_used: int = 0  # MB
    memory_total: int = 0  # MB
    inference_queue_size: int = 0
    avg_wait_time_ms: float = 0.0


@dataclass
class LeanServerStatus:
    """Status of the Lean server."""
    address: str = ""
    port: int = 0
    connected: bool = False
    available_processes: int = 0
    # None when the last poll failed or hasn't run yet; we surface the gap
    # rather than a misleading 0.  Metrics loggers skip None-valued fields
    # so wandb shows a gap instead of a spurious zero on disconnected steps.
    used_processes: int | None = None
    max_processes: int = 0
    cpu_percent: list[float] = field(default_factory=list)
    ram_percent: float | None = None
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    # Resident memory of the leanserver Python process itself (not the host).
    # Grow-over-time here is a memory leak in the leanserver; we've seen
    # one instance grow to 110 GiB in production while its sibling stayed
    # at 50 MiB on the same workload. Surfaced in /status via commit
    # leantree/5a4b499. Populated only when the server is new enough to
    # report it; 0 otherwise (treat as "unknown", not "zero").
    leanserver_rss_gb: float = 0.0
    # Currently-tracked proof branches on the server side. Useful to
    # correlate with rss_gb to distinguish "branches_dict is big" from
    # "something else in the python heap is big".
    total_branches: int = 0
    last_update: float = field(default_factory=time.time)
    error: str = ""


@dataclass
class CollectionStats:
    """Statistics for the current collection phase."""
    num_actors: int = 0
    samples_collected: int = 0
    target_samples: int = 0
    proofs_attempted: int = 0
    proofs_successful: int = 0
    expansions: int = 0
    start_time: float = 0.0
    wait_times: list[float] = field(default_factory=list)
    _wait_times_lock: threading.Lock = field(default_factory=threading.Lock)

    def record_wait_time(self, wait_time: float):
        with self._wait_times_lock:
            self.wait_times.append(wait_time)
            # Keep only last 1000 samples
            if len(self.wait_times) > 1000:
                self.wait_times = self.wait_times[-1000:]

    def get_wait_time_stats(self) -> tuple[float, float, float]:
        """Returns (min, max, median) wait times, or (0, 0, 0) if no data."""
        with self._wait_times_lock:
            if not self.wait_times:
                return (0.0, 0.0, 0.0)
            return (min(self.wait_times), max(self.wait_times), median(self.wait_times))

    def reset(self):
        self.samples_collected = 0
        self.target_samples = 0
        self.proofs_attempted = 0
        self.proofs_successful = 0
        self.expansions = 0
        self.start_time = time.time()
        with self._wait_times_lock:
            self.wait_times = []

    @property
    def success_rate(self) -> float:
        if self.proofs_attempted == 0:
            return 0.0
        return self.proofs_successful / self.proofs_attempted

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time > 0 else 0.0
    
    def to_dict(self) -> dict:
        wait_min, wait_max, wait_med = self.get_wait_time_stats()
        return {
            "num_actors": self.num_actors,
            "samples_collected": self.samples_collected,
            "target_samples": self.target_samples,
            "proofs_attempted": self.proofs_attempted,
            "proofs_successful": self.proofs_successful,
            "success_rate": self.success_rate,
            "expansions": self.expansions,
            "elapsed": self.elapsed,
            "wait_time_min": wait_min,
            "wait_time_max": wait_max,
            "wait_time_median": wait_med,
        }


@dataclass
class EvalResult:
    """Result from an evaluation run."""
    step: int
    dataset: str
    success_rate: float
    solved: int
    total: int
    errors: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class EvalProgress:
    """Progress of the current evaluation."""
    dataset: str = ""
    current: int = 0
    total: int = 0
    solved: int = 0
    errors: int = 0
    active: bool = False

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "current": self.current,
            "total": self.total,
            "solved": self.solved,
            "errors": self.errors,
            "active": self.active,
            "progress_percent": (self.current / self.total * 100) if self.total > 0 else 0,
        }


@dataclass
class TrainingStats:
    """Statistics for the current training step."""
    step: int = 0
    loss: float = 0.0
    num_tokens: int = 0
    learning_rate: float = 0.0


Phase = Literal["idle", "collecting", "evaluating", "training"]


# -----------------------------------------------------------------------------
# Instrumentation payload helpers
# -----------------------------------------------------------------------------

# Max wall-clock spread between rank-duplicated phase events for the same
# (name, action). DDP ranks sit behind a barrier so real transitions fire
# within a handful of seconds; consecutive real transitions of the same kind
# are much farther apart (whole phase durations). 10s is comfortably in the gap.
_PHASE_DEDUP_WINDOW = 10.0

def _compact_instrumentation(
    actors: dict[Any, list[dict]],
    phases: list[dict],
    mode: str,
    since: float,
) -> dict:
    """Encode actor timelines and phase events into a compact, gzip-friendly shape.

    Instead of a list of `{type, start, end}` objects per actor (lots of repeated
    keys and braces), emit flat typed arrays: for each actor, a `llm` and `lean`
    array of interleaved [start, end, start, end, ...] floats. This halves the
    JSON size before gzip and removes most of the parse overhead on the client.

    `since` filters to events that started strictly after that absolute
    timestamp (for delta fetches). Pass -inf to return everything.
    """
    out_actors: dict[str, dict[str, list[float]]] = {}
    max_cursor = since if since != float("-inf") else 0.0
    for aid, events in actors.items():
        llm: list[float] = []
        lean: list[float] = []
        for ev in events:
            start = ev["start"]
            if start <= since:
                continue
            bucket = llm if ev["type"] == "llm" else lean
            bucket.append(start)
            bucket.append(ev["end"])
            if start > max_cursor:
                max_cursor = start
        if llm or lean:
            out_actors[str(aid)] = {"llm": llm, "lean": lean}

    # Phase events are semantically global (all DDP ranks transition together
    # behind a barrier), but older run logs were written from every rank, so
    # one transition shows up as N near-duplicate entries. Collapse groups
    # with the same (name, action) that land within _PHASE_DEDUP_WINDOW of
    # each other, keeping the earliest timestamp (which is what you'd want
    # for "start of this phase").
    out_phases = []
    sorted_phases = sorted(phases, key=lambda p: p["time"])
    last_by_key: dict[tuple[str, str], float] = {}
    for ph in sorted_phases:
        t = ph["time"]
        if t <= since:
            continue
        key = (ph["name"], ph["action"])
        last_t = last_by_key.get(key)
        if last_t is not None and t - last_t < _PHASE_DEDUP_WINDOW:
            # Extend the dedup window so a slow cascade (rank 7 lagging rank 0
            # by 15s) still collapses as long as each step is within the window.
            last_by_key[key] = t
            continue
        last_by_key[key] = t
        out_phases.append({"name": ph["name"], "action": ph["action"], "t": t})
        if t > max_cursor:
            max_cursor = t

    return {
        "actors": out_actors,
        "phases": out_phases,
        "mode": mode,
        "cursor": max_cursor,
    }


def _gzip_json(payload: dict) -> Response:
    """Serialize payload as compact JSON, gzip, return with Content-Encoding header."""
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    body = gzip.compress(raw, compresslevel=6)
    return Response(
        body,
        mimetype="application/json",
        headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"},
    )


# -----------------------------------------------------------------------------
# Web Monitor
# -----------------------------------------------------------------------------

class WebMonitor:
    """
    Web-based monitor for the RL training loop.
    
    Runs a Flask server in a background thread that serves:
    - A React web app for visualization
    - API endpoints for state polling and log streaming
    """

    def __init__(self, num_actors: int = 0, enabled: bool = True, port: int = 5050):
        self.enabled = enabled
        self.port = port
        self._lock = threading.Lock()

        # Output directory for replay buffers and logs
        self.output_dir: str | None = None

        # Current state
        self.phase: Phase = "idle"
        self.step: int = 0
        self.replay_buffer_size: int = 0
        self.replay_buffer_base_size: int = 0  # Size at start of collection

        # Live transitions (for displaying during collection)
        self.live_transitions: deque = deque(maxlen=500)  # Increased from 200
        
        # Live tactics (for displaying in distributed mode)
        self.live_tactics: deque = deque(maxlen=200)

        # Collection stats
        self.collection = CollectionStats(num_actors=num_actors)
        
        # Training start time (for calculating overall expansions/sec)
        self.training_start_time: float = time.time()

        # Training stats
        self.training = TrainingStats()

        # Evaluation history
        self.eval_history: deque[EvalResult] = deque(maxlen=50)

        # Current evaluation progress
        self.eval_progress = EvalProgress()

        # Local actors
        self.local_actors: dict[int, LocalActorStatus] = {}

        # GPU status
        self.gpus: list[GPUStatus] = []

        # Lean server status (single server for local mode)
        self.lean_server: LeanServerStatus = LeanServerStatus()
        
        # Multiple lean servers (for distributed mode monitoring)
        self.lean_servers: list[LeanServerStatus] = []

        # Timeline instrumentation
        self.actor_timelines: dict[int, deque] = {}  # actor_id -> deque of event dicts
        self.phase_events: list[dict] = []  # global phase start/end markers
        self._timeline_file: TextIO | None = None
        self.mode: str = "live"  # "live" or "standalone"
        self._max_timeline_events_per_actor = 10000

        # Cache for standalone (post-hoc) instrumentation reads. Keyed by
        # timeline.jsonl mtime so we only re-parse when the file changes.
        self._instr_file_cache: dict[str, Any] = {"mtime": None, "body": None}

        # Server thread
        self._server_thread: threading.Thread | None = None
        self._gpu_monitor_thread: threading.Thread | None = None
        self._lean_monitor_thread: threading.Thread | None = None
        self._lean_servers_monitor_thread: threading.Thread | None = None
        self._stop_monitors = threading.Event()
        self._app: Flask | None = None

        if enabled:
            self._start_server()
            self._start_gpu_monitor()

    def _start_server(self):
        """Start the Flask server in a background thread."""
        self._app = self._create_app()
        
        def run_server():
            log_handler = logging.getLogger('werkzeug')
            log_handler.setLevel(logging.ERROR)
            self._app.run(host="0.0.0.0", port=self.port, threaded=True)
        
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        
        url = f"http://localhost:{self.port}"
        print(f"\n{'='*60}")
        print(f"  Web Monitor: {url}")
        print(f"{'='*60}\n")

    def _start_gpu_monitor(self):
        """Start a background thread to monitor GPU status."""
        try:
            import torch
            if not torch.cuda.is_available():
                return
        except ImportError:
            return

        def monitor_gpus():
            # Map PyTorch indices to physical GPU IDs
            # If CUDA_VISIBLE_DEVICES is set, parse it to get physical IDs
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if cuda_visible:
                physical_ids = [int(x.strip()) for x in cuda_visible.split(',') if x.strip()]
            else:
                physical_ids = list(range(torch.cuda.device_count()))
            
            while not self._stop_monitors.wait(timeout=2.0):
                try:
                    # Query all GPUs at once for efficiency and reliability
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', 
                         '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=5.0
                    )
                    if result.returncode != 0:
                        continue  # Keep previous values on failure
                    
                    all_gpu_stats = {}
                    for line in result.stdout.strip().split('\n'):
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 4:
                            gpu_idx = int(parts[0])
                            all_gpu_stats[gpu_idx] = {
                                'utilization': float(parts[1]),
                                'memory_used': int(parts[2]),
                                'memory_total': int(parts[3]),
                            }
                    
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        physical_id = physical_ids[i] if i < len(physical_ids) else i
                        
                        if physical_id in all_gpu_stats:
                            stats = all_gpu_stats[physical_id]
                            self.update_gpu(
                                gpu_id=i,
                                name=props.name,
                                utilization=stats['utilization'],
                                memory_used=stats['memory_used'],
                                memory_total=stats['memory_total'],
                            )
                        # If physical_id not found, keep previous values
                except Exception:
                    pass  # Keep previous values on error
        
        self._gpu_monitor_thread = threading.Thread(target=monitor_gpus, daemon=True)
        self._gpu_monitor_thread.start()

    def set_lean_server(self, address: str, port: int):
        """Configure the Lean server address and start monitoring."""
        with self._lock:
            self.lean_server.address = address
            self.lean_server.port = port
        
        # Start the Lean server monitor if not already running
        if self._lean_monitor_thread is None and self.enabled:
            self._start_lean_monitor()

    def set_lean_servers(self, server_urls: list[str]):
        """
        Configure multiple Lean servers for monitoring (distributed mode).
        
        Args:
            server_urls: List of server URLs in format "host:port"
        """
        with self._lock:
            self.lean_servers = []
            for url in server_urls:
                if ":" in url:
                    host, port = url.rsplit(":", 1)
                    try:
                        port_int = int(port)
                    except ValueError:
                        port_int = 8000
                else:
                    host = url
                    port_int = 8000
                
                server = LeanServerStatus(address=host, port=port_int)
                self.lean_servers.append(server)
        
        # Start the multi-server monitor if not already running
        if self._lean_servers_monitor_thread is None and self.enabled:
            self._start_lean_servers_monitor()

    def _start_lean_servers_monitor(self):
        """Start a background thread to monitor multiple Lean servers."""
        def monitor_lean_servers():
            while not self._stop_monitors.wait(timeout=3.0):
                with self._lock:
                    servers = self.lean_servers[:]
                
                for server in servers:
                    if not server.address or not server.port:
                        continue
                    
                    try:
                        url = f"http://{server.address}:{server.port}/status"
                        req = urllib.request.Request(url, method="GET")
                        req.add_header("Accept", "application/json")
                        
                        with urllib.request.urlopen(req, timeout=5.0) as response:
                            data = json.loads(response.read().decode())
                        
                        with self._lock:
                            server.connected = True
                            server.available_processes = data.get("available_processes", 0)
                            server.used_processes = data.get("used_processes", 0)
                            server.max_processes = data.get("max_processes", 0)
                            server.cpu_percent = data.get("cpu_percent_per_core", [])
                            ram = data.get("ram", {})
                            server.ram_percent = ram.get("percent", 0.0)
                            server.ram_used_gb = ram.get("used_bytes", 0) / (1024**3)
                            server.ram_total_gb = ram.get("total_bytes", 0) / (1024**3)
                            # 0 when the leanserver is old enough not to report
                            # this field yet; the metric reader can filter those.
                            rss_bytes = data.get("leanserver_rss_bytes") or 0
                            server.leanserver_rss_gb = rss_bytes / (1024**3)
                            server.total_branches = data.get("total_branches", 0)
                            server.last_update = time.time()
                            server.error = ""
                    except Exception as e:
                        with self._lock:
                            server.connected = False
                            server.error = str(e)
                            # Null the observables we surface so callers can
                            # distinguish "down" from "idle".
                            server.ram_percent = None
                            server.used_processes = None

        self._lean_servers_monitor_thread = threading.Thread(target=monitor_lean_servers, daemon=True)
        self._lean_servers_monitor_thread.start()

    def _start_lean_monitor(self):
        """Start a background thread to monitor Lean server status."""
        def monitor_lean():
            while not self._stop_monitors.wait(timeout=3.0):
                with self._lock:
                    address = self.lean_server.address
                    port = self.lean_server.port
                
                if not address or not port:
                    continue
                
                try:
                    url = f"http://{address}:{port}/status"
                    req = urllib.request.Request(url, method="GET")
                    req.add_header("Accept", "application/json")
                    
                    with urllib.request.urlopen(req, timeout=5.0) as response:
                        data = json.loads(response.read().decode())
                    
                    with self._lock:
                        self.lean_server.connected = True
                        self.lean_server.available_processes = data.get("available_processes", 0)
                        self.lean_server.used_processes = data.get("used_processes", 0)
                        self.lean_server.max_processes = data.get("max_processes", 0)
                        self.lean_server.cpu_percent = data.get("cpu_percent_per_core", [])
                        ram = data.get("ram", {})
                        self.lean_server.ram_percent = ram.get("percent", 0.0)
                        self.lean_server.ram_used_gb = ram.get("used_bytes", 0) / (1024**3)
                        self.lean_server.ram_total_gb = ram.get("total_bytes", 0) / (1024**3)
                        self.lean_server.last_update = time.time()
                        self.lean_server.error = ""
                except Exception as e:
                    with self._lock:
                        self.lean_server.connected = False
                        self.lean_server.error = str(e)
        
        self._lean_monitor_thread = threading.Thread(target=monitor_lean, daemon=True)
        self._lean_monitor_thread.start()

    def _create_app(self) -> Flask:
        """Create the Flask application."""
        # Determine static folder path
        web_dist = os.path.join(os.path.dirname(__file__), "web", "dist")
        if not os.path.exists(web_dist):
            web_dist = None
        
        app = Flask(__name__, static_folder=web_dist, static_url_path="")

        @app.route("/")
        def index():
            if web_dist and os.path.exists(os.path.join(web_dist, "index.html")):
                return send_from_directory(web_dist, "index.html")
            return self._fallback_html()

        @app.route("/api/state")
        def get_state():
            return jsonify(self._get_state())

        @app.route("/api/logs")
        def list_log_components():
            with _log_buffers_lock:
                components = list(_log_buffers.keys())
            return jsonify({"components": components})

        @app.route("/api/logs/<component>")
        def get_logs(component: str):
            """Get recent logs for a component."""
            buffer = _get_log_buffer(component)
            with _log_buffers_lock:
                logs = list(buffer)
            return jsonify({"logs": logs})

        @app.route("/api/logs/<component>/stream")
        def stream_logs(component: str):
            """SSE stream for logs."""
            def generate():
                buffer = _get_log_buffer(component)
                last_len = 0
                while True:
                    with _log_buffers_lock:
                        current_len = len(buffer)
                        if current_len > last_len:
                            new_logs = list(buffer)[last_len:]
                            last_len = current_len
                            for entry in new_logs:
                                yield f"data: {json.dumps(entry)}\n\n"
                    time.sleep(0.5)
            
            return Response(generate(), mimetype="text/event-stream")

        @app.route("/api/replay_buffers")
        def list_replay_buffers():
            """List available replay buffer files."""
            with self._lock:
                output_dir = self.output_dir
            if not output_dir or not os.path.exists(output_dir):
                return jsonify({"files": []})
            
            files = []
            for f in sorted(os.listdir(output_dir)):
                if f.startswith("replay_buffer_") and f.endswith(".jsonl"):
                    filepath = os.path.join(output_dir, f)
                    files.append({
                        "name": f,
                        "size": os.path.getsize(filepath),
                        "step": int(f.replace("replay_buffer_", "").replace(".jsonl", "")),
                    })
            return jsonify({"files": files})

        @app.route("/api/replay_buffers/<filename>")
        def get_replay_buffer(filename: str):
            """Get contents of a specific replay buffer file."""
            with self._lock:
                output_dir = self.output_dir
            if not output_dir:
                return jsonify({"error": "No output directory"}), 404
            
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath) or not filename.startswith("replay_buffer_"):
                return jsonify({"error": "File not found"}), 404
            
            try:
                with open(filepath, "r") as f:
                    data = [json.loads(line) for line in f if line.strip()]
                return jsonify({"transitions": data})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @app.route("/api/tactics")
        def get_tactics():
            """Get recent tactics from tactics.txt or live tactics from distributed mode."""
            with self._lock:
                output_dir = self.output_dir
                # Include live tactics from distributed mode
                live_tactics = list(self.live_tactics)
            
            tactics = []
            
            # Try to read from tactics.txt (local mode)
            if output_dir:
                filepath = os.path.join(output_dir, "tactics.txt")
                if os.path.exists(filepath):
                    try:
                        with open(filepath, "r") as f:
                            lines = f.readlines()
                            # Return last 200 lines
                            for line in lines[-200:]:
                                parts = line.strip().split("\t")
                                if len(parts) >= 3:
                                    tactics.append({
                                        "status": parts[0],
                                        "state": parts[1],
                                        "tactic": parts[2],
                                    })
                    except Exception:
                        pass
            
            # Also include live tactics from distributed mode
            for t in live_tactics:
                tactics.append({
                    "status": t.get("status", "error"),
                    "state": str(t.get("state", "")),
                    "tactic": t.get("tactic", ""),
                })
            
            # Return most recent 200
            return jsonify({"tactics": tactics[-200:]})

        @app.route("/api/live_transitions")
        def get_live_transitions():
            """Get live transitions collected during the current collection phase."""
            with self._lock:
                transitions = list(self.live_transitions)
            return jsonify({"transitions": transitions})

        @app.route("/api/instrumentation")
        def get_instrumentation():
            """Get live timeline instrumentation data (compact + gzipped).

            Query params:
              since (float, optional): absolute seconds. Only events with
                start > since (actors) or t > since (phases) are returned.
                Clients use this to fetch deltas so long runs don't re-download
                history on every poll.
            """
            try:
                since = float(request.args.get("since", "-inf"))
            except ValueError:
                since = float("-inf")
            with self._lock:
                actors_src = {
                    aid: list(evs) for aid, evs in self.actor_timelines.items()
                }
                phases_src = list(self.phase_events)
                mode = self.mode
            payload = _compact_instrumentation(actors_src, phases_src, mode, since)
            return _gzip_json(payload)

        @app.route("/api/instrumentation/file")
        def get_instrumentation_file():
            """Serve timeline.jsonl contents in compact+gzipped form, cached by mtime."""
            with self._lock:
                output_dir = self.output_dir
            if not output_dir:
                return jsonify({"error": "No output directory"}), 404

            timeline_path = os.path.join(output_dir, "timeline.jsonl")
            if not os.path.exists(timeline_path):
                return _gzip_json({"actors": {}, "phases": [], "mode": self.mode})

            try:
                mtime = os.path.getmtime(timeline_path)
            except OSError as e:
                return jsonify({"error": str(e)}), 500

            cached_body = None
            if self._instr_file_cache["mtime"] == mtime:
                cached_body = self._instr_file_cache["body"]

            if cached_body is None:
                actors: dict[str, list] = {}
                phases: list[dict] = []
                try:
                    with open(timeline_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            entry = json.loads(line)
                            t = entry.get("type")
                            if t == "phase":
                                phases.append(entry)
                            elif t == "actor":
                                aid = str(entry["actor"])
                                actors.setdefault(aid, []).append({
                                    "type": entry["event"],
                                    "start": entry["start"],
                                    "end": entry["end"],
                                })
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
                payload = _compact_instrumentation(actors, phases, self.mode, float("-inf"))
                cached_body = gzip.compress(
                    json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                    compresslevel=6,
                )
                self._instr_file_cache["mtime"] = mtime
                self._instr_file_cache["body"] = cached_body

            return Response(
                cached_body,
                mimetype="application/json",
                headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"},
            )

        return app

    def _fallback_html(self) -> str:
        """Fallback HTML when web app is not built."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>NanoProof Monitor</title>
    <style>
        body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #00d9ff; }
        .card { background: #16213e; border-radius: 8px; padding: 16px; margin: 16px 0; }
        .stat { display: inline-block; margin-right: 24px; }
        .stat-value { font-size: 24px; font-weight: bold; color: #00d9ff; }
        .stat-label { font-size: 12px; color: #888; }
        .phase { padding: 4px 12px; border-radius: 4px; font-weight: bold; }
        .phase-collecting { background: #22c55e; color: #000; }
        .phase-training { background: #3b82f6; color: #fff; }
        .phase-evaluating { background: #eab308; color: #000; }
        .phase-idle { background: #666; color: #fff; }
        pre { background: #0f0f1a; padding: 12px; border-radius: 4px; overflow-x: auto; max-height: 300px; }
        .prover-grid { display: flex; gap: 8px; flex-wrap: wrap; }
        .prover-server { background: #1e3a5f; padding: 12px; border-radius: 8px; min-width: 200px; }
        .thread-grid { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 8px; }
        .thread { width: 20px; height: 20px; border-radius: 4px; }
        .thread-running { background: #22c55e; }
        .thread-idle { background: #666; }
        .thread-blocked { background: #eab308; }
        .thread-error { background: #ef4444; }
        .gpu-bar { height: 8px; background: #333; border-radius: 4px; overflow: hidden; margin-top: 4px; }
        .gpu-bar-fill { height: 100%; background: linear-gradient(90deg, #22c55e, #eab308, #ef4444); }
        .logs { font-family: monospace; font-size: 12px; line-height: 1.4; }
        .log-entry { padding: 2px 0; border-bottom: 1px solid #222; }
        .log-time { color: #666; }
        .log-component { color: #00d9ff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>NanoProof Monitor</h1>
        <div id="app">Loading...</div>
    </div>
    <script>
        async function fetchState() {
            try {
                const res = await fetch('/api/state');
                const state = await res.json();
                renderState(state);
            } catch (e) {
                document.getElementById('app').innerHTML = '<p>Error loading state</p>';
            }
        }
        
        function renderState(s) {
            const phaseClass = 'phase-' + s.phase;
            let html = `
                <div class="card">
                    <span class="phase ${phaseClass}">${s.phase.toUpperCase()}</span>
                    <span style="margin-left: 16px;">Step ${s.step}</span>
                </div>
                
                <div class="card">
                    <h3>Stats</h3>
                    <div class="stat"><div class="stat-value">${s.collection.samples_collected}/${s.collection.target_samples}</div><div class="stat-label">Samples</div></div>
                    <div class="stat"><div class="stat-value">${s.collection.proofs_successful}</div><div class="stat-label">Proofs Found</div></div>
                    <div class="stat"><div class="stat-value">${(s.collection.success_rate * 100).toFixed(1)}%</div><div class="stat-label">Success Rate</div></div>
                    <div class="stat"><div class="stat-value">${s.replay_buffer_size}</div><div class="stat-label">Buffer Size</div></div>
                </div>
                
                <div class="card">
                    <h3>Training</h3>
                    <div class="stat"><div class="stat-value">${s.training.loss.toFixed(6)}</div><div class="stat-label">Loss</div></div>
                    <div class="stat"><div class="stat-value">${s.training.num_tokens.toLocaleString()}</div><div class="stat-label">Tokens</div></div>
                </div>
            `;
            
            // GPUs
            if (s.gpus.length > 0) {
                html += '<div class="card"><h3>GPUs</h3>';
                for (const gpu of s.gpus) {
                    const memPct = gpu.memory_total > 0 ? (gpu.memory_used / gpu.memory_total * 100) : 0;
                    html += `<div style="margin: 8px 0;">
                        <div>GPU ${gpu.id}: ${gpu.name}</div>
                        <div>Util: ${gpu.utilization.toFixed(0)}% | Mem: ${gpu.memory_used}/${gpu.memory_total} MB | Queue: ${gpu.inference_queue_size} | Wait: ${gpu.avg_wait_time_ms.toFixed(1)}ms</div>
                        <div class="gpu-bar"><div class="gpu-bar-fill" style="width: ${memPct}%"></div></div>
                    </div>`;
                }
                html += '</div>';
            }
            
            // Eval history
            if (s.eval_history.length > 0) {
                html += '<div class="card"><h3>Evaluations</h3><pre>';
                for (const e of s.eval_history.slice(-10)) {
                    html += `Step ${e.step} | ${e.dataset}: ${(e.success_rate * 100).toFixed(1)}% (${e.solved}/${e.total})\n`;
                }
                html += '</pre></div>';
            }
            
            // Logs
            html += `<div class="card">
                <h3>Logs <button onclick="fetchLogs()">Refresh</button></h3>
                <div id="logs" class="logs" style="max-height: 200px; overflow-y: auto;"></div>
            </div>`;
            
            document.getElementById('app').innerHTML = html;
            fetchLogs();
        }
        
        async function fetchLogs() {
            try {
                const res = await fetch('/api/logs/_merged');
                const data = await res.json();
                const logsDiv = document.getElementById('logs');
                if (logsDiv) {
                    logsDiv.innerHTML = data.logs.slice(-50).map(l => 
                        `<div class="log-entry"><span class="log-time">${l.timestamp}</span> <span class="log-component">[${l.component}]</span> ${l.message}</div>`
                    ).join('');
                    logsDiv.scrollTop = logsDiv.scrollHeight;
                }
            } catch (e) {}
        }
        
        fetchState();
        setInterval(fetchState, 1000);
    </script>
</body>
</html>
"""

    def _get_state(self) -> dict:
        """Get the current state as a JSON-serializable dict."""
        with self._lock:
            return {
                "mode": self.mode,
                "phase": self.phase,
                "step": self.step,
                # During collection, show live count (base + collected)
                "replay_buffer_size": (
                    self.replay_buffer_base_size + self.collection.samples_collected
                    if self.phase == "collecting"
                    else self.replay_buffer_size
                ),
                "output_dir": self.output_dir,
                "collection": {
                    **self.collection.to_dict(),
                    "total_elapsed": time.time() - self.training_start_time,
                },
                "training": {
                    "step": self.training.step,
                    "loss": self.training.loss,
                    "num_tokens": self.training.num_tokens,
                    "learning_rate": self.training.learning_rate,
                },
                "eval_history": [
                    {
                        "step": e.step,
                        "dataset": e.dataset,
                        "success_rate": e.success_rate,
                        "solved": e.solved,
                        "total": e.total,
                        "errors": e.errors,
                        "timestamp": e.timestamp,
                    }
                    for e in self.eval_history
                ],
                "eval_progress": self.eval_progress.to_dict(),
                "local_actors": {
                    str(actor_id): {
                        "id": a.id,
                        "state": a.state,
                        "games_played": a.games_played,
                        "games_solved": a.games_solved,
                        "current_theorem": a.current_theorem[:60] if a.current_theorem else "",
                    }
                    for actor_id, a in self.local_actors.items()
                },
                "gpus": [
                    {
                        "id": g.id,
                        "name": g.name,
                        "utilization": g.utilization,
                        "memory_used": g.memory_used,
                        "memory_total": g.memory_total,
                        "inference_queue_size": g.inference_queue_size,
                        "avg_wait_time_ms": g.avg_wait_time_ms,
                    }
                    for g in self.gpus
                ],
                "lean_server": {
                    "address": self.lean_server.address,
                    "port": self.lean_server.port,
                    "connected": self.lean_server.connected,
                    "available_processes": self.lean_server.available_processes,
                    "used_processes": self.lean_server.used_processes,
                    "max_processes": self.lean_server.max_processes,
                    "cpu_percent": self.lean_server.cpu_percent,
                    "ram_percent": self.lean_server.ram_percent,
                    "ram_used_gb": self.lean_server.ram_used_gb,
                    "ram_total_gb": self.lean_server.ram_total_gb,
                    "error": self.lean_server.error,
                },
                "lean_servers": [
                    {
                        "address": s.address,
                        "port": s.port,
                        "connected": s.connected,
                        "available_processes": s.available_processes,
                        "used_processes": s.used_processes,
                        "max_processes": s.max_processes,
                        "cpu_percent": s.cpu_percent,
                        "ram_percent": s.ram_percent,
                        "ram_used_gb": s.ram_used_gb,
                        "ram_total_gb": s.ram_total_gb,
                        "leanserver_rss_gb": s.leanserver_rss_gb,
                        "total_branches": s.total_branches,
                        "error": s.error,
                    }
                    for s in self.lean_servers
                ],
            }

    # --- State update methods ---

    def set_phase(self, phase: Phase):
        with self._lock:
            self.phase = phase
            if phase == "collecting":
                self.collection.reset()

    def set_step(self, step: int):
        with self._lock:
            self.step = step

    def set_replay_buffer_size(self, size: int):
        with self._lock:
            self.replay_buffer_size = size

    def set_output_dir(self, output_dir: str):
        with self._lock:
            self.output_dir = output_dir
            # Open timeline file for append
            if self._timeline_file is not None:
                self._timeline_file.close()
            timeline_path = os.path.join(output_dir, "timeline.jsonl")
            self._timeline_file = open(timeline_path, "a")

    def start_collection(self, target_samples: int, num_actors: int):
        with self._lock:
            self.phase = "collecting"
            self.collection.reset()
            self.collection.target_samples = target_samples
            self.collection.num_actors = num_actors
            self.replay_buffer_base_size = self.replay_buffer_size
            self.live_transitions.clear()
            self.live_tactics.clear()

    def record_proof_attempt(self, successful: bool, transitions: int = 0):
        with self._lock:
            self.collection.proofs_attempted += 1
            if successful:
                self.collection.proofs_successful += 1
                self.collection.samples_collected += transitions

    def record_transitions(self, transitions: list):
        """Record live transitions during collection."""
        with self._lock:
            for t in transitions:
                self.live_transitions.append(t)
            # Also update samples_collected count for the progress bar
            self.collection.samples_collected += len(transitions)

    def clear_live_transitions(self):
        """Clear live transitions (called when collection ends)."""
        with self._lock:
            self.live_transitions.clear()
            self.live_tactics.clear()

    def record_tactics(self, tactics: list):
        """Record tactics from distributed provers."""
        with self._lock:
            for t in tactics:
                self.live_tactics.append(t)

    def record_expansion(self):
        with self._lock:
            self.collection.expansions += 1

    def record_batch_wait(self, wait_time: float):
        self.collection.record_wait_time(wait_time)

    def update_collection_stats(self, proofs_attempted: int = 0, proofs_successful: int = 0, expansions: int = 0):
        """Update collection stats from distributed mode metrics."""
        with self._lock:
            self.collection.proofs_attempted = proofs_attempted
            self.collection.proofs_successful = proofs_successful
            if expansions > 0:
                self.collection.expansions = expansions

    def update_training(self, step: int, loss: float, num_tokens: int = 0, lr: float = 0.0):
        with self._lock:
            self.phase = "training"
            self.training.step = step
            self.training.loss = loss
            self.training.num_tokens = num_tokens
            self.training.learning_rate = lr

    def record_eval(self, step: int, dataset: str, success_rate: float, solved: int, total: int, errors: int):
        with self._lock:
            self.eval_history.append(EvalResult(
                step=step,
                dataset=dataset,
                success_rate=success_rate,
                solved=solved,
                total=total,
                errors=errors,
            ))
            # Clear eval progress when recording final result
            self.eval_progress = EvalProgress()

    def start_eval(self, dataset: str, total: int):
        """Start tracking evaluation progress."""
        with self._lock:
            self.eval_progress = EvalProgress(
                dataset=dataset,
                current=0,
                total=total,
                solved=0,
                errors=0,
                active=True,
            )

    def update_eval_progress(self, current: int, solved: int, errors: int):
        """Update evaluation progress."""
        with self._lock:
            self.eval_progress.current = current
            self.eval_progress.solved = solved
            self.eval_progress.errors = errors

    # --- Prover server updates ---

    # --- Local actor updates ---

    def update_local_actor(self, actor_id: int, state: str = "running", 
                           games_played: int | None = None, games_solved: int | None = None,
                           current_theorem: str = ""):
        """Update status of a local actor."""
        with self._lock:
            if actor_id not in self.local_actors:
                self.local_actors[actor_id] = LocalActorStatus(id=actor_id)
            
            actor = self.local_actors[actor_id]
            actor.state = state
            if games_played is not None:
                actor.games_played = games_played
            if games_solved is not None:
                actor.games_solved = games_solved
            actor.current_theorem = current_theorem
            actor.last_update = time.time()

    def clear_local_actors(self):
        """Clear all local actors (called when collection ends)."""
        with self._lock:
            self.local_actors.clear()

    # --- Metrics exporter for wandb/goodseed ---

    def lean_server_metrics(self) -> dict[str, float]:
        """Per-leanserver metrics flattened into a wandb-friendly dict.

        Call once per training step and splat into ``wandb_run.log(...)``.
        Returned keys look like::

            monitoring/lean/10_10_25_36/ram_percent
            monitoring/lean/10_10_25_36/used_processes
            monitoring/lean/10_10_25_36/connected   # 1 if last poll succeeded

        Dots in IP addresses are replaced with underscores so wandb groups
        the series sensibly under the "monitoring/" section in the sidebar
        rather than being treated as nested fields.  All values are scalars
        so they pass the safe-filter in MetricsLogger.log.
        """
        metrics: dict[str, float] = {}
        with self._lock:
            servers = list(self.lean_servers)
        for s in servers:
            if not s.address:
                continue
            host_key = s.address.replace(".", "_")
            prefix = f"monitoring/lean/{host_key}"
            metrics[f"{prefix}/connected"] = 1.0 if s.connected else 0.0
            if s.ram_percent is not None:
                metrics[f"{prefix}/ram_percent"] = float(s.ram_percent)
            if s.used_processes is not None:
                metrics[f"{prefix}/used_processes"] = float(s.used_processes)
        return metrics

    # --- Timeline instrumentation ---

    def record_timeline_events(self, actor_id: int, events: list[TimelineEvent]):
        """Record timeline events from a completed proof attempt."""
        with self._lock:
            if actor_id not in self.actor_timelines:
                self.actor_timelines[actor_id] = deque(maxlen=self._max_timeline_events_per_actor)
            buf = self.actor_timelines[actor_id]
            for ev in events:
                d = ev.to_dict()
                buf.append(d)
                if self._timeline_file is not None:
                    self._timeline_file.write(
                        json.dumps({"type": "actor", "actor": actor_id, "event": d["type"],
                                    "start": d["start"], "end": d["end"]}) + "\n")
            if self._timeline_file is not None:
                self._timeline_file.flush()

    def record_phase_event(self, name: str, action: str):
        """Record a global phase event (start/end of collect, eval, train).

        Phase transitions are globally synchronized across DDP ranks, so we
        only log from rank 0. Without this guard, an 8-rank run would write
        8 near-duplicate entries per transition (one per process) into the
        shared timeline.jsonl.
        """
        if not _is_master_process:
            return
        with self._lock:
            entry = {"type": "phase", "name": name, "action": action, "time": time.time()}
            self.phase_events.append(entry)
            if self._timeline_file is not None:
                self._timeline_file.write(json.dumps(entry) + "\n")
                self._timeline_file.flush()

    # --- GPU updates ---

    def update_gpu(self, gpu_id: int, name: str = "", utilization: float = 0.0,
                   memory_used: int = 0, memory_total: int = 0,
                   inference_queue_size: int = 0, avg_wait_time_ms: float = 0.0):
        """Update GPU status."""
        with self._lock:
            # Find or create GPU entry
            gpu = None
            for g in self.gpus:
                if g.id == gpu_id:
                    gpu = g
                    break
            
            if gpu is None:
                gpu = GPUStatus(id=gpu_id)
                self.gpus.append(gpu)
            
            gpu.name = name or gpu.name
            gpu.utilization = utilization
            gpu.memory_used = memory_used
            gpu.memory_total = memory_total
            gpu.inference_queue_size = inference_queue_size
            gpu.avg_wait_time_ms = avg_wait_time_ms


# Alias for backwards compatibility
RLMonitor = WebMonitor

# Global monitor instance
_monitor: WebMonitor | None = None


def get_monitor() -> WebMonitor | None:
    """Get the global monitor instance."""
    return _monitor


def set_monitor(monitor: WebMonitor):
    """Set the global monitor instance."""
    global _monitor
    _monitor = monitor


def create_monitor(num_actors: int = 0, enabled: bool = True, port: int = 5050) -> WebMonitor:
    """Create and set a new global monitor."""
    monitor = WebMonitor(num_actors=num_actors, enabled=enabled, port=port)
    set_monitor(monitor)
    return monitor


def run_standalone(run_dir: str, port: int = 5050):
    """Launch the web monitor in standalone mode on a finished run directory.

    Only the Profiler tab is shown (no live monitor data).
    """
    if not os.path.isdir(run_dir):
        print(f"Error: {run_dir} is not a directory")
        sys.exit(1)

    monitor = WebMonitor(enabled=True, port=port)
    monitor.mode = "standalone"
    monitor.output_dir = run_dir
    set_monitor(monitor)

    print(f"Serving profiler for: {run_dir}")
    # Block forever (the Flask server runs in a daemon thread)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch nanoproof web monitor on a run directory", allow_abbrev=False)
    parser.add_argument("run_dir", help="Path to the RL run output directory")
    parser.add_argument("--port", type=int, default=5050, help="Port for the web server (default: 5050)")
    args = parser.parse_args()
    run_standalone(args.run_dir, args.port)
