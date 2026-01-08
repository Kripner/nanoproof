"""
RL Server - coordinates distributed RL training.

This module provides:
- TheoremDispatcher: supplies theorems to provers and collects results
- Dynamic prover registration/unregistration
- Coordinator endpoints for theorem dispatch and result collection

Architecture:
    Coordinator:
      - /get_theorem: supplies next theorem to prove
      - /submit_result: receives proof results
      - /generate: proxies to inference servers
    
    Prover:
      - Requests theorems from coordinator
      - Submits results (proof tree or error)

The inference server is started using nanoproof.inference module.
Provers connect to the coordinator for both theorem supply and inference.

Used by rl.py when distributed=True.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import requests
from flask import Flask, request as flask_request, jsonify

from nanoproof.search import Node, extract_transitions
from nanoproof.experience_collection import TheoremsSampler
from nanoproof.cli import log, get_monitor


# -----------------------------------------------------------------------------
# Prover Registry
# -----------------------------------------------------------------------------

class ProverRegistry:
    """Thread-safe registry of prover server addresses."""
    
    def __init__(self):
        self._provers: set[str] = set()
        self._collecting: bool = False
        self._lock = threading.Lock()
    
    def register(self, address: str):
        """Register a prover server. If collection is running, start it immediately."""
        with self._lock:
            self._provers.add(address)
            log(f"Prover registered: {address} (total: {len(self._provers)})", component="Registry")
        
            if self._collecting:
                start_prover(address)
    
    def unregister(self, address: str):
        """Unregister a prover server."""
        with self._lock:
            self._provers.discard(address)
            log(f"Prover unregistered: {address} (total: {len(self._provers)})", component="Registry")
    
    def get_all(self) -> list[str]:
        """Get list of all registered provers."""
        with self._lock:
            return list(self._provers)
    
    def count(self) -> int:
        """Get number of registered provers."""
        with self._lock:
            return len(self._provers)
    
    def set_collecting(self, collecting: bool):
        """Set whether collection is currently running."""
        with self._lock:
            self._collecting = collecting
    
    def is_collecting(self) -> bool:
        """Check if collection is currently running."""
        with self._lock:
            return self._collecting


# Global registry instance
_registry = ProverRegistry()


def get_registry() -> ProverRegistry:
    """Get the global prover registry."""
    return _registry


# -----------------------------------------------------------------------------
# Theorem Dispatcher
# -----------------------------------------------------------------------------

@dataclass
class ProofResult:
    """Result of a proof attempt."""
    theorem_id: str
    theorem: str
    proof_tree: Optional[dict]  # Serialized Node tree, None if not proven
    error: Optional[str] = None  # Error message, None if no error
    
    @property
    def is_solved(self) -> bool:
        return self.proof_tree is not None and self.error is None


class TheoremDispatcher:
    """
    Simple dispatcher with pluggable theorem supplier and result handler.
    
    The actual state management is done by distributed_collect/distributed_eval,
    which set up the get_theorem and submit_result functions.
    """
    
    def __init__(self):
        self.get_theorem: Callable[[], Optional[tuple[str, str]]] = lambda: None
        self.submit_result: Callable[[ProofResult], None] = lambda r: None


# Global dispatcher instance
_dispatcher = TheoremDispatcher()


def get_dispatcher() -> TheoremDispatcher:
    """Get the global theorem dispatcher."""
    return _dispatcher


# -----------------------------------------------------------------------------
# Inference Router (round-robin across inference servers)
# -----------------------------------------------------------------------------

class InferenceRouter:
    """Routes inference requests to backend servers in round-robin fashion."""
    
    def __init__(self, inference_ports: list[int], host: str = "127.0.0.1", timeout: float = 30.0):
        self.endpoints = [f"http://{host}:{port}" for port in inference_ports]
        self.timeout = timeout
        self._next_idx = 0
        self._lock = threading.Lock()
    
    def forward_request(self, states: list[str]) -> list[list[str]]:
        """Forward inference request to next backend server."""
        if not states:
            return []
        
        with self._lock:
            idx = self._next_idx
            self._next_idx = (self._next_idx + 1) % len(self.endpoints)
            endpoint = self.endpoints[idx]
        
        try:
            response = requests.post(
                f"{endpoint}/generate",
                json={"states": states},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("tactics", [])
        except Exception as e:
            log(f"Inference to {endpoint} failed: {e}", component="Coordinator")
            return [[] for _ in states]


# -----------------------------------------------------------------------------
# Coordinator Flask App (prover registration + inference proxy)
# -----------------------------------------------------------------------------

def create_coordinator_app(registry: ProverRegistry, router: InferenceRouter, dispatcher: TheoremDispatcher):
    """Create Flask app for coordinator (handles registration, inference, and theorem dispatch)."""
    app = Flask(__name__)
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "provers": registry.count(),
            "collecting": registry.is_collecting(),
        })
    
    @app.route("/generate", methods=["POST"])
    def generate():
        """Proxy inference request to backend servers."""
        data = flask_request.get_json()
        states = data.get("states", [])
        tactics = router.forward_request(states)
        return jsonify({"tactics": tactics})
    
    @app.route("/get_theorem", methods=["GET"])
    def get_theorem():
        """
        Get next theorem to prove.
        
        Response:
            {"id": "train_123", "theorem": "..."} or {"done": true}
        """
        result = dispatcher.get_theorem()
        if result is None:
            return jsonify({"done": True})
        theorem_id, theorem = result
        return jsonify({"id": theorem_id, "theorem": theorem})
    
    @app.route("/submit_result", methods=["POST"])
    def submit_result():
        """
        Submit proof result.
        
        Request body:
            {
                "id": "train_123",
                "theorem": "...",
                "proof_tree": {...} or null,
                "error": "..." or null
            }
        """
        data = flask_request.get_json()
        result = ProofResult(
            theorem_id=data["id"],
            theorem=data["theorem"],
            proof_tree=data.get("proof_tree"),
            error=data.get("error"),
        )
        dispatcher.submit_result(result)
        return jsonify({"status": "ok"})
    
    @app.route("/register", methods=["POST"])
    def register():
        """Register a prover server."""
        data = flask_request.get_json()
        address = data.get("address")
        if not address:
            return jsonify({"error": "address required"}), 400
        registry.register(address)
        return jsonify({"status": "registered", "collecting": registry.is_collecting()})
    
    @app.route("/unregister", methods=["POST"])
    def unregister():
        """Unregister a prover server."""
        data = flask_request.get_json()
        address = data.get("address")
        if not address:
            return jsonify({"error": "address required"}), 400
        registry.unregister(address)
        return jsonify({"status": "unregistered"})
    
    @app.route("/provers", methods=["GET"])
    def list_provers():
        """List all registered provers."""
        return jsonify({"provers": registry.get_all()})
    
    return app


def start_coordinator(coordinator_port: int, inference_ports: list[int]):
    """Start coordinator server in background thread."""
    registry = get_registry()
    router = InferenceRouter(inference_ports)
    dispatcher = get_dispatcher()
    app = create_coordinator_app(registry, router, dispatcher)
    
    def run_server():
        log_handler = logging.getLogger('werkzeug')
        log_handler.setLevel(logging.ERROR)
        app.run(host="0.0.0.0", port=coordinator_port, threaded=True)
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    log(f"Coordinator started on port {coordinator_port}, proxying to {len(inference_ports)} inference server(s)", 
        component="Coordinator")
    return thread


# -----------------------------------------------------------------------------
# Prover Coordination
# -----------------------------------------------------------------------------

def start_prover(addr: str, max_retries: int = 3, retry_delay: float = 2.0):
    """Start a single prover server with retries for startup race conditions."""
    log(f"Starting prover at {addr}", component="Coordinator")
    for attempt in range(max_retries):
        try:
            response = requests.post(f"http://{addr}/start", timeout=10.0)
            response.raise_for_status()
            log(f"Started prover at {addr}", component="Coordinator")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                log(f"Failed to start prover at {addr} after {max_retries} attempts: {e}", component="Coordinator")


def start_all_provers(prover_addresses: list[str]):
    """Instruct all prover servers to start collection."""
    for addr in prover_addresses:
        start_prover(addr)


def pause_all_provers(prover_addresses: list[str]):
    """Instruct all prover servers to pause collection."""
    for addr in prover_addresses:
        try:
            response = requests.post(f"http://{addr}/pause", timeout=10.0)
            response.raise_for_status()
            log(f"Paused prover at {addr}", component="Coordinator")
        except Exception as e:
            log(f"Failed to pause prover at {addr}: {e}", component="Coordinator")


def poll_all_provers(prover_addresses: list[str], monitor) -> int:
    """Poll all prover servers for their status and update the monitor.
    
    Returns total expansions across all provers.
    """
    total_expansions = 0
    
    for addr in prover_addresses:
        try:
            response = requests.get(f"http://{addr}/poll", timeout=5.0)
            response.raise_for_status()
            data = response.json()
            total_expansions += data.get("expansions", 0)
            if monitor:
                monitor.update_prover_server(
                    address=addr,
                    games_played=data.get("games_played", 0),
                    games_solved=data.get("games_solved", 0),
                    thread_states=data.get("thread_states", []),
                    num_threads=len(data.get("thread_states", [])),
                )
        except Exception:
            # Prover might be busy or disconnected
            pass
    
    return total_expansions


def distributed_collect(
    sampler: TheoremsSampler,
    target_transitions: int,
    poll_interval: float,
    replay_buffer,  # ReplayBuffer
) -> int:
    """
    Collect transitions from distributed provers.
    
    Args:
        sampler: TheoremsSampler for generating training theorems
        target_transitions: Target number of transitions to collect
        poll_interval: How often to check progress
        replay_buffer: Buffer to add extracted transitions to
    
    Returns:
        Number of transitions collected.
    """
    registry = get_registry()
    dispatcher = get_dispatcher()
    monitor = get_monitor()
    
    log(f"Starting distributed collection, target={target_transitions}", component="Coordinator")
    
    # Local state for this collection run
    results: list[ProofResult] = []  # All results (for metrics)
    new_results: list[ProofResult] = []  # Unprocessed results (for transition extraction)
    results_lock = threading.Lock()
    theorem_counter = 0
    
    def get_theorem() -> Optional[tuple[str, str]]:
        """Supply theorems infinitely from sampler."""
        nonlocal theorem_counter
        theorem_counter += 1
        theorem = sampler.sample_theorem()
        return (f"train_{theorem_counter}", theorem)
    
    def submit_result(result: ProofResult):
        """Collect proof results."""
        # Ignore straggler results from previous eval phase
        if not result.theorem_id.startswith("train_"):
            return
        
        with results_lock:
            results.append(result)
            new_results.append(result)
            n = len(results)
            solved = sum(1 for r in results if r.is_solved)
            if n % 10 == 0:
                log(f"Progress: {n} results, {solved} solved", component="Dispatcher")
    
    def extract_new_transitions() -> list[tuple[str, str, float]]:
        """Extract transitions from new results, clearing the new_results list."""
        transitions = []
        with results_lock:
            for result in new_results:
                if result.is_solved:
                    root = Node.deserialize(result.proof_tree)
                    transitions.extend(extract_transitions(root))
            new_results.clear()
        return transitions
    
    def get_metrics() -> dict:
        """Get summary metrics."""
        with results_lock:
            total = len(results)
            solved = sum(1 for r in results if r.is_solved)
            errors = sum(1 for r in results if r.error)
            return {
                "success_rate": solved / total if total > 0 else 0.0,
                "solved": solved,
                "total": total,
                "errors": errors,
            }
    
    # Set up dispatcher
    dispatcher.get_theorem = get_theorem
    dispatcher.submit_result = submit_result
    registry.set_collecting(True)
    
    try:
        # Wait for at least one prover
        while registry.count() == 0:
            log("Waiting for provers to register...", component="Coordinator")
            time.sleep(poll_interval)
        
        # Start all currently registered provers
        start_all_provers(registry.get_all())
        
        collected = 0
        
        while collected < target_transitions:
            time.sleep(poll_interval)
            
            new_transitions = extract_new_transitions()
            if new_transitions:
                replay_buffer.local_buffer.extend(new_transitions)
                collected += len(new_transitions)
                
                if monitor:
                    monitor.record_transitions(new_transitions)
                    monitor.set_replay_buffer_size(len(replay_buffer.local_buffer))
                
                log(f"Transitions: {collected}/{target_transitions}", component="Coordinator")
            
            # Poll prover servers for status updates (and get expansions)
            total_expansions = poll_all_provers(registry.get_all(), monitor)
            
            if monitor:
                metrics = get_metrics()
                monitor.update_collection_stats(
                    proofs_attempted=metrics['total'],
                    proofs_successful=metrics['solved'],
                    expansions=total_expansions,
                )
        
        pause_all_provers(registry.get_all())
        
        # Clear dispatcher (provers will get None)
        dispatcher.get_theorem = lambda: None
        dispatcher.submit_result = lambda r: None
        
        log(f"Distributed collection complete: {collected} transitions", component="Coordinator")
        return collected
    finally:
        registry.set_collecting(False)


def distributed_eval(theorems: list[str], dataset_name: str = "eval") -> dict:
    """
    Evaluate theorems using distributed provers.
    
    Args:
        theorems: List of theorem strings to evaluate
        dataset_name: Name of the dataset for monitor display
    
    Returns:
        Dict with 'success_rate', 'solved', 'total', 'errors'.
    """
    registry = get_registry()
    dispatcher = get_dispatcher()
    monitor = get_monitor()
    
    log(f"Starting distributed evaluation on {len(theorems)} theorems", component="Coordinator")
    
    # Local state for this eval run
    results: list[ProofResult] = []
    results_lock = threading.Lock()
    done = threading.Event()
    index = 0
    expected = len(theorems)
    
    def get_theorem() -> Optional[tuple[str, str]]:
        """Supply theorems from the list, None when exhausted."""
        nonlocal index
        with results_lock:
            if index >= len(theorems):
                return None
            theorem = theorems[index]
            tid = f"eval_{index}"
            index += 1
            return (tid, theorem)
    
    def submit_result(result: ProofResult):
        """Collect proof results and signal done when complete."""
        # Ignore straggler results from previous collection phase
        if not result.theorem_id.startswith("eval_"):
            return
        
        with results_lock:
            results.append(result)
            n = len(results)
            solved = sum(1 for r in results if r.is_solved)
            if n % 10 == 0:
                log(f"Progress: {n} results, {solved} solved", component="Dispatcher")
            if n >= expected:
                done.set()
    
    def get_metrics() -> dict:
        """Get summary metrics."""
        with results_lock:
            total = len(results)
            solved = sum(1 for r in results if r.is_solved)
            errors = sum(1 for r in results if r.error)
            return {
                "success_rate": solved / total if total > 0 else 0.0,
                "solved": solved,
                "total": total,
                "errors": errors,
            }
    
    # Update monitor to show eval is starting
    if monitor:
        monitor.start_eval(dataset_name, len(theorems))
    
    # Wait for at least one prover
    while registry.count() == 0:
        log("Waiting for provers to register...", component="Coordinator")
        time.sleep(1.0)
    
    # Set up dispatcher
    dispatcher.get_theorem = get_theorem
    dispatcher.submit_result = submit_result
    
    # Start provers
    start_all_provers(registry.get_all())
    
    # Poll for results with progress updates (10 min overall timeout)
    poll_interval = 2.0
    max_wait_time = 600.0
    start_time = time.time()
    
    while not done.wait(timeout=poll_interval):
        # Update monitor with current progress
        if monitor:
            metrics = get_metrics()
            monitor.update_eval_progress(
                current=metrics['total'],
                solved=metrics['solved'],
                errors=metrics['errors']
            )
        
        # Poll prover servers for status updates
        poll_all_provers(registry.get_all(), monitor)
        
        # Check overall timeout
        if time.time() - start_time > max_wait_time:
            log("Eval timed out", component="Coordinator")
            break
    
    pause_all_provers(registry.get_all())
    
    # Clear dispatcher
    dispatcher.get_theorem = lambda: None
    dispatcher.submit_result = lambda r: None
    
    # Verify all theorems (and only the requested theorems) were attempted
    assert len(results) == len(theorems), \
        f"Expected {len(theorems)} results, got {len(results)}"
    
    submitted_ids = {r.theorem_id for r in results}
    expected_ids = {f"eval_{i}" for i in range(len(theorems))}
    assert submitted_ids == expected_ids, \
        f"Theorem ID mismatch: missing={expected_ids - submitted_ids}, extra={submitted_ids - expected_ids}"
    
    for result in results:
        idx = int(result.theorem_id.split("_")[1])
        assert result.theorem == theorems[idx], \
            f"Theorem mismatch for {result.theorem_id}: expected {theorems[idx]!r}, got {result.theorem!r}"
    
    metrics = get_metrics()
    log(f"Eval complete: {metrics['solved']}/{metrics['total']} solved", component="Coordinator")
    return metrics
