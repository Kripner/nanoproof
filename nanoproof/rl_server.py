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
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

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
    Dispatches theorems to provers and collects results.
    
    Two modes:
    - Training: start_training(sampler) - infinite theorems from sampler
    - Eval: start_eval(theorems) - fixed list, done when all processed
    """
    
    def __init__(self):
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._results: list[ProofResult] = []
        self._lock = threading.Lock()
        self._done = threading.Event()
        
        # Mode state
        self._mode: str = "idle"  # "idle", "training", "eval"
        self._sampler: Optional[TheoremsSampler] = None
        self._expected: Optional[int] = None
        self._theorem_counter = 0
    
    def _clear(self):
        """Clear queue and results."""
        self._results = []
        self._done.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
    
    def start_training(self, sampler: TheoremsSampler):
        """Start training mode with infinite theorems from sampler."""
        with self._lock:
            self._clear()
            self._mode = "training"
            self._sampler = sampler
            self._expected = None
        log("Dispatcher: training mode started", component="Dispatcher")
    
    def start_eval(self, theorems: list[str]):
        """Start eval mode with fixed theorem list. Done when all processed."""
        with self._lock:
            self._clear()
            self._mode = "eval"
            self._sampler = None
            self._expected = len(theorems)
            for i, theorem in enumerate(theorems):
                self._queue.put((f"eval_{i}", theorem))
        log(f"Dispatcher: eval mode started with {len(theorems)} theorems", component="Dispatcher")
    
    def stop(self):
        """Stop dispatching (provers will get None from get_theorem)."""
        with self._lock:
            self._mode = "idle"
            self._sampler = None
    
    def get_theorem(self) -> Optional[tuple[str, str]]:
        """Get next theorem. Returns None if idle or eval queue exhausted."""
        if self._mode == "idle":
            return None
        
        # In training mode, refill from sampler
        if self._mode == "training" and self._sampler:
            if self._queue.qsize() < 10:
                for _ in range(50):
                    theorem = self._sampler.sample_theorem()
                    self._theorem_counter += 1
                    self._queue.put((f"train_{self._theorem_counter}", theorem))
        
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None
    
    def submit_result(self, result: ProofResult):
        """Submit a proof result."""
        with self._lock:
            self._results.append(result)
            n = len(self._results)
            solved = sum(1 for r in self._results if r.is_solved)
            
            if n % 10 == 0:
                log(f"Progress: {n} results, {solved} solved", component="Dispatcher")
            
            # In eval mode, signal done when all results received
            if self._mode == "eval" and self._expected and n >= self._expected:
                self._done.set()
    
    def wait(self, timeout: float = None) -> bool:
        """Wait for eval completion. Returns True if done, False on timeout."""
        return self._done.wait(timeout=timeout)
    
    @property
    def results(self) -> list[ProofResult]:
        """Get all collected results."""
        with self._lock:
            return list(self._results)
    
    def metrics(self) -> dict:
        """Get summary metrics."""
        with self._lock:
            total = len(self._results)
            solved = sum(1 for r in self._results if r.is_solved)
            errors = sum(1 for r in self._results if r.error)
            return {
                "success_rate": solved / total if total > 0 else 0.0,
                "solved": solved,
                "total": total,
                "errors": errors,
            }
    
    def get_transitions(self) -> list[tuple[str, str, float]]:
        """Extract all transitions from solved proof trees."""
        transitions = []
        with self._lock:
            for result in self._results:
                if result.is_solved:
                    root = Node.deserialize(result.proof_tree)
                    transitions.extend(extract_transitions(root))
        return transitions


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
    
    def __init__(self, inference_ports: list[int], host: str = "127.0.0.1"):
        self.endpoints = [f"http://{host}:{port}" for port in inference_ports]
        self._next_idx = 0
        self._lock = threading.Lock()
    
    def forward_request(self, states: list[str]) -> list[list[str]]:
        """Forward inference request to next backend server."""
        if len(states) == 0:
            return []
        
        with self._lock:
            idx = self._next_idx
            self._next_idx = (self._next_idx + 1) % len(self.endpoints)
            endpoint = self.endpoints[idx]
        
        try:
            response = requests.post(
                f"{endpoint}/generate",
                json={"states": states},
                timeout=60.0
            )
            response.raise_for_status()
            return response.json().get("tactics", [])
        except Exception as e:
            log(f"Inference request to {endpoint} failed: {e}", component="Coordinator")
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
            "queue_size": dispatcher.queue_size,
            "results": len(dispatcher.results),
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
    
    dispatcher.start_training(sampler)
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
            
            # Get current transitions
            transitions = dispatcher.get_transitions()
            new_transitions = transitions[collected:]
            
            if new_transitions:
                # Add to replay buffer
                replay_buffer.local_buffer.extend(new_transitions)
                collected = len(transitions)
                
                # Update monitor with transitions
                if monitor:
                    monitor.record_transitions(new_transitions)
                    monitor.set_replay_buffer_size(len(replay_buffer.local_buffer))
                
                log(f"Transitions: {collected}/{target_transitions}", component="Coordinator")
            
            # Update monitor with proof stats from dispatcher
            if monitor:
                metrics = dispatcher.metrics()
                monitor.update_collection_stats(
                    proofs_attempted=metrics['total'],
                    proofs_successful=metrics['solved'],
                )
        
        pause_all_provers(registry.get_all())
        dispatcher.stop()
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
    
    # Update monitor to show eval is starting
    if monitor:
        monitor.start_eval(dataset_name, len(theorems))
    
    # Wait for at least one prover
    while registry.count() == 0:
        log("Waiting for provers to register...", component="Coordinator")
        time.sleep(1.0)
    
    dispatcher.start_eval(theorems)
    
    # Start provers
    start_all_provers(registry.get_all())
    
    # Poll for results with progress updates (10 min overall timeout)
    poll_interval = 2.0
    max_wait_time = 600.0
    start_time = time.time()
    
    while not dispatcher.wait(timeout=poll_interval):
        # Update monitor with current progress
        if monitor:
            metrics = dispatcher.metrics()
            monitor.update_eval_progress(
                current=metrics['total'],
                solved=metrics['solved'],
                errors=metrics['errors']
            )
        
        # Check overall timeout
        if time.time() - start_time > max_wait_time:
            log("Eval timed out", component="Coordinator")
            break
    
    metrics = dispatcher.metrics()
    pause_all_provers(registry.get_all())
    
    log(f"Eval complete: {metrics['solved']}/{metrics['total']} solved", component="Coordinator")
    return metrics
