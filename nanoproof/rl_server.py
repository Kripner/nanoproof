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
from nanoproof.experience_collection import TheoremsSampler, ReplayBuffer
from nanoproof.cli import log, get_monitor, log_tactic


# -----------------------------------------------------------------------------
# Prover Registry
# -----------------------------------------------------------------------------

class ProverRegistry:
    """Thread-safe registry of prover server addresses."""
    
    def __init__(self):
        self._provers: set[str] = set()
        self._autostart: bool = False
        self._lock = threading.Lock()
    
    def register(self, address: str) -> bool:
        """Register a prover server. If autostart is enabled, start it immediately.
        
        Returns True if this was a new registration, False if already registered.
        """
        with self._lock:
            if address in self._provers:
                return False  # Already registered, no-op
            
            self._provers.add(address)
            log(f"Prover registered: {address} (total: {len(self._provers)})", component="Registry")
        
            if self._autostart:
                start_prover(address)
            
            return True
    
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
    
    def set_autostart(self, autostart: bool):
        """Set whether newly registered provers should be auto-started."""
        with self._lock:
            self._autostart = autostart
    
    def is_autostart(self) -> bool:
        """Check if autostart is enabled."""
        with self._lock:
            return self._autostart


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
    proof_tree: Optional[dict]  # Serialized (simplified) Node tree, None if not proven
    unsimplified_proof_tree: Optional[dict] = None  # Serialized (unsimplified) Node tree, None if not proven
    linearized_proof: Optional[str] = None  # Human-readable proof source, None if not proven
    error: Optional[str] = None  # Error message, None if no error
    num_iterations: int = 0  # Number of MCTS iterations run (always max when proof is None)
    
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
    
    def __init__(self, endpoints: list[str], timeout: float = 30.0):
        """
        Create an inference router.
        
        Args:
            endpoints: List of full URLs like ["http://127.0.0.1:5001", "http://10.0.0.2:5002"]
            timeout: Request timeout in seconds
        """
        self.endpoints = endpoints
        self.timeout = timeout
        self._next_idx = 0
        self._lock = threading.Lock()
    
    @classmethod
    def from_ports(cls, inference_ports: list[int], host: str = "127.0.0.1", timeout: float = 30.0):
        """Create router for inference servers on specified ports (single-node setup)."""
        endpoints = [f"http://{host}:{port}" for port in inference_ports]
        return cls(endpoints, timeout)
    
    def wait_for_servers(self, startup_timeout: float = 30.0, check_interval: float = 0.5) -> bool:
        """Wait for all inference servers to be healthy.
        
        Returns True if all servers are healthy, False if timeout.
        """
        import time
        start_time = time.time()
        
        while time.time() - start_time < startup_timeout:
            all_healthy = True
            for endpoint in self.endpoints:
                try:
                    response = requests.get(f"{endpoint}/health", timeout=2.0)
                    if response.status_code != 200:
                        all_healthy = False
                        break
                except Exception:
                    all_healthy = False
                    break
            
            if all_healthy:
                log(f"All {len(self.endpoints)} inference servers are healthy", component="Coordinator")
                return True
            
            time.sleep(check_interval)
        
        # Log which servers failed
        for endpoint in self.endpoints:
            try:
                response = requests.get(f"{endpoint}/health", timeout=2.0)
                if response.status_code == 200:
                    log(f"Inference server {endpoint}: OK", component="Coordinator")
                else:
                    log(f"Inference server {endpoint}: unhealthy (status {response.status_code})", component="Coordinator")
            except Exception as e:
                log(f"Inference server {endpoint}: unreachable ({e})", component="Coordinator")
        
        return False
    
    def _get_next_endpoint(self) -> str:
        """Get next endpoint in round-robin fashion."""
        with self._lock:
            idx = self._next_idx
            self._next_idx = (self._next_idx + 1) % len(self.endpoints)
            return self.endpoints[idx]
    
    def forward_generate(self, states: list[str]) -> list[dict]:
        """Forward tactic generation request to next backend server.
        
        Returns list of dicts, each containing either {"tactics": [...]} or {"error": "..."}.
        """
        if not states:
            return []
        
        endpoint = self._get_next_endpoint()
        
        try:
            response = requests.post(
                f"{endpoint}/generate",
                json={"states": states},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as e:
            log(f"Inference to {endpoint} failed: {e}", component="Coordinator")
            return [{"error": str(e)} for _ in states]

    def forward_predict_value(self, states: list[str]) -> list[dict]:
        """Forward value prediction request to next backend server.
        
        Returns list of dicts, each containing either {"value": ...} or {"error": "..."}.
        """
        if not states:
            return []
        
        endpoint = self._get_next_endpoint()
        
        try:
            response = requests.post(
                f"{endpoint}/predict_value",
                json={"states": states},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as e:
            log(f"Value prediction to {endpoint} failed: {e}", component="Coordinator")
            return [{"error": str(e)} for _ in states]


# -----------------------------------------------------------------------------
# Coordinator Flask App (prover registration + inference proxy)
# -----------------------------------------------------------------------------

# Global shutdown flag for coordinator
_coordinator_shutdown = threading.Event()


def create_coordinator_app(registry: ProverRegistry, router: InferenceRouter, dispatcher: TheoremDispatcher):
    """Create Flask app for coordinator (handles registration, inference, and theorem dispatch)."""
    app = Flask(__name__)
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "provers": registry.count(),
            "autostart": registry.is_autostart(),
        })
    
    @app.route("/generate", methods=["POST"])
    def generate():
        """Proxy inference request to backend servers."""
        if _coordinator_shutdown.is_set():
            return jsonify({"results": [{"error": "shutting_down"}]}), 503
        data = flask_request.get_json()
        states = data.get("states", [])
        results = router.forward_generate(states)
        return jsonify({"results": results})

    @app.route("/predict_value", methods=["POST"])
    def predict_value():
        """Proxy value prediction request to backend servers."""
        if _coordinator_shutdown.is_set():
            return jsonify({"results": [{"error": "shutting_down"}]}), 503
        data = flask_request.get_json()
        states = data.get("states", [])
        results = router.forward_predict_value(states)
        return jsonify({"results": results})
    
    @app.route("/get_theorem", methods=["GET"])
    def get_theorem():
        """
        Get next theorem to prove.
        
        Response:
            {"id": "train_123", "theorem": "..."} or {"done": true}
        """
        if _coordinator_shutdown.is_set():
            return jsonify({"done": True})  # Signal provers to stop
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
                "unsimplified_proof_tree": {...} or null,
                "error": "..." or null
            }
        """
        if _coordinator_shutdown.is_set():
            return jsonify({"status": "ignored"})
        data = flask_request.get_json()
        result = ProofResult(
            theorem_id=data["id"],
            theorem=data["theorem"],
            proof_tree=data.get("proof_tree"),
            unsimplified_proof_tree=data.get("unsimplified_proof_tree"),
            linearized_proof=data.get("linearized_proof"),
            error=data.get("error"),
            num_iterations=data.get("num_iterations", 0),
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
        return jsonify({"status": "registered", "autostart": registry.is_autostart()})
    
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


def start_coordinator(
    coordinator_port: int, 
    inference_endpoints: list[str],
    startup_timeout: float = 30.0
):
    """Start coordinator server in background thread.
    
    Args:
        coordinator_port: Port for the coordinator to listen on
        inference_endpoints: List of inference server URLs (e.g., ["http://127.0.0.1:5001"])
        startup_timeout: Maximum time to wait for inference servers to be healthy
    
    Returns:
        Tuple of (coordinator_thread, inference_router).
        
    Note: Call shutdown_coordinator() before exit to stop accepting new requests.
    """
    registry = get_registry()
    router = InferenceRouter(inference_endpoints)
    dispatcher = get_dispatcher()
    app = create_coordinator_app(registry, router, dispatcher)
    
    def run_server():
        log_handler = logging.getLogger('werkzeug')
        log_handler.setLevel(logging.ERROR)
        app.run(host="0.0.0.0", port=coordinator_port, threaded=True)
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    log(f"Coordinator started on port {coordinator_port}, proxying to {len(inference_endpoints)} inference server(s): {inference_endpoints}", 
        component="Coordinator")
    
    # Wait for all inference servers to be healthy
    if not router.wait_for_servers(startup_timeout=startup_timeout):
        log("WARNING: Not all inference servers are healthy! Distributed collection may fail.", 
            component="Coordinator")
    
    return thread, router


def shutdown_coordinator():
    """Signal the coordinator to stop accepting new work and pause all provers."""
    if not _coordinator_shutdown.is_set():
        _coordinator_shutdown.set()
        log("Coordinator shutdown signaled", component="Coordinator")
        pause_all_provers(get_registry().get_all())


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
    Also collects tactics and writes them to tactics.txt.
    """
    total_expansions = 0
    
    for addr in prover_addresses:
        try:
            response = requests.get(f"http://{addr}/poll", timeout=5.0)
            response.raise_for_status()
            data = response.json()
            total_expansions += data.get("expansions", 0)
            
            # Collect tactics from this prover and log them locally
            tactics = data.get("tactics", [])
            for t in tactics:
                log_tactic(t.get("state", ""), t.get("tactic", ""), t.get("status", "error"))
            
            # Also add to monitor's live tactics for the UI
            if monitor and tactics:
                monitor.record_tactics(tactics)
            
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
    replay_buffer: ReplayBuffer,
    no_progress_timeout: float = 300.0,  # 5 minute timeout if no progress
) -> int:
    """
    Collect transitions from distributed provers.
    
    Args:
        sampler: TheoremsSampler for generating training theorems
        target_transitions: Target number of transitions to collect
        poll_interval: How often to check progress
        replay_buffer: Buffer to add extracted transitions to
        no_progress_timeout: Maximum time to wait without any progress (seconds)
    
    Returns:
        Number of transitions collected. May be less than target if stalled.
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
    registry.set_autostart(True)
    
    try:
        # Wait for at least one prover
        while registry.count() == 0:
            log("Waiting for provers to register...", component="Coordinator")
            time.sleep(poll_interval)
        
        # Start all currently registered provers
        start_all_provers(registry.get_all())
        
        collected = 0
        last_progress_time = time.time()
        last_collected = 0
        
        while collected < target_transitions:
            time.sleep(poll_interval)
            
            new_transitions = extract_new_transitions()
            if new_transitions:
                replay_buffer.add_transitions(new_transitions)
                collected += len(new_transitions)
                
                if monitor:
                    monitor.record_transitions(new_transitions)
                    monitor.set_replay_buffer_size(len(replay_buffer.local_buffer))
                
                log(f"Transitions: {collected}/{target_transitions}", component="Coordinator")
            
            # Track progress for timeout detection
            if collected > last_collected:
                last_progress_time = time.time()
                last_collected = collected
            
            # Check for no-progress timeout or provers disconnected
            no_progress_elapsed = time.time() - last_progress_time
            prover_count = registry.count()
            if no_progress_elapsed > no_progress_timeout or prover_count == 0:
                log(f"Collection stalled: no progress for {no_progress_elapsed:.0f}s, "
                    f"provers={prover_count}, collected={collected}/{target_transitions}", 
                    component="Coordinator")
                break
            
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
        registry.set_autostart(False)


def distributed_eval(theorems: list[str], dataset_name: str = "eval", no_progress_timeout: float = 120.0) -> dict:
    """
    Evaluate theorems using distributed provers.
    
    Args:
        theorems: List of theorem strings to evaluate
        dataset_name: Name of the dataset for monitor display
        no_progress_timeout: Maximum time to wait without progress (seconds), default 2 minutes
    
    Returns:
        Dict with 'success_rate', 'solved', 'total', 'errors', 'timed_out'.
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
    timed_out = False
    invalid = False  # Set if we receive mismatched/duplicate/unknown results
    
    # Track sent theorems for validation: id -> theorem
    sent_theorems: dict[str, str] = {}
    received_ids: set[str] = set()
    
    def get_theorem() -> Optional[tuple[str, str]]:
        """Supply theorems from the list, None when exhausted."""
        nonlocal index
        with results_lock:
            if index >= len(theorems):
                return None
            theorem = theorems[index]
            tid = f"eval_{index}"
            sent_theorems[tid] = theorem
            index += 1
            return (tid, theorem)
    
    def submit_result(result: ProofResult):
        """Collect proof results and signal done when complete."""
        nonlocal invalid
        
        # Ignore straggler results from previous collection phase
        if not result.theorem_id.startswith("eval_"):
            return
        
        with results_lock:
            # Validate that we actually sent this theorem
            if result.theorem_id not in sent_theorems:
                log(f"Received result for unknown theorem ID: {result.theorem_id}", component="Dispatcher")
                invalid = True
                return
            
            # Validate theorem content matches what we sent
            expected_theorem = sent_theorems[result.theorem_id]
            if result.theorem != expected_theorem:
                log(f"Theorem mismatch for {result.theorem_id}: "
                    f"expected {expected_theorem[:100]!r}..., got {result.theorem[:100]!r}...", 
                    component="Dispatcher")
                invalid = True
                return
            
            # Check for duplicate results
            if result.theorem_id in received_ids:
                log(f"Duplicate result for {result.theorem_id}", component="Dispatcher")
                invalid = True
                return
            
            received_ids.add(result.theorem_id)
            results.append(result)
            n = len(results)
            solved = sum(1 for r in results if r.is_solved)
            if n % 10 == 0:
                log(f"Progress: {n} results, {solved} solved", component="Dispatcher")
            if n >= expected:
                done.set()
    
    def get_metrics(include_detailed: bool = False) -> dict:
        """Get summary metrics based on results received so far."""
        with results_lock:
            total = len(results)
            solved = sum(1 for r in results if r.is_solved)
            errors = sum(1 for r in results if r.error)
            metrics = {
                "success_rate": solved / total if total > 0 else 0.0,
                "solved": solved,
                "total": total,
                "errors": errors,
                "timed_out": timed_out,
                "invalid": invalid,
            }
            if include_detailed:
                # Convert ProofResult to detailed_results format
                metrics["detailed_results"] = [
                    {
                        "theorem": r.theorem,
                        "proof_tree": r.proof_tree,
                        "unsimplified_proof_tree": r.unsimplified_proof_tree,
                        "linearized_proof": r.linearized_proof,
                        "num_iterations": r.num_iterations,
                        "error": r.error,
                    }
                    for r in results
                ]
            return metrics
    
    # Update monitor to show eval is starting
    if monitor:
        monitor.start_eval(dataset_name, len(theorems))
    
    # Wait for at least one prover
    while registry.count() == 0:
        log("Waiting for provers to register...", component="Coordinator")
        time.sleep(3.0)
    
    # Set up dispatcher
    dispatcher.get_theorem = get_theorem
    dispatcher.submit_result = submit_result
    registry.set_autostart(True)
    
    try:
        # Start provers
        start_all_provers(registry.get_all())
        
        # Poll for results with progress updates
        poll_interval = 2.0
        last_progress_time = time.time()
        last_result_count = 0
        
        while not done.wait(timeout=poll_interval):
            # Update monitor with current progress
            if monitor:
                metrics = get_metrics()
                monitor.update_eval_progress(
                    current=metrics['total'],
                    solved=metrics['solved'],
                    errors=metrics['errors']
                )
            
            # Track progress for stall detection
            current_count = len(results)
            if current_count > last_result_count:
                last_progress_time = time.time()
                last_result_count = current_count
            
            # Poll prover servers for status updates
            poll_all_provers(registry.get_all(), monitor)
            
            # Check for no-progress timeout or provers disconnected
            no_progress_elapsed = time.time() - last_progress_time
            prover_count = registry.count()
            if no_progress_elapsed > no_progress_timeout or prover_count == 0:
                missing = expected - len(results)
                log(f"Eval stalled: no progress for {no_progress_elapsed:.0f}s, "
                    f"provers={prover_count}, missing {missing} results", component="Coordinator")
                timed_out = True
                break
        
        pause_all_provers(registry.get_all())
        
        # Clear dispatcher
        dispatcher.get_theorem = lambda: None
        dispatcher.submit_result = lambda r: None
        
        # Check for missing results (only if we didn't time out - timeout is expected to have missing results)
        with results_lock:
            missing_ids = set(sent_theorems.keys()) - received_ids
            if missing_ids and not timed_out:
                log(f"Missing results for {len(missing_ids)} theorems: "
                    f"{sorted(missing_ids)[:10]}{'...' if len(missing_ids) > 10 else ''}", 
                    component="Dispatcher")
                invalid = True
        
        metrics = get_metrics(include_detailed=True)
        status = "timed out" if timed_out else "complete"
        log(f"Eval {status}: {metrics['solved']}/{metrics['total']} solved (expected {expected})", 
            component="Coordinator")
        return metrics
    finally:
        registry.set_autostart(False)
