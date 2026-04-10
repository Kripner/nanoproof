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

from nanoproof.cli import log, log_tactic


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
