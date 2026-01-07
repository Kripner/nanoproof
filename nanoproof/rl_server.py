"""
RL Server - coordinates distributed RL training.

This module provides:
- Inference servers that run on each DDP rank (one per GPU)
- Inference proxy that load-balances across inference servers
- Dynamic prover registration/unregistration
- Prover coordination (start/pause/poll)
- Distributed collection of transitions

Architecture (with 2 GPUs):
    prover_server → Coordinator (port 5000) → Inference servers
                                               ├─ Rank 0 (port 5001, GPU 0)
                                               └─ Rank 1 (port 5002, GPU 1)

Used by rl.py when distributed=True.
"""

import threading
import time

import requests
import torch
from flask import Flask, request as flask_request, jsonify

from nanoproof.search import TacticModel
from nanoproof.experience_collection import ReplayBuffer
from nanoproof.cli import log
from nanoproof.cli import get_monitor


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
# Inference Handler (runs on each DDP rank)
# -----------------------------------------------------------------------------

class InferenceHandler:
    """
    Handles inference requests locally using a TacticModel.
    Thread-safe wrapper around TacticModel.
    Each DDP rank runs one of these on its GPU.
    """
    
    def __init__(self, tactic_model: TacticModel):
        self.tactic_model = tactic_model
        self._lock = threading.Lock()
    
    @torch.no_grad()
    def generate_tactics_batch(self, state_strs: list[str]) -> list[list[str]]:
        """Generate tactics for a batch of states."""
        with self._lock:
            return self.tactic_model.sample_tactic_from_str_batch(state_strs)


def create_inference_only_app(handler: InferenceHandler, rank: int):
    """Create Flask app for inference-only server (no registration, just /generate)."""
    app = Flask(__name__)
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "rank": rank})
    
    @app.route("/generate", methods=["POST"])
    def generate():
        data = flask_request.get_json()
        states = data.get("states", [])
        if len(states) == 0:
            return jsonify({"tactics": []})
        tactics = handler.generate_tactics_batch(states)
        return jsonify({"tactics": tactics})
    
    return app


def start_inference_only_server(handler: InferenceHandler, port: int, rank: int):
    """Start inference-only server in background thread."""
    app = create_inference_only_app(handler, rank)
    
    def run_server():
        import logging
        log_handler = logging.getLogger('werkzeug')
        log_handler.setLevel(logging.ERROR)
        app.run(host="0.0.0.0", port=port, threaded=True)
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    log(f"Inference server started on port {port} (rank {rank})", component="InferenceServer")
    return thread


# -----------------------------------------------------------------------------
# Inference Proxy (load-balances across inference servers)
# -----------------------------------------------------------------------------

class InferenceProxy:
    """
    Proxies inference requests to multiple inference servers in round-robin.
    Used by the coordinator to distribute load across GPUs.
    """
    
    def __init__(self, inference_ports: list[int], host: str = "127.0.0.1"):
        self.endpoints = [f"http://{host}:{port}" for port in inference_ports]
        self._next_idx = 0
        self._lock = threading.Lock()
        self._requests_per_endpoint = [0] * len(self.endpoints)
        log(f"Inference proxy initialized with endpoints: {self.endpoints}", component="InferenceProxy")
    
    def _get_next_endpoint(self) -> tuple[int, str]:
        """Get the next endpoint in round-robin order."""
        with self._lock:
            idx = self._next_idx
            self._next_idx = (self._next_idx + 1) % len(self.endpoints)
            self._requests_per_endpoint[idx] += 1
            return idx, self.endpoints[idx]
    
    def generate_tactics_batch(self, state_strs: list[str]) -> list[list[str]]:
        """Forward request to next inference server."""
        if len(state_strs) == 0:
            return []
        
        idx, endpoint = self._get_next_endpoint()
        
        try:
            response = requests.post(
                f"{endpoint}/generate",
                json={"states": state_strs},
                timeout=30.0
            )
            response.raise_for_status()
            return response.json().get("tactics", [])
        except Exception as e:
            log(f"Inference request to {endpoint} failed: {e}", component="InferenceProxy")
            # Return empty tactics on failure
            return [[] for _ in state_strs]
    
    def get_stats(self) -> dict:
        """Get load-balancing stats."""
        with self._lock:
            return {
                "endpoints": self.endpoints,
                "requests_per_endpoint": self._requests_per_endpoint.copy(),
            }


# -----------------------------------------------------------------------------
# Coordinator Flask App (prover registration + inference proxy)
# -----------------------------------------------------------------------------

def create_coordinator_app(proxy: InferenceProxy, registry: ProverRegistry):
    """Create Flask app for coordinator (handles registration and proxies inference)."""
    app = Flask(__name__)
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "provers": registry.count(),
            "collecting": registry.is_collecting(),
            "inference_stats": proxy.get_stats(),
        })
    
    @app.route("/generate", methods=["POST"])
    def generate():
        """Proxy inference request to backend servers."""
        data = flask_request.get_json()
        states = data.get("states", [])
        if len(states) == 0:
            return jsonify({"tactics": []})
        tactics = proxy.generate_tactics_batch(states)
        return jsonify({"tactics": tactics})
    
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


def start_coordinator(inference_ports: list[int], coordinator_port: int):
    """Start coordinator server in background thread."""
    registry = get_registry()
    proxy = InferenceProxy(inference_ports)
    app = create_coordinator_app(proxy, registry)
    
    def run_server():
        import logging
        log_handler = logging.getLogger('werkzeug')
        log_handler.setLevel(logging.ERROR)
        app.run(host="0.0.0.0", port=coordinator_port, threaded=True)
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    log(f"Coordinator started on port {coordinator_port}", component="Coordinator")
    return thread


# Legacy function for backward compatibility
def start_inference_server(handler: InferenceHandler, port: int):
    """Start inference server in background thread (legacy, single-GPU mode)."""
    return start_inference_only_server(handler, port, rank=0)


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


def poll_all_provers(prover_addresses: list[str]) -> tuple[list, list, int, int, int]:
    """
    Poll all prover servers for transitions.
    Also updates the monitor with prover status.
    Returns (transitions, tactics, games_played, games_solved, expansions).
    """
    from nanoproof.cli import get_monitor
    
    all_transitions = []
    all_tactics = []
    total_games_played = 0
    total_games_solved = 0
    total_expansions = 0
    monitor = get_monitor()
    
    for addr in prover_addresses:
        try:
            response = requests.get(f"http://{addr}/poll", timeout=30.0)
            response.raise_for_status()
            data = response.json()
            all_transitions.extend(data.get("transitions", []))
            all_tactics.extend(data.get("tactics", []))
            total_games_played += data.get("games_played", 0)
            total_games_solved += data.get("games_solved", 0)
            total_expansions += data.get("expansions", 0)
            
            # Update monitor with prover status
            if monitor:
                monitor.update_prover_server(
                    address=addr,
                    games_played=data.get("games_played", 0),
                    games_solved=data.get("games_solved", 0),
                    transitions=len(data.get("transitions", [])),
                    num_threads=data.get("num_threads", 0),
                    thread_states=data.get("thread_states"),
                )
        except Exception as e:
            log(f"Failed to poll prover at {addr}: {e}", component="Coordinator")
            # Mark prover as disconnected in monitor
            if monitor:
                monitor.update_prover_server(address=addr, num_threads=0, thread_states=[])
    
    return all_transitions, all_tactics, total_games_played, total_games_solved, total_expansions


def distributed_collect(
    target_transitions: int,
    poll_interval: float,
    replay_buffer: ReplayBuffer,
) -> int:
    """
    Collect transitions from distributed provers.
    
    Uses the global prover registry. If no provers are registered,
    waits until at least one is available.
    
    Returns the number of transitions collected.
    """
    
    registry = get_registry()
    monitor = get_monitor()
    log(f"Starting distributed collection, target={target_transitions}", component="Coordinator")
    
    # Mark collection as running (new provers will be started automatically)
    registry.set_collecting(True)
    
    try:
        # Wait for at least one prover
        while registry.count() == 0:
            log("Waiting for provers to register...", component="Coordinator")
            time.sleep(poll_interval)
        
        # Start all currently registered provers
        prover_addresses = registry.get_all()
        start_all_provers(prover_addresses)
        
        collected = 0
        total_games_played = 0
        total_games_solved = 0
        total_expansions = 0
        
        while collected < target_transitions:
            time.sleep(poll_interval)
            
            # Get current list of provers (may have changed)
            prover_addresses = registry.get_all()
            if not prover_addresses:
                log("No provers registered, waiting...", component="Coordinator")
                continue
            
            transitions, tactics, games_played, games_solved, expansions = poll_all_provers(prover_addresses)
            
            # Add transitions to replay buffer
            for t in transitions:
                replay_buffer.local_buffer.append(t)
            
            collected += len(transitions)
            total_games_played += games_played
            total_games_solved += games_solved
            total_expansions += expansions
            
            # Update monitor with collection stats
            if monitor and (games_played > 0 or len(transitions) > 0 or expansions > 0):
                # Record proof attempts (successful ones have transitions)
                for _ in range(games_solved):
                    monitor.record_proof_attempt(successful=True, transitions=0)
                for _ in range(games_played - games_solved):
                    monitor.record_proof_attempt(successful=False, transitions=0)
                # Record expansions
                for _ in range(expansions):
                    monitor.record_expansion()
                # Add live transitions for web display
                if transitions:
                    monitor.record_transitions(transitions)
                # Record tactics for web display
                if tactics:
                    monitor.record_tactics(tactics)
                # Update replay buffer size
                monitor.set_replay_buffer_size(len(replay_buffer.local_buffer))
            
            if len(transitions) > 0:
                log(f"Collected {len(transitions)} transitions (total: {collected}/{target_transitions})", 
                    component="Coordinator")
        
        # Pause all provers
        prover_addresses = registry.get_all()
        pause_all_provers(prover_addresses)
        
        log(f"Distributed collection complete: {collected} transitions", component="Coordinator")
        return collected
    finally:
        # Mark collection as stopped
        registry.set_collecting(False)
