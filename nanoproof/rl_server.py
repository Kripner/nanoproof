"""
RL Server - coordinates distributed RL training.

This module provides:
- Inference server that runs alongside the training loop
- Dynamic prover registration/unregistration
- Prover coordination (start/pause/poll)
- Distributed collection of transitions

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
            should_start = self._collecting
            log(f"Prover registered: {address} (total: {len(self._provers)})", component="Registry")
        
        # Start the prover outside the lock to avoid blocking
        if should_start:
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
# Inference Handler
# -----------------------------------------------------------------------------

class InferenceHandler:
    """
    Handles inference requests from remote prover servers.
    Thread-safe wrapper around TacticModel.
    """
    
    def __init__(self, tactic_model: TacticModel):
        self.tactic_model = tactic_model
        self._lock = threading.Lock()
    
    @torch.inference_mode()
    def generate_tactics_batch(self, state_strs: list[str]) -> list[list[str]]:
        """Generate tactics for a batch of states."""
        with self._lock:
            return self.tactic_model.sample_tactic_from_str_batch(state_strs)


# -----------------------------------------------------------------------------
# Flask App for Inference + Registration
# -----------------------------------------------------------------------------

def create_inference_app(handler: InferenceHandler, registry: ProverRegistry):
    """Create Flask app for inference server."""
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
        data = flask_request.get_json()
        states = data.get("states", [])
        if len(states) == 0:
            return jsonify({"tactics": []})
        tactics = handler.generate_tactics_batch(states)
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


def start_inference_server(handler: InferenceHandler, port: int):
    """Start inference server in background thread."""
    registry = get_registry()
    app = create_inference_app(handler, registry)
    
    def run_server():
        # Disable Flask's default logging
        import logging
        log_handler = logging.getLogger('werkzeug')
        log_handler.setLevel(logging.ERROR)
        app.run(host="0.0.0.0", port=port, threaded=True)
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    log(f"Inference server started on port {port}", component="InferenceServer")
    return thread


# -----------------------------------------------------------------------------
# Prover Coordination
# -----------------------------------------------------------------------------

def start_prover(addr: str):
    """Start a single prover server."""
    try:
        response = requests.post(f"http://{addr}/start", timeout=5.0)
        response.raise_for_status()
        log(f"Started prover at {addr}", component="Coordinator")
    except Exception as e:
        log(f"Failed to start prover at {addr}: {e}", component="Coordinator")


def start_all_provers(prover_addresses: list[str]):
    """Instruct all prover servers to start collection."""
    for addr in prover_addresses:
        start_prover(addr)


def pause_all_provers(prover_addresses: list[str]):
    """Instruct all prover servers to pause collection."""
    for addr in prover_addresses:
        try:
            response = requests.post(f"http://{addr}/pause", timeout=5.0)
            response.raise_for_status()
            log(f"Paused prover at {addr}", component="Coordinator")
        except Exception as e:
            log(f"Failed to pause prover at {addr}: {e}", component="Coordinator")


def poll_all_provers(prover_addresses: list[str]) -> tuple[list, int, int]:
    """
    Poll all prover servers for transitions.
    Also updates the monitor with prover status.
    Returns (transitions, games_played, games_solved).
    """
    from nanoproof.cli import get_monitor
    
    all_transitions = []
    total_games_played = 0
    total_games_solved = 0
    monitor = get_monitor()
    
    for addr in prover_addresses:
        try:
            response = requests.get(f"http://{addr}/poll", timeout=5.0)
            response.raise_for_status()
            data = response.json()
            all_transitions.extend(data["transitions"])
            total_games_played += data["games_played"]
            total_games_solved += data["games_solved"]
            
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
    
    return all_transitions, total_games_played, total_games_solved


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
        while collected < target_transitions:
            time.sleep(poll_interval)
            
            # Get current list of provers (may have changed)
            prover_addresses = registry.get_all()
            if not prover_addresses:
                log("No provers registered, waiting...", component="Coordinator")
                continue
            
            transitions, games_played, games_solved = poll_all_provers(prover_addresses)
            
            # Add transitions to replay buffer
            for t in transitions:
                replay_buffer.local_buffer.append(t)
            
            collected += len(transitions)
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
