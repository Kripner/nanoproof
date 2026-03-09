"""
Prover Server - runs on CPU-only nodes.

This server connects to a Lean server for proof verification and uses
RemoteTacticModel for tactic generation via the coordinator.

On startup, it registers itself with the coordinator. On shutdown, it unregisters.

Uses ProverWorker from experience_collection.py with callbacks that:
  - Request theorems from coordinator (/get_theorem)
  - Submit results to coordinator (/submit_result) with serialized proof tree

Usage:
    # With infra.toml (recommended):
    python -m nanoproof.prover_server --infra-file infra.toml

    # Or with explicit arguments:
    python -m nanoproof.prover_server --rl-server <host:port> --lean-server <host:port>
"""

import argparse
import atexit
import logging
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from flask import Flask, jsonify

from nanoproof.search import Config, Game
from nanoproof.common import linearize_proof, construct_proof_source
from nanoproof.experience_collection import ProverWorker
from nanoproof.inference import RemoteTacticModel
from nanoproof.cli import get_and_clear_tactics_buffer
from nanoproof.infra import load_infra_config


# -----------------------------------------------------------------------------
# Local Stats Buffer (for monitoring)
# -----------------------------------------------------------------------------

@dataclass
class LocalBuffer:
    """Tracks stats for monitoring."""
    games_played: int = 0
    games_solved: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def record_game(self, solved: bool):
        with self.lock:
            self.games_played += 1
            if solved:
                self.games_solved += 1
    
    def get_stats(self) -> dict:
        with self.lock:
            return {
                "games_played": self.games_played,
                "games_solved": self.games_solved,
            }


# -----------------------------------------------------------------------------
# Remote Prover Worker Factory
# -----------------------------------------------------------------------------

class ConnectionFailureTracker:
    """Tracks consecutive connection failures and waits before retrying."""
    
    def __init__(self, retry_delay: float = 5.0):
        self.consecutive_failures = 0
        self.retry_delay = retry_delay
    
    def record_success(self):
        self.consecutive_failures = 0
    
    def record_failure(self, error: Exception, url: str = ""):
        self.consecutive_failures += 1
        url_info = f" (url={url})" if url else ""
        print(f"[Prover] Connection failed ({self.consecutive_failures}): {error}{url_info}")
        print(f"[Prover] Retrying in {self.retry_delay}s...")
        time.sleep(self.retry_delay)


def create_remote_prover_worker(
    config: Config,
    tactic_model: RemoteTacticModel,
    coordinator_url: str,
    lean_address: str,
    lean_port: int,
    buffer: LocalBuffer,
    num_actors: int,
) -> ProverWorker:
    """
    Create a ProverWorker that gets theorems from coordinator and submits results.
    """
    
    failure_tracker = ConnectionFailureTracker(retry_delay=5.0)
    
    def get_theorem() -> Optional[tuple[str, str]]:
        """Request next theorem from coordinator. Retries on connection failure."""
        url = f"{coordinator_url}/get_theorem"
        while True:
            try:
                response = requests.get(url, timeout=5.0)
                response.raise_for_status()
                data = response.json()
                failure_tracker.record_success()
                if data.get("done"):
                    return None
                return data["id"], data["theorem"]
            except Exception as e:
                failure_tracker.record_failure(e, url)
    
    def on_result(theorem_id: str, theorem: str, game: Optional["Game"], error: str | None):
        """Submit proof result to coordinator."""
        # Only include proof_tree if actually solved
        is_solved = game and game.root and game.root.is_solved
        num_iterations = game.num_iterations if game else 0
        
        # Compute linearized proof if solved
        linearized = None
        if is_solved:
            tactics = linearize_proof(game.root)
            linearized = construct_proof_source(theorem, tactics)
        
        result = {
            "id": theorem_id,
            "theorem": theorem,
            "proof_tree": game.root.serialize() if is_solved else None,
            "unsimplified_proof_tree": game.unsimplified_root.serialize() if is_solved and game.unsimplified_root else None,
            "linearized_proof": linearized,
            "error": error,
            "num_iterations": num_iterations,
        }
        
        buffer.record_game(is_solved)
        if is_solved:
            print(f"[Prover] SOLVED {theorem_id} after {num_iterations} iterations")
        else:
            error_info = f" (error: {error})" if error else ""
            print(f"[Prover] FAILED {theorem_id} after {num_iterations} iterations{error_info}")
        
        # Submit to coordinator
        url = f"{coordinator_url}/submit_result"
        try:
            response = requests.post(url, json=result, timeout=30.0)
            response.raise_for_status()
        except Exception as e:
            print(f"[Prover] Failed to submit result to {url}: {e}")
    
    return ProverWorker(
        config=config,
        tactic_model=tactic_model,
        lean_address=lean_address,
        lean_port=lean_port,
        get_theorem=get_theorem,
        on_result=on_result,
        num_actors=num_actors,
        paused=True,  # Start paused, coordinator will tell us to start
    )


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

def register_with_rl_server(rl_server: str, my_address: str, retry_delay: float = 5.0) -> bool:
    """Register this prover with the RL server. Returns True on success, False on failure."""
    try:
        response = requests.post(
            f"http://{rl_server}/register",
            json={"address": my_address},
            timeout=5.0
        )
        response.raise_for_status()
        return True
    except Exception as e:
        return False


def register_with_rl_server_blocking(rl_server: str, my_address: str, retry_delay: float = 5.0):
    """Register this prover with the RL server. Retries until successful."""
    while True:
        if register_with_rl_server(rl_server, my_address):
            print(f"[Registration] Registered with RL server at {rl_server}")
            return
        print(f"[Registration] Failed to register with RL server, retrying in {retry_delay}s...")
        time.sleep(retry_delay)


def unregister_from_rl_server(rl_server: str, my_address: str):
    """Unregister this prover from the RL server."""
    try:
        response = requests.post(
            f"http://{rl_server}/unregister",
            json={"address": my_address},
            timeout=5.0
        )
        response.raise_for_status()
        print(f"[Registration] Unregistered from RL server")
    except Exception as e:
        print(f"[Registration] Failed to unregister from RL server: {e}")


def get_local_ip() -> str:
    """Get the local IP address that can be reached from outside."""
    try:
        # Connect to an external address to find our outgoing IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def start_registration_loop(
    rl_server: str,
    my_address: str,
    shutdown_event: threading.Event,
    poll_interval: float = 5.0
) -> threading.Thread:
    """
    Start a background thread that periodically re-registers with the RL server.
    
    This ensures the prover stays registered even if the RL server restarts.
    The thread will poll every poll_interval seconds until shutdown_event is set.
    """
    def registration_loop():
        last_success = False
        while not shutdown_event.is_set():
            success = register_with_rl_server(rl_server, my_address)
            if success and not last_success:
                print(f"[Registration] Registered with RL server at {rl_server}")
            elif not success and last_success:
                print(f"[Registration] Lost connection to RL server, will keep trying...")
            last_success = success
            
            # Wait for poll_interval, checking shutdown_event periodically
            for _ in range(int(poll_interval * 10)):
                if shutdown_event.is_set():
                    break
                time.sleep(0.1)
    
    thread = threading.Thread(target=registration_loop, daemon=True)
    thread.start()
    return thread


# -----------------------------------------------------------------------------
# Flask Server
# -----------------------------------------------------------------------------

def create_app(prover_worker: ProverWorker, buffer: LocalBuffer):
    app = Flask(__name__)
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})
    
    @app.route("/start", methods=["POST"])
    def start():
        """Start or resume experience collection."""
        prover_worker.start()
        prover_worker.resume()
        return jsonify({"status": "started"})
    
    @app.route("/pause", methods=["POST"])
    def pause():
        """Pause experience collection."""
        prover_worker.pause()
        return jsonify({"status": "paused"})
    
    @app.route("/poll", methods=["GET"])
    def poll():
        """Poll for stats and tactics."""
        result = buffer.get_stats()
        result["thread_states"] = prover_worker.get_thread_states()
        result["expansions"] = prover_worker.get_expansions()
        # Include tactics collected since last poll
        result["tactics"] = get_and_clear_tactics_buffer()
        return jsonify(result)
    
    @app.route("/stats", methods=["GET"])
    def stats():
        """Get current stats."""
        return jsonify(buffer.get_stats())
    
    return app


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prover Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=6001, help="Port to bind to")
    parser.add_argument("--infra-file", default=None, help="Path to infra.toml file (recommended). If provided, --rl-server and --lean-server are read from it.")
    parser.add_argument("--rl-server", default=None, help="RL server address (host:port) - handles both registration and inference (ignored if --infra-file is set)")
    parser.add_argument("--lean-server", default=None, help="Lean server address (host:port) (ignored if --infra-file is set)")
    parser.add_argument("--num-actors", type=int, default=4, help="Number of parallel actors")
    parser.add_argument("--num-simulations", type=int, default=50, help="MCTS simulations per game")
    parser.add_argument("--num-sampled-tactics", type=int, default=6, help="Tactics to sample per state")
    parser.add_argument("--my-address", default=None, help="Address to register (auto-detected if not set)")
    args = parser.parse_args()
    
    # Determine RL server and Lean server addresses
    if args.infra_file:
        # Load configuration from infra.toml
        infra_config = load_infra_config(args.infra_file)
        rl_server = infra_config.rl_server
        
        # Get this prover's address (IP:port) to look up which lean server to use
        my_ip = get_local_ip()
        my_prover_address = f"{my_ip}:{args.port}"
        lean_server_config = infra_config.get_lean_server_for_prover(my_prover_address)
        if lean_server_config is None:
            print(f"ERROR: No lean server mapping found for prover {my_prover_address} in {args.infra_file}")
            print(f"Available mappings: {infra_config.prover_to_lean}")
            sys.exit(1)
        lean_host = lean_server_config.address
        lean_port = lean_server_config.port
        print(f"Loaded config from {args.infra_file}:")
        print(f"  RL server: {rl_server}")
        print(f"  Lean server for {my_prover_address}: {lean_host}:{lean_port}")
    else:
        # Use explicit arguments
        if not args.rl_server:
            parser.error("--rl-server is required when --infra-file is not provided")
        if not args.lean_server:
            parser.error("--lean-server is required when --infra-file is not provided")
        
        rl_server = args.rl_server
        lean_host, lean_port = args.lean_server.split(":")
        lean_port = int(lean_port)
    
    # Determine our address for registration
    if args.my_address:
        my_address = args.my_address
    else:
        my_ip = get_local_ip()
        my_address = f"{my_ip}:{args.port}"
    
    # Create components
    config = Config(
        num_simulations=args.num_simulations,
        num_actors=args.num_actors,
        num_sampled_tactics=args.num_sampled_tactics,
        server_address=lean_host,
        server_port=lean_port,
    )
    
    coordinator_url = f"http://{rl_server}"
    tactic_model = RemoteTacticModel(rl_server)
    buffer = LocalBuffer()
    prover_worker = create_remote_prover_worker(
        config=config,
        tactic_model=tactic_model,
        coordinator_url=coordinator_url,
        lean_address=lean_host,
        lean_port=lean_port,
        buffer=buffer,
        num_actors=args.num_actors,
    )
    
    app = create_app(prover_worker, buffer)
    
    # Shutdown coordination
    shutdown_event = threading.Event()
    cleanup_done = threading.Event()
    
    def cleanup():
        if cleanup_done.is_set():
            return  # Already cleaned up
        cleanup_done.set()
        print("Shutting down...")
        prover_worker.stop()
        unregister_from_rl_server(rl_server, my_address)
    
    atexit.register(cleanup)
    
    # Handle signals for graceful shutdown
    def signal_handler(signum, frame):
        if shutdown_event.is_set():
            # Second Ctrl+C - exit via sys.exit so atexit handlers run
            print("\nExiting...")
            sys.exit(0)
        print("\nShutdown requested (press Ctrl+C again to force)...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"Prover server starting on {args.host}:{args.port}")
    print(f"  RL server: {rl_server}")
    print(f"  Lean server: {lean_host}:{lean_port}")
    print(f"  My address: {my_address}")
    print(f"  Actors: {args.num_actors}")
    print(f"  MCTS simulations: {args.num_simulations}")
    
    # Disable Flask's default request logging
    werkzeug_log = logging.getLogger('werkzeug')
    werkzeug_log.setLevel(logging.ERROR)
    
    # Run Flask in a background thread
    flask_thread = threading.Thread(
        target=lambda: app.run(host=args.host, port=args.port, threaded=True, use_reloader=False),
        daemon=True
    )
    flask_thread.start()
    
    # Wait for Flask server to be ready before registering
    print("Waiting for Flask server to be ready...")
    for _ in range(50):  # Up to 5 seconds
        try:
            response = requests.get(f"http://127.0.0.1:{args.port}/health", timeout=1.0)
            if response.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.1)
    else:
        print("WARNING: Flask server may not be ready")
    
    # Start registration polling loop (polls every 5 seconds to re-register with RL server)
    # This ensures we stay registered even if the RL server restarts
    print(f"Starting registration loop as {my_address} with RL server at {rl_server}")
    registration_thread = start_registration_loop(rl_server, my_address, shutdown_event, poll_interval=5.0)
    
    print("\nCtrl+C to exit\n")
    
    # Main loop: keep running until shutdown is requested
    # The prover remains running even after actors finish their current batch,
    # so it can be restarted when the RL server starts a new collection/eval phase.
    try:
        while not shutdown_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[ProverServer] Interrupted by user")
    
    cleanup()


if __name__ == "__main__":
    main()
