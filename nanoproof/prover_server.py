"""
Prover Server - runs on CPU-only nodes.

This server connects to a Lean server for proof verification and uses
RemoteTacticModel for tactic generation via the coordinator.

On startup, it registers itself with the coordinator. On shutdown, it unregisters.

Uses ProverWorker from experience_collection.py with callbacks that:
  - Request theorems from coordinator (/get_theorem)
  - Submit results to coordinator (/submit_result) with serialized proof tree

Usage:
    python -m nanoproof.prover_server --rl-server <host:port> --lean-server <host:port>
"""

import argparse
import atexit
import logging
import signal
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from flask import Flask, jsonify

from nanoproof.search import Config, Game
from nanoproof.experience_collection import ProverWorker
from nanoproof.inference import RemoteTacticModel


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
    """Tracks consecutive connection failures and raises after threshold."""
    
    def __init__(self, max_failures: int = 3):
        self.consecutive_failures = 0
        self.max_failures = max_failures
    
    def record_success(self):
        self.consecutive_failures = 0
    
    def record_failure(self, error: Exception, url: str = ""):
        self.consecutive_failures += 1
        url_info = f" (url={url})" if url else ""
        print(f"[Prover] Failed to get theorem ({self.consecutive_failures}/{self.max_failures}): {error}{url_info}")
        if self.consecutive_failures >= self.max_failures:
            raise RuntimeError(f"RL server unreachable after {self.max_failures} consecutive failures")


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
    
    failure_tracker = ConnectionFailureTracker(max_failures=3)
    
    def get_theorem() -> Optional[tuple[str, str]]:
        """Request next theorem from coordinator."""
        url = f"{coordinator_url}/get_theorem"
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
            return None
    
    def on_result(theorem_id: str, theorem: str, game: Optional["Game"], error: str | None):
        """Submit proof result to coordinator."""
        # Only include proof_tree if actually solved
        is_solved = game and game.root and game.root.is_solved
        print(game.root.pp_tree())
        result = {
            "id": theorem_id,
            "theorem": theorem,
            "proof_tree": game.root.serialize() if is_solved else None,
            "error": error,
        }
        
        # Track stats locally
        if game and game.root:
            buffer.record_game(game.root.is_solved)
            if game.root.is_solved:
                print(f"[Prover] SOLVED {theorem_id}")
        else:
            buffer.record_game(False)
        
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

def register_with_rl_server(rl_server: str, my_address: str) -> bool:
    """Register this prover with the RL server."""
    try:
        response = requests.post(
            f"http://{rl_server}/register",
            json={"address": my_address},
            timeout=5.0
        )
        response.raise_for_status()
        print(f"[Registration] Registered with RL server at {rl_server}")
        return True
    except Exception as e:
        print(f"[Registration] Failed to register with RL server: {e}")
        return False


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
        """Poll for stats."""
        result = buffer.get_stats()
        result["thread_states"] = prover_worker.get_thread_states()
        result["expansions"] = prover_worker.get_expansions()
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
    parser.add_argument("--port", type=int, default=5001, help="Port to bind to")
    parser.add_argument("--rl-server", required=True, help="RL server address (host:port) - handles both registration and inference")
    parser.add_argument("--lean-server", required=True, help="Lean server address (host:port)")
    parser.add_argument("--num-actors", type=int, default=4, help="Number of parallel actors")
    parser.add_argument("--num-simulations", type=int, default=50, help="MCTS simulations per game")
    parser.add_argument("--num-sampled-tactics", type=int, default=6, help="Tactics to sample per state")
    parser.add_argument("--my-address", default=None, help="Address to register (auto-detected if not set)")
    args = parser.parse_args()
    
    # Parse Lean server address
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
    
    coordinator_url = f"http://{args.rl_server}"
    tactic_model = RemoteTacticModel(args.rl_server)
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
    
    # Setup cleanup on exit
    def cleanup():
        print("Shutting down...")
        prover_worker.stop()
        unregister_from_rl_server(args.rl_server, my_address)
    
    atexit.register(cleanup)
    
    # Handle signals for graceful shutdown
    def signal_handler(signum, frame):
        cleanup()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"Prover server starting on {args.host}:{args.port}")
    print(f"  RL server: {args.rl_server}")
    print(f"  Lean server: {args.lean_server}")
    print(f"  My address: {my_address}")
    print(f"  Actors: {args.num_actors}")
    
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
    
    # Register with RL server (after Flask is ready)
    print(f"Registering as {my_address} with RL server at {args.rl_server}")
    register_with_rl_server(args.rl_server, my_address)
    
    print("\nCtrl+C to exit\n")
    
    # Main loop: wait for actors to start, then monitor for exit
    try:
        while not prover_worker.has_started_actors():
            time.sleep(1.0)
        
        while not prover_worker.all_actors_exited():
            time.sleep(1.0)
        
        print("\n[ProverServer] All actors have exited. Shutting down...")
    except KeyboardInterrupt:
        print("\n[ProverServer] Interrupted by user")
    
    cleanup()


if __name__ == "__main__":
    main()
