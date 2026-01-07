"""
Prover Server - runs on CPU-only nodes.

This server connects to a Lean server for proof verification and to an LLM server
for tactic generation. It provides HTTP endpoints for the main LLM server to control
experience collection.

On startup, it registers itself with the RL server. On shutdown, it unregisters.

Usage:
    python -m nanoproof.prover_server --rl-server <host:port> --lean-server <host:port>
"""

import argparse
import asyncio
import atexit
import random
import signal
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Optional
import logging
import traceback

import requests
from flask import Flask, request, jsonify

from leantree.repl_adapter.server import LeanClient
from nanoproof.search import Node, Player, Game, run_mcts, Config, MCTSAbortedError
from nanoproof.data.leanworkbook import list_theorems


# -----------------------------------------------------------------------------
# RemoteTacticModel - calls the RL server for inference
# -----------------------------------------------------------------------------

class RLServerDisconnectedError(Exception):
    """Raised when the RL server appears to be permanently disconnected."""
    pass


class RemoteTacticModel:
    """
    Tactic model that calls a remote RL server for inference.
    
    This is used by prover servers running on CPU-only nodes.
    The RL server handles batching and GPU inference.
    """
    
    def __init__(self, rl_server_address: str, timeout: float = 60.0, max_pool_size: int = 50):
        self.rl_server_address = rl_server_address
        self.timeout = timeout
        # Configure session with larger connection pool for many concurrent actors
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_pool_size,
            pool_maxsize=max_pool_size,
            max_retries=0,  # We handle retries ourselves
        )
        self._session.mount('http://', adapter)
        self._session.mount('https://', adapter)
        self._consecutive_failures = 0
        self._max_failures = 10  # After this many failures, raise exception
        self._last_error_log_time = 0
        self._error_log_interval = 5.0  # Only log errors every N seconds
        self._lock = threading.Lock()
    
    def sample_tactic(self, state) -> list[str]:
        """Sample tactics for a single state by calling the remote RL server."""
        assert len(state) == 1, \
            f"expected single branch in state, got {len(state)}"
        
        state_str = str(state[0].state).strip()
        
        try:
            response = self._session.post(
                f"http://{self.rl_server_address}/generate",
                json={"states": [state_str]},
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            # Reset failure counter on success
            with self._lock:
                self._consecutive_failures = 0
            
            return result["tactics"][0]
        except requests.exceptions.Timeout:
            self._record_failure("Timeout calling RL server")
            return []
        except requests.exceptions.RequestException as e:
            self._record_failure(f"Error calling RL server: {e}")
            return []
    
    def _record_failure(self, message: str):
        """Record a failure and potentially raise if too many consecutive failures."""
        with self._lock:
            self._consecutive_failures += 1
            failures = self._consecutive_failures
            
            # Rate-limit error logging
            now = time.time()
            if now - self._last_error_log_time >= self._error_log_interval:
                print(f"[RemoteTacticModel] {message} (failures: {failures}/{self._max_failures})")
                self._last_error_log_time = now
            
            # If too many failures, signal that RL server is down
            if failures >= self._max_failures:
                raise RLServerDisconnectedError(
                    f"RL server appears disconnected after {failures} consecutive failures"
                )
    
    def shutdown(self):
        """No-op, exists for API compatibility."""
        pass


# -----------------------------------------------------------------------------
# Theorems Sampler
# -----------------------------------------------------------------------------

class TheoremsSampler:
    def __init__(self, seed: int | None = 0):
        self.theorems = list_theorems(split="train")
        self.rng = random.Random(seed)

    def sample_theorem(self) -> str:
        return self.rng.choice(self.theorems)


# -----------------------------------------------------------------------------
# Local Replay Buffer (stores transitions until polled)
# -----------------------------------------------------------------------------

@dataclass
class LocalBuffer:
    transitions: list = field(default_factory=list)
    tactics: list = field(default_factory=list)  # (state, tactic, success) tuples
    games_played: int = 0
    games_solved: int = 0
    expansions: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def add_transitions(self, new_transitions: list, solved: bool):
        with self.lock:
            self.transitions.extend(new_transitions)
            self.games_played += 1
            if solved:
                self.games_solved += 1
    
    def add_tactic(self, state: str, tactic: str, success: bool):
        """Record a tactic application for display in the web UI."""
        with self.lock:
            self.tactics.append({"state": state, "tactic": tactic, "success": success})
            # Keep only last 1000 tactics to avoid memory bloat
            if len(self.tactics) > 1000:
                self.tactics = self.tactics[-1000:]
    
    def add_expansion(self):
        """Record an MCTS expansion."""
        with self.lock:
            self.expansions += 1
    
    def pop_transitions(self) -> dict:
        """Return all transitions and tactics, then clear them."""
        with self.lock:
            result = {
                "transitions": self.transitions.copy(),
                "tactics": self.tactics.copy(),
            }
            self.transitions = []
            self.tactics = []
            return result
    
    def get_stats(self) -> dict:
        """Return current stats without modifying them."""
        with self.lock:
            return {
                "transitions_pending": len(self.transitions),
                "games_played": self.games_played,
                "games_solved": self.games_solved,
                "expansions": self.expansions,
            }


# -----------------------------------------------------------------------------
# Keyboard Monitor Thread
# -----------------------------------------------------------------------------

class KeyboardThread(threading.Thread):
    """Background thread that monitors keyboard input and prints stats on Enter."""
    
    def __init__(self, prover_worker: "ProverWorker", buffer: "LocalBuffer"):
        super().__init__(daemon=True)
        self.prover_worker = prover_worker
        self.buffer = buffer
        self._stop_flag = threading.Event()
    
    def stop(self):
        """Signal the thread to stop."""
        self._stop_flag.set()
    
    def run(self):
        import sys
        import select
        
        while not self._stop_flag.is_set():
            # Check for keyboard input (non-blocking on Unix)
            try:
                if sys.stdin in select.select([sys.stdin], [], [], 0.5)[0]:
                    line = sys.stdin.readline()
                    self._print_stats()
            except (ValueError, OSError):
                # stdin might be closed
                break
    
    def _print_stats(self):
        """Print current stats."""
        stats = self.prover_worker.get_stats_summary()
        buffer_stats = self.buffer.get_stats()
        print("\n" + "=" * 60)
        print("PROVER SERVER STATS")
        print("=" * 60)
        print(f"Actors: {stats['active_threads']}/{stats['total_actors']} active")
        print(f"  Exited: {stats['exited_threads']}")
        print(f"  States: {stats['state_counts']}")
        print(f"  Paused: {stats['paused']}")
        print(f"Buffer:")
        print(f"  Transitions pending: {buffer_stats['transitions_pending']}")
        print(f"  Games played: {buffer_stats['games_played']}")
        print(f"  Games solved: {buffer_stats['games_solved']}")
        print(f"  Expansions: {buffer_stats['expansions']}")
        print("=" * 60 + "\n")


# -----------------------------------------------------------------------------
# Prover Worker
# -----------------------------------------------------------------------------

class ProverWorker:
    """Runs the prover in a background thread."""
    
    def __init__(
        self,
        config: Config,
        tactic_model,  # RemoteTacticModel or any model with sample_tactic()
        lean_server_address: str,
        lean_server_port: int,
        buffer: LocalBuffer,
        num_actors: int,
        theorems_seed: int = 0,
    ):
        self.config = config
        self.tactic_model = tactic_model
        self.lean_address = lean_server_address
        self.lean_port = lean_server_port
        self.buffer = buffer
        self.num_actors = num_actors
        
        self._running = False
        self._paused = True
        self._stop_flag = threading.Event()
        self._threads: list[threading.Thread] = []
        self._theorems_sampler = TheoremsSampler(seed=theorems_seed)
        
        # Thread states for monitoring: 'idle', 'running', 'blocked', 'error'
        self._thread_states: dict[int, str] = {}
        self._thread_states_lock = threading.Lock()
    
    def start(self):
        """Start the prover workers."""
        if self._running:
            return
        
        self._running = True
        self._paused = False
        self._stop_flag.clear()
        
        print(f"[ProverWorker] Starting {self.num_actors} actor threads")
        for i in range(self.num_actors):
            t = threading.Thread(target=self._actor_loop, args=(i,), daemon=True)
            t.start()
            self._threads.append(t)
    
    def pause(self):
        """Pause collection (actors will finish current game then wait)."""
        self._paused = True
        print("[ProverWorker] Paused")
    
    def resume(self):
        """Resume collection."""
        self._paused = False
        print("[ProverWorker] Resumed")
    
    def stop(self):
        """Stop all workers."""
        self._stop_flag.set()
        self._running = False
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads = []
        print("[ProverWorker] Stopped")
    
    def set_thread_state(self, thread_id: int, state: str):
        """Set the state of a thread."""
        with self._thread_states_lock:
            self._thread_states[thread_id] = state
    
    def get_thread_states(self) -> list[str]:
        """Get the states of all threads as a list."""
        with self._thread_states_lock:
            return [self._thread_states.get(i, "idle") for i in range(self.num_actors)]
    
    def get_active_thread_count(self) -> int:
        """Get the number of threads that are still alive."""
        return sum(1 for t in self._threads if t.is_alive())
    
    def has_started_actors(self) -> bool:
        """Check if actors have been started (threads list is non-empty)."""
        return len(self._threads) > 0 and self._running
    
    def all_actors_exited(self) -> bool:
        """Check if all actor threads have exited (only valid after actors started)."""
        if not self._running or len(self._threads) == 0:
            return False  # Not started yet, so not "exited"
        return all(not t.is_alive() for t in self._threads)
    
    def get_stats_summary(self) -> dict:
        """Get a summary of actor stats for CLI display."""
        states = self.get_thread_states()
        active = self.get_active_thread_count()
        state_counts = {}
        for s in states:
            state_counts[s] = state_counts.get(s, 0) + 1
        return {
            "total_actors": self.num_actors,
            "active_threads": active,
            "exited_threads": self.num_actors - active,
            "state_counts": state_counts,
            "paused": self._paused,
            "running": self._running,
        }
    
    def _actor_loop(self, actor_id: int):
        """Main loop for a single actor thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Create local theorem sampler with unique seed
        local_sampler = TheoremsSampler(
            seed=self._theorems_sampler.rng.randint(0, 2**31) + actor_id
        )
        
        # Connect to Lean server
        self.set_thread_state(actor_id, "blocked")
        print(f"[Actor {actor_id}] Connecting to Lean server at {self.lean_address}:{self.lean_port}")
        client = LeanClient(self.lean_address, self.lean_port)
        print(f"[Actor {actor_id}] Connected")
        self.set_thread_state(actor_id, "idle")
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while not self._stop_flag.is_set():
            # Check if paused
            if self._paused:
                self.set_thread_state(actor_id, "idle")
                time.sleep(0.5)
                continue
            
            self.set_thread_state(actor_id, "running")
            try:
                game = self._play_game(client, local_sampler)
                consecutive_errors = 0
            except RLServerDisconnectedError as e:
                # RL server is down - exit gracefully
                self.set_thread_state(actor_id, "error")
                print(f"[Actor {actor_id}] {e}")
                break
            except MCTSAbortedError:
                # MCTS was aborted due to pause - this is expected, just continue
                # The Lean process is now freed for evaluation
                print(f"[Actor {actor_id}] MCTS aborted (paused for evaluation)")
                continue
            except ConnectionResetError:
                consecutive_errors += 1
                self.set_thread_state(actor_id, "error")
                print(f"[Actor {actor_id}] Connection reset ({consecutive_errors}/{max_consecutive_errors})")
                if consecutive_errors >= max_consecutive_errors:
                    break
                try:
                    self.set_thread_state(actor_id, "blocked")
                    client = LeanClient(self.lean_address, self.lean_port)
                except Exception as e:
                    print(f"[Actor {actor_id}] Reconnect failed: {e}")
                continue
            except Exception as e:
                consecutive_errors += 1
                self.set_thread_state(actor_id, "error")
                print(f"[Actor {actor_id}] Error: {e} ({consecutive_errors}/{max_consecutive_errors})")
                print(f"[Actor {actor_id}] Traceback:\n{traceback.format_exc()}")
                if consecutive_errors >= max_consecutive_errors:
                    break
                continue
            if game is None:
                continue
            
            # Extract and store transitions
            is_solved = game.root.is_solved
            transitions = []
            if is_solved:
                transitions = self._extract_transitions(game.root)
                print(f"[Actor {actor_id}] SOLVED! {len(transitions)} transitions")
            
            self.buffer.add_transitions(transitions, is_solved)
        
        self.set_thread_state(actor_id, "idle")
        print(f"[Actor {actor_id}] Exiting")
    
    def _play_game(self, client: LeanClient, sampler: TheoremsSampler) -> Optional[Game]:
        """Play a single game."""
        theorem = sampler.sample_theorem()
        
        process = client.get_process()
        if process is None:
            raise RuntimeError("Failed to acquire Lean process (server overloaded?)")
        
        with process as env:
            env.send_command("""
                open scoped Real
                open scoped Nat
                open scoped Topology
                open scoped Polynomial
            """)
            init_branch = env.proof_from_sorry(theorem)
            if not init_branch.is_success():
                return None
            init_branch = init_branch.value
            
            game = Game(theorem, self.config.num_simulations)
            game.root = Node(
                action=None,
                prior=None,
                state=[init_branch],
                to_play=Player.OR,
                reward=None,
            )
            
            # Use callbacks to track expansions and tactics
            # Pass pause check as abort_check so MCTS exits early when paused,
            # freeing the Lean process for evaluation
            run_mcts(
                self.config, game, self.tactic_model,
                expansion_callback=self.buffer.add_expansion,
                tactic_callback=self.buffer.add_tactic,
                abort_check=lambda: self._paused,
            )
            return game
    
    def _extract_transitions(self, node: Node) -> list[tuple[str, str, float]]:
        """Extract transitions from a solved proof tree."""
        if not node.is_solved:
            return []
        
        transitions = []
        while node.to_play == Player.OR and not node.is_terminal:
            assert len(node.state) == 1
            # Select shortest valid tactic
            actions = [a for a in node.children if node.children[a].is_solved]
            if not actions:
                break
            action = min(actions, key=lambda a: len(a))
            transitions.append((
                str(node.state[0].state).strip(),
                action.strip(),
                node.value_target
            ))
            node = node.children[action]
        
        if node.to_play == Player.AND:
            for _, child in node.children.items():
                transitions.extend(self._extract_transitions(child))
        
        return transitions


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
        """
        Poll for collected transitions and stats.
        Returns and clears the local buffer.
        Also includes thread states for monitoring.
        """
        result = buffer.pop_transitions()
        result.update(buffer.get_stats())
        result["num_threads"] = prover_worker.num_actors
        result["thread_states"] = prover_worker.get_thread_states()
        return jsonify(result)
    
    @app.route("/stats", methods=["GET"])
    def stats():
        """Get current stats without clearing buffer."""
        return jsonify(buffer.get_stats())
    
    return app


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prover Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5001, help="Port to bind to")
    parser.add_argument("--rl-server", required=True, help="RL server address (host:port)")
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
    
    tactic_model = RemoteTacticModel(args.rl_server)
    buffer = LocalBuffer()
    prover_worker = ProverWorker(
        config=config,
        tactic_model=tactic_model,
        lean_server_address=lean_host,
        lean_server_port=lean_port,
        buffer=buffer,
        num_actors=args.num_actors,
    )
    
    app = create_app(prover_worker, buffer)
    
    # Register with RL server
    print(f"Registering as {my_address} with RL server at {args.rl_server}")
    register_with_rl_server(args.rl_server, my_address)
    
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
    print(f"\nPress Enter to show stats, Ctrl+C to exit\n")
    
    # Disable Flask's default request logging (very noisy with frequent /poll requests)
    werkzeug_log = logging.getLogger('werkzeug')
    werkzeug_log.setLevel(logging.ERROR)
    
    # Start keyboard monitoring thread
    keyboard_thread = KeyboardThread(prover_worker, buffer)
    keyboard_thread.start()
    
    # Run Flask in a background thread
    flask_thread = threading.Thread(
        target=lambda: app.run(host=args.host, port=args.port, threaded=True, use_reloader=False),
        daemon=True
    )
    flask_thread.start()
    
    # Main loop: wait for actors to start, then monitor for all actors exiting
    try:
        # Wait for actors to start
        while not prover_worker.has_started_actors():
            time.sleep(1.0)
        
        # Monitor for all actors exiting
        while not prover_worker.all_actors_exited():
            time.sleep(1.0)
        
        print("\n[ProverServer] All actors have exited. Shutting down...")
    except KeyboardInterrupt:
        print("\n[ProverServer] Interrupted by user")
    
    keyboard_thread.stop()
    cleanup()


if __name__ == "__main__":
    main()
