"""
Prover abstraction: collects RL experience and evaluates theorems.

This module hides the difference between local and distributed prover backends
behind a single ``Prover`` interface so the training loop in ``rl.py`` does not
need to branch on mode for either collection or evaluation.

Contents
--------
- ``TheoremsSampler``: weighted sampler over the configured training datasets.
- ``LocalBuffer``, ``ConnectionFailureTracker`` and the registration helpers
  (``register_with_rl_server``, ``unregister_from_rl_server``,
  ``start_registration_loop``, ``get_local_ip``): used by the CPU-node prover
  daemon (``nanoproof.prover_server``) to register with the coordinator.
- ``ProverWorker``: thread pool that plays proof games via ``run_mcts``. The
  same worker class is used in local mode (with a ``BlockingTacticModel``) and
  in distributed mode (with a ``RemoteTacticModel``); the difference is in the
  ``get_theorem`` / ``on_result`` callbacks.
- ``create_remote_prover_worker``: factory used by the prover daemon to wire a
  ``ProverWorker`` against the coordinator's HTTP endpoints.
- ``distributed_collect`` / ``distributed_eval``: coordinator-side collection
  and evaluation loops. They drive the ``TheoremDispatcher`` from
  ``nanoproof.rl_server`` and extract transitions from results sent by remote
  provers.
- ``Prover`` ABC + ``LocalProver`` + ``DistributedProver``: the abstraction the
  training loop talks to.
"""

import abc
import asyncio
import random
import socket
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests
import torch
import torch.distributed as dist
from leantree.repl_adapter.server import LeanClient
from leantree.repl_adapter.interaction import LeanProcessException

from nanoproof.common import (
    Player,
    active_barrier_master,
    active_barrier_wait,
    construct_proof_source,
    get_dist_info,
    linearize_proof,
    theorem_to_example,
)
from nanoproof.cli import get_monitor, log
from nanoproof.data.rl import deepseek_prover, leanworkbook, numinamath
from nanoproof.inference import BlockingTacticModel, RemoteTacticModel
from nanoproof.replay_buffer import (
    ReplayBuffer,
    compute_value_target,
    extract_transitions,
    prune_redundant_nodes,
)
from nanoproof.search import Game, Node, SearchConfig, run_mcts, verify_node


# -----------------------------------------------------------------------------
# Theorem sampler (used by both local and distributed collection)
# -----------------------------------------------------------------------------

class TheoremsSampler:
    """Samples theorems for local experience collection from multiple datasets.

    Samples uniformly at random from one of the available datasets, then
    uniformly at random from that dataset.

    Thread-safe: the sampler may be called from multiple Flask request threads
    in distributed mode when provers concurrently request theorems.
    """

    def __init__(self, seed: int | None = 0):
        # Load theorems from all datasets
        self.datasets = {
            "leanworkbook": leanworkbook.list_theorems(split="train"),
            "deepseek_prover": deepseek_prover.list_theorems(split="train"),
            "numinamath": numinamath.list_theorems(split="train"),
        }
        self.dataset_names = list(self.datasets.keys())
        self.rng = random.Random(seed)
        self._lock = threading.Lock()

        # Log dataset sizes
        for name, theorems in self.datasets.items():
            log(f"Loaded {len(theorems)} theorems from {name}", component="Sampler")

    def sample_theorem(self) -> str:
        with self._lock:
            # First, pick a dataset uniformly at random
            dataset_name = self.rng.choice(self.dataset_names)
            # Then, sample a theorem uniformly from that dataset
            return self.rng.choice(self.datasets[dataset_name])


# -----------------------------------------------------------------------------
# CPU-node prover daemon helpers (LocalBuffer, registration, failure tracking)
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
# ProverWorker: shared by local and distributed modes
# -----------------------------------------------------------------------------

class ProverWorker:
    """
    Runs prover threads that play proof games.

    Configurable with callbacks for:
    - get_theorem: returns (id, theorem) or None
    - on_result: called with (id, theorem, game_or_none, error_or_none)

    Used by both local mode (with TheoremsSampler) and distributed mode (with coordinator).
    """

    def __init__(
        self,
        config: SearchConfig,
        tactic_model,  # TacticModel | BlockingTacticModel | RemoteTacticModel
        lean_address: str,
        lean_port: int,
        get_theorem: Callable[[], Optional[tuple[str, str]]],
        on_result: Callable[[str, str, Optional[Game], Optional[str]], None],
        num_actors: Optional[int] = None,
        paused: bool = False,
    ):
        self.config = config
        self.tactic_model = tactic_model
        self.lean_address = lean_address
        self.lean_port = lean_port
        self.get_theorem = get_theorem
        self.on_result = on_result
        self.num_actors = num_actors or config.num_actors

        self._running = False
        self._paused = paused
        self._stop_flag = threading.Event()
        self._threads: list[threading.Thread] = []
        self._thread_states: dict[int, str] = {}
        self._thread_states_lock = threading.Lock()

        self.games_played = 0
        self.games_solved = 0
        self.expansions = 0
        self._stats_lock = threading.Lock()

    def start(self):
        """Start the actor threads. Can be called again to restart after actors exit."""
        # If already running with live threads, do nothing
        if self._running and not self.all_actors_exited():
            return

        # Clean up dead threads from previous run
        self._threads = [t for t in self._threads if t.is_alive()]

        self._running = True
        self._paused = False
        self._stop_flag.clear()

        for i in range(self.num_actors):
            t = threading.Thread(target=self._actor_loop, args=(i,), daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        """Stop all actor threads."""
        self._stop_flag.set()
        self._running = False
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads = []

    def pause(self):
        """Pause collection (actors will finish current game then wait)."""
        self._paused = True

    def resume(self):
        """Resume collection."""
        self._paused = False

    def _set_thread_state(self, actor_id: int, state: str):
        with self._thread_states_lock:
            self._thread_states[actor_id] = state

    def get_thread_states(self) -> list[str]:
        with self._thread_states_lock:
            return [self._thread_states.get(i, "idle") for i in range(self.num_actors)]

    def get_expansions(self) -> int:
        with self._stats_lock:
            return self.expansions

    def has_started_actors(self) -> bool:
        return len(self._threads) > 0 and self._running

    def all_actors_exited(self) -> bool:
        return all(not t.is_alive() for t in self._threads)

    def _actor_loop(self, actor_id: int):
        """Main loop for a single actor thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Connect to Lean server
        self._set_thread_state(actor_id, "blocked")
        client = LeanClient(self.lean_address, self.lean_port)
        self._set_thread_state(actor_id, "idle")

        consecutive_errors = 0
        max_consecutive_errors = 5
        max_retries = 3

        while not self._stop_flag.is_set():
            # Check if paused
            if self._paused:
                self._set_thread_state(actor_id, "idle")
                time.sleep(0.5)
                continue

            # Get theorem
            theorem_data = self.get_theorem()
            if theorem_data is None:
                self._set_thread_state(actor_id, "idle")
                time.sleep(0.5)
                continue

            theorem_id, theorem = theorem_data
            self._set_thread_state(actor_id, "running")

            # Try to prove it (with retries for transient connection errors)
            game = None
            error = None
            skip_report = False
            for attempt in range(max_retries):
                try:
                    game = self._play_game(client, theorem)
                    consecutive_errors = 0
                    break  # Success (game may be None if Lean couldn't init proof)
                except (ConnectionResetError, ConnectionRefusedError, BrokenPipeError, LeanProcessException) as e:
                    if attempt < max_retries - 1:
                        # Reconnect and retry
                        self._set_thread_state(actor_id, "retry")
                        log(f"[Actor {actor_id}] Connection error (attempt {attempt + 1}/{max_retries}): '{e}', reconnecting...", component="Collection")
                        time.sleep(1.0 * (attempt + 1))  # Increasing delay
                    else:
                        error = str(e)
                        consecutive_errors += 1
                except Exception as e:
                    if "Model paused for training" in str(e):
                        self._set_thread_state(actor_id, "idle")
                        skip_report = True
                        break  # Don't report, theorem not consumed
                    error = str(e)
                    consecutive_errors += 1
                    log(f"[Actor {actor_id}] Error (lean={self.lean_address}:{self.lean_port}): {e}", component="Collection")
                    traceback.print_exc()
                    break  # Don't retry other exceptions

            # Always report result (unless model was paused mid-attempt)
            if not skip_report:
                if game is not None and game.root is not None:
                    with self._stats_lock:
                        self.games_played += 1
                        if game.root.is_solved:
                            self.games_solved += 1

                if error is not None:
                    self._set_thread_state(actor_id, "error")

                self.on_result(theorem_id, theorem, game, error)

            if consecutive_errors >= max_consecutive_errors:
                log(f"[Actor {actor_id}] Too many consecutive errors, exiting", component="Collection")
                break

        self._set_thread_state(actor_id, "idle")

    def _play_game(self, client, theorem: str):
        """Play a single proof game."""
        process = client.get_process()
        if process is None:
            log(f"FAILED: Could not get Lean process for theorem", component="Collection")
            return None

        with process as env:
            env.send_command("""
                open scoped Real
                open scoped Nat
                open scoped Topology
                open scoped Polynomial
            """)
            init_branch = env.proof_from_sorry(theorem_to_example(theorem))
            if not init_branch.is_success():
                log(f"FAILED: Could not initialize proof - {init_branch.error if hasattr(init_branch, 'error') else 'unknown error'}", component="Collection")
                return None
            init_branch = init_branch.value

            game = Game(theorem, self.config.num_simulations)
            game.root = Node(
                parent=None,
                action=None,
                prior=None,
                state=[init_branch],
                to_play=Player.OR,
                reward=None,
            )

            def on_expansion():
                with self._stats_lock:
                    self.expansions += 1

            run_mcts(
                self.config, game, self.tactic_model,
                expansion_callback=on_expansion,
            )
            if game.root.is_solved:
                try:
                    verify_node(game.root)
                except AssertionError as e:
                    log(f"FAILED: Verification failed after {game.num_iterations} iterations: '{e}'\nTheorem: '{theorem}'\nProof tree:\n{game.root.pp_tree()}", component="Collection")
                    game.root.is_solved = False
                    return game
                game.unsimplified_root = game.root.clone()
                prune_redundant_nodes(game.root)
                compute_value_target(game.root)

                verify_node(game.root)

                # Verify the linearized proof compiles correctly
                tactics = linearize_proof(game.root)
                proof_source = construct_proof_source(theorem, tactics)
                if not env.is_valid_source(proof_source):
                    log(f"FAILED: Linearized proof verification failed after {game.num_iterations} iterations:\n\"\"\"\n{proof_source}\n\"\"\"\n... proof tree:\n{game.root.pp_tree()}\n", component="Collection")
                    game.root.is_solved = False

            return game


# -----------------------------------------------------------------------------
# Remote prover worker factory (used by nanoproof.prover_server)
# -----------------------------------------------------------------------------

def create_remote_prover_worker(
    config: SearchConfig,
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
# Distributed coordinator-side collection / evaluation
# -----------------------------------------------------------------------------

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
    # Local import to avoid a circular dependency: rl_server imports from prover.
    from nanoproof.rl_server import (
        ProofResult,
        get_dispatcher,
        get_registry,
        pause_all_provers,
        poll_all_provers,
        start_all_provers,
    )

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
    from nanoproof.rl_server import (
        ProofResult,
        get_dispatcher,
        get_registry,
        pause_all_provers,
        poll_all_provers,
        start_all_provers,
    )

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


# -----------------------------------------------------------------------------
# Prover ABC and concrete implementations
# -----------------------------------------------------------------------------

class Prover(abc.ABC):
    """Collects RL experience and evaluates theorems.

    Hides the difference between local (in-process actors driving a
    BlockingTacticModel against a Lean server) and distributed (master
    coordinator dispatching to remote prover processes) backends.
    """

    @abc.abstractmethod
    def collect(
        self,
        sampler: TheoremsSampler,
        target_transitions: int,
        replay_buffer: ReplayBuffer,
    ) -> int:
        """Collect transitions into ``replay_buffer``. Returns count added."""

    @abc.abstractmethod
    def evaluate(self, theorems: list[str], dataset_name: str) -> dict:
        """Run evaluation on ``theorems``. Returns a metrics dict with the keys
        ``success_rate``, ``solved``, ``total``, ``errors``. Distributed runs
        additionally include ``timed_out``, ``invalid`` and ``detailed_results``.
        """

    @abc.abstractmethod
    def pause(self) -> None:
        """Pause inference so training can use the GPU exclusively."""

    @abc.abstractmethod
    def resume(self) -> None: ...

    @abc.abstractmethod
    def shutdown(self) -> None: ...


class LocalProver(Prover):
    """Runs MCTS actors in-process against a Lean server.

    All work happens on the calling rank; in DDP we still use ``all_reduce`` to
    aggregate the per-rank ``local_buffer`` size against ``target_transitions``.
    """

    def __init__(
        self,
        config: SearchConfig,
        tactic_model: BlockingTacticModel,
        lean_address: str,
        lean_port: int,
        num_simulations_eval: int = 50,
    ):
        self.config = config
        self.tactic_model = tactic_model
        self.lean_address = lean_address
        self.lean_port = lean_port
        self.num_simulations_eval = num_simulations_eval

    @torch.no_grad()
    def collect(
        self,
        sampler: TheoremsSampler,
        target_transitions: int,
        replay_buffer: ReplayBuffer,
    ) -> int:
        """Run parallel actors locally until the global (DDP-summed) replay
        buffer has at least ``target_transitions`` new transitions.

        Uses ``ProverWorker`` to manage actor threads. LLM calls are
        automatically batched via the ``BlockingTacticModel``.
        """
        ddp, rank, _, world_size = get_dist_info()
        master_process = rank == 0
        device = self.tactic_model.network.get_device()
        num_actors = self.config.num_actors

        # Counter for generating theorem IDs
        theorem_counter = [0]

        if master_process:
            log(f"Starting collection with {num_actors} actors/rank, target={target_transitions} transitions",
                component="Collection")

        def get_theorem():
            """Sample a theorem from the local sampler."""
            theorem = sampler.sample_theorem()
            theorem_counter[0] += 1
            return (f"local_{theorem_counter[0]}", theorem)

        def on_result(theorem_id: str, theorem: str, game, error: str | None):
            """Handle proof result by extracting transitions."""
            if error is not None or game is None or game.root is None:
                return
            if not game.root.is_solved:
                return
            # Extract transitions and add to buffer
            transitions = extract_transitions(game.root)
            replay_buffer.add_transitions(transitions)
            # Update monitor
            monitor = get_monitor()
            if monitor is not None:
                monitor.record_transitions(transitions)

        worker = ProverWorker(
            config=self.config,
            tactic_model=self.tactic_model,
            lean_address=self.lean_address,
            lean_port=self.lean_port,
            get_theorem=get_theorem,
            on_result=on_result,
            num_actors=num_actors,
        )
        worker.start()

        monitor = get_monitor()

        def check_global_collected() -> int:
            """Check how many transitions have been collected globally (across all DDP ranks)."""
            with replay_buffer._lock:
                local_collected = len(replay_buffer.local_buffer)
            if ddp:
                collected_tensor = torch.tensor([local_collected], dtype=torch.long, device=device)
                dist.all_reduce(collected_tensor, op=dist.ReduceOp.SUM)
                return collected_tensor.item()
            return local_collected

        # Wait until target is reached
        loop_count = 0
        try:
            while True:
                global_collected = check_global_collected()
                if global_collected >= target_transitions:
                    if master_process:
                        log(f"Target reached: {global_collected}/{target_transitions} transitions collected",
                            component="Collection")
                    break

                # Update monitor with actor status
                if monitor is not None:
                    states = worker.get_thread_states()
                    for i, state in enumerate(states):
                        monitor.update_local_actor(i, state=state)

                # Check if all actors have exited (error condition)
                if worker.has_started_actors() and worker.all_actors_exited():
                    log("WARNING: All actors have exited unexpectedly", component="Collection")
                    break

                # Periodic status update (master only)
                loop_count += 1
                if loop_count % 50 == 0 and master_process:  # Every 5 seconds
                    log(f"Progress: {global_collected}/{target_transitions} transitions",
                        component="Collection")

                time.sleep(0.1)
        finally:
            if master_process:
                log(f"Stopping actors...", component="Collection")
            worker.stop()

            if monitor is not None:
                monitor.clear_local_actors()

        if master_process:
            log(f"Collection complete: {len(replay_buffer.local_buffer)} local transitions",
                component="Collection")
        return len(replay_buffer.local_buffer)

    @torch.no_grad()
    def evaluate(self, theorems: list[str], dataset_name: str) -> dict:
        """Run MCTS over ``theorems`` and return success metrics.

        Theorems are sharded across DDP ranks; results are gathered on rank 0
        via ``all_gather_object``.
        """
        ddp, rank, _, world_size = get_dist_info()
        master_process = rank == 0
        monitor = get_monitor()

        # Shard theorems across ranks
        theorems_subset = theorems[rank::world_size]
        expected = len(theorems_subset)

        if monitor and master_process:
            monitor.start_eval(dataset_name, len(theorems))

        # Eval uses its own SearchConfig with num_simulations_eval
        eval_config = SearchConfig(
            num_simulations=self.num_simulations_eval,
            num_actors=self.config.num_actors,
        )

        # Per-theorem one-shot supply
        index = [0]
        index_lock = threading.Lock()

        def get_theorem() -> Optional[tuple[str, str]]:
            with index_lock:
                if index[0] >= expected:
                    return None
                tid = f"eval_{rank}_{index[0]}"
                theorem = theorems_subset[index[0]]
                index[0] += 1
                return (tid, theorem)

        local_results: list[dict] = []
        results_lock = threading.Lock()

        def on_result(theorem_id: str, theorem: str, game, error: str | None):
            is_solved = bool(game and game.root and game.root.is_solved)
            num_iterations = game.num_iterations if game else 0

            proof_tree = None
            unsimplified_proof_tree = None
            linearized = None
            if is_solved:
                proof_tree = game.root.serialize()
                if game.unsimplified_root is not None:
                    unsimplified_proof_tree = game.unsimplified_root.serialize()
                tactics = linearize_proof(game.root)
                linearized = construct_proof_source(theorem, tactics)

            with results_lock:
                local_results.append({
                    "theorem": theorem,
                    "is_solved": is_solved,
                    "error": error,
                    "proof_tree": proof_tree,
                    "unsimplified_proof_tree": unsimplified_proof_tree,
                    "linearized_proof": linearized,
                    "num_iterations": num_iterations,
                })

                # Live monitor progress (master rank only — non-master shards
                # can't drive the dataset-wide bar coherently).
                if monitor and master_process:
                    n = len(local_results)
                    solved = sum(1 for r in local_results if r["is_solved"])
                    errors = sum(1 for r in local_results if r["error"])
                    monitor.update_eval_progress(current=n, solved=solved, errors=errors)

        worker = ProverWorker(
            config=eval_config,
            tactic_model=self.tactic_model,
            lean_address=self.lean_address,
            lean_port=self.lean_port,
            get_theorem=get_theorem,
            on_result=on_result,
            num_actors=eval_config.num_actors,
        )
        worker.start()

        try:
            # Wait until every theorem in this rank's shard has produced a result.
            while True:
                with results_lock:
                    done = len(local_results) >= expected
                if done:
                    break
                if worker.has_started_actors() and worker.all_actors_exited():
                    log("WARNING: All actors exited before eval completed", component="Eval")
                    break
                time.sleep(0.1)
        finally:
            worker.stop()

        # Aggregate across ranks
        if ddp:
            gathered: list[list[dict]] = [None] * world_size  # type: ignore
            dist.all_gather_object(gathered, local_results)
            all_results = [r for shard in gathered for r in shard]
        else:
            all_results = local_results

        total = len(all_results)
        solved = sum(1 for r in all_results if r["is_solved"])
        errors = sum(1 for r in all_results if r["error"])
        return {
            "success_rate": solved / total if total > 0 else 0.0,
            "solved": solved,
            "total": total,
            "errors": errors,
            "timed_out": False,
            "invalid": False,
            "detailed_results": all_results,
        }

    def pause(self) -> None:
        self.tactic_model.pause()

    def resume(self) -> None:
        self.tactic_model.resume()

    def shutdown(self) -> None:
        self.tactic_model.shutdown()


class DistributedProver(Prover):
    """Drives the coordinator + remote prover daemons.

    All ranks construct one of these, but only the master rank runs the
    coordinator-side collection / eval loops; the other ranks block on the
    matching ``active_barrier_wait`` so they don't busy-loop while still being
    available to serve inference requests.
    """

    def __init__(
        self,
        inference_model: BlockingTacticModel,
        poll_interval: float,
    ):
        self.inference_model = inference_model
        self.poll_interval = poll_interval

    def collect(
        self,
        sampler: TheoremsSampler,
        target_transitions: int,
        replay_buffer: ReplayBuffer,
    ) -> int:
        ddp, rank, _, _ = get_dist_info()
        master_process = rank == 0
        # The barrier key uses the current local_buffer size as a tiebreaker
        # so consecutive calls don't reuse the same key (collect can be called
        # multiple times per training step in principle).
        key = f"collection_done_{id(replay_buffer)}_{len(replay_buffer.local_buffer)}"

        collected_total = 0
        if master_process:
            try:
                # Retry until we have enough transitions
                while len(replay_buffer.local_buffer) < target_transitions:
                    collected = distributed_collect(
                        sampler=sampler,
                        target_transitions=target_transitions,
                        poll_interval=self.poll_interval,
                        replay_buffer=replay_buffer,
                    )
                    collected_total += collected
                    if collected < target_transitions:
                        log(f"Collection incomplete ({collected}/{target_transitions}), retrying...",
                            component="Coordinator")
            finally:
                # Signal completion to other ranks via store.
                active_barrier_master(key)
        else:
            # Active waiting to not block the Python thread (which is needed for inference).
            active_barrier_wait(key)

        # Wait for provers to abort their MCTS searches and release Lean processes
        time.sleep(3.0)
        return collected_total

    def evaluate(self, theorems: list[str], dataset_name: str) -> dict:
        ddp, rank, _, _ = get_dist_info()
        master_process = rank == 0
        key = f"eval_done_{dataset_name}_{len(theorems)}"

        result: dict = {}
        if master_process:
            try:
                result = distributed_eval(theorems, dataset_name=dataset_name)
            finally:
                active_barrier_master(key)
        else:
            active_barrier_wait(key)
        return result

    def pause(self) -> None:
        self.inference_model.pause()

    def resume(self) -> None:
        self.inference_model.resume()

    def shutdown(self) -> None:
        # Local import to avoid loading rl_server (and Flask) just to get the
        # shutdown function in non-distributed code paths.
        from nanoproof.rl_server import shutdown_coordinator
        ddp, rank, _, _ = get_dist_info()
        if rank == 0:
            shutdown_coordinator()
        self.inference_model.shutdown()
