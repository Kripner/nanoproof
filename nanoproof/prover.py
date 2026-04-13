"""
Prover: runs MCTS proof searches for experience collection and evaluation.

Components:
- ``TheoremsSampler``: weighted sampler over configured training datasets.
- ``ProverWorker``: thread pool that plays proof games via ``run_mcts``. Each
  actor thread gets a dedicated ``LeanClient`` connection (1:1 mapping).
- ``Prover``: high-level API used by the training loop. Manages a
  ``ProverWorker`` for both collection and evaluation.
"""

import asyncio
import random
import threading
import time
import traceback
from typing import Callable, Optional
import torch
from leantree.repl_adapter.server import LeanClient
from leantree.repl_adapter.interaction import LeanProcessException

from nanoproof.common import (
    Player,
    construct_proof_source,
    linearize_proof,
    theorem_to_example,
)
from nanoproof.cli import get_monitor, log
from nanoproof.data.rl import deepseek_prover, leanworkbook, numinamath
from nanoproof.replay_buffer import (
    ReplayBuffer,
    compute_value_target,
    extract_transitions,
    prune_redundant_nodes,
)
from nanoproof.search import Game, Node, SearchConfig, run_mcts, verify_node
from nanoproof.inference import InferenceBalancer


# -----------------------------------------------------------------------------
# Theorem sampler
# -----------------------------------------------------------------------------

class TheoremsSampler:
    """Samples theorems for experience collection from multiple datasets.

    Samples uniformly at random from one of the available datasets, then
    uniformly at random from that dataset. Thread-safe.
    """

    ALL_DATASETS = {
        "leanworkbook": lambda: leanworkbook.list_theorems(split="train"),
        "deepseek_prover": lambda: deepseek_prover.list_theorems(split="train"),
        "numinamath": lambda: numinamath.list_theorems(split="train"),
    }

    def __init__(self, seed: int | None = 0, datasets: list[str] | None = None):
        if datasets is None:
            datasets = list(self.ALL_DATASETS.keys())
        self.datasets = {name: self.ALL_DATASETS[name]() for name in datasets}
        self.dataset_names = list(self.datasets.keys())
        self.rng = random.Random(seed)
        self._lock = threading.Lock()

        for name, theorems in self.datasets.items():
            log(f"Loaded {len(theorems)} theorems from {name}", component="Sampler")

    def sample_theorem(self) -> str:
        with self._lock:
            dataset_name = self.rng.choice(self.dataset_names)
            return self.rng.choice(self.datasets[dataset_name])


# -----------------------------------------------------------------------------
# ProverWorker
# -----------------------------------------------------------------------------

class ProverWorker:
    """
    Runs actor threads that play proof games via MCTS.

    Each actor thread gets a dedicated Lean server connection (1:1 mapping
    via the ``lean_servers`` list). Configurable with callbacks:
    - get_theorem: returns (id, theorem) or None
    - on_result: called with (id, theorem, game_or_none, error_or_none)
    """

    def __init__(
        self,
        config: SearchConfig,
        tactic_model: InferenceBalancer,
        lean_servers: list[tuple[str, int]],
        get_theorem: Callable[[], Optional[tuple[str, str]]],
        on_result: Callable[[str, str, Optional[Game], Optional[str]], None],
        paused: bool = False,
    ):
        self.config = config
        self.tactic_model = tactic_model
        self.lean_servers = lean_servers  # one (host, port) per actor
        self.get_theorem = get_theorem
        self.on_result = on_result
        self.num_actors = len(lean_servers)

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
        if self._running and not self.all_actors_exited():
            return

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

        # Connect to this actor's dedicated Lean server
        lean_address, lean_port = self.lean_servers[actor_id]
        self._set_thread_state(actor_id, "blocked")
        client = LeanClient(lean_address, lean_port)
        self._set_thread_state(actor_id, "idle")

        consecutive_errors = 0
        max_consecutive_errors = 5
        max_retries = 3

        while not self._stop_flag.is_set():
            if self._paused:
                self._set_thread_state(actor_id, "idle")
                time.sleep(0.5)
                continue

            theorem_data = self.get_theorem()
            if theorem_data is None:
                self._set_thread_state(actor_id, "idle")
                time.sleep(0.5)
                continue

            theorem_id, theorem = theorem_data
            self._set_thread_state(actor_id, "running")

            game = None
            error = None
            skip_report = False
            for attempt in range(max_retries):
                try:
                    game = self._play_game(client, theorem)
                    consecutive_errors = 0
                    break
                except (ConnectionResetError, ConnectionRefusedError, BrokenPipeError, LeanProcessException) as e:
                    if attempt < max_retries - 1:
                        self._set_thread_state(actor_id, "retry")
                        log(f"[Actor {actor_id}] Connection error (attempt {attempt + 1}/{max_retries}): '{e}', reconnecting...", component="Collection")
                        time.sleep(1.0 * (attempt + 1))
                    else:
                        error = str(e)
                        consecutive_errors += 1
                except Exception as e:
                    if "Model paused for training" in str(e):
                        self._set_thread_state(actor_id, "idle")
                        skip_report = True
                        break
                    error = str(e)
                    consecutive_errors += 1
                    log(f"[Actor {actor_id}] Error (lean={lean_address}:{lean_port}): {e}", component="Collection")
                    traceback.print_exc()
                    break

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
# Prover
# -----------------------------------------------------------------------------

class Prover:
    """High-level API for MCTS proof search (collection and evaluation).

    Manages a ``ProverWorker`` and exposes ``collect`` / ``evaluate`` for the
    training loop. Only runs on the master rank (rank 0).
    """

    def __init__(
        self,
        tactic_model: InferenceBalancer,
        lean_servers: list[tuple[str, int]],
        num_actors: int | None = None,
    ):
        self.tactic_model = tactic_model
        self.lean_servers = lean_servers
        self.num_actors = num_actors if num_actors is not None else len(lean_servers)

    @torch.no_grad()
    def collect(
        self,
        sampler: TheoremsSampler,
        target_transitions: int,
        replay_buffer: ReplayBuffer,
        num_simulations: int,
    ) -> int:
        """Collect transitions into replay_buffer.local_buffer.

        Spawns a ProverWorker, runs until target is reached, then stops.
        """
        monitor = get_monitor()

        log(f"Starting collection with {self.num_actors} actors, target={target_transitions} transitions",
            component="Collection")

        theorem_counter = [0]

        def get_theorem():
            theorem = sampler.sample_theorem()
            theorem_counter[0] += 1
            return (f"collect_{theorem_counter[0]}", theorem)

        def on_result(theorem_id, theorem, game, error):
            if error is not None or game is None or game.root is None:
                return
            if not game.root.is_solved:
                return
            transitions = extract_transitions(game.root)
            replay_buffer.add_transitions(transitions)
            if monitor is not None:
                monitor.record_transitions(transitions)

        config = SearchConfig(num_simulations=num_simulations)
        worker = ProverWorker(
            config=config,
            tactic_model=self.tactic_model,
            lean_servers=self.lean_servers[:self.num_actors],
            get_theorem=get_theorem,
            on_result=on_result,
        )
        worker.start()

        loop_count = 0
        try:
            while True:
                with replay_buffer._lock:
                    collected = len(replay_buffer.local_buffer)
                if collected >= target_transitions:
                    log(f"Target reached: {collected}/{target_transitions} transitions collected",
                        component="Collection")
                    break

                if monitor is not None:
                    states = worker.get_thread_states()
                    for i, state in enumerate(states):
                        monitor.update_local_actor(i, state=state)

                if worker.has_started_actors() and worker.all_actors_exited():
                    log("WARNING: All actors have exited unexpectedly", component="Collection")
                    break

                loop_count += 1
                if loop_count % 50 == 0:
                    log(f"Progress: {collected}/{target_transitions} transitions",
                        component="Collection")

                time.sleep(0.1)
        finally:
            log(f"Stopping actors...", component="Collection")
            worker.stop()
            if monitor is not None:
                monitor.clear_local_actors()

        log(f"Collection complete: {len(replay_buffer.local_buffer)} transitions",
            component="Collection")
        return len(replay_buffer.local_buffer)

    @torch.no_grad()
    def evaluate(self, theorems: list[str], dataset_name: str, num_simulations: int) -> dict:
        """Evaluate theorems using MCTS. Returns metrics dict."""
        monitor = get_monitor()

        if monitor:
            monitor.start_eval(dataset_name, len(theorems))

        eval_config = SearchConfig(num_simulations=num_simulations)

        index = [0]
        index_lock = threading.Lock()

        def get_theorem():
            with index_lock:
                if index[0] >= len(theorems):
                    return None
                tid = f"eval_{index[0]}"
                theorem = theorems[index[0]]
                index[0] += 1
                return (tid, theorem)

        results: list[dict] = []
        results_lock = threading.Lock()

        def on_result(theorem_id, theorem, game, error):
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
                results.append({
                    "theorem": theorem,
                    "is_solved": is_solved,
                    "error": error,
                    "proof_tree": proof_tree,
                    "unsimplified_proof_tree": unsimplified_proof_tree,
                    "linearized_proof": linearized,
                    "num_iterations": num_iterations,
                })

                if monitor:
                    n = len(results)
                    solved = sum(1 for r in results if r["is_solved"])
                    errors = sum(1 for r in results if r["error"])
                    monitor.update_eval_progress(current=n, solved=solved, errors=errors)

        worker = ProverWorker(
            config=eval_config,
            tactic_model=self.tactic_model,
            lean_servers=self.lean_servers[:self.num_actors],
            get_theorem=get_theorem,
            on_result=on_result,
        )
        worker.start()

        try:
            while True:
                with results_lock:
                    done = len(results) >= len(theorems)
                if done:
                    break
                if worker.has_started_actors() and worker.all_actors_exited():
                    log("WARNING: All actors exited before eval completed", component="Eval")
                    break
                time.sleep(0.1)
        finally:
            worker.stop()

        total = len(results)
        solved = sum(1 for r in results if r["is_solved"])
        errors = sum(1 for r in results if r["error"])
        return {
            "success_rate": solved / total if total > 0 else 0.0,
            "solved": solved,
            "total": total,
            "errors": errors,
            "detailed_results": results,
        }

