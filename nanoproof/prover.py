"""
Prover: runs MCTS proof searches for experience collection and evaluation.

Components:
- ``Prover``: stateless, thread-safe single-theorem proof search via MCTS.
- ``ProverWorker``: manages parallel proof search with actor threads. Provides
  ``collect()`` and ``evaluate()`` for the training loop.
"""

import asyncio
import faulthandler
import logging
import threading
import time
import traceback
from typing import Callable, Optional
import torch
from leantree.repl_adapter.server import LeanClient
from leantree.repl_adapter.interaction import LeanProcessException
from leantree.utils import RemoteException

from nanoproof.common import (
    Player,
    TimelineRecorder,
    construct_proof_source,
    linearize_proof,
    theorem_to_example,
)
from nanoproof.cli import get_monitor, log
from nanoproof.data.bench.common import BenchTheorem
from nanoproof.experience_collection import (
    TheoremsSampler,
    ReplayBuffer,
    compute_value_target,
    extract_transitions,
    prune_redundant_nodes,
)
from nanoproof.search import Game, MCTSAbortedError, Node, SearchConfig, run_mcts, verify_node
from nanoproof.inference import InferenceBalancer

logger = logging.getLogger(__name__)

import json as json_mod
from urllib.request import urlopen


# -----------------------------------------------------------------------------
# Prover
# -----------------------------------------------------------------------------

class Prover:
    """Proves a single theorem using MCTS. Stateless and thread-safe.

    Multiple actor threads can call :meth:`prove` concurrently; the method
    has no side-effects beyond the provided ``expansion_callback``.
    """

    def __init__(self, config: SearchConfig, tactic_model: InferenceBalancer):
        self.config = config
        self.tactic_model = tactic_model

    @staticmethod
    def _get_process_interruptible(client: LeanClient, abort_check: Callable[[], bool] | None,
                                   poll_interval: float = 10.0, max_wait: float = 300.0):
        """Get a Lean process, polling abort_check between short blocking calls.

        Uses short server-side timeouts so that abort_check is tested every
        *poll_interval* seconds.  Returns None if aborted or *max_wait* is
        exceeded.
        """
        deadline = time.time() + max_wait
        while True:
            if abort_check is not None and abort_check():
                return None
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            timeout = min(poll_interval, remaining)
            process = client.get_process(timeout=timeout)
            if process is not None:
                return process

    def prove(self, client: LeanClient, theorem: BenchTheorem, expansion_callback: Callable[[], None] | None = None,
              abort_check: Callable[[], bool] | None = None,
              timeline: TimelineRecorder | None = None) -> Game | None:
        """Run a single MCTS proof game.

        Returns a :class:`Game` with results, or ``None`` if Lean setup fails.
        """
        logger.debug(f"Proving: {theorem.source[:80]}...")
        process = self._get_process_interruptible(client, abort_check)
        if process is None:
            log(f"FAILED: Could not get Lean process for theorem", component="Prover")
            return None

        with process as env:
            example = theorem_to_example(theorem.source)
            init_branch = env.proof_from_sorry(example)
            if not init_branch.is_success():
                err = init_branch.error if hasattr(init_branch, 'error') else 'unknown error'
                log(f"FAILED: Could not initialize proof - {err}\nLean code:\n{example}", component="Prover")
                return None
            init_branch = init_branch.value

            game = Game(theorem.source, self.config.num_simulations)
            game.root = Node(
                parent=None,
                action=None,
                prior=None,
                state=[init_branch],
                to_play=Player.OR,
                reward=None,
            )

            run_mcts(
                self.config, game, self.tactic_model,
                expansion_callback=expansion_callback,
                abort_check=abort_check,
                timeline=timeline,
            )
            if game.root.is_solved:
                verify_err = verify_node(game.root, timeout=self.config.verify_timeout)
                if verify_err:
                    log(f"FAILED: Verification failed after {game.num_iterations} iterations: '{verify_err}'\nTheorem: '{theorem.source}'\nProof tree:\n{game.root.pp_tree()}", component="Prover")
                    game.root.is_solved = False
                    return game
                game.unsimplified_root = game.root.clone()
                prune_redundant_nodes(game.root)
                compute_value_target(game.root)

                verify_err = verify_node(game.root, timeout=self.config.verify_timeout)
                if verify_err:
                    log(f"FAILED: Post-prune verification failed after {game.num_iterations} iterations: '{verify_err}'\nTheorem: '{theorem.source}'\nProof tree:\n{game.root.pp_tree()}", component="Prover")
                    game.root.is_solved = False
                    return game

                # Verify the linearized proof compiles correctly
                tactics = linearize_proof(game.root)
                proof_source = construct_proof_source(theorem.source, tactics)
                if not env.is_valid_source(proof_source):
                    log(f"FAILED: Linearized proof verification failed after {game.num_iterations} iterations:\n\"\"\"\n{proof_source}\n\"\"\"\n... proof tree:\n{game.root.pp_tree()}\n", component="Prover")
                    game.root.is_solved = False

            return game


# -----------------------------------------------------------------------------
# ProverWorker
# -----------------------------------------------------------------------------

class ProverWorker:
    """Manages parallel proof search across multiple Lean servers.

    Runs actor threads that each get a dedicated ``LeanClient`` and call
    :meth:`Prover.prove`. Provides :meth:`collect` and :meth:`evaluate`
    as the two main modes of operation.
    """

    def __init__(
        self,
        tactic_model: InferenceBalancer,
        lean_addrs: list[str],
    ):
        self.tactic_model = tactic_model
        self.lean_servers = self._query_lean_servers(lean_addrs)
        self.num_actors = len(self.lean_servers)

    @staticmethod
    def _query_lean_servers(raw_addrs: list[str]) -> list[tuple[str, int]]:
        """Query each Lean server for capacity; return a flat (host, port) list with one entry per process."""
        servers = []
        for addr in raw_addrs:
            host, port_str = addr.split(":") if ":" in addr else (addr, "8000")
            port = int(port_str)
            try:
                with urlopen(f"http://{host}:{port}/status", timeout=10) as resp:
                    status = json_mod.loads(resp.read())
                max_procs = status.get("max_processes", 0)
            except Exception as e:
                raise ConnectionError(f"Could not reach Lean server {addr}: {e}") from e
            if max_procs == 0:
                raise ConnectionError(f"Lean server {addr} reports 0 available processes")
            log(f"Lean server {host}:{port}: {max_procs} processes", component="LeanPool")
            servers.extend([(host, port)] * max_procs)
        return servers

    @torch.no_grad()
    def collect(
        self,
        sampler: TheoremsSampler,
        target_transitions: int,
        replay_buffer: ReplayBuffer,
        num_simulations: int,
    ) -> int:
        """Collect transitions into replay_buffer.local_buffer.

        Runs actor threads until the target number of transitions is reached.
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

        def done_check():
            with replay_buffer._lock:
                return len(replay_buffer.local_buffer) >= target_transitions

        loop_count = [0]

        def poll_callback(thread_states):
            if monitor is not None:
                for i, state in enumerate(thread_states):
                    monitor.update_local_actor(i, state=state)
            loop_count[0] += 1
            if loop_count[0] % 50 == 0:
                with replay_buffer._lock:
                    collected = len(replay_buffer.local_buffer)
                log(f"Progress: {collected}/{target_transitions} transitions",
                    component="Collection")

        prover = Prover(SearchConfig(num_simulations=num_simulations), self.tactic_model)
        self._run_pool(prover, get_theorem, on_result, done_check, poll_callback,
                       record_timeline=True)

        if monitor is not None:
            monitor.clear_local_actors()

        log(f"Collection complete: {len(replay_buffer.local_buffer)} transitions",
            component="Collection")
        return len(replay_buffer.local_buffer)

    @torch.no_grad()
    def evaluate(self, theorems: list[BenchTheorem], dataset_name: str, num_simulations: int,
                 progress_callback: Callable[[int, int, int, int], None] | None = None,
                 verify_timeout: int = 5000) -> dict:
        """Evaluate theorems using MCTS. Returns metrics dict.

        *progress_callback*, if given, is called as
        ``progress_callback(started, finished, solved, errors)`` whenever a
        theorem is picked up or completed.
        """
        monitor = get_monitor()

        if monitor:
            monitor.start_eval(dataset_name, len(theorems))

        index = [0]
        index_lock = threading.Lock()

        def get_theorem():
            with index_lock:
                if index[0] >= len(theorems):
                    return None
                tid = f"eval_{index[0]}"
                theorem = theorems[index[0]]
                index[0] += 1
                started = index[0]
            if progress_callback:
                with results_lock:
                    finished = len(results)
                    solved = sum(1 for r in results if r["is_solved"])
                    errors = sum(1 for r in results if r["error"])
                progress_callback(started, finished, solved, errors)
            return (tid, theorem)

        results: list[dict] = []
        results_lock = threading.Lock()

        def on_result(theorem_id, theorem: BenchTheorem, game, error):
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
                linearized = construct_proof_source(theorem.source, tactics)

            with results_lock:
                results.append({
                    "theorem": theorem.source,
                    "name": theorem.name,
                    "is_solved": is_solved,
                    "error": error,
                    "proof_tree": proof_tree,
                    "unsimplified_proof_tree": unsimplified_proof_tree,
                    "linearized_proof": linearized,
                    "num_iterations": num_iterations,
                })

                n = len(results)
                solved = sum(1 for r in results if r["is_solved"])
                errors = sum(1 for r in results if r["error"])

                if monitor:
                    monitor.update_eval_progress(current=n, solved=solved, errors=errors)

                if progress_callback:
                    with index_lock:
                        started = index[0]
                    progress_callback(started, n, solved, errors)

        def done_check():
            with results_lock:
                return len(results) >= len(theorems)

        prover = Prover(SearchConfig(num_simulations=num_simulations, verify_timeout=verify_timeout), self.tactic_model)
        self._run_pool(prover, get_theorem, on_result, done_check,
                       record_timeline=True)

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

    def _run_pool(
        self,
        prover: Prover,
        get_theorem: Callable[[], Optional[tuple[str, BenchTheorem]]],
        on_result: Callable[[str, BenchTheorem, Optional[Game], Optional[str]], None],
        done_check: Callable[[], bool],
        poll_callback: Callable[[list[str]], None] | None = None,
        record_timeline: bool = False,
    ):
        """Run actor threads until *done_check* returns ``True``.

        All thread state is local to this call; the ``ProverWorker`` instance
        remains reusable across multiple ``collect`` / ``evaluate`` invocations.
        """
        stop_flag = threading.Event()
        thread_states: dict[int, str] = {}
        thread_states_lock = threading.Lock()

        games_played = [0]
        games_solved = [0]
        expansions = [0]
        stats_lock = threading.Lock()

        def set_thread_state(actor_id: int, state: str):
            with thread_states_lock:
                thread_states[actor_id] = state

        def get_thread_states() -> list[str]:
            with thread_states_lock:
                return [thread_states.get(i, "idle") for i in range(self.num_actors)]

        def on_expansion():
            with stats_lock:
                expansions[0] += 1

        def actor_loop(actor_id: int):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            lean_address, lean_port = self.lean_servers[actor_id]
            set_thread_state(actor_id, "blocked")
            client = LeanClient(lean_address, lean_port)
            set_thread_state(actor_id, "idle")

            consecutive_errors = 0
            max_consecutive_errors = 5
            max_retries = 3

            while not stop_flag.is_set():
                theorem_data = get_theorem()
                if theorem_data is None:
                    set_thread_state(actor_id, "idle")
                    time.sleep(0.5)
                    continue

                theorem_id, theorem = theorem_data
                set_thread_state(actor_id, "running")

                game = None
                error = None
                skip_report = False
                timeline = TimelineRecorder() if record_timeline else None
                for attempt in range(max_retries):
                    try:
                        game = prover.prove(client, theorem, expansion_callback=on_expansion,
                                            abort_check=stop_flag.is_set,
                                            timeline=timeline)
                        consecutive_errors = 0
                        break
                    except (ConnectionError, LeanProcessException, RemoteException, TimeoutError) as e:
                        if attempt < max_retries - 1:
                            set_thread_state(actor_id, "retry")
                            log(f"[Actor {actor_id}] Connection error (attempt {attempt + 1}/{max_retries}): '{e}', reconnecting...", component="Prover")
                            time.sleep(1.0 * (attempt + 1))
                        else:
                            error = str(e)
                            consecutive_errors += 1
                    except MCTSAbortedError:
                        skip_report = True
                        break
                    except Exception as e:
                        if "Model paused for training" in str(e):
                            set_thread_state(actor_id, "idle")
                            time.sleep(0.5)
                            skip_report = True
                            break
                        error = str(e)
                        consecutive_errors += 1
                        log(f"[Actor {actor_id}] Error (lean={lean_address}:{lean_port}): {e}", component="Prover")
                        traceback.print_exc()
                        break

                if not skip_report:
                    is_solved = game and game.root and game.root.is_solved
                    logger.debug(f"Actor {actor_id}: {theorem_id} {'solved' if is_solved else 'unsolved'} in {game.num_iterations if game else 0} iters")
                    if is_solved:
                        logger.debug(f"Actor {actor_id}: proof tree for {theorem_id}:\n{game.root.pp_tree()}")
                    if game is not None and game.root is not None:
                        with stats_lock:
                            games_played[0] += 1
                            if game.root.is_solved:
                                games_solved[0] += 1

                    if error is not None:
                        set_thread_state(actor_id, "error")

                    on_result(theorem_id, theorem, game, error)

                    # Flush timeline events to the monitor
                    if timeline and timeline.events:
                        monitor = get_monitor()
                        if monitor is not None:
                            monitor.record_timeline_events(actor_id, timeline.events)
                        timeline.events.clear()

                if consecutive_errors >= max_consecutive_errors:
                    log(f"[Actor {actor_id}] Too many consecutive errors, exiting", component="Prover")
                    break

            set_thread_state(actor_id, "idle")

        # Start actor threads
        threads: list[threading.Thread] = []
        for i in range(self.num_actors):
            t = threading.Thread(target=actor_loop, args=(i,), daemon=True)
            t.start()
            threads.append(t)

        # Poll until done
        try:
            while True:
                if done_check():
                    break
                if all(not t.is_alive() for t in threads):
                    log("WARNING: All actors have exited unexpectedly", component="Prover")
                    break
                if poll_callback is not None:
                    poll_callback(get_thread_states())
                time.sleep(0.1)
        finally:
            stop_flag.set()
            deadline = time.time() + 60.0
            for t in threads:
                t.join(timeout=max(0.0, deadline - time.time()))
            alive = sum(1 for t in threads if t.is_alive())
            if alive:
                log(f"WARNING: {alive}/{len(threads)} actor threads still alive after 60s", component="Prover")
                faulthandler.dump_traceback()
