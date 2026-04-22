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
from dataclasses import dataclass
from typing import Callable, Literal, Optional
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
from nanoproof.cli import get_monitor, log, log_actionable_error
from nanoproof.data.bench.common import BenchTheorem
from nanoproof.experience_collection import (
    TheoremsSampler,
    CollectedExperience,
    compute_value_target,
    prune_redundant_nodes,
)
from nanoproof.search import Game, MCTSAbortedError, Node, SearchConfig, run_mcts, verify_node
from nanoproof.inference import InferenceBalancer

logger = logging.getLogger(__name__)

import json as json_mod
from urllib.request import urlopen


def _flush_timeline(
    actor_id: int,
    timeline: TimelineRecorder,
    *,
    game: "Game | None",
    error: str | None,
    interrupted: bool,
) -> None:
    """Ship an actor's accumulated timeline events and a single outcome marker
    to the monitor, then reset the recorder.

    Called once per theorem iteration regardless of ``skip_report``: aborted
    attempts still did real LLM/Lean work and are worth showing in the
    profiler. The outcome kind classifies what the "productive only" toggle
    should hide.
    """
    monitor = get_monitor()
    if monitor is None:
        timeline.events.clear()
        return
    if timeline.events:
        monitor.record_timeline_events(actor_id, timeline.events)
    kind = _outcome_kind(game=game, error=error, interrupted=interrupted)
    if kind is not None:
        monitor.record_outcome(actor_id, kind)
    timeline.events.clear()


def _outcome_kind(*, game, error, interrupted) -> str | None:
    if interrupted:
        return "interrupted"
    if game is not None and game.root is not None:
        return "solved" if game.root.is_solved else "gave_up"
    if error is not None:
        return "gave_up"
    return None


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
                                   poll_interval: float = 10.0, max_wait: float = 300.0) -> tuple:
        """Get a Lean process, polling abort_check between short blocking calls.

        Uses short server-side timeouts so that abort_check is tested every
        *poll_interval* seconds.  Returns ``(process, reason)`` where reason
        is ``"ok"`` (process is not None), ``"aborted"`` (caller asked to
        stop — typically end of a collect cycle), or ``"timeout"`` (waited
        *max_wait* without success — real pool saturation).
        """
        deadline = time.time() + max_wait
        while True:
            if abort_check is not None and abort_check():
                return None, "aborted"
            remaining = deadline - time.time()
            if remaining <= 0:
                return None, "timeout"
            timeout = min(poll_interval, remaining)
            process = client.get_process(timeout=timeout)
            if process is not None:
                return process, "ok"

    def prove(self, client: LeanClient, theorem: BenchTheorem, expansion_callback: Callable[[], None] | None = None,
              abort_check: Callable[[], bool] | None = None,
              timeline: TimelineRecorder | None = None) -> Game | None:
        """Run a single MCTS proof game.

        Returns a :class:`Game` with results, or ``None`` if Lean setup fails.
        """
        logger.debug(f"Proving: {theorem.source[:80]}...")
        process, reason = self._get_process_interruptible(client, abort_check)
        if process is None:
            # "aborted" is the eval-release path (ProverWorker sets the
            # release event so mid-proof actors drop their Lean leases);
            # only surface true pool-saturation timeouts.
            if reason == "timeout":
                log(f"FAILED: Could not get Lean process for theorem (300s pool timeout)", component="Prover")
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

@dataclass
class _Job:
    """One unit of work delivered to the actor pool.

    Actors read ``get_theorem`` / ``on_result`` from the CURRENTLY installed
    job at each boundary (before fetching, after a proof completes), not
    from the job that was current when a proof started. That is what lets a
    mid-MCTS actor whose proof started in step N deliver its result into
    step N+1's experience once training is done.
    """
    get_theorem: Callable[[], Optional[tuple[str, BenchTheorem]]]
    on_result: Callable[[str, BenchTheorem, Optional[Game], Optional[str]], None]
    done_check: Callable[[], bool]
    poll_callback: Optional[Callable[[list[str]], None]]
    record_timeline: bool
    prover: Prover


class ProverWorker:
    """Long-lived actor pool that parallelises MCTS proof search across
    Lean servers.

    The pool is created once in ``__init__`` and runs until ``close()``.
    Callers drive it with ``collect()`` and ``evaluate()``; the two differ
    only in how they park actors on exit:

    - ``collect()`` parks with a *pause*: the installed job stays live, so
      an actor that is blocked deep inside MCTS (waiting on paused
      inference during a training step) resumes its proof on the same
      tree when inference resumes. A completed proof is attributed to
      whatever job is current at completion time, so mid-proof work from
      step N naturally lands in step N+1's experience. Lean leases held
      by mid-proof actors stay held across the training step.
    - ``evaluate()`` parks with a *release*: the release event is raised,
      mid-proof actors abort and return their Lean leases, and the job
      is cleared. This keeps eval-theorem results from leaking into the
      next collect cycle and ensures leases are not tied to theorems
      that will never be completed in the new job.

    Actors carry a single abort signal (``_release_event``) and a single
    blocking point (paused inference inside ``sample_tactic``). There is
    no per-call stop flag and no "model paused for training" fast-fail
    path: paused inference simply blocks the caller until resume.
    """

    def __init__(
        self,
        tactic_model: InferenceBalancer,
        lean_addrs: list[str],
    ):
        self.tactic_model = tactic_model
        self.lean_servers = self._query_lean_servers(lean_addrs)
        self.num_actors = len(self.lean_servers)

        # Pool state. ``_current_job`` is the single source of truth for
        # what actors do; it is swapped under ``_lock`` and actors wait on
        # ``_job_cv`` when it is None. ``_release_event`` is the only abort
        # signal seen by Prover.prove. ``_shutdown_event`` is set in close()
        # and wakes every waiting actor so they can exit.
        self._lock = threading.Lock()
        self._job_cv = threading.Condition(self._lock)
        self._current_job: Optional[_Job] = None
        self._release_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._actors_mid_proof = 0
        self._thread_states: dict[int, str] = {i: "idle" for i in range(self.num_actors)}

        self._threads: list[threading.Thread] = []
        for actor_id in range(self.num_actors):
            t = threading.Thread(
                target=self._actor_loop,
                args=(actor_id,),
                daemon=True,
                name=f"prover-actor-{actor_id}",
            )
            t.start()
            self._threads.append(t)

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

    def close(self):
        """Signal shutdown and join actor threads. Called at process teardown."""
        self._shutdown_event.set()
        with self._lock:
            self._release_event.set()
            self._current_job = None
            self._job_cv.notify_all()
        deadline = time.time() + 60.0
        for t in self._threads:
            t.join(timeout=max(0.0, deadline - time.time()))
        alive = sum(1 for t in self._threads if t.is_alive())
        if alive:
            log(f"WARNING: {alive}/{len(self._threads)} actor threads still alive after close", component="Prover")
            faulthandler.dump_traceback()

    @torch.no_grad()
    def collect(
        self,
        sampler: TheoremsSampler,
        target_transitions: int,
        experience: CollectedExperience,
        num_simulations: int,
    ) -> int:
        """Record successful proofs into ``experience`` until the target
        number of transitions is reached. Parks with *pause* on exit."""
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
            transitions_before = experience.num_transitions()
            experience.record_proof(theorem, game)
            if monitor is not None:
                monitor.add_collected_samples(experience.num_transitions() - transitions_before)

        def done_check():
            return experience.num_transitions() >= target_transitions

        loop_count = [0]

        def poll_callback(thread_states):
            if monitor is not None:
                for i, state in enumerate(thread_states):
                    monitor.update_local_actor(i, state=state)
            loop_count[0] += 1
            if loop_count[0] % 50 == 0:
                log(f"Progress: {experience.num_transitions()}/{target_transitions} transitions",
                    component="Collection")

        job = _Job(
            get_theorem=get_theorem,
            on_result=on_result,
            done_check=done_check,
            poll_callback=poll_callback,
            record_timeline=True,
            prover=Prover(SearchConfig(num_simulations=num_simulations), self.tactic_model),
        )
        self._run_job(job, park="pause")

        if monitor is not None:
            monitor.clear_local_actors()

        total = experience.num_transitions()
        log(f"Collection complete: {total} transitions", component="Collection")
        return total

    @torch.no_grad()
    def evaluate(self, theorems: list[BenchTheorem], dataset_name: str, num_simulations: int,
                 progress_callback: Callable[[int, int, int, int], None] | None = None,
                 verify_timeout: int = 5000) -> dict:
        """Evaluate theorems using MCTS. Returns metrics dict. Parks with
        *release* on exit so eval state does not leak into the next job
        and Lean leases are returned.

        *progress_callback*, if given, is called as
        ``progress_callback(started, finished, solved, errors)`` whenever a
        theorem is picked up or completed.
        """
        monitor = get_monitor()

        if monitor:
            monitor.start_eval(dataset_name, len(theorems))

        index = [0]
        index_lock = threading.Lock()
        results: list[dict] = []
        results_lock = threading.Lock()

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

        job = _Job(
            get_theorem=get_theorem,
            on_result=on_result,
            done_check=done_check,
            poll_callback=None,
            record_timeline=True,
            prover=Prover(SearchConfig(num_simulations=num_simulations, verify_timeout=verify_timeout), self.tactic_model),
        )
        self._run_job(job, park="release")

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

    def _run_job(self, job: _Job, *, park: Literal["pause", "release"]):
        """Install *job*, drive the done-check loop, then park the pool.

        ``park == "pause"`` leaves the job installed on exit so a
        straggler completing after done_check fires (common for collect)
        is attributed to whichever job is current when the proof lands.
        Between ``collect()`` calls this lets step-N mid-proof work
        become step-N+1 training data.

        ``park == "release"`` raises the release event, waits for
        mid-proof actors to abort, then clears the job. Eval uses it on
        *both* ends: a pre-install release drains any collect stragglers
        so their results do not get mis-attributed to eval's on_result;
        a post-done release frees Lean leases that would otherwise sit
        idle tied to eval theorems that will never be resumed.
        """
        if park == "release":
            self._release_and_idle()

        with self._lock:
            self._current_job = job
            self._release_event.clear()
            self._job_cv.notify_all()

        while not self._shutdown_event.is_set():
            if job.done_check():
                break
            if job.poll_callback is not None:
                job.poll_callback(self._snapshot_thread_states())
            time.sleep(0.1)

        if park == "release":
            self._release_and_idle()

    def _release_and_idle(self):
        """Abort mid-proof actors, wait for leases to drop, then clear the job."""
        with self._lock:
            self._release_event.set()
        deadline = time.time() + 60.0
        while True:
            with self._lock:
                mid = self._actors_mid_proof
            if mid == 0:
                break
            if time.time() > deadline:
                log(f"WARNING: release timed out with {mid} actors still mid-proof",
                    component="Prover")
                break
            time.sleep(0.05)
        with self._lock:
            self._current_job = None
            self._release_event.clear()

    def _set_thread_state(self, actor_id: int, state: str):
        with self._lock:
            self._thread_states[actor_id] = state

    def _snapshot_thread_states(self) -> list[str]:
        with self._lock:
            return [self._thread_states[i] for i in range(self.num_actors)]

    def _actor_loop(self, actor_id: int):
        asyncio.set_event_loop(asyncio.new_event_loop())
        lean_address, lean_port = self.lean_servers[actor_id]
        self._set_thread_state(actor_id, "blocked")
        client = LeanClient(lean_address, lean_port)
        self._set_thread_state(actor_id, "idle")

        consecutive_errors = 0
        max_consecutive_errors = 5
        max_retries = 3

        while not self._shutdown_event.is_set():
            with self._job_cv:
                while self._current_job is None and not self._shutdown_event.is_set():
                    self._job_cv.wait()
                if self._shutdown_event.is_set():
                    break
                job = self._current_job

            theorem_data = job.get_theorem()
            if theorem_data is None:
                self._set_thread_state(actor_id, "idle")
                time.sleep(0.5)
                continue

            theorem_id, theorem = theorem_data
            self._set_thread_state(actor_id, "running")

            game = None
            error = None
            skip_report = False
            interrupted = False
            timeline = TimelineRecorder() if job.record_timeline else None

            with self._lock:
                self._actors_mid_proof += 1
            try:
                for attempt in range(max_retries):
                    try:
                        game = job.prover.prove(
                            client, theorem,
                            abort_check=self._release_event.is_set,
                            timeline=timeline,
                        )
                        consecutive_errors = 0
                        break
                    except (ConnectionError, LeanProcessException, RemoteException, TimeoutError) as e:
                        if attempt < max_retries - 1:
                            self._set_thread_state(actor_id, "retry")
                            short_err = str(e).split('\n', 1)[0]
                            log(f"[Actor {actor_id}] Connection error (attempt {attempt + 1}/{max_retries}): '{short_err}', reconnecting...",
                                component="Prover")
                            time.sleep(1.0 * (attempt + 1))
                        else:
                            error = str(e)
                            consecutive_errors += 1
                            log(f"[Actor {actor_id}] Error (lean={lean_address}:{lean_port}): {e}", component="Prover")
                            log_actionable_error("Prover", str(e),
                                                 actor=actor_id, lean=f"{lean_address}:{lean_port}",
                                                 retries_exhausted=True)
                    except MCTSAbortedError:
                        skip_report = True
                        interrupted = True
                        break
                    except Exception as e:
                        error = str(e)
                        consecutive_errors += 1
                        log(f"[Actor {actor_id}] Error (lean={lean_address}:{lean_port}): {e}", component="Prover")
                        log_actionable_error("Prover", str(e),
                                             actor=actor_id, lean=f"{lean_address}:{lean_port}")
                        traceback.print_exc()
                        break
            finally:
                with self._lock:
                    self._actors_mid_proof -= 1

            # Route the result to whichever job is current NOW, not the
            # one this proof started under. For collect->train->collect
            # that lets step-N mid-proof work land in step-N+1's
            # experience. For eval, a pre-install release in _run_job
            # guarantees no collect stragglers are alive at this point,
            # so same-type attribution still holds. If current_job is
            # None (released, between jobs) the straggler's result is
            # dropped, which is the intended release semantic.
            with self._lock:
                current_job = self._current_job

            if not skip_report and current_job is not None:
                is_solved = bool(game and game.root and game.root.is_solved)
                logger.debug(f"Actor {actor_id}: {theorem_id} {'solved' if is_solved else 'unsolved'} in {game.num_iterations if game else 0} iters")
                if is_solved:
                    logger.debug(f"Actor {actor_id}: proof tree for {theorem_id}:\n{game.root.pp_tree()}")
                if error is not None:
                    self._set_thread_state(actor_id, "error")
                current_job.on_result(theorem_id, theorem, game, error)

            # Always flush the timeline, including for aborted proofs:
            # the LLM/Lean work they did belongs on the profiler, and
            # the outcome marker classifies what the "productive only"
            # toggle hides.
            if timeline is not None:
                _flush_timeline(actor_id, timeline, game=game, error=error,
                                interrupted=interrupted)

            if consecutive_errors >= max_consecutive_errors:
                log(f"[Actor {actor_id}] {consecutive_errors} consecutive errors; backing off 60s",
                    component="Prover")
                time.sleep(60.0)
                consecutive_errors = 0

            self._set_thread_state(actor_id, "idle")

        self._set_thread_state(actor_id, "idle")
