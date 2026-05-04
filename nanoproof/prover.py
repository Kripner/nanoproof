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
from typing import Callable, Literal, Optional
import json as json_mod
from urllib.request import urlopen

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
from nanoproof.cli import get_monitor, log_actionable_error
from nanoproof.data.bench.common import BenchTheorem
from nanoproof.experience_collection import (
    CollectExperienceHolder,
    Matchmaker,
    Outcome,
    compute_value_target,
    prune_redundant_nodes,
)
from nanoproof.search import (
    Game,
    MCTSAbortedError,
    Node,
    SearchConfig,
    close_leaves_with_grind,
    run_mcts,
    verify_node,
)
from nanoproof.inference import InferenceBalancer

logger = logging.getLogger(__name__)


class LeanPoolTimeoutError(Exception):
    """Raised when waiting on the Lean process pool exceeds the deadline.

    Surfaced as an attempt-level error (not "unsolved") so eval results
    distinguish "we tried and failed" from "we never got to try".
    """


class ProofInitError(Exception):
    """Raised when ``proof_from_sorry`` cannot initialize a proof.

    The Lean process rejected the theorem before any search ran, so the
    attempt should be recorded as an error rather than a silent
    "unsolved".
    """


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


class Prover:
    """Runs proof search on a single theorem. Stateless and thread-safe."""

    def __init__(
        self,
        config: SearchConfig,
        tactic_model: InferenceBalancer,
        simplify_proofs: bool = True,
        post_search_grind: bool = False,
        expand_inject_grind: bool = False,
    ):
        self.config = config
        self.tactic_model = tactic_model
        self.simplify_proofs = simplify_proofs
        self.post_search_grind = post_search_grind
        self.expand_inject_grind = expand_inject_grind

    def prove(
        self,
        client: LeanClient,
        theorem: BenchTheorem,
        num_simulations: int,
        expansion_callback: Callable[[], None] | None = None,
        abort_check: Callable[[], bool] | None = None,
        timeline: TimelineRecorder | None = None,
        tactic_sink: Callable[[str, list[tuple[str, str, int]]], None] | None = None,
    ) -> Game | None:
        """
        Run a single MCTS proof game with the supplied per-call simulation budget.
        Returns a :class:`Game` with results, or ``None`` if Lean setup fails.
        """
        logger.debug(f"Proving: {theorem.source[:80]}...")
        process, reason = self._get_process_interruptible(client, abort_check)
        if process is None:
            # "aborted" is the eval-release path (ProverWorker sets the
            # release event so mid-proof actors drop their Lean leases).
            # Pool-saturation timeouts are real failures: raise so the
            # actor records them as errors instead of silent "unsolved".
            if reason == "timeout":
                raise LeanPoolTimeoutError(
                    "Could not get Lean process for theorem (300s pool timeout)"
                )
            return None

        with process as env:
            example = theorem_to_example(theorem.source)
            init_branch = env.proof_from_sorry(example)
            if not init_branch.is_success():
                err = (
                    init_branch.error
                    if hasattr(init_branch, "error")
                    else "unknown error"
                )
                raise ProofInitError(
                    f"Could not initialize proof - {err}\nLean code:\n{example}"
                )
            init_branch = init_branch.value

            game = Game(theorem.source, num_simulations)
            game.root = Node(
                parent=None,
                action=None,
                prior=None,
                state=[init_branch],
                to_play=Player.OR,
                reward=None,
            )

            run_mcts(
                self.config,
                game,
                self.tactic_model,
                expansion_callback=expansion_callback,
                abort_check=abort_check,
                timeline=timeline,
                tactic_sink=tactic_sink,
                inject_grind=self.expand_inject_grind,
            )
            if not game.root.is_solved and self.post_search_grind:
                close_leaves_with_grind(
                    game.root, timeout=self.config.verify_timeout
                )
            if game.root.is_solved:
                verify_err = verify_node(game.root, timeout=self.config.verify_timeout)
                if verify_err:
                    logger.warning(
                        f"FAILED: Verification failed after {game.num_iterations} iterations: '{verify_err}'\nTheorem: '{theorem.source}'\nProof tree:\n{game.root.pp_tree()}"
                    )
                    game.root.is_solved = False
                    return game

                if self.simplify_proofs:
                    game.unsimplified_root = game.root.clone()
                    prune_redundant_nodes(game.root)
                compute_value_target(game.root)

                if self.simplify_proofs:
                    verify_err = verify_node(game.root, timeout=self.config.verify_timeout)
                    if verify_err:
                        logger.warning(
                            f"FAILED: Post-prune verification failed after {game.num_iterations} iterations: '{verify_err}'\nTheorem: '{theorem.source}'\nProof tree:\n{game.root.pp_tree()}"
                        )
                        game.root.is_solved = False
                        return game

                # Verify the linearized proof compiles correctly
                tactics = linearize_proof(game.root)
                proof_source = construct_proof_source(theorem.source, tactics)
                if not env.is_valid_source(proof_source):
                    logger.warning(
                        f'FAILED: Linearized proof verification failed after {game.num_iterations} iterations:\n"""\n{proof_source}\n"""\n... proof tree:\n{game.root.pp_tree()}\n'
                    )
                    game.root.is_solved = False

            return game

    @staticmethod
    def _get_process_interruptible(
        client: LeanClient,
        abort_check: Callable[[], bool] | None,
        poll_interval: float = 10.0,
        max_wait: float = 300.0,
    ) -> tuple:
        """Get a Lean process, polling abort_check between short blocking calls.

        Uses short server-side timeouts so that abort_check is tested every
        *poll_interval* seconds.  Returns ``(process, reason)`` where reason
        is ``"ok"`` (process is not None), ``"aborted"`` (caller asked to
        stop, typically end of a collect cycle), or ``"timeout"`` (waited
        *max_wait* without success, real pool saturation).
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


class ProverWorker:
    """Long-lived actor pool that parallelises MCTS proof search across
    Lean servers.

    The pool is created once in ``__init__`` and runs until ``close()``.
    Callers drive it with two modes:

    - **Collect** is persistent: ``install_collect()`` is called once at
      startup with the matchmaker, the experience holder, and the search
      config. Actors then sample theorems from the matchmaker, write
      attempts and tactics into the holder, and feed outcomes back to
      the matchmaker. ``collect(target_transitions)`` is just a wait
      barrier on the holder's transition counter.
    - **Evaluate** is a temporary interrupt: ``evaluate()`` drains
      in-flight collect actors, swaps to eval state, runs to completion,
      drains again, and restores collect. Eval bookkeeping (per-call
      results list and theorem iterator) lives on the worker only for
      the duration of the call.

    Actors carry a single abort signal (``_release_event``) used by
    ``evaluate`` to drop in-flight Lean leases at the boundaries; collect
    never sets it. ``_shutdown_event`` is set in :meth:`close` and wakes
    every waiting actor so they can exit.
    """

    def __init__(
        self,
        tactic_model: InferenceBalancer,
        lean_addrs: list[str],
    ):
        self.tactic_model = tactic_model
        self.lean_servers = self._query_lean_servers(lean_addrs)
        self.num_actors = len(self.lean_servers)

        self._lock = threading.Lock()
        self._mode_cv = threading.Condition(self._lock)
        self._mode: Literal["idle", "collect", "eval"] = "idle"
        self._release_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._actors_mid_proof = 0
        self._thread_states: dict[int, str] = {
            i: "idle" for i in range(self.num_actors)
        }

        # Persistent collect state. Set once via install_collect().
        self._matchmaker: Optional[Matchmaker] = None
        self._collect_holder: Optional[CollectExperienceHolder] = None
        self._collect_prover: Optional[Prover] = None
        self._collect_tactic_sink: Optional[
            Callable[[str, list[tuple[str, str, int]]], None]
        ] = None
        self._collect_theorem_counter = 0
        self._collect_disable_solvers: bool = False

        # Per-call eval state. Set/cleared inside evaluate().
        self._eval_get_theorem: Optional[
            Callable[[], Optional[tuple[str, BenchTheorem, int]]]
        ] = None
        self._eval_on_result: Optional[
            Callable[[str, BenchTheorem, int, Optional[Game], Optional[str]], None]
        ] = None
        self._eval_prover: Optional[Prover] = None
        self._eval_tactic_sink: Optional[
            Callable[[str, list[tuple[str, str, int]]], None]
        ] = None

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
                raise ConnectionError(
                    f"Lean server {addr} reports 0 available processes"
                )
            logger.info(f"Lean server {host}:{port}: {max_procs} processes")
            servers.extend([(host, port)] * max_procs)
        return servers

    def close(self):
        """Signal shutdown and join actor threads. Called at process teardown."""
        self._shutdown_event.set()
        with self._mode_cv:
            self._release_event.set()
            self._mode = "idle"
            self._mode_cv.notify_all()
        deadline = time.time() + 60.0
        for t in self._threads:
            t.join(timeout=max(0.0, deadline - time.time()))
        alive = sum(1 for t in self._threads if t.is_alive())
        if alive:
            logger.warning(
                f"{alive}/{len(self._threads)} actor threads still alive after close"
            )
            faulthandler.dump_traceback()

    def install_collect(
        self,
        matchmaker: Matchmaker,
        holder: CollectExperienceHolder,
        search_config: SearchConfig,
        simplify_proofs: bool = True,
        disable_solvers: bool = False,
    ) -> None:
        """Configure persistent collect mode. Must be called once before
        the first :meth:`collect`. Actors begin sampling from the matchmaker
        and writing into ``holder`` immediately; ``collect()`` is just a
        wait-for-target barrier on top of that.

        ``disable_solvers``: when True, the collect Prover runs ``grind`` on
        each unexpanded OR leaf after MCTS exhausts its budget without a
        proof; successful grinds are kept in the proof tree (and verified)
        but filtered out of the replay buffer.
        """
        prover = Prover(
            search_config,
            self.tactic_model,
            simplify_proofs=simplify_proofs,
            post_search_grind=disable_solvers,
        )
        with self._mode_cv:
            assert self._matchmaker is None, "install_collect called twice"
            self._matchmaker = matchmaker
            self._collect_holder = holder
            self._collect_prover = prover
            self._collect_tactic_sink = holder.record_tactic
            self._collect_disable_solvers = disable_solvers
            self._mode = "collect"
            self._release_event.clear()
            self._mode_cv.notify_all()

    @torch.no_grad()
    def collect(self, target_transitions: int) -> int:
        """Wait until the installed holder accumulates ``target_transitions``
        new transitions beyond its current count. Returns the delta.

        Collect mode runs continuously between calls (actors keep proving
        during train), so this is just a barrier; actor results land in
        whichever inner experience the holder has at report time."""
        assert self._collect_holder is not None, (
            "collect() before install_collect()"
        )
        monitor = get_monitor()
        holder = self._collect_holder

        baseline = holder.num_transitions()
        target_total = baseline + target_transitions
        logger.info(
            f"Starting collection with {self.num_actors} actors, target=+{target_transitions} transitions (from {baseline})"
        )

        loop_count = 0
        while not self._shutdown_event.is_set():
            now = holder.num_transitions()
            if now >= target_total:
                break
            if monitor is not None:
                states = self._snapshot_thread_states()
                for i, state in enumerate(states):
                    monitor.update_local_actor(i, state=state)
            loop_count += 1
            if loop_count % 100 == 0:
                logger.info(
                    f"Progress: {now - baseline}/{target_transitions} transitions"
                )
            time.sleep(0.1)

        if monitor is not None:
            monitor.clear_local_actors()

        delta = holder.num_transitions() - baseline
        logger.info(f"Collection complete: {delta} new transitions")
        return delta

    def pause_actors(self) -> None:
        """Park all actors at the mode condvar and drain in-flight proofs.

        Used to free the GIL on the master rank before phases where the
        main thread needs uncontested model access (policy eval, training).
        Without this, the 126 actor threads keep POST'ing to the inference
        balancer; the resulting GIL pressure can starve the main thread's
        forward pass long enough for the NCCL watchdog to fire on workers
        waiting at the next collective."""
        self._park_and_drain(target_label="pause")

    def resume_actors(self) -> None:
        """Restart actors in collect mode. Required after :meth:`pause_actors`
        and after :meth:`evaluate` (which always returns in idle), if the
        caller wants collect to resume. No-op if collect was never installed."""
        with self._mode_cv:
            self._mode = "collect" if self._matchmaker is not None else "idle"
            self._release_event.clear()
            self._mode_cv.notify_all()

    @torch.no_grad()
    def evaluate(
        self,
        theorems: list[BenchTheorem],
        dataset_name: str,
        num_simulations: int,
        search_config: SearchConfig,
        progress_callback: Callable[[int, int, int, int], None] | None = None,
        tactic_sink: Callable[[str, list[tuple[str, str, int]]], None] | None = None,
        disable_solvers: bool = False,
    ) -> dict:
        """Evaluate theorems using MCTS. Drains in-flight collect actors at
        entry and exit so eval state cannot leak into collect (or vice
        versa) and Lean leases held by mid-proof collect actors are
        returned before eval starts.

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
            return (tid, theorem, num_simulations)

        def on_result(theorem_id, theorem: BenchTheorem, num_simulations, game, error):
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
                results.append(
                    {
                        "theorem": theorem.source,
                        "dataset": theorem.dataset,
                        "id": theorem.id,
                        "is_solved": is_solved,
                        "error": error,
                        "proof_tree": proof_tree,
                        "unsimplified_proof_tree": unsimplified_proof_tree,
                        "linearized_proof": linearized,
                        "num_iterations": num_iterations,
                    }
                )

                n = len(results)
                solved = sum(1 for r in results if r["is_solved"])
                errors = sum(1 for r in results if r["error"])

                if monitor:
                    monitor.update_eval_progress(
                        current=n, solved=solved, errors=errors
                    )

                if progress_callback:
                    with index_lock:
                        started = index[0]
                    progress_callback(started, n, solved, errors)

        eval_prover = Prover(
            search_config,
            self.tactic_model,
            expand_inject_grind=disable_solvers,
        )
        self._switch_to_eval(get_theorem, on_result, eval_prover, tactic_sink)
        try:
            while not self._shutdown_event.is_set():
                with results_lock:
                    if len(results) >= len(theorems):
                        break
                time.sleep(0.1)
        finally:
            self._switch_back_from_eval()

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

    def _switch_to_eval(
        self,
        get_theorem_fn,
        on_result_fn,
        eval_prover: "Prover",
        tactic_sink: Optional[Callable[[str, list[tuple[str, str, int]]], None]],
    ) -> None:
        """Park actors, drain in-flight proofs, then install eval state."""
        self._park_and_drain(target_label="eval")
        with self._mode_cv:
            self._eval_get_theorem = get_theorem_fn
            self._eval_on_result = on_result_fn
            self._eval_prover = eval_prover
            self._eval_tactic_sink = tactic_sink
            self._mode = "eval"
            self._release_event.clear()
            self._mode_cv.notify_all()

    def _switch_back_from_eval(self) -> None:
        """Park actors, drain in-flight eval proofs, and clear eval state.

        Always returns mode to "idle". Caller is responsible for calling
        :meth:`resume_actors` if they want collect to restart; this keeps
        back-to-back evaluate() calls from transiently flipping to collect
        between them, and matches the explicit pause/resume contract used
        by the rl loop."""
        self._park_and_drain(target_label="post-eval")
        with self._mode_cv:
            self._eval_get_theorem = None
            self._eval_on_result = None
            self._eval_prover = None
            self._eval_tactic_sink = None
            self._mode = "idle"
            self._release_event.clear()
            self._mode_cv.notify_all()

    def _park_and_drain(self, *, target_label: str) -> None:
        """Flip mode to "idle" so actors looping back will park on the cv,
        signal in-flight proofs to abort via ``_release_event``, then poll
        ``_actors_mid_proof`` until it reaches zero (or 60s timeout).

        Without the "idle" flip the drain races against actors: each
        aborted proof decrements the counter, but the actor immediately
        re-enters the cv block, sees the still-active mode, takes a
        fresh theorem, and increments the counter again. With many
        actors this tight oscillation keeps the counter above zero
        indefinitely. Parking the mode first lets the counter actually
        reach zero."""
        with self._mode_cv:
            self._mode = "idle"
            self._release_event.set()
        deadline = time.time() + 60.0
        while True:
            with self._lock:
                mid = self._actors_mid_proof
            if mid == 0:
                break
            if time.time() > deadline:
                logger.warning(
                    f"drain ({target_label}) timed out with {mid} actors still mid-proof"
                )
                break
            time.sleep(0.05)

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
            # Snapshot mode and the mode-specific dispatch state under the
            # condition lock. The increment of ``_actors_mid_proof`` happens
            # later (only once we have a theorem in hand) so the drain in
            # ``_drain_actors`` does not have to wait for actors that ended
            # up idling on a None get_theorem.
            with self._mode_cv:
                while self._mode == "idle" and not self._shutdown_event.is_set():
                    self._mode_cv.wait()
                if self._shutdown_event.is_set():
                    break
                mode = self._mode
                if mode == "collect":
                    matchmaker = self._matchmaker
                    holder = self._collect_holder
                    prover = self._collect_prover
                    tactic_sink = self._collect_tactic_sink
                    self._collect_theorem_counter += 1
                    collect_theorem_id = (
                        f"collect_{self._collect_theorem_counter}"
                    )
                    eval_get_theorem = None
                    eval_on_result = None
                else:  # "eval"
                    matchmaker = None
                    holder = None
                    prover = self._eval_prover
                    tactic_sink = self._eval_tactic_sink
                    collect_theorem_id = None
                    eval_get_theorem = self._eval_get_theorem
                    eval_on_result = self._eval_on_result

            try:
                if mode == "collect":
                    theorem, num_simulations = matchmaker.next_assignment()
                    theorem_id = collect_theorem_id
                else:
                    theorem_data = eval_get_theorem()
                    if theorem_data is None:
                        self._set_thread_state(actor_id, "idle")
                        time.sleep(0.5)
                        continue
                    theorem_id, theorem, num_simulations = theorem_data
            except Exception as e:
                logger.exception(
                    f"[Actor {actor_id}] get_theorem failed: {e}"
                )
                self._set_thread_state(actor_id, "idle")
                time.sleep(0.5)
                continue

            self._set_thread_state(actor_id, "running")

            game = None
            error = None
            skip_report = False
            interrupted = False
            timeline = TimelineRecorder()

            with self._lock:
                self._actors_mid_proof += 1
            try:
                for attempt in range(max_retries):
                    try:
                        game = prover.prove(
                            client,
                            theorem,
                            num_simulations=num_simulations,
                            abort_check=self._release_event.is_set,
                            timeline=timeline,
                            tactic_sink=tactic_sink,
                        )
                        consecutive_errors = 0
                        break
                    except LeanPoolTimeoutError as e:
                        # Pool saturation: don't retry (each attempt would
                        # block another 300s). Record as an error so the
                        # theorem can be retried later via --continue.
                        error = str(e)
                        consecutive_errors += 1
                        logger.warning(
                            f"[Actor {actor_id}] {e} (lean={lean_address}:{lean_port})"
                        )
                        log_actionable_error(
                            "Prover",
                            str(e),
                            actor=actor_id,
                            lean=f"{lean_address}:{lean_port}",
                        )
                        break
                    except ProofInitError as e:
                        # proof_from_sorry rejected the theorem before
                        # search ran. Record as an error (not silent
                        # "unsolved") so --continue can retry it.
                        error = str(e)
                        consecutive_errors += 1
                        short_err = str(e).split("\n", 1)[0]
                        logger.warning(
                            f"[Actor {actor_id}] {short_err} (lean={lean_address}:{lean_port})"
                        )
                        log_actionable_error(
                            "Prover",
                            short_err,
                            actor=actor_id,
                            lean=f"{lean_address}:{lean_port}",
                        )
                        break
                    except (
                        ConnectionError,
                        LeanProcessException,
                        RemoteException,
                        TimeoutError,
                    ) as e:
                        if attempt < max_retries - 1:
                            self._set_thread_state(actor_id, "retry")
                            short_err = str(e).split("\n", 1)[0]
                            logger.warning(
                                f"[Actor {actor_id}] Connection error (attempt {attempt + 1}/{max_retries}): '{short_err}', reconnecting..."
                            )
                            time.sleep(1.0 * (attempt + 1))
                        else:
                            error = str(e)
                            consecutive_errors += 1
                            logger.error(
                                f"[Actor {actor_id}] Error (lean={lean_address}:{lean_port}): {e}"
                            )
                            log_actionable_error(
                                "Prover",
                                str(e),
                                actor=actor_id,
                                lean=f"{lean_address}:{lean_port}",
                                retries_exhausted=True,
                            )
                    except MCTSAbortedError:
                        skip_report = True
                        interrupted = True
                        break
                    except Exception as e:
                        error = str(e)
                        consecutive_errors += 1
                        logger.exception(
                            f"[Actor {actor_id}] Error (lean={lean_address}:{lean_port}): {e}"
                        )
                        log_actionable_error(
                            "Prover",
                            str(e),
                            actor=actor_id,
                            lean=f"{lean_address}:{lean_port}",
                        )
                        break

                # Route by the mode captured at iteration start. Cross-mode
                # boundaries are handled by ``_drain_actors``: by the time
                # ``_mode`` flips, ``_actors_mid_proof`` is zero, so no
                # in-flight actor can route to the wrong sink.
                if not skip_report:
                    is_solved = bool(game and game.root and game.root.is_solved)
                    logger.debug(
                        f"Actor {actor_id}: {theorem_id} {'solved' if is_solved else 'unsolved'} in {game.num_iterations if game else 0} iters"
                    )
                    if is_solved:
                        logger.debug(
                            f"Actor {actor_id}: proof tree for {theorem_id}:\n{game.root.pp_tree()}"
                        )
                    if error is not None:
                        self._set_thread_state(actor_id, "error")
                    if mode == "collect":
                        self._report_collect(
                            matchmaker, holder, theorem, num_simulations, game, error
                        )
                    else:
                        eval_on_result(
                            theorem_id, theorem, num_simulations, game, error
                        )
            finally:
                with self._lock:
                    self._actors_mid_proof -= 1

            # Always flush the timeline, including for aborted proofs:
            # the LLM/Lean work they did belongs on the profiler, and
            # the outcome marker classifies what the "productive only"
            # toggle hides.
            _flush_timeline(
                actor_id, timeline, game=game, error=error, interrupted=interrupted
            )

            if consecutive_errors >= max_consecutive_errors:
                logger.warning(
                    f"[Actor {actor_id}] {consecutive_errors} consecutive errors; backing off 60s"
                )
                time.sleep(60.0)
                consecutive_errors = 0

            self._set_thread_state(actor_id, "idle")

        self._set_thread_state(actor_id, "idle")

    def _report_collect(
        self,
        matchmaker: Matchmaker,
        holder: CollectExperienceHolder,
        theorem: BenchTheorem,
        num_simulations: int,
        game: "Game | None",
        error: Optional[str],
    ) -> None:
        outcome = self._derive_outcome(game, error)
        proof_size = (
            len(linearize_proof(game.root)) if outcome == "proven" else None
        )
        transitions_before = holder.num_transitions()
        holder.record_attempt(
            theorem,
            outcome,
            num_simulations,
            game,
            error,
            proof_size,
            filter_grind=self._collect_disable_solvers,
        )
        matchmaker.send_result(theorem, outcome, proof_size)
        monitor = get_monitor()
        if monitor is not None and outcome != "error":
            monitor.record_proof_attempt(
                successful=outcome == "proven",
                transitions=holder.num_transitions() - transitions_before,
            )

    @staticmethod
    def _derive_outcome(game, error) -> Outcome:
        if error is not None:
            return "error"
        if game is not None and game.root is not None and game.root.is_solved:
            return "proven"
        return "unproven"
