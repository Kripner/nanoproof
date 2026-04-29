"""
Experience collection: replay buffer, matchmaker, and proof-tree-to-transitions pipeline.

This module owns:
- Matchmaker / MatchmakerConfig: per-theorem stat tracking, weighted problem
  selection, and adaptive simulation budgets (AlphaProof-style). Stats are
  fully derived from on-disk theorems.jsonl shards so a resumed run
  reconstructs identical state via :meth:`Matchmaker.reconstruct_from_run_dir`.
- ReplayBuffer: DDP-aware buffer of (context, tactic, value_target) transitions
  sampled during training.
- CollectedExperience: per-phase accumulator of every prove attempt (proven /
  unproven / error) plus generated tactics. ``record_tactic`` is wired up as
  the per-job ``tactic_sink`` passed to ``prover.collect()`` /
  ``prover.evaluate()``; results are written to disk by
  :meth:`CollectedExperience.save` (``theorems.jsonl`` /
  ``generated_tactics.jsonl``).
- compute_value_target: assigns regression targets to nodes of a solved proof tree
- extract_transitions: walks a solved proof tree and yields training transitions
- prune_redundant_nodes / prune_redundant_node: tree-editing pass that removes
  redundant OR nodes before transition extraction

Tree-walking helpers depend on Node / Player / execute_tree from search.py.
"""

import glob
import json
import logging
import os
import random
import threading
from dataclasses import dataclass, asdict, field
from typing import Literal

import torch.distributed as dist

from nanoproof.common import get_dist_info, Player, GLOBAL_CONFIG, info0

logger = logging.getLogger(__name__)
from nanoproof.data.bench.common import BenchTheorem
from nanoproof.data.rl import deepseek_prover, leanworkbook, numinamath
from nanoproof.search import Node, execute_tree
from nanoproof.tokenizer import get_tokenizer


# -----------------------------------------------------------------------------
# On-disk layout helpers (one place that knows about the step_/evals/ dirs)
# -----------------------------------------------------------------------------

_STEP_PREFIX = "step_"
THEOREMS_FILENAME = "theorems.jsonl"

Outcome = Literal["proven", "unproven", "error"]


def step_dir(run_dir: str, step: int) -> str:
    """Path to the per-step collection directory under ``run_dir``."""
    return os.path.join(run_dir, f"{_STEP_PREFIX}{step:05d}")


def eval_dir(run_dir: str, step: int) -> str:
    """Path to the per-step eval directory under ``run_dir``."""
    return os.path.join(run_dir, "evals", f"{step:05d}")


def list_step_shards(run_dir: str) -> list[tuple[int, str]]:
    """Return ``[(step, theorems.jsonl_path), ...]`` sorted by step."""
    shards: list[tuple[int, str]] = []
    for shard_path in glob.glob(
        os.path.join(run_dir, f"{_STEP_PREFIX}*", THEOREMS_FILENAME)
    ):
        shard_step = int(
            os.path.basename(os.path.dirname(shard_path))[len(_STEP_PREFIX) :]
        )
        shards.append((shard_step, shard_path))
    shards.sort()
    return shards


def load_collected_transitions(run_dir: str) -> list[tuple[str, str, float]]:
    """Concatenate transitions from every ``step_<s>/theorems.jsonl`` in
    ``run_dir``, preserving step order across shards and file order within a
    shard. Skips entries with no transitions (unproven / errored). Callers
    apply FIFO truncation or length filtering themselves.
    """
    transitions: list[tuple[str, str, float]] = []
    for _, shard_path in list_step_shards(run_dir):
        with open(shard_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for t in obj.get("transitions", []):
                    transitions.append((t[0], t[1], t[2]))
    return transitions


# -----------------------------------------------------------------------------
# Matchmaker
# -----------------------------------------------------------------------------


_DATASET_LOADERS = {
    "leanworkbook": leanworkbook.list_theorems,
    "deepseek_prover": deepseek_prover.list_theorems,
    "numinamath": numinamath.list_theorems,
}


def list_available_datasets() -> list[str]:
    return list(_DATASET_LOADERS.keys())


@dataclass(frozen=True)
class MatchmakerConfig:
    """AlphaProof-style matchmaker hyperparameters (Sup. Table 7, prove-only).

    The "last N" window for weight and budget computation walks only
    proven/unproven outcomes; errors are filtered out -- with one exception
    handled inside :class:`Matchmaker`: two consecutive raw error outcomes
    drop the theorem permanently.

    All fields are required; use :meth:`defaults` and
    :func:`nanoproof.common.add_dataclass_args` to expose them as CLI flags.
    """

    # window size (in decided proven/unproven outcomes) used for weight tier
    # classification and per-attempt simulation budget
    trust_count: int
    # number of recent consecutive proven outcomes that demote a theorem to
    # the fully-proved (low) weight tier
    trust_count_proved: int
    # sampling weight for theorems still in the interesting tier
    # (unseen, under-trusted, or recently mixed)
    weight_interesting: float
    # sampling weight for theorems with no proofs in the trust window
    # (look unprovable for now)
    weight_undecided: float
    # sampling weight for theorems consistently proven over the last
    # trust_count_proved attempts
    weight_fully_proved: float
    # baseline per-attempt simulation budget before failure-based scaling
    base_simulations: int
    # simulation budget multiplier applied per unproven outcome in the trust window
    failure_multiplier: float
    # hard upper bound on the per-attempt simulation budget after failure scaling
    cap_simulations: int

    @classmethod
    def defaults(cls) -> dict:
        return {
            "trust_count": 4,
            "trust_count_proved": 6,
            "weight_interesting": 1.0,
            "weight_undecided": 0.1,
            "weight_fully_proved": 1e-3,
            "base_simulations": 64,
            "failure_multiplier": 1.5,
            "cap_simulations": 1024,
        }


@dataclass
class TheoremStats:
    """Per-theorem attempt log used by the matchmaker. Public so the web UI
    can replay the same logic over an on-disk attempt history.

    ``history`` is the full arrival-order log of ``(outcome, proof_size)``
    pairs. ``proof_size`` is the number of tactics in the linearized proof
    when ``outcome == "proven"``, otherwise ``None``. Weight + budget
    computation filters out errors on the fly.
    """

    history: list[tuple[Outcome, int | None]] = field(default_factory=list)

    def update(self, outcome: Outcome, proof_size: int | None = None) -> None:
        self.history.append((outcome, proof_size))

    def weight(self, config: MatchmakerConfig) -> float:
        if (
            len(self.history) >= 2
            and self.history[-1][0] == "error"
            and self.history[-2][0] == "error"
        ):
            return 0.0
        if any(o == "proven" and ps == 1 for o, ps in self.history):
            return config.weight_fully_proved
        decided = [o for o, _ in self.history if o != "error"]
        if not decided:
            return config.weight_interesting
        if len(decided) < config.trust_count:
            return config.weight_interesting
        proved = any(o == "proven" for o in decided)
        if not proved:
            return config.weight_undecided
        recent = decided[-config.trust_count_proved :]
        if len(recent) >= config.trust_count_proved and all(o == "proven" for o in recent):
            return config.weight_fully_proved
        return config.weight_interesting

    def num_simulations(self, config: MatchmakerConfig) -> int:
        decided = [o for o, _ in self.history if o != "error"]
        recent = decided[-config.trust_count :]
        num_failures = sum(1 for o in recent if o == "unproven")
        budget = config.base_simulations * (config.failure_multiplier**num_failures)
        return min(config.cap_simulations, int(budget))


class Matchmaker:
    """Tracks per-theorem outcomes and assigns (theorem, num_simulations)
    pairs to actors. Thread-safe.

    Replaces the old per-dataset uniform sampler: weighting is flat across
    all loaded theorems so interestingness alone drives selection.
    """

    def __init__(
        self,
        datasets: list[str],
        lean_version: str | None,
        config: MatchmakerConfig,
        seed: int,
    ):
        self.config = config
        self.datasets = list(datasets)
        self.theorems: list[BenchTheorem] = []
        for name in self.datasets:
            assert name in _DATASET_LOADERS, (
                f"Unknown dataset: {name!r} (known: {list(_DATASET_LOADERS)})"
            )
            theorems = _DATASET_LOADERS[name](split="train", lean_version=lean_version)
            info0(logger, f"Loaded {len(theorems)} theorems from {name}")
            self.theorems.extend(theorems)
        self._index: dict[tuple[str, str], int] = {
            (t.dataset, t.id): i for i, t in enumerate(self.theorems)
        }
        assert len(self._index) == len(self.theorems), (
            "Matchmaker: duplicate (dataset, id) across loaded theorems"
        )
        self._stats: list[TheoremStats] = [
            TheoremStats() for _ in range(len(self.theorems))
        ]
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def next_assignment(self) -> tuple[BenchTheorem, int]:
        """Return ``(theorem, num_simulations)`` for the next attempt."""
        with self._lock:
            weights = [s.weight(self.config) for s in self._stats]
            total = sum(weights)
            assert total > 0, "Matchmaker: all theorems have weight 0 (every theorem dropped or fully proved at minimum weight 0)"
            (idx,) = self._rng.choices(range(len(self.theorems)), weights=weights, k=1)
            theorem = self.theorems[idx]
            num_sims = self._stats[idx].num_simulations(self.config)
            return theorem, num_sims

    def send_result(
        self,
        theorem: BenchTheorem,
        outcome: Outcome,
        proof_size: int | None = None,
    ) -> None:
        key = (theorem.dataset, theorem.id)
        with self._lock:
            idx = self._index.get(key)
            assert idx is not None, (
                f"Matchmaker.send_result: unknown theorem {key!r}"
            )
            self._stats[idx].update(outcome, proof_size)

    def weight_for(self, theorem: BenchTheorem) -> float:
        with self._lock:
            idx = self._index[(theorem.dataset, theorem.id)]
            return self._stats[idx].weight(self.config)

    def proven_counts_by_dataset(self) -> dict[str, int]:
        """Per-dataset count of theorems with at least one ``proven`` outcome
        in their history. Datasets with zero proven theorems still appear
        (mapped to 0) so downstream metric series stay stable."""
        counts: dict[str, int] = {name: 0 for name in self.datasets}
        with self._lock:
            for theorem, stats in zip(self.theorems, self._stats):
                if any(o == "proven" for o, _ in stats.history):
                    counts[theorem.dataset] += 1
        return counts

    def reconstruct_from_run_dir(self, run_dir: str) -> int:
        """Replay every saved attempt under ``run_dir/step_*/theorems.jsonl``
        through :meth:`send_result`. Returns the number of attempts replayed.

        The run's ``args.json`` must declare the same ``--datasets`` as this
        matchmaker; otherwise per-theorem stats would silently mix across
        dataset configurations.
        """
        args_path = os.path.join(run_dir, "args.json")
        assert os.path.exists(args_path), (
            f"Matchmaker.reconstruct_from_run_dir: missing {args_path}"
        )
        with open(args_path, "r") as f:
            prior_args = json.load(f)
        prior_datasets = prior_args.get("datasets")
        assert prior_datasets is not None, (
            f"Matchmaker.reconstruct_from_run_dir: {args_path} has no 'datasets' field"
        )
        assert sorted(prior_datasets) == sorted(self.datasets), (
            f"Matchmaker.reconstruct_from_run_dir: datasets mismatch -- "
            f"resumed run had {sorted(prior_datasets)}, current run has {sorted(self.datasets)}"
        )

        replayed = 0
        for _, shard_path in list_step_shards(run_dir):
            with open(shard_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    key = (obj["dataset"], obj["id"])
                    idx = self._index.get(key)
                    if idx is None:
                        # Theorem present in the prior run but not in the
                        # current loaded list (e.g. whitelist drift). Skip.
                        continue
                    self._stats[idx].update(obj["outcome"], obj.get("proof_size"))
                    replayed += 1
        info0(logger, f"Matchmaker: replayed {replayed} attempts from {run_dir}")
        return replayed


class ReplayBuffer:
    """DDP-aware FIFO replay buffer of (context, tactic, value_target) transitions.

    Rank 0 is the only rank that collects new transitions. At the end of each
    collection phase, :meth:`extend_and_sync` merges in the new transitions
    (rank 0 passes the real list, other ranks pass ``[]``), truncates to
    ``window_size``, and broadcasts the result so all ranks sample from the
    same buffer during training.
    """

    def __init__(self, window_size: int, seed: int):
        self.window_size = window_size
        self.buffer: list[tuple[str, str, float]] = []
        self.rng = random.Random(seed)

    def extend_and_sync(self, new_transitions: list[tuple[str, str, float]]) -> None:
        """Append newly-collected transitions (rank 0 only), FIFO-truncate, broadcast."""
        ddp, _, _, _ = get_dist_info()

        self.buffer.extend(new_transitions)
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size :]

        if ddp:
            buffer_list = [self.buffer]
            dist.broadcast_object_list(buffer_list, src=0)
            self.buffer = buffer_list[0]

    def load_from(self, run_dir: str) -> None:
        """Repopulate ``self.buffer`` from every collection shard in ``run_dir``.

        Each rank reads the same files independently, so no broadcast is
        needed. FIFO-truncates to ``window_size`` to match the eviction
        policy of the live buffer.
        """
        info0(logger, f"Loading replay buffer from {run_dir}")
        transitions = load_collected_transitions(run_dir)
        if len(transitions) > self.window_size:
            transitions = transitions[-self.window_size :]
        self.buffer = transitions
        info0(logger, f"Loaded {len(self.buffer)} transitions from {run_dir}")

    def sample_transition(self) -> tuple[str, str, float]:
        return self.rng.choice(self.buffer)


# -----------------------------------------------------------------------------
# CollectedExperience: per-phase artifacts (proofs + generated tactics)
# -----------------------------------------------------------------------------


@dataclass
class TheoremAttempt:
    """One prove attempt from a collection phase.

    Always recorded -- proven, unproven, and errored attempts alike --
    because the matchmaker's stats are reconstructed from these records on
    resume. The tree fields are populated only for ``outcome == "proven"``.
    """

    dataset: str
    id: str
    theorem: str  # BenchTheorem.source
    num_simulations: int  # budget allocated by matchmaker
    num_iterations: int  # MCTS iterations actually run (0 if error)
    outcome: Outcome
    error: str | None
    full_tree: dict | None  # pre-prune tree (game.unsimplified_root.serialize())
    simplified_tree: dict | None  # post-prune tree (game.root.serialize())
    transitions: list[tuple[str, str, float]]
    proof_size: int | None  # number of tactics in the linearized proof; None unless proven


class CollectedExperience:
    """Per-phase accumulator used during collection and evaluation.

    Attempts and tactics are populated by worker threads during the phase:
    :meth:`record_attempt` is called once per attempt (proven/unproven/error)
    from the prover callback; :meth:`record_tactic` is wired in as the
    job's per-expansion ``tactic_sink`` and fires once per MCTS expansion
    with the full batch of (tactic, status) pairs generated for that
    state. At the end of the phase the RL loop pulls
    :meth:`transitions` into :class:`ReplayBuffer` and calls :meth:`save`
    to write ``theorems.jsonl`` and ``generated_tactics.jsonl``.
    """

    TRAIN_SUBSAMPLE_K = 100

    def __init__(self):
        self.attempts: list[TheoremAttempt] = []
        self.tactics: list[dict] = []
        self.train_samples: list[dict] = []
        self._train_seen: int = 0
        self._train_rng = random.Random(0)
        self._lock = threading.Lock()

    def record_attempt(
        self,
        theorem: BenchTheorem,
        outcome: Outcome,
        num_simulations: int,
        game,
        error: str | None,
        proof_size: int | None = None,
    ) -> None:
        """Snapshot one prove attempt and append it to ``self.attempts``."""
        if outcome == "proven":
            assert game is not None and game.root is not None
            transitions = extract_transitions(game.root)
            full_tree = (
                game.unsimplified_root.serialize()
                if getattr(game, "unsimplified_root", None)
                else None
            )
            simplified_tree = game.root.serialize()
        else:
            transitions = []
            full_tree = None
            simplified_tree = None
        num_iterations = game.num_iterations if game is not None else 0
        attempt = TheoremAttempt(
            dataset=theorem.dataset,
            id=theorem.id,
            theorem=theorem.source,
            num_simulations=num_simulations,
            num_iterations=num_iterations,
            outcome=outcome,
            error=error,
            full_tree=full_tree,
            simplified_tree=simplified_tree,
            transitions=transitions,
            proof_size=proof_size,
        )
        with self._lock:
            self.attempts.append(attempt)

    def record_tactic(
        self, state: str, tactics_with_status: list[tuple[str, str, int]]
    ) -> None:
        """Buffer one node-expansion's worth of tactic attempts as a single
        ``{state, tactics: [{tactic, status, count}, ...]}`` entry. ``count``
        is how many times the model sampled this tactic before dedup.
        Lock-free (``list.append`` is atomic under the GIL)."""
        self.tactics.append(
            {
                "state": state.replace("\n", "\\n"),
                "tactics": [
                    {"tactic": tactic, "status": status, "count": count}
                    for tactic, status, count in tactics_with_status
                ],
            }
        )

    def record_train_samples(
        self, inputs, targets, per_token_loss, sources: list
    ) -> None:
        """Reservoir-sample rows of one training micro-batch into ``self.train_samples``.

        ``inputs`` / ``targets`` / ``per_token_loss`` are all shape ``(B, T)``
        (with -1 in ``targets`` and 0 in the loss at masked positions).
        ``sources`` is a length-``B`` list of per-row source tags (``"rl"`` /
        ``"sft"`` / ``None``). Call per micro-step; reservoir sampling (Vitter
        Algorithm R) keeps the subsample size bounded at
        :attr:`TRAIN_SUBSAMPLE_K` regardless of total micro-batches.
        """
        tokenizer = get_tokenizer()
        pad_token_id = tokenizer.encode_special("<|pad|>")
        value_delim_tok = tokenizer.encode_special("<|value|>")

        inputs_cpu = inputs.detach().cpu()
        targets_cpu = targets.detach().cpu()
        loss_cpu = per_token_loss.detach().float().cpu()

        B = inputs_cpu.size(0)
        for b in range(B):
            real_len = int((inputs_cpu[b] != pad_token_id).sum().item())
            if real_len == 0:
                continue
            ids = inputs_cpu[b, :real_len].tolist()
            tgts = targets_cpu[b, :real_len].tolist()
            losses_row = loss_cpu[b, :real_len].tolist()
            # losses[t] aligns with tokens[t] ("loss of predicting this token")
            # by shifting the upstream per-token-loss (which is indexed by the
            # predictor position, not the predicted position). losses[0] is
            # always None (BOS is never a target); losses[t>=1] is the loss
            # computed when predicting tokens[t] from tokens[:t], gated on
            # supervision via tgts[t-1] >= 0.
            shifted_losses: list[float | None] = [None]
            for t in range(1, real_len):
                shifted_losses.append(losses_row[t - 1] if tgts[t - 1] >= 0 else None)
            tokens = [tokenizer.id_to_token(i) for i in ids]
            is_value = value_delim_tok in ids
            sample = {
                "source": sources[b] if b < len(sources) else None,
                "is_value": is_value,
                "tokens": tokens,
                "losses": shifted_losses,
            }
            self._train_reservoir_offer(sample)

    def _train_reservoir_offer(self, sample: dict) -> None:
        k = self.TRAIN_SUBSAMPLE_K
        if len(self.train_samples) < k:
            self.train_samples.append(sample)
        else:
            j = self._train_rng.randrange(self._train_seen + 1)
            if j < k:
                self.train_samples[j] = sample
        self._train_seen += 1

    def num_transitions(self) -> int:
        with self._lock:
            return sum(len(a.transitions) for a in self.attempts)

    def transitions(self) -> list[tuple[str, str, float]]:
        """All transitions across all proven attempts, filtered by the global
        state/tactic length limits."""
        with self._lock:
            raw = [t for a in self.attempts for t in a.transitions]
        out: list[tuple[str, str, float]] = []
        for context, tactic, value_target in raw:
            context = context.strip()
            tactic = tactic.strip()
            if (
                len(context) > GLOBAL_CONFIG.state_max_len
                or len(tactic) > GLOBAL_CONFIG.tactic_max_len
            ):
                continue
            assert len(context) != 0, (
                f"Empty context in transition: tactic={tactic}, value_target={value_target}"
            )
            assert len(tactic) != 0, (
                f"Empty tactic in transition: context={context}, value_target={value_target}"
            )
            assert value_target is not None, (
                f"None value_target in transition: context={context}, tactic={tactic}"
            )
            out.append((context, tactic, value_target))
        return out

    def save(self, phase_dir: str) -> None:
        """Write ``theorems.jsonl``, ``generated_tactics.jsonl`` and
        ``train_subsample.jsonl`` under ``phase_dir``.

        All files are always written (possibly empty) so an absent file is
        an unambiguous signal that the phase did not run.
        """
        os.makedirs(phase_dir, exist_ok=True)
        with open(os.path.join(phase_dir, THEOREMS_FILENAME), "w") as f:
            for a in self.attempts:
                f.write(json.dumps(asdict(a)) + "\n")
        with open(os.path.join(phase_dir, "generated_tactics.jsonl"), "w") as f:
            for t in self.tactics:
                f.write(json.dumps(t) + "\n")
        with open(os.path.join(phase_dir, "train_subsample.jsonl"), "w") as f:
            for s in self.train_samples:
                f.write(json.dumps(s) + "\n")


class CollectExperienceHolder:
    """Persistent, swappable wrapper around a :class:`CollectedExperience`.

    Collect actors record into the *current* inner experience. The master
    rotates the inner at step-save time: :meth:`rotate` atomically swaps in a
    fresh ``CollectedExperience`` and returns the old one for serialization.

    Without this indirection, an actor whose proof crosses a step boundary
    appends to a ``CollectedExperience`` whose ``save()`` already ran, so the
    record is silently lost (while the paired ``Matchmaker.send_result``
    persists, leaving the matchmaker ahead of disk). The holder lock
    serializes record/rotate so a record can never land in an instance that
    has already been returned for save.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inner = CollectedExperience()

    def record_attempt(self, *args, **kwargs) -> None:
        with self._lock:
            self._inner.record_attempt(*args, **kwargs)

    def record_tactic(self, *args, **kwargs) -> None:
        with self._lock:
            self._inner.record_tactic(*args, **kwargs)

    def record_train_samples(self, *args, **kwargs) -> None:
        with self._lock:
            self._inner.record_train_samples(*args, **kwargs)

    def num_transitions(self) -> int:
        with self._lock:
            return self._inner.num_transitions()

    def transitions(self) -> list[tuple[str, str, float]]:
        with self._lock:
            return self._inner.transitions()

    def rotate(self) -> CollectedExperience:
        """Atomically replace the inner with a fresh experience; return the old."""
        new_inner = CollectedExperience()
        with self._lock:
            old = self._inner
            self._inner = new_inner
        return old


def compute_value_target(node: Node) -> float:
    """Computes the actual value for a node, to be used as a target in learning."""
    assert node.is_solved, (
        f"Node is not solved in compute_value_target (is root={node.action is None}, is terminal={node.is_terminal}, to_play={node.to_play})"
    )
    if node.is_terminal:
        node.value_target = 0
        return 0
    elif node.to_play == Player.OR:
        max_child_value = max(
            compute_value_target(child)
            for child in node.children.values()
            if child.is_solved
        )
        value = -1 + max_child_value
        node.value_target = value
        return value
    elif node.to_play == Player.AND:
        value = min(compute_value_target(child) for child in node.children.values())
        node.value_target = value
        return value
    else:
        raise ValueError(f"Unknown to_play: {node.to_play}")


def prune_redundant_nodes(root: Node) -> int:
    pruned_count = 0
    while True:
        pruned = prune_redundant_node(root)
        if not pruned:
            break
        pruned_count += 1
    return pruned_count


# TODO: when executing the tree, we can stop once the states match
def prune_redundant_node(root: Node) -> bool:
    # All solved interior OR nodes that don't directly finish the proof - candidates for pruning.
    # Sorted in BFS order to delete as much as possible early.
    nodes = [
        n
        for n in root.get_tree_nodes()
        if (
            n.is_solved
            and n.to_play == Player.OR
            and not n.is_terminal
            and not any(child.is_terminal for child in n.children.values())
        )
    ]
    for to_consider in nodes:
        solved_actions = [
            a for a in to_consider.children if to_consider.children[a].is_solved
        ]
        assert len(solved_actions) == 1, (
            f"prune_redundant_node: Expected 1 solved action, got {len(solved_actions)}"
        )
        action = solved_actions[0]
        child = to_consider.children[action]
        child_solved_actions = [
            a for a in child.children if child.children[a].is_solved
        ]
        assert child_solved_actions, f"prune_redundant_node: No solved actions in child"
        # TODO: instead of selecting the shortest, execute all of them and filter out the failed ones
        min_len = min(len(str(a)) for a in child_solved_actions)
        shortest_actions = [a for a in child_solved_actions if len(str(a)) == min_len]
        child_solved_action = shortest_actions[0]
        if child.to_play == Player.OR:
            assert len(to_consider.state) == 1, (
                f"prune_redundant_node: Expected 1 branch at OR node, got {len(to_consider.state)}"
            )
            try:
                # Skip the action, execute the subtree without it.
                # TODO: set allow_premature_end=True and then potentially remove unnecessary nodes
                node_to_state = execute_tree(
                    child, to_consider.state[0], allow_premature_end=False
                )
            except AssertionError as e:
                # The tree is not valid anymore.
                continue
            # Found a redundant edge - remove it, update the subtree, and return.
            for n, new_state in node_to_state:
                n.state = new_state
            del to_consider.children[action]
            grandchild = child.children[child_solved_action]
            grandchild.parent = to_consider
            to_consider.children[child_solved_action] = grandchild
            return True
        elif child.to_play == Player.AND:
            pass  # TODO
        else:
            raise AssertionError(
                f"prune_redundant_node: Unknown node type: {child.to_play}"
            )
    return False


def extract_transitions(node: Node) -> list[tuple[str, str, float]]:
    """
    Extract (context, tactic, value_target) transitions from a solved proof tree.

    Walks the solved path, extracting (state, action, value) for each OR node.
    Works with both live LeanProofBranch states and deserialized MockProofBranch states.
    """
    if not node.is_solved:
        return []

    transitions = []
    _extract_transitions_recursive(node, transitions)
    return transitions


def _extract_transitions_recursive(node: Node, transitions: list):
    """Recursively extract transitions from solved paths."""
    # if not node.is_solved:
    #     return

    # Walk down the OR nodes
    while node.to_play == Player.OR and not node.is_terminal:
        assert len(node.state) == 1, (
            f"extract_transitions: Expected 1 branch at OR node, got {len(node.state)}"
        )
        assert node.children, f"extract_transitions: No children at OR node"

        # Find solved actions
        solved_actions = [a for a in node.children if node.children[a].is_solved]
        assert solved_actions, f"extract_transitions: No solved actions at OR node"

        # Pick shortest tactic (more than one terminal node can be solved when expanding)
        # Note: if we ever let the search run even after proof is found, we should here select also based on the sub-tree depth.
        action = min(solved_actions, key=lambda a: len(a))

        # Extract transition: (context, tactic, value_target)
        context = str(node.state[0].state).strip()
        transitions.append((context, action.strip(), node.value_target))

        node = node.children[action]

    # Handle AND nodes (multiple subgoals)
    if node.to_play == Player.AND and node.children:
        for child in node.children.values():
            _extract_transitions_recursive(child, transitions)
