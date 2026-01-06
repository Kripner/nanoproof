import os
from dataclasses import dataclass, field
import enum
import time
import threading
from contextlib import nullcontext
from typing import Self, Protocol, TYPE_CHECKING
import math

# Lightweight imports (fast)
from leantree.repl_adapter.server import LeanClient, LeanProofBranch
from nanoproof.common import pretty_print_tree
from nanoproof.cli import get_monitor, log, log_tactic

# Heavy imports are done lazily inside TacticModel to speed up prover_server startup
# These are only imported when TacticModel.create() is called:
# - torch
# - nanoproof.checkpoints (load_model)
# - nanoproof.engine (Engine)
# - nanoproof.model (Transformer)
# - nanoproof.tokenizer (HuggingFaceTokenizer)

if TYPE_CHECKING:
    # For type hints only - not actually imported at runtime
    import torch
    from nanoproof.model import Transformer
    from nanoproof.tokenizer import HuggingFaceTokenizer
    from nanoproof.engine import Engine

"""
leanserver --project-path ~/troja/nanoproof/leantree_project/ --repl-exe ~/repos/leantree/lean-repl/.lake/build/bin/repl --imports Mathlib FormalConjectures.ForMathlib.Analysis.SpecialFunctions.NthRoot FormalConjectures.Util.Answer --max-processes 2 --address=<PUBLIC_IP> --log-level=DEBUG
"""


@dataclass
class Config:
    # Acting
    num_simulations: int = 50
    num_actors: int = 4
    num_sampled_tactics: int = 6

    # UCB formula
    pb_c_base: int = 3200
    pb_c_init: float = 0.01
    value_discount: float = 0.98
    prior_temperature: float = 1.5

    # Other MCTS parameters
    no_legal_actions_value: float = -40.0

    # Progressive sampling parameters
    ps_c: float = 0.03
    ps_alpha: float = 0.8

    # Value predictions
    num_value_bins: int = 64

    # Training
    training_steps: int = int(500e3)
    batch_size: int = 64
    sequence_length: int = 32
    window_size: int = 20000
    lr: float = 1e-4
    value_weight: float = 0.002

    # Lean server
    server_address: str = "10.10.25.40"
    server_port: int = 8000


class Player(enum.Enum):
    OR = 1
    AND = 2


Action = str | int

State = list[LeanProofBranch]


@dataclass
class Node:
    """Node in the search tree."""
    # Action that was taken to reach this node.
    action: Action | None
    # Prior probability of the node according to the policy.
    prior: float | None
    # State after the action has been applied.
    state: State
    # Per-step reward obtained after applying the action.
    reward: float | None
    # Whether the node is an OR or AND node.
    to_play: Player
    is_solved: bool = False

    visit_count: int = 0
    evaluations: int = 0
    value_sum: float = 0
    children: dict[Action, Self] | None = None

    # Not used in search, but used as a regression target in RL.
    value_target: float = 0

    def expanded(self) -> bool:
        return self.children is not None

    def value(self) -> float:
        if self.visit_count == 0:
            return 0  # TODO: isn't this also weird?
        return self.value_sum / self.visit_count

    def prior_sum(self) -> float:
        return sum(child.prior for child in self.children.values())

    @property
    def is_terminal(self) -> bool:
        return len(self.state) == 0

    def calculate_solved(self) -> bool:
        if self.is_terminal:
            self.is_solved = True
        elif not self.expanded():
            self.is_solved = False
        else:
            if self.to_play == Player.OR:
                self.is_solved = any(child.calculate_solved() for child in self.children.values())
            else:
                self.is_solved = all(child.calculate_solved() for child in self.children.values())
        return self.is_solved

    def pp_tree(self) -> str:
        def get_children(node: Node):
            return node.children.values() if node.children is not None else []

        def get_node_label(node: Node):
            state_str = "\n\n".join(str(branch.state) for branch in node.state) if len(node.state) > 0 else "<empty>"
            type_str = "AND" if node.to_play == Player.AND else "OR"
            solved_str = " (SOLVED)" if node.is_solved else ""
            return f"[{type_str}{solved_str}]\nvis={node.visit_count} evals={node.evaluations} val={node.value():.2f}\n{state_str}"

        def get_edge_label(node: Node):
            if node.action is None:
                return None
            prior_str = f"p={node.prior:.2f}" if node.prior is not None else "p=None"
            reward_str = f"r={node.reward:.2f}" if node.reward is not None else "r=None"
            return f"[{prior_str} {reward_str}] {str(node.action)}"

        return pretty_print_tree(self, get_children, get_node_label, get_edge_label, max_label_len=200,
                                 max_edge_label_len=50)


class Game:
    """A single episode of interaction with the environment."""

    def __init__(self, theorem: str, num_simulations: int | None = None):
        self.theorem = theorem
        # Number of simulations to run.
        self.num_simulations = num_simulations
        self.root: Node = None


@dataclass
class TacticModel:
    # Type hints use strings for lazy imports
    network: "Transformer"
    tokenizer: "HuggingFaceTokenizer"
    engine: "Engine"
    num_samples: int = 6

    def __post_init__(self):
        import torch
        self.rng = torch.Generator(device=self.network.get_device())
        self.rng.manual_seed(0)

    def sample_tactic(self, state: State) -> list[str]:
        import torch
        assert len(state) == 1, \
            f"expected single branch in state when generating tactic, got {len(state)} - choose one goal first"
        device = self.network.get_device()
        assert device.type == "cuda"

        state_str = str(state[0].state).strip()
        tokens = self.tokenizer(state_str + "\n<|tactic|>", prepend=self.tokenizer.get_bos_token_id())
        seed = torch.randint(torch.iinfo(torch.int32).max, (1,), device=device, generator=self.rng).item()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            sample_toks, masks = self.engine.generate_batch(
                tokens, num_samples=self.num_samples, min_tokens=1, max_tokens=64, seed=seed
            )
        tactics = []
        for i in range(self.num_samples):
            tactic_toks = [token for token, mask in zip(sample_toks[i], masks[i]) if mask == 1]
            tactic = self.tokenizer.decode(tactic_toks)
            if "sorry" in tactic or "admit" in tactic:
                continue
            tactics.append(tactic)
        return tactics

    def sample_tactic_batch(self, states: list[State]) -> list[list[str]]:
        """Batched version of sample_tactic for multiple states at once."""
        state_strs = []
        for state in states:
            assert len(state) == 1, \
                f"expected single branch in state when generating tactic, got {len(state)} - choose one goal first"
            state_strs.append(str(state[0].state).strip())
        return self.sample_tactic_from_str_batch(state_strs)

    def sample_tactic_from_str(self, state_str: str) -> list[str]:
        """Generate tactics from a state string directly (no State object needed)."""
        return self.sample_tactic_from_str_batch([state_str])[0]

    def sample_tactic_from_str_batch(self, state_strs: list[str]) -> list[list[str]]:
        """Batched tactic generation from state strings directly."""
        import torch
        device = self.network.get_device()
        assert device.type == "cuda"

        # Prepare tokenized prompts
        prompts = []
        for state_str in state_strs:
            tokens = self.tokenizer(state_str + "\n<|tactic|>", prepend=self.tokenizer.get_bos_token_id())
            prompts.append(tokens)

        seed = torch.randint(torch.iinfo(torch.int32).max, (1,), device=device, generator=self.rng).item()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # prompts is list[list[int]], which triggers batched generation
            sample_toks_batch, masks_batch = self.engine.generate_batch(
                prompts, num_samples=self.num_samples, min_tokens=1, max_tokens=64, seed=seed
            )

        # sample_toks_batch shape: (num_prompts, num_samples, seq_len)
        results = []
        for prompt_idx in range(len(state_strs)):
            tactics = []
            for sample_idx in range(self.num_samples):
                tactic_toks = [
                    token for token, mask in zip(
                        sample_toks_batch[prompt_idx][sample_idx],
                        masks_batch[prompt_idx][sample_idx]
                    ) if mask == 1
                ]
                tactic = self.tokenizer.decode(tactic_toks)
                if "sorry" in tactic or "admit" in tactic:
                    continue
                tactics.append(tactic)
            results.append(tactics)
        return results

    def shutdown(self):
        """No-op for non-batched model. Exists for API compatibility with BatchedTacticModel."""
        pass

    @classmethod
    def create(cls, num_samples: int = 6) -> Self:
        # Lazy imports - only load heavy dependencies when actually creating the model
        import torch
        from nanoproof.checkpoints import load_model
        from nanoproof.engine import Engine
        
        source = "sft"  # which checkpoint to load the model from
        model_tag = "d26"  # model tag to load the model from
        device = torch.device("cuda")

        model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag)
        engine = Engine(model, tokenizer)
        return cls(model, tokenizer, engine, num_samples=num_samples)


class BatchedTacticModel:
    """
    Thread-safe wrapper around TacticModel that batches LLM calls from multiple threads.
    
    When sample_tactic is called, it blocks until either:
    1. batch_size inputs have been accumulated, or
    2. timeout_seconds has elapsed since the first pending request
    
    Then a single batched LLM call is made and results are distributed to waiting threads.
    """

    def __init__(self, inner_model: TacticModel, batch_size: int = 32, timeout_seconds: float = 0.1):
        self.inner_model = inner_model
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds

        # Synchronization primitives
        self._lock = threading.Lock()
        self._batch_ready = threading.Condition(self._lock)

        # Pending requests: list of (state, result_event, result_slot)
        self._pending: list[tuple[State, threading.Event, list]] = []
        self._first_request_time: float | None = None
        self._batch_in_progress = False
        
        # Shutdown flag to unblock waiting threads
        self._shutdown = False

        # Stats for logging
        self._total_requests = 0
        self._total_batches = 0
        log(f"Initialized with batch_size={batch_size}, timeout={timeout_seconds}s", component="BatchedTacticModel")

    @property
    def network(self):
        """Expose network for compatibility with code that accesses model.network."""
        return self.inner_model.network

    def shutdown(self):
        """
        Signal shutdown to unblock all waiting threads.
        Waiting threads will raise RuntimeError.
        """
        with self._lock:
            self._shutdown = True
            # Set all pending result events to unblock waiting threads
            for _, result_event, result_slot in self._pending:
                result_slot.append([])  # Empty result
                result_event.set()
            self._pending = []
            self._batch_ready.notify_all()
        log("Shutdown initiated, all waiting threads unblocked", component="BatchedTacticModel")

    def sample_tactic(self, state: State) -> list[str]:
        """
        Thread-safe sample_tactic that batches calls from multiple threads.
        Blocks until result is available.
        
        Returns empty list if shutdown is in progress.
        """
        if self._shutdown:
            return []
            
        wait_start = time.time()
        result_event = threading.Event()
        result_slot: list[list[str]] = []  # Will hold the result

        with self._lock:
            if self._shutdown:
                return []
                
            self._total_requests += 1
            request_num = self._total_requests
            pending_count = len(self._pending) + 1

            # Add this request to the pending batch
            self._pending.append((state, result_event, result_slot))

            if self._first_request_time is None:
                self._first_request_time = time.time()
                log(f"Request #{request_num}: first in batch", component="BatchedTacticModel")

            # Log every 10th request or when batch is full
            if request_num % 10 == 0 or pending_count >= self.batch_size:
                log(f"Request #{request_num}: pending={pending_count}/{self.batch_size}",
                    component="BatchedTacticModel")

            # Check if we should trigger the batch
            should_process = len(self._pending) >= self.batch_size

            if should_process and not self._batch_in_progress:
                self._process_batch_locked()

        # If not processing immediately, wait for batch to be ready or timeout
        if not result_event.is_set():
            self._wait_for_batch_or_timeout(result_event)

        # Wait for the result with timeout to avoid hanging forever
        # Use a loop with short timeouts to stay responsive to shutdown
        max_wait = 60.0  # Maximum total wait time
        wait_interval = 1.0  # Check for shutdown every second
        total_waited = 0.0
        while not result_event.is_set() and total_waited < max_wait:
            if self._shutdown:
                return []
            result_event.wait(timeout=wait_interval)
            total_waited += wait_interval
        
        if not result_event.is_set():
            log(f"Request timed out after {max_wait}s", component="BatchedTacticModel")
            return []

        # Record wait time for monitoring
        wait_time = time.time() - wait_start
        monitor = get_monitor()
        if monitor is not None:
            monitor.record_batch_wait(wait_time)

        return result_slot[0] if result_slot else []

    def _wait_for_batch_or_timeout(self, my_event: threading.Event):
        """Wait until batch is ready or timeout, then potentially trigger processing."""
        max_iterations = 100  # Prevent infinite loop
        iterations = 0
        while not my_event.is_set() and iterations < max_iterations:
            iterations += 1
            with self._lock:
                if self._shutdown:
                    return
                    
                if my_event.is_set():
                    return

                if self._first_request_time is None:
                    # Batch was already processed, but our event should be set
                    # Wait briefly and check again (handles race condition)
                    self._batch_ready.wait(timeout=0.1)
                    continue

                elapsed = time.time() - self._first_request_time
                remaining = self.timeout_seconds - elapsed

                if remaining <= 0 or len(self._pending) >= self.batch_size:
                    # Time to process
                    if not self._batch_in_progress and len(self._pending) > 0:
                        self._process_batch_locked()
                    return

                # Wait with timeout
                self._batch_ready.wait(timeout=min(remaining, 0.5))  # Cap wait to check shutdown

    def _process_batch_locked(self):
        """Process the current batch. Must be called with lock held."""
        if len(self._pending) == 0:
            return

        self._batch_in_progress = True
        self._total_batches += 1
        batch_num = self._total_batches

        # Grab the current batch
        batch = self._pending[:]
        self._pending = []
        self._first_request_time = None

        log(f"Batch #{batch_num}: processing {len(batch)} requests", component="BatchedTacticModel")

        # Release lock during LLM inference
        self._lock.release()
        results = None
        inference_error = None
        try:
            states = [item[0] for item in batch]
            inference_start = time.time()
            results = self.inner_model.sample_tactic_batch(states)
            inference_time = time.time() - inference_start
            log(f"Batch #{batch_num}: LLM inference took {inference_time:.2f}s", component="BatchedTacticModel")
        except Exception as e:
            inference_error = e
            log(f"Batch #{batch_num}: LLM inference FAILED: {type(e).__name__}: {e}", component="BatchedTacticModel")
        finally:
            self._lock.acquire()
            # CRITICAL: Always reset _batch_in_progress, even on error
            self._batch_in_progress = False

        # Distribute results to waiting threads
        for i, (_, result_event, result_slot) in enumerate(batch):
            if results is not None:
                result_slot.append(results[i])
            else:
                # On error, return empty list so threads don't hang
                result_slot.append([])
            result_event.set()

        if inference_error:
            log(f"Batch #{batch_num}: returned empty results to {len(batch)} threads due to error", component="BatchedTacticModel")
        else:
            log(f"Batch #{batch_num}: results distributed to {len(batch)} threads", component="BatchedTacticModel")

        # Notify any threads waiting for this batch
        self._batch_ready.notify_all()


def run_mcts(config: Config, game: Game, model: TacticModel | BatchedTacticModel,
              expansion_callback=None, tactic_callback=None):
    """
    Run MCTS to find a proof.
    
    Args:
        config: MCTS configuration
        game: The game to solve
        model: Tactic model (local or remote)
        expansion_callback: Optional callable() to call on each expansion
        tactic_callback: Optional callable(state_str, tactic, success) to call on each tactic
    """
    root = game.root
    for i in range(game.num_simulations):
        node = root
        search_path = [node]

        while node.expanded() and len(node.children) > 0 and not progressive_sample(node, config):
            _, node = select_child(config, node)
            search_path.append(node)

        assert node.state is not None
        tactics = model.sample_tactic(node.state)
        tactic_logprobs = [1.0] * len(tactics)  # TODO: use the actual action logprobs

        # Get state string for tactic callback
        state_str = str(node.state[0].state).strip() if len(node.state) == 1 else ""

        expand_node(node, tactics, tactic_logprobs, config.prior_temperature)

        # Record expansion for monitoring
        monitor = get_monitor()
        if monitor is not None:
            monitor.record_expansion()
        if expansion_callback is not None:
            expansion_callback()
        
        # Record tactics for monitoring (if we have children, tactics were applied)
        if tactic_callback is not None and node.children:
            for tactic, child in node.children.items():
                # A tactic is successful if it led to a valid state
                success = child.state is not None and len(child.state) > 0
                tactic_callback(state_str, tactic, success)

        backpropagate(
            search_path,
            1.0,  # TODO: use the actual value
            config,
        )

        # print(root.pp_tree())
        # print("-" * 80)
        if root.is_solved:
            break


def progressive_sample(node: Node, config: Config) -> bool:
    """Whether to expand a node in the search tree again (progressive sampling)."""
    return (
            node.to_play == Player.OR
            and node.evaluations <= config.ps_c * node.visit_count ** config.ps_alpha
    )


def select_child(config: Config, node: Node) -> tuple[Action, Node]:
    """Selects the child with the highest UCB score."""
    # Cache prior_sum once for all children (avoids O(children^2) recomputation)
    prior_sum = node.prior_sum()
    _, action, child = max(
        (ucb_score(config, node, child, prior_sum), action, child)
        for action, child in node.children.items()
    )
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: Config, parent: Node, child: Node, prior_sum: float) -> float:
    pb_c = (
            math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
            + config.pb_c_init
    )
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    # Due to progressive sampling, we normalise priors here.
    prior_score = pb_c * child.prior / prior_sum
    if child.visit_count > 0:
        value = child.reward + child.value()
        value_score = config.value_discount ** (- 1 - value)
    else:
        value_score = 0  # TODO: this is from the official pseudocode, but probably could be improved

    if parent.to_play == Player.AND:
        # Invert value score for AND nodes.
        value_score = 1 - value_score
        if child.is_solved:
            # Avoid re-selecting proven subgoals.
            value_score = -1e9
    return prior_score + value_score


# We expand a node using the value and sampled actions obtained from the neural
# network. Immediately attempt the actions in the environment.
def expand_node(
        node: Node,
        actions: list[str],
        action_logprobs: list[float],
        temperature: float,
):
    node.evaluations += 1
    policy = {
        a: math.exp(logprob / temperature)
        for a, logprob in zip(actions, action_logprobs)
    }
    node.children = {}
    state_str = str(node.state[0].state).strip() if len(node.state) == 1 else "<multi-branch>"
    for action, p in policy.items():
        # Check if action is duplicate.
        if action in node.children:
            node.children[action].prior += p  # TODO: wtf is this?
            continue
        # Immediately apply the actions in the environment.
        assert len(node.state) == 1
        branch = node.state[0]
        new_branches = branch.try_apply_tactic(action)
        if not new_branches.is_success():
            # Invalid action encountered.
            log_tactic(state_str, action, success=False)
            continue
        if len(new_branches.value) == 1 and new_branches.value[0].state.semantic_equals(node.state[0].state):
            # Tactic made no progress.
            log_tactic(state_str, action, success=False)
            continue
        log_tactic(state_str, action, success=True)
        new_branches = [b for b in new_branches.value if not b.state.is_solved()]
        child = Node(
            action=action,
            prior=p,
            state=new_branches,
            to_play=Player.AND if len(new_branches) > 1 else Player.OR,
            reward=-1.0,
        )
        if child.is_terminal:
            child.is_solved = True
            node.is_solved = True
        node.children[action] = child
        if len(new_branches) > 1:
            # For AND nodes, immediately add children with pseudo-actions to focus on each goal.
            child.children = {}
            for i, branch in enumerate(new_branches):
                grandchild = Node(
                    action=i,
                    prior=1.0 / len(new_branches),
                    state=[branch],
                    to_play=Player.OR,
                    reward=None,  # this reward is never used
                )
                child.children[i] = grandchild


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(
        search_path: list[Node],
        value: float,
        config: Config,
):
    if len(search_path[-1].children) == 0:
        value = config.no_legal_actions_value
    is_solved = False
    for ix, node in reversed(list(enumerate(search_path))):
        node.value_sum += value
        node.visit_count += 1
        if node.to_play == Player.AND:
            is_solved = all(child.is_solved for child in node.children.values())
        else:
            is_solved |= node.is_solved
        node.is_solved = is_solved

        if ix != 0:  # we are not at the root yet - calculate the value for parent
            if search_path[ix - 1].to_play == Player.AND:  # our parent is an AND node
                value = backprop_value_towards_min(search_path[ix - 1])
            else:
                value = node.reward + value


def backprop_value_towards_min(node):
    """Computes the value for an AND node by propagating the min value from children, corresponding to the longest/hardest unsolved proof branch."""
    value = 1
    for child in node.children.values():
        if not child.is_solved and child.visit_count > 0:
            value = min(value, child.value())
    return value


def run_bfs(game: Game, model: TacticModel):
    open_nodes = [game.root]
    while open_nodes:
        node = open_nodes.pop(0)
        assert node.to_play == Player.OR
        assert len(node.state) == 1
        branch = node.state[0]
        print("-" * 80 + f"Solving state:\n{branch.state}\n")
        for retry_idx in range(10):
            print("Generating ..." + f" (retry {retry_idx})" if retry_idx != 0 else "")
            tactics = model.sample_tactic(node.state)
            # [tactic] = tactics
            # print(f"Trying tactic:\n'{tactic}'")
            options = []
            selected_tactic, selected_new_branches = None, None
            for tactic in tactics:
                new_branches = branch.try_apply_tactic(tactic)
                if new_branches.is_success():
                    new_branches = [b for b in new_branches.value if not b.state.is_solved()]
                    if selected_tactic is None or len(tactic) < len(selected_tactic):
                        selected_tactic = tactic
                        selected_new_branches = new_branches
                        options.append((tactic, new_branches, True))
                    else:
                        options.append((tactic, new_branches, False))
                else:
                    options.append((tactic, None, False))
                    # print(f"Error: '{new_branches.error}'")
            for tactic, new_branches, is_selected in options:
                print("✅" if new_branches is not None else "❌", tactic, "(SELECTED)" if is_selected else "")
            if selected_new_branches is not None:
                new_branches = selected_new_branches
                break
        else:
            print("Could not generate a valid tactic in 10 retries, terminating BFS.")
            return False
        node.children = {}
        print(f"Got {len(new_branches)} new branch(es)!")
        if len(new_branches) <= 1:
            child = Node(
                action=tactic,
                to_play=Player.OR,
                prior=None,
                state=new_branches,
                reward=None,
            )
            node.children[tactic] = child
            if not child.is_terminal:
                open_nodes.append(child)
        else:
            child = Node(
                action=tactic,
                prior=None,
                state=new_branches,
                to_play=Player.AND,
                reward=None,
            )
            node.children[tactic] = child
            for i, branch in enumerate(new_branches):
                grandchild = Node(
                    action=i,
                    prior=None,
                    state=[branch],
                    to_play=Player.OR,
                    reward=None,
                )
                child.children[i] = grandchild
                open_nodes.append(grandchild)
            break
    game.root.calculate_solved()
    assert game.root.is_solved
    return True


def _main():
    base_dir = get_base_dir()
    server_address = "10.10.25.39"
    server_port = 8000

    project_dir = os.path.join(base_dir, "leantree_project")
    if not os.path.exists(project_dir) or not os.listdir(project_dir):
        # TODO: we need to add this to LeantreeProject.lean:
        # """
        # import FormalConjectures.ForMathlib.Analysis.SpecialFunctions.NthRoot
        # import FormalConjectures.Util.Answer
        # """
        formal_conjectures = LeanLibrary(
            name="formal_conjectures",
            scope="google-deepmind",
            git="https://github.com/google-deepmind/formal-conjectures",
            rev="d3d568c9b6ba0b0609b8dd61d0019cd77462e96a",
        )
        LeanProject.create(project_dir, libraries=[LeanLibraries.MATHLIB, formal_conjectures])

    model = TacticModel.create()

    time_start = time.time()
    theorems = list_theorems(split="train")
    print(f"Retrieved {len(theorems)} theorems in {time.time() - time_start} seconds")
    theorem = theorems[1]
    print(theorem + "\n-----")

    # We expect that the server has these imports:
    # import Mathlib
    # import FormalConjectures.ForMathlib.Analysis.SpecialFunctions.NthRoot
    # import FormalConjectures.Util.Answer

    client = LeanClient(server_address, server_port)
    print(f"Connected to server at {server_address}:{server_port}")
    print(f"Server status: {client.check_status()}")
    with client.get_process() as env:
        print("Sending `open scoped` commands...")
        env.send_command("""
    open scoped Real
    open scoped Nat
    open scoped Topology
    open scoped Polynomial
    """)
        print("Starting proof...")
        init_branch = env.proof_from_sorry(theorem)
        if not init_branch.is_success():
            print(f"Error when starting proof: '{init_branch.error}'")
            return
        init_branch = init_branch.value
        print(f"Initial state:\n{init_branch.state}")

        game = Game(theorem)
        game.root = Node(
            action=None,
            prior=None,
            state=[init_branch],
            to_play=Player.OR,
            reward=None,
        )
        run_bfs(game, model)


if __name__ == "__main__":
    _main()
