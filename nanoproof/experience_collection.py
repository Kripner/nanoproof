"""
Experience collection: replay buffer, theorem sampling, and proof-tree → transitions pipeline.

This module owns:
- TheoremsSampler: weighted sampler over configured training datasets.
- ReplayBuffer: thread-safe DDP-aware buffer of (context, tactic, value_target) transitions
- compute_value_target: assigns regression targets to nodes of a solved proof tree
- extract_transitions: walks a solved proof tree and yields training transitions
- prune_redundant_nodes / prune_redundant_node: tree-editing pass that removes
  redundant OR nodes before transition extraction

Tree-walking helpers depend on Node / Player / execute_tree from search.py.
"""

import random
import threading

import torch.distributed as dist

from nanoproof.cli import log
from nanoproof.common import get_dist_info, Player, GLOBAL_CONFIG
from nanoproof.data.bench.common import BenchTheorem
from nanoproof.data.rl import deepseek_prover, leanworkbook, numinamath
from nanoproof.search import Node, execute_tree


# -----------------------------------------------------------------------------
# Theorem sampler
# -----------------------------------------------------------------------------

class TheoremsSampler:
    """Samples theorems for experience collection from multiple datasets.

    Samples uniformly at random from one of the available datasets, then
    uniformly at random from that dataset. Thread-safe.
    """

    ALL_DATASETS = {
        "leanworkbook": lambda lean_version: leanworkbook.list_theorems(split="train", lean_version=lean_version),
        "deepseek_prover": lambda lean_version: deepseek_prover.list_theorems(split="train", lean_version=lean_version),
        "numinamath": lambda lean_version: numinamath.list_theorems(split="train", lean_version=lean_version),
    }

    def __init__(self, seed: int | None = 0, datasets: list[str] | None = None, lean_version: str | None = None):
        if datasets is None:
            datasets = list(self.ALL_DATASETS.keys())
        self.datasets = {name: self.ALL_DATASETS[name](lean_version) for name in datasets}
        self.dataset_names = list(self.datasets.keys())
        self.rng = random.Random(seed)
        self._lock = threading.Lock()

        for name, theorems in self.datasets.items():
            log(f"Loaded {len(theorems)} theorems from {name}", component="Sampler")

    def sample_theorem(self) -> BenchTheorem:
        with self._lock:
            dataset_name = self.rng.choice(self.dataset_names)
            return self.rng.choice(self.datasets[dataset_name])


class ReplayBuffer:
    """
    Replay buffer for storing proof transitions.

    Supports DDP synchronization across multiple ranks.
    """
    def __init__(self, window_size: int, seed: int):
        self.window_size = window_size
        self.local_buffer = []
        self.buffer = []
        self.rng = random.Random(seed)
        self._lock = threading.Lock()  # Thread-safe access to local_buffer

    def add_transitions(self, transitions: list[tuple[str, str, float]]):
        with self._lock:
            transitions = [
                (context.strip(), tactic.strip(), value_target)
                for context, tactic, value_target in transitions
                if len(context.strip()) <= GLOBAL_CONFIG.state_max_len and len(tactic.strip()) <= GLOBAL_CONFIG.tactic_max_len
            ]
            # log(f"Adding {len(transitions)}/{received_count} transitions to replay buffer:" + "\n".join(f"  {context} {tactic} {value_target}" for context, tactic, value_target in transitions), component="Collection")
            for context, tactic, value_target in transitions:
                assert len(context) != 0, f"Empty context in transition: tactic={tactic}, value_target={value_target}"
                assert len(tactic) != 0, f"Empty tactic in transition: context={context}, value_target={value_target}"
                assert value_target is not None, f"None value_target in transition: context={context}, tactic={tactic}"
            self.local_buffer.extend(transitions)

    def synchronize(self):
        """Merge local_buffer into buffer and broadcast to all DDP ranks.

        Only rank 0 collects transitions (into local_buffer). This method
        moves them into the shared buffer and broadcasts the result so all
        ranks can sample from it during training.
        """
        ddp, _, _, _ = get_dist_info()

        # Move local_buffer → buffer (only rank 0 has data, others are empty)
        self.buffer.extend(self.local_buffer)
        self.local_buffer = []
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]

        # Broadcast from rank 0 so all ranks have the same buffer
        if ddp:
            buffer_list = [self.buffer]
            dist.broadcast_object_list(buffer_list, src=0)
            self.buffer = buffer_list[0]

    def sample_transition(self) -> tuple[str, str, float]:
        return self.rng.choice(self.buffer)


def compute_value_target(node: Node) -> float:
    """Computes the actual value for a node, to be used as a target in learning."""
    assert node.is_solved, f"Node is not solved in compute_value_target (is root={node.action is None}, is terminal={node.is_terminal}, to_play={node.to_play})"
    if node.is_terminal:
        node.value_target = 0
        return 0
    elif node.to_play == Player.OR:
        max_child_value = max(compute_value_target(child) for child in node.children.values() if child.is_solved)
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
        n for n in root.get_tree_nodes() if (
            n.is_solved and
            n.to_play == Player.OR and
            not n.is_terminal and
            not any(child.is_terminal for child in n.children.values())
        )
    ]
    for to_consider in nodes:
        solved_actions = [a for a in to_consider.children if to_consider.children[a].is_solved]
        assert len(solved_actions) == 1, f"prune_redundant_node: Expected 1 solved action, got {len(solved_actions)}"
        action = solved_actions[0]
        child = to_consider.children[action]
        child_solved_actions = [a for a in child.children if child.children[a].is_solved]
        assert child_solved_actions, f"prune_redundant_node: No solved actions in child"
        # TODO: instead of selecting the shortest, execute all of them and filter out the failed ones
        min_len = min(len(str(a)) for a in child_solved_actions)
        shortest_actions = [a for a in child_solved_actions if len(str(a)) == min_len]
        child_solved_action = shortest_actions[0]
        if child.to_play == Player.OR:
            assert len(to_consider.state) == 1, f"prune_redundant_node: Expected 1 branch at OR node, got {len(to_consider.state)}"
            try:
                # Skip the action, execute the subtree without it.
                # TODO: set allow_premature_end=True and then potentially remove unnecessary nodes
                node_to_state = execute_tree(child, to_consider.state[0], allow_premature_end=False)
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
            raise AssertionError(f"prune_redundant_node: Unknown node type: {child.to_play}")
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
        assert len(node.state) == 1, f"extract_transitions: Expected 1 branch at OR node, got {len(node.state)}"
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
