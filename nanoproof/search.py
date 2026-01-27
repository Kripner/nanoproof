from dataclasses import dataclass
import enum
from typing import Self
import math
import uuid

from leantree.repl_adapter.server import LeanProofBranch, LeanClient
from leantree.repl_adapter.interaction import LeanProcess

from nanoproof.common import pretty_print_tree, ValueOrError, theorem_to_example, Player
from nanoproof.cli import get_monitor, log_tactic
from nanoproof.inference import TacticModel, BlockingTacticModel


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
    window_size: int = 60_000_000
    lr: float = 1e-4
    value_weight: float = 0.002

    # Lean server
    server_address: str = "10.10.25.35"
    server_port: int = 8000


Action = str | int

State = list[LeanProofBranch]


@dataclass
class Node:
    """Node in the search tree."""

    parent: Self | None  # Not serialized.
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
    value_target: float | None = None

    # Unique ID for this node, assigned in __post_init__ if not provided.
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        assert (self.parent is None) == (self.action is None), f"Node __post_init__: parent={self.parent} action={self.action}"

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
            value_target_str = f"[v={node.value_target:.2f}]" if node.value_target is not None else ""
            return f"[{type_str}{solved_str}{value_target_str}]\nvis={node.visit_count} evals={node.evaluations} val={node.value():.2f}\n{state_str}"

        def get_edge_label(node: Node):
            if node.action is None:
                return None
            prior_str = f"p={node.prior:.2f}" if node.prior is not None else "p=None"
            reward_str = f"r={node.reward:.2f}" if node.reward is not None else "r=None"
            return f"[{prior_str} {reward_str}] {str(node.action)}"

        return pretty_print_tree(self, get_children, get_node_label, get_edge_label, max_label_len=200,
                                 max_edge_label_len=50)

    def serialize(self) -> dict:
        """Serialize the node tree to a JSON-compatible dict."""
        # Serialize state as list of state strings (LeanProofBranch objects can't be serialized)
        state_strs = [str(branch.state) for branch in self.state] if self.state else []
        
        # Serialize children recursively
        children_data = None
        if self.children is not None:
            children_data = {
                str(action): child.serialize() 
                for action, child in self.children.items()
            }
        
        return {
            "id": self.id,
            "parent_id": self.parent.id if self.parent else None,
            "action": self.action,
            "prior": self.prior,
            "state": state_strs,
            "reward": self.reward,
            "to_play": self.to_play.value,
            "is_solved": self.is_solved,
            "visit_count": self.visit_count,
            "evaluations": self.evaluations,
            "value_sum": self.value_sum,
            "value_target": self.value_target,
            "children": children_data,
        }

    @classmethod
    def deserialize(cls, data: dict, id_to_node: dict[str, Self] | None = None) -> Self:
        """
        Deserialize a node tree from a dict.
        
        Creates MockProofBranch objects for the state so that transition
        extraction code (which expects branch.state) works correctly.
        
        Args:
            data: The serialized node data.
            id_to_node: Dict mapping node ids to node instances, used to look up parents.
                        If None, a new dict is created (for the root node).
        """
        if id_to_node is None:
            id_to_node = {}
        
        # Create mock proof branches with .state attribute
        state_strs = data.get("state", [])
        state = [MockProofBranch(s) for s in state_strs]
        
        # Look up parent from dict using parent_id
        parent_id = data.get("parent_id")
        parent = None
        if parent_id:
            assert parent_id in id_to_node, f"deserialize: Parent node not found: {parent_id}"
            parent = id_to_node[parent_id]
        
        # Create the node first (without children)
        node = cls(
            parent=parent,
            action=data["action"],
            prior=data["prior"],
            state=state,
            reward=data["reward"],
            to_play=Player(data["to_play"]),
            is_solved=data["is_solved"],
            visit_count=data["visit_count"],
            evaluations=data["evaluations"],
            value_sum=data["value_sum"],
            value_target=data.get("value_target"),
            children=None,
            id=data["id"],
        )
        
        # Add node to dict so children can look it up
        id_to_node[node.id] = node
        
        # Deserialize children recursively, passing the dict
        if data.get("children") is not None:
            children = {}
            for action_str, child_data in data["children"].items():
                # Try to convert action back to int if it was an int (for AND node children)
                try:
                    action = int(action_str)
                except ValueError:
                    action = action_str
                children[action] = cls.deserialize(child_data, id_to_node)
            node.children = children
        
        return node

    def clone(self) -> Self:
        return self.deserialize(self.serialize())

    def get_tree_nodes(self) -> list[Self]:
        result = []
        q = [self]
        while q:
            node = q.pop(0)
            result.append(node)
            if node.children is not None:
                q.extend(node.children.values())
        return result

    def find_node_by_id(self, id: str) -> Self | None:
        for node in self.get_tree_nodes():
            if node.id == id:
                return node
        return None

class MockProofBranch:
    """Mock proof branch for deserialized nodes. Mimics LeanProofBranch.state."""
    
    def __init__(self, state_str: str):
        self.state = state_str
    
    def __str__(self):
        return self.state


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

# TODO: deduplicate with execute_tree (just execute & check that all states equal to the expected values)
def verify_node(node: Node):
    assert node.to_play == Player.OR, f"verify_node: Expected OR root, got {node.to_play}"
    assert len(node.state) == 1, f"verify_node: Expected 1 branch at root, got {len(node.state)}"
    init_branch = node.state[0]
    to_verify = [(node, [init_branch])]
    i = 0
    while to_verify:
        node, branches = to_verify.pop(0)
        if node.to_play == Player.AND:
            assert len(branches) == len(node.state), f"verify_node: {len(branches)=} != {len(node.state)=}"
            for (action, child) in node.children.items():
                assert isinstance(action, int), f"verify_node: Expected int action below AND node, got {type(action)}"
                assert child.to_play == Player.OR, f"verify_node: Expected OR node below AND node, got {child.to_play}"
                to_verify.append((child, child.state))
        elif node.to_play == Player.OR:
            assert len(branches) == 1, f"verify_node: Expected 1 branch at OR node, got {len(branches)}"
            branch = branches[0]
            solved_actions = [a for a in node.children if node.children[a].is_solved]
            # More than one terminal node can be solved when expanding.
            for action in solved_actions:
                child = node.children[action]

                result = branch.try_apply_tactic(action, timeout=5000)
                assert result.is_success(), f"verify_node: Tactic application error: '{result.error}'; state: '{branch.state}'; action: `{action}`"
                
                new_branches = result.value
                if len(new_branches) != len(child.state):
                    return f"Unexpected number of branches after tactic application: {len(new_branches)=} != {len(child.state)=}; state: '{branch.state}'; action: `{action}`"
                if len(new_branches) > 0:
                    to_verify.append((child, new_branches))
        else:
            raise AssertionError(f"verify_node: Unknown node type: {node.to_play}")

        i += 1
        if i > 1000:
            raise AssertionError(f"verify_node: Exceeded maximum number of iterations ({i=})")


def execute_tree(root: Node, init_branch: LeanProofBranch, allow_premature_end: bool = False) -> list[tuple[Node, State]]:
    """
    Execute the tree starting from the initial branch. Return the actual obtained state for each node.
    """
    assert root.to_play == Player.OR, f"execute_tree: Expected OR root, got {root.to_play}"
    assert len(root.state) == 1, f"execute_tree: Expected 1 branch at root, got {len(root.state)}"

    node_to_state = []
    to_execute = [(root, [init_branch])]
    i = 0
    while to_execute:
        node, branches = to_execute.pop(0)
        node_to_state.append((node, branches))
        if node.to_play == Player.AND:
            assert len(branches) == len(node.state) == len(node.children), f"execute_tree (AND): {len(branches)=} != {len(node.state)=} != {len(node.children)=}"
            for branch, (action, child) in zip(branches, node.children.items()):
                assert isinstance(action, int), f"execute_tree (AND): Expected int action below AND node, got {type(action)}"
                assert child.to_play == Player.OR, f"execute_tree (AND): Expected OR node below AND node, got {child.to_play}"
                to_execute.append((child, [branch]))
        elif node.to_play == Player.OR:
            assert len(branches) == 1, f"execute_tree (OR): Expected 1 branch at OR node, got {len(branches)}"
            branch = branches[0]
            solved_actions = [a for a in node.children if node.children[a].is_solved]
            # More than one terminal node can be solved when expanding.
            for action in solved_actions:
                child = node.children[action]

                result = branch.try_apply_tactic(action, timeout=5000)
                assert result.is_success(), f"execute_tree (OR): Tactic application error: '{result.error}'; state: '{branch.state}'; action: `{action}`"
                
                new_branches = result.value
                if len(new_branches) != len(child.state) and not (allow_premature_end and len(new_branches) == 0):
                    raise AssertionError(f"execute_tree (OR): Unexpected number of branches after tactic application: {len(new_branches)=} != {len(child.state)=}; state: '{branch.state}'; action: `{action}`")
                if len(new_branches) > 0:
                    to_execute.append((child, new_branches))
        else:
            raise AssertionError(f"execute_tree: Unknown node type: {node.to_play}")

        i += 1
        if i > 1000:
            raise AssertionError(f"execute_tree: Exceeded maximum number of iterations ({i=})")
    return node_to_state

def revive_tree_states(root: Node, theorem_str: str, lean_process: LeanProcess):
    init_branch = lean_process.proof_from_sorry(theorem_to_example(theorem_str))
    assert init_branch.is_success(), f"revive_tree_states: Failed to create initial branch: '{init_branch.error}'"
    init_branch = init_branch.value
    node_to_state = execute_tree(root, init_branch)
    for node, state in node_to_state:
        node.state = state


def prune_redundant_nodes(root: Node) -> int:
    pruned_count = 0
    while True:
        pruned = prune_redundant_node(root)
        if not pruned:
            break
        pruned_count += 1
    return pruned_count

def prune_redundant_node(root: Node) -> bool:
    # all solved interior OR nodes that don't directly finish the proof - candidates for pruning
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


class Game:
    """A single episode of interaction with the environment."""
    def __init__(self, theorem: str, num_simulations: int | None = None):
        self.theorem = theorem
        # Number of simulations to run.
        self.num_simulations = num_simulations
        # Number of iterations actually run (set by run_mcts)
        self.num_iterations: int = 0
        self.root: Node = None
        self.unsimplified_root: Node = None


class MCTSAbortedError(Exception):
    """Raised when MCTS is aborted early (e.g., prover paused during evaluation)."""
    pass


def run_mcts(config: Config, game: Game, model: "TacticModel | BlockingTacticModel", expansion_callback=None, tactic_callback=None, abort_check=None) -> int:
    """
    Run MCTS to find a proof.
    
    Args:
        config: MCTS configuration
        game: The game to solve
        model: Tactic model (local or remote)
        expansion_callback: Optional callable() to call on each expansion
        tactic_callback: Optional callable(state_str, tactic, success) to call on each tactic
        abort_check: Optional callable() -> bool that returns True if MCTS should abort early.
                     This is checked each iteration and allows callers to cancel search
                     (e.g., when prover is paused and needs to free Lean processes).
    
    Returns:
        The number of iterations (simulations) that were run.
    
    Raises:
        MCTSAbortedError: If abort_check returns True during search.
    """
    root = game.root
    num_iterations = 0
    for i in range(game.num_simulations):
        num_iterations = i + 1
        # Check if we should abort early (e.g., prover paused during evaluation)
        if abort_check is not None and abort_check():
            raise MCTSAbortedError("MCTS aborted: prover paused")
        node = root
        search_path = [node]

        while node.expanded() and len(node.children) > 0 and not progressive_sample(node, config):
            _, node = select_child(config, node)
            search_path.append(node)

        assert node.state is not None, f"run_mcts: node.state is None, node.id={node.id}"
        result = model.sample_tactic(node.state)
        if not result.is_success():
            if "State too long for model's rotary cache" in str(result.error):
                continue
            raise RuntimeError(f"Tactic/value prediction failed: {result.error}")
        tactics, value = result.value
        tactic_logprobs = [1.0] * len(tactics)  # TODO (!): use the actual action logprobs
        value = -value  # convert to MCTS value scale (negative proof depth)

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
                tactic_callback(str(node.state[0].state).strip(), tactic, success)

        backpropagate(
            search_path,
            value,
            config,
        )

        # print(root.pp_tree())
        # print("-" * 80)
        if root.is_solved:
            break
    
    game.num_iterations = num_iterations
    return num_iterations


def progressive_sample(node: Node, config: Config) -> bool:
    """Whether to expand a node in the search tree again (progressive sampling)."""
    return (
            node.to_play == Player.OR
            and node.evaluations <= config.ps_c * node.visit_count ** config.ps_alpha
    )


def select_child(config: Config, node: Node) -> tuple[Action, Node]:
    """Selects the child with the highest UCB score."""
    _, action, child = max(
        (ucb_score(config, node, child), action, child)
        for action, child in node.children.items()
    )
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: Config, parent: Node, child: Node) -> float:
    pb_c = (
        math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
        + config.pb_c_init
    )
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    # Due to progressive sampling, we normalise priors here.
    prior_score = pb_c * child.prior / parent.prior_sum()
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

# If a new state is equal to the state of a parent, we are in a cycle.
def is_cycling(node: Node, new_branches: list[LeanProofBranch]) -> bool:
    p = node.parent
    while p is not None:
        if len(p.state) == len(new_branches) and all(branch.state.semantic_equals(p_branch.state) for branch, p_branch in zip(new_branches, p.state)):
            return True
        p = p.parent
    return False

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
        # TODO (!): investigate how often this times out; log timeout errors differently
        new_branches = branch.try_apply_tactic(action)
        if not new_branches.is_success():
            # Invalid action encountered.
            log_tactic(state_str, action, status="error")
            #print(f"TACTIC ERROR (action='{action}'): '{new_branches.error}'")
            continue
        if is_cycling(node, new_branches.value):
            # Cycle detected.
            log_tactic(state_str, action, status="cycle")
            continue
        log_tactic(state_str, action, status="success")
        # new_branches = [b for b in new_branches.value if not b.state.is_solved()]
        new_branches = new_branches.value
        child = Node(
            parent=node,
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
                    parent=child,
                    action=i,
                    prior=1.0 / len(new_branches),
                    state=[branch],
                    to_play=Player.OR,
                    reward=0.0,
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


def run_bfs(game: Game, model: "TacticModel"):
    open_nodes = [game.root]
    while open_nodes:
        node = open_nodes.pop(0)
        assert node.to_play == Player.OR
        assert len(node.state) == 1
        branch = node.state[0]
        print("-" * 80 + f"Solving state:\n{branch.state}\n")
        for retry_idx in range(10):
            print("Generating ..." + f" (retry {retry_idx})" if retry_idx != 0 else "")
            result = model.sample_tactic(node.state)
            if isinstance(result, ValueOrError):
                if not result.is_success():
                    raise RuntimeError(f"Tactic generation failed: {result.error}")
                tactics, _value = result.value
            else:
                tactics, _value = result
            # [tactic] = tactics
            # print(f"Trying tactic:\n'{tactic}'")
            options = []
            selected_tactic, selected_new_branches = None, None
            for tactic in tactics:
                new_branches = branch.try_apply_tactic(tactic)
                if new_branches.is_success():
                    # new_branches = [b for b in new_branches.value if not b.state.is_solved()]
                    new_branches = new_branches.value
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
                parent=node,
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
                parent=node,
                action=tactic,
                prior=None,
                state=new_branches,
                to_play=Player.AND,
                reward=None,
            )
            node.children[tactic] = child
            for i, branch in enumerate(new_branches):
                grandchild = Node(
                    parent=child,
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
    theorem = "example (n : Nat) : n + 0 = n := by sorry"
    # theorem = "example : 1 + 2 = 3 := by sorry"

    client = LeanClient(address="10.10.25.30", port=8000)
    process = client.get_process()
    with process as env:
        env.send_command("""
            open scoped Real
            open scoped Nat
            open scoped Topology
            open scoped Polynomial
        """)
        init_branch = env.proof_from_sorry(theorem_to_example(theorem))
        assert init_branch.is_success()
        init_branch = init_branch.value
        
        root = Node(
            parent=None,
            action=None,
            prior=None,
            state=[init_branch],
            to_play=Player.OR,
            reward=None,
        )

        result = init_branch.try_apply_tactic("induction n")
        assert result.is_success()
        new_branches = result.value
        assert len(new_branches) == 2
        child = Node(
            parent=root,
            action="induction n",
            prior=None,
            state=new_branches,
            to_play=Player.AND,
            reward=None,
        )
        root.children = {"induction n": child}

        gchild1 = Node(
            parent=child,
            action=0,
            prior=None,
            state=[new_branches[0]],
            to_play=Player.OR,
            reward=None,
        )
        gchild2 = Node(
            parent=child,
            action=1,
            prior=None,
            state=[new_branches[1]],
            to_play=Player.OR,
            reward=None,
        )
        child.children = {0: gchild1, 1: gchild2}

        result = gchild1.state[0].try_apply_tactic("rfl")
        assert result.is_success()
        new_branches = result.value
        assert len(new_branches) == 0
        ggchild1 = Node(
            parent=gchild1,
            action="rfl",
            prior=None,
            state=[],
            to_play=Player.OR,
            reward=None,
        )
        gchild1.children = {"rfl": ggchild1}

        result = gchild2.state[0].try_apply_tactic("rfl")
        assert result.is_success()
        new_branches = result.value
        assert len(new_branches) == 0
        ggchild2 = Node(
            parent=gchild2,
            action="rfl",
            prior=None,
            state=[],
            to_play=Player.OR,
            reward=None,
        )
        gchild2.children = {"rfl": ggchild2}

        root.calculate_solved()

        print(root.pp_tree())
        print()

        verify_node(root)
        print("Verification: Success!")

        # root = create_node_tree(
        #     init_branch,
        #     [
        #         "rfl"
        #     ]
        # )

if __name__ == "__main__":
    _main()