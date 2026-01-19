from dataclasses import dataclass
import enum
from typing import Self, TYPE_CHECKING
import math

from leantree.repl_adapter.server import LeanProofBranch
from nanoproof.common import pretty_print_tree, ValueOrError
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


class Player(enum.Enum):
    OR = 1
    AND = 2


Action = str | int

State = list[LeanProofBranch]


@dataclass
class Node:
    """Node in the search tree."""
    parent: Self | None = None  # Not serialized.
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
    def deserialize(cls, data: dict) -> Self:
        """
        Deserialize a node tree from a dict.
        
        Creates MockProofBranch objects for the state so that transition
        extraction code (which expects branch.state) works correctly.
        """
        # Deserialize children recursively
        children = None
        if data.get("children") is not None:
            children = {}
            for action_str, child_data in data["children"].items():
                # Try to convert action back to int if it was an int (for AND node children)
                try:
                    action = int(action_str)
                except ValueError:
                    action = action_str
                children[action] = cls.deserialize(child_data)
        
        # Create mock proof branches with .state attribute
        state_strs = data.get("state", [])
        state = [MockProofBranch(s) for s in state_strs]
        
        return cls(
            action=data["action"],
            prior=data["prior"],
            state=state,
            reward=data["reward"],
            to_play=Player(data["to_play"]),
            is_solved=data["is_solved"],
            visit_count=data["visit_count"],
            evaluations=data["evaluations"],
            value_sum=data["value_sum"],
            value_target=data.get("value_target", 0),
            children=children,
        )


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
    if not node.is_solved:
        return
    
    # Walk down the OR nodes
    while node.to_play == Player.OR and not node.is_terminal:
        if len(node.state) != 1:
            break
        if not node.children:
            break
        
        # Find solved actions
        solved_actions = [a for a in node.children if node.children[a].is_solved]
        if not solved_actions:
            break
        
        # Pick shortest tactic
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
        self.root: Node = None


class MCTSAbortedError(Exception):
    """Raised when MCTS is aborted early (e.g., prover paused during evaluation)."""
    pass


def run_mcts(config: Config, game: Game, model: "TacticModel | BlockingTacticModel", expansion_callback=None, tactic_callback=None, abort_check=None):
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
    
    Raises:
        MCTSAbortedError: If abort_check returns True during search.
    """
    root = game.root
    for i in range(game.num_simulations):
        # Check if we should abort early (e.g., prover paused during evaluation)
        if abort_check is not None and abort_check():
            raise MCTSAbortedError("MCTS aborted: prover paused")
        node = root
        search_path = [node]

        while node.expanded() and len(node.children) > 0 and not progressive_sample(node, config):
            _, node = select_child(config, node)
            search_path.append(node)

        assert node.state is not None
        tactics = model.sample_tactic(node.state)
        if not tactics.is_success():
            raise RuntimeError(f"Tactic generation failed: {tactics.error}")
        tactics = tactics.value
        tactic_logprobs = [1.0] * len(tactics)  # TODO (!): use the actual action logprobs

        value = model.predict_value(node.state)
        if not value.is_success():
            raise RuntimeError(f"Value prediction failed: {value.error}")
        value = -value.value  # convert to MCTS value scale (negative proof depth)

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
        new_branches = branch.try_apply_tactic(action)
        if not new_branches.is_success():
            # Invalid action encountered.
            log_tactic(state_str, action, status="error")
            continue
        if is_cycling(node, new_branches.value):
            # Cycle detected.
            log_tactic(state_str, action, status="cycle")
            continue
        log_tactic(state_str, action, status="success")
        new_branches = [b for b in new_branches.value if not b.state.is_solved()]
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
            tactics_result = model.sample_tactic(node.state)
            if isinstance(tactics_result, ValueOrError):
                if not tactics_result.is_success():
                    raise RuntimeError(f"Tactic generation failed: {tactics_result.error}")
                tactics = tactics_result.value
            else:
                tactics = tactics_result
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


