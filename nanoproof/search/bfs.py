from ast import Or
import os
from dataclasses import dataclass, field
import enum
import time
from contextlib import nullcontext
from typing import Self

import torch
from leantree import LeanProject, LeanLibrary, LeanLibraries, LeanProofState
from leantree.repl_adapter.server import LeanClient, LeanProofBranch

from nanoproof.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type
from nanoproof.checkpoints import load_model, save_checkpoint
from nanoproof.engine import Engine
from nanoproof.data.leanworkbook import list_theorems
from nanoproof.model import Transformer
from nanoproof.tokenizer import HuggingFaceTokenizer

"""
leanserver --project-path ~/troja/nanoproof/leantree_project/ --repl-exe ~/repos/leantree/lean-repl/.lake/build/bin/repl --imports Mathlib FormalConjectures.ForMathlib.Analysis.SpecialFunctions.NthRoot FormalConjectures.Util.Answer --max-processes 2 --address=<PUBLIC_IP> --log-level=DEBUG
"""


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
            return 0
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

class Game:
    """A single episode of interaction with the environment."""
    def __init__(self, theorem: str, num_simulations: int | None = None):
        self.theorem = theorem
        # Number of simulations to run.
        self.num_simulations = num_simulations
        self.root = None


@dataclass
class TacticModel:
    network: Transformer
    tokenizer: HuggingFaceTokenizer
    engine: Engine

    def __post_init__(self):
        self.rng = torch.Generator(device=self.network.get_device())
        self.rng.manual_seed(0)

    def sample_tactic(self, state: State) -> str:
        assert len(state) == 1, f"expected single branch in state when generating tactic, got {len(state)} - choose one goal first"
        device = self.network.get_device()
        assert device.type == "cuda"

        state_str = str(state[0]).strip()
        tokens = self.tokenizer(state_str + "\n<|tactic|>", prepend=self.tokenizer.get_bos_token_id())
        seed = torch.randint(torch.iinfo(torch.int32).max, (1,), device=device, generator=self.rng).item()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            sample_toks, masks = self.engine.generate_batch(tokens, num_samples=1, min_tokens=1, max_tokens=64, seed=seed)
        tactic_toks = [token for token, mask in zip(sample_toks[0], masks[0]) if mask == 1]
        return self.tokenizer.decode(tactic_toks)

    @classmethod
    def create(cls) -> Self:
        source = "sft"  # which checkpoint to load the model from
        model_tag = "d26"  # model tag to load the model from
        device = torch.device("cuda")

        model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag)
        engine = Engine(model, tokenizer)
        return cls(model, tokenizer, engine)


def run_bfs(game: Game, model: TacticModel):
    open_nodes = [game.root]
    while open_nodes:
        node = open_nodes.pop(0)
        assert node.to_play == Player.OR
        assert len(node.state) == 1
        branch = node.state[0]
        # print("-" * 80 + f"Solving state:\n{state_str}\n")
        new_branches = None
        for retry_idx in range(10):
            # print("Generating ..." + f" (retry {retry_idx})" if retry_idx != 0 else "")
            tactic = model.sample_tactic(node.state)
            # print(f"Trying tactic:\n'{tactic}'")
            new_branches = branch.try_apply_tactic(tactic)
            if new_branches.is_success():
                new_branches = [b for b in new_branches.value if not b.state.is_solved()]
                break
        else:
            print("Could not generate a valid tactic in 10 retries, terminating BFS.")
            return False
        node.children = {}
        # print(f"Got {len(new_branches)} new branch(es)!")
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
            # print(f"Error: '{new_branches.error}'\n")
    game.root.calculate_solved()
    assert game.root.is_solved
    return True


def _main():
    base_dir = get_base_dir()
    server_address = "10.10.25.40"
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
    theorems = list_theorems()
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
        print(f"Initial state:\n{init_branch.state}")

        game = Game(theorem)
        game.root = Node(
            action=None,
            prior=None,
            branch=[init_branch],
            to_play=Player.OR,
            reward=None,
        )
        run_bfs(game, model)


if __name__ == "__main__":
    _main()