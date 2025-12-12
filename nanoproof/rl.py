from typing import Callable
import random

import torch
from leantree import LeanProject, LeanLibrary, LeanLibraries
from leantree.repl_adapter.server import LeanClient

from nanoproof.model import Transformer
from nanoproof.search.bfs import Node, Player, Game, run_bfs, TacticModel
from nanoproof.data.leanworkbook import list_theorems


class Config:
    def __init__(
            self,
            num_simulations: int,
            batch_size: int,
            num_actors: int,
            lr: float,
    ):
        # Acting
        self.num_actors = num_actors

        self.num_simulations = num_simulations

        # UCB formula
        self.pb_c_base = 3200
        self.pb_c_init = 0.001
        self.value_discount = 0.99
        self.prior_temperature = 200

        # Other MCTS parameters
        self.no_legal_actions_value = -40

        # Progressive sampling parameters
        self.ps_c = 0.01
        self.ps_alpha = 0.6

        # Value predictions
        self.num_value_bins = 64

        # Training
        self.training_steps = int(1000e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.sequence_length = 32
        self.lr = lr
        self.value_weight = 0.001

        # Lean server
        self.server_address = "10.10.25.40"
        self.server_port = 8000


class SharedModelStorage:
    def __init__(self):
        self._params = {}

    def latest_params(self) -> dict:
        return self._params[max(self._params.keys())]

    def save_params(self, step: int, params: dict):
        self._params[step] = params


class TheoremsSampler:
    def __init__(self):
        self.theorems = list_theorems()

    def sample_theorem(self) -> str:
        return random.choice(self.theorems)


class ReplayBuffer:
    def __init__(self, config: Config):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.buffer = []

    def save_game(self, game):
        transitions = self._extract_transitions(game.root)
        self.buffer.extend(transitions)
        self.buffer = self.buffer[-self.window_size:]

    def extract_transitions(self, node: Node) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
        """Extracts transitions from all proven nodes in the game."""
        ...

    def sample_batch(self) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
        return [self.sample_transition() for _ in range(self.batch_size)]

    def sample_transition(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        ...


# Each acting job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the learner by writing it
# to a shared replay buffer.
def run_actor(config: Config, storage: SharedModelStorage, replay_buffer: ReplayBuffer, theorems_sampler: TheoremsSampler):
    network = Transformer(config)
    model = TacticModel(network, tokenizer, engine)
    while True:
        network.params = storage.latest_params()
        game = play_game(config, model, theorems_sampler)
        if game.root.is_solved:
            replay_buffer.save_game(game)


# Each game is produced by starting from the initial Lean state, and executing
# BFS/MCTS to find a proof. If one is found, we extract from the search tree the
# state-tactic-value transitions in the proof, which are added to a replay
# buffer for training.
def play_game(config: Config, model: TacticModel, theorems_sampler: TheoremsSampler) -> Game:
    theorem = theorems_sampler.sample_theorem()

    client = LeanClient(config.server_address, config.server_port)
    with client.get_process() as env:
        init_branch = env.proof_from_sorry(theorem)
        game = Game(theorem, config.num_simulations)

        game.root = Node(
            action=None,
            observation=init_branch.state,
            prior=None,
            branch=init_branch,
            to_play=Player.OR,
            reward=None,
        )

        run_bfs(game, model)
        if game.root.is_solved:
            # TODO: Perform final check to ensure the proof is valid.
            # game.root.is_solved = final_check(game)

            # TODO: Compute value targets for the proof.
            # compute_value_target(game.root)
            pass

        return game


# def train_network(config: Config, storage: SharedModelStorage, replay_buffer: ReplayBuffer):
#     network = Network(config)

#     for i in range(config.training_steps):
#         if i % config.checkpoint_interval == 0:
#             storage.save_params(i, network.params)
#         batch = replay_buffer.sample_batch()
#         network.update(batch)
#     storage.save_params(config.training_steps, network.params)


def _main():
    pass


if __name__ == "__main__":
    _main()