import random
from dataclasses import dataclass, field

import torch
from leantree.repl_adapter.server import LeanClient

from nanoproof.search.bfs import Node, Player, Game, run_bfs, TacticModel, Action, State
from nanoproof.data.leanworkbook import list_theorems



@dataclass
class Config:
    # Acting
    num_simulations: int = 50
    num_actors: int = 4

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
    checkpoint_interval: int = int(2e3)
    window_size: int = int(250e3)
    batch_size: int = 64
    sequence_length: int = 32
    lr: float = 1e-4
    value_weight: float = 0.002

    # Lean server
    server_address: str = "10.10.25.40"
    server_port: int = 8000

class TheoremsSampler:
    def __init__(self, seed: int | None = 0):
        self.theorems = list_theorems()
        self.rng = random.Random(seed)

    def sample_theorem(self) -> str:
        # return "theorem lean_workbook_42924 (h : 1 / 2 * 30 * 23 * 6 = 2070) : 1 / 2 * 30 * 23 * 6 = 2070  :=  by sorry"
        return self.rng.choice(self.theorems)


class ReplayBuffer:
    def __init__(self, config: Config):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.buffer = []

    def save_game(self, game: Game):
        transitions = self._extract_transitions(game.root)
        print("! New transitions !")
        for transition in transitions:
            print(transition)
        print("--")

        self.buffer.extend(transitions)
        self.buffer = self.buffer[-self.window_size:]

    def _extract_transitions(self, node: Node) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
        """Extracts transitions from a proof."""
        assert node.to_play == Player.OR
        if not node.is_solved:
            return []
        transitions = []
        while node.to_play == Player.OR and not node.is_terminal:
            assert len(node.state) == 1
            action = self._select_optimal_action(node)
            transitions.append((node.state[0].state, action, node.value_target))
            node = node.children[action]
        if node.to_play == Player.AND:
            for _, child in node.children.items():
                transitions.extend(self._extract_transitions(child))
        return transitions

    def _select_optimal_action(self, node: Node) -> Action:
        assert node.to_play == Player.OR
        [action] = [ action for action in node.children if node.children[action].is_solved ]
        return action

    def sample_batch(self) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
        return [self.sample_transition() for _ in range(self.batch_size)]

    def sample_transition(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        ...


# Each acting job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the learner by writing it
# to a shared replay buffer.
def run_actor(config: Config, model: TacticModel, replay_buffer: ReplayBuffer, theorems_sampler: TheoremsSampler):
    while True:
        game = play_game(config, model, theorems_sampler)
        if game.root.is_solved:
            replay_buffer.save_game(game)
        else:
            print("pešek\n--")


# Each game is produced by starting from the initial Lean state, and executing
# BFS/MCTS to find a proof. If one is found, we extract from the search tree the
# state-tactic-value transitions in the proof, which are added to a replay
# buffer for training.
def play_game(config: Config, model: TacticModel, theorems_sampler: TheoremsSampler) -> Game:
    theorem = theorems_sampler.sample_theorem()
    print(f"Playing game for theorem:\n{theorem}\n")

    client = LeanClient(config.server_address, config.server_port)
    with client.get_process() as env:
        init_branch = env.proof_from_sorry(theorem)
        game = Game(theorem, config.num_simulations)

        game.root = Node(
            action=None,
            prior=None,
            state=[init_branch],
            to_play=Player.OR,
            reward=None,
        )

        success = run_bfs(game, model)
        if game.root.is_solved:
            # TODO: Perform final check to ensure the proof is valid.
            # game.root.is_solved = final_check(game)

            # TODO: try to remove each tactic from the proof and check if the proof is still valid

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
    config = Config()
    model = TacticModel.create()
    replay_buffer = ReplayBuffer(config)
    theorems_sampler = TheoremsSampler()
    run_actor(config, model, replay_buffer, theorems_sampler)


if __name__ == "__main__":
    _main()