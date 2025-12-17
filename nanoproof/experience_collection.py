import random
import torch
import torch.distributed as dist

from nanoproof.common import get_dist_info
from leantree.repl_adapter.server import LeanClient

from nanoproof.search import Node, Player, Game, run_bfs, run_mcts, TacticModel, Action, State, Config
from nanoproof.data.leanworkbook import list_theorems
from nanoproof.common import SimpleTimer

class TheoremsSampler:
    def __init__(self, seed: int | None = 0):
        self.theorems = list_theorems(split="train")
        self.rng = random.Random(seed)

    def sample_theorem(self) -> str:
        # return "theorem lean_workbook_42924 (h : 1 / 2 * 30 * 23 * 6 = 2070) : 1 / 2 * 30 * 23 * 6 = 2070  :=  by sorry"
        return self.rng.choice(self.theorems)


class ReplayBuffer:
    def __init__(self, config: Config, seed: int = 0):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.local_buffer = []
        self.buffer = []
        self.rng = random.Random(seed)

    def save_game(self, game: Game) -> int:
        transitions = self._extract_transitions(game.root)
        print("! New transitions !")
        for transition in transitions:
            print(transition)

        self.local_buffer.extend(transitions)

        print(f"Local buffer size: {len(self.local_buffer)}")

        return len(transitions)

    def synchronize(self):
        ddp, _, _, world_size = get_dist_info()
        if ddp:
            gathered_buffers = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_buffers, self.local_buffer)
            for buffer in gathered_buffers:
                self.buffer.extend(buffer)
        else:
            self.buffer.extend(self.local_buffer)
        
        self.local_buffer = []
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]

    def _extract_transitions(self, node: Node) -> list[tuple[str, str, float]]:
        """Extracts transitions from a proof."""
        assert node.to_play == Player.OR
        if not node.is_solved:
            return []
        transitions = []
        while node.to_play == Player.OR and not node.is_terminal:
            assert len(node.state) == 1
            action = self._select_optimal_action(node)
            assert isinstance(action, str)
            transitions.append((str(node.state[0].state).strip(), action.strip(), node.value_target))
            node = node.children[action]
        if node.to_play == Player.AND:
            for _, child in node.children.items():
                transitions.extend(self._extract_transitions(child))
        return transitions

    def _select_optimal_action(self, node: Node) -> Action:
        assert node.to_play == Player.OR
        actions = [action for action in node.children if node.children[action].is_solved]
        assert len(actions) > 0
        # select the shortest tactic
        return min(actions, key=lambda a: len(a))

    def sample_transition(self) -> tuple[str, str, float]:
        return self.rng.choice(self.buffer)


# Each acting job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the learner by writing it
# to a shared replay buffer.
@torch.inference_mode()
def run_actor(total_to_collect: int, config: Config, model: TacticModel, replay_buffer: ReplayBuffer, theorems_sampler: TheoremsSampler) -> SimpleTimer:
    collected = 0
    ddp, _, _, world_size = get_dist_info()
    timer = SimpleTimer()

    while True:
        # Check if we have collected enough proofs globally
        if ddp:
            collected_tensor = torch.tensor([collected], dtype=torch.long, device=model.network.get_device())
            dist.all_reduce(collected_tensor, op=dist.ReduceOp.SUM)
            global_collected = collected_tensor.item()
        else:
            global_collected = collected
        if global_collected >= total_to_collect:
            break

        game = play_game(config, model, theorems_sampler, timer)
        if game is None:
            # print("Invalid theorem statement.")
            continue
        if game.root.is_solved:
            collected += replay_buffer.save_game(game)
    
    return timer


# Each game is produced by starting from the initial Lean state, and executing
# BFS/MCTS to find a proof. If one is found, we extract from the search tree the
# state-tactic-value transitions in the proof, which are added to a replay
# buffer for training.
def play_game(config: Config, model: TacticModel, theorems_sampler: TheoremsSampler, timer: SimpleTimer) -> Game | None:
    theorem = theorems_sampler.sample_theorem()
    client = LeanClient(config.server_address, config.server_port)
    with client.get_process() as env:
        env.send_command("""
            open scoped Real
            open scoped Nat
            open scoped Topology
            open scoped Polynomial
        """)
        init_branch = env.proof_from_sorry(theorem)
        if not init_branch.is_success():
            return None
        init_branch = init_branch.value
        game = Game(theorem, config.num_simulations)

        game.root = Node(
            action=None,
            prior=None,
            state=[init_branch],
            to_play=Player.OR,
            reward=None,
        )

        # success = run_bfs(game, model)
        run_mcts(config, game, model, timer)
        if game.root.is_solved:
            # TODO: Perform final check to ensure the proof is valid.
            # game.root.is_solved = final_check(game)

            # TODO: try to remove each tactic from the proof and check if the proof is still valid (maybe even more iterations of this)

            # TODO: Compute value targets for the proof.
            # compute_value_target(game.root)
            print(theorem)
            pass

        return game


def _main():
    config = Config()
    model = TacticModel.create()
    replay_buffer = ReplayBuffer(config)
    theorems_sampler = TheoremsSampler()
    timer = run_actor(config, model, replay_buffer, theorems_sampler)
    timer.log_times()


if __name__ == "__main__":
    _main()