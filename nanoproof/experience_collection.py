import asyncio
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import torch
import torch.distributed as dist

from nanoproof.common import get_dist_info
from leantree.repl_adapter.server import LeanClient

from nanoproof.search import Node, Player, Game, run_bfs, run_mcts, TacticModel, BatchedTacticModel, Action, State, \
    Config
from nanoproof.data.leanworkbook import list_theorems
from nanoproof.cli import get_monitor, log, log_error


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
        self._lock = threading.Lock()  # Thread-safe access to local_buffer

    def save_game(self, game: Game) -> int:
        transitions = self._extract_transitions(game.root)
        log(f"New transitions: {len(transitions)}", component="ReplayBuffer")
        for transition in transitions:
            log(f"  {transition}", component="ReplayBuffer")

        with self._lock:
            self.local_buffer.extend(transitions)
            log(f"Local buffer size: {len(self.local_buffer)}", component="ReplayBuffer")

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
def run_actor(total_to_collect: int, config: Config, model: BatchedTacticModel, replay_buffer: ReplayBuffer,
              theorems_sampler: TheoremsSampler):
    """
    Runs parallel actors to collect proofs.
    
    Runs config.num_actors threads in parallel, each with its own LeanClient.
    LLM calls are automatically batched via the BatchedTacticModel.
    """
    ddp, _, _, world_size = get_dist_info()
    device = model.network.get_device()
    num_actors = config.num_actors

    log(f"Starting collection with {num_actors} actors, target={total_to_collect} transitions", component="Collection")

    # Thread-safe counter for collected transitions
    collected_lock = threading.Lock()
    collected = [0]  # Use list to allow mutation in nested function
    stop_flag = threading.Event()
    actors_started = [0]  # Track how many actors have started

    def check_global_collected() -> int:
        """Check how many transitions have been collected globally (across all DDP ranks)."""
        with collected_lock:
            local_collected = collected[0]
        if ddp:
            collected_tensor = torch.tensor([local_collected], dtype=torch.long, device=device)
            dist.all_reduce(collected_tensor, op=dist.ReduceOp.SUM)
            return collected_tensor.item()
        return local_collected

    def actor_thread(actor_id: int):
        """Single actor thread that continuously plays games until stop_flag is set."""
        # Each thread needs its own asyncio event loop for the LeanClient
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        log(f"Starting, connecting to Lean server at {config.server_address}:{config.server_port}", actor_id=actor_id)
        # Each actor has its own LeanClient
        client = LeanClient(config.server_address, config.server_port)
        log(f"Connected to Lean server", actor_id=actor_id)

        # Each actor has its own RNG for theorem sampling (seeded by actor_id for reproducibility)
        local_theorems_sampler = TheoremsSampler(seed=theorems_sampler.rng.randint(0, 2 ** 31) + actor_id)

        with collected_lock:
            actors_started[0] += 1
        log(f"Ready, starting game loop", actor_id=actor_id)

        games_played = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        while not stop_flag.is_set():
            log(f"Starting game {games_played + 1}", actor_id=actor_id)
            try:
                game = play_game_with_client(config, model, local_theorems_sampler, client)
                consecutive_errors = 0  # Reset on success
            except ConnectionResetError:
                consecutive_errors += 1
                log(f"Game {games_played + 1} connection reset by Lean server ({consecutive_errors}/{max_consecutive_errors})", actor_id=actor_id)
                if consecutive_errors >= max_consecutive_errors:
                    log(f"Too many consecutive connection errors, stopping actor", actor_id=actor_id)
                    break
                # Try to reconnect
                try:
                    client = LeanClient(config.server_address, config.server_port)
                    log(f"Reconnected to Lean server", actor_id=actor_id)
                except Exception as e:
                    log(f"Failed to reconnect: {e}", actor_id=actor_id)
                continue
            except Exception as e:
                consecutive_errors += 1
                log(f"Game {games_played + 1} unexpected error ({consecutive_errors}/{max_consecutive_errors}): {type(e).__name__}: {e}", actor_id=actor_id)
                if consecutive_errors >= max_consecutive_errors:
                    log(f"Too many consecutive errors, stopping actor", actor_id=actor_id)
                    break
                continue
            games_played += 1

            if game is None:
                log(f"Game {games_played} failed to start (invalid theorem?)", actor_id=actor_id)
                continue

            # Record proof attempt for monitoring
            monitor = get_monitor()
            is_solved = game.root.is_solved
            transitions = 0

            if is_solved:
                with collected_lock:
                    transitions = replay_buffer.save_game(game)
                    collected[0] += transitions
                log(f"Game {games_played} SOLVED! Got {transitions} transitions", actor_id=actor_id)
            else:
                log(f"Game {games_played} not solved", actor_id=actor_id)

            if monitor is not None:
                monitor.record_proof_attempt(successful=is_solved, transitions=transitions)

    with ThreadPoolExecutor(max_workers=num_actors) as executor:
        # Start all actor threads
        log(f"Submitting {num_actors} actor threads", component="Collection")
        futures = [executor.submit(actor_thread, i) for i in range(num_actors)]
        log(f"All actor threads submitted, waiting for them to start...", component="Collection")

        # Wait for actors to start (with timeout)
        startup_timeout = 30.0
        start_time = time.time()
        while actors_started[0] < num_actors:
            if time.time() - start_time > startup_timeout:
                log(f"WARNING: Only {actors_started[0]}/{num_actors} actors started after {startup_timeout}s",
                    component="Collection")
                break
            # Check if any actors crashed during startup
            for future in futures:
                if future.done():
                    try:
                        future.result()  # This will raise if the thread crashed
                    except Exception as e:
                        log_error(f"Actor crashed during startup", exception=e, component="Collection")
                        stop_flag.set()
                        break
            if stop_flag.is_set():
                break
            threading.Event().wait(0.5)

        log(f"{actors_started[0]}/{num_actors} actors started, beginning collection loop", component="Collection")

        # Monitor progress and stop when we've collected enough
        loop_count = 0
        while not stop_flag.is_set():
            global_collected = check_global_collected()
            if global_collected >= total_to_collect:
                log(f"Target reached: {global_collected}/{total_to_collect} transitions collected",
                    component="Collection")
                stop_flag.set()
                break

            # Check for crashed actors
            for future in futures:
                if future.done():
                    try:
                        future.result()
                    except Exception as e:
                        log_error(f"Actor crashed during collection", exception=e, component="Collection")
                        stop_flag.set()
                        break

            if stop_flag.is_set():
                break

            # Periodic status update
            loop_count += 1
            if loop_count % 50 == 0:  # Every 5 seconds
                log(f"Progress: {global_collected}/{total_to_collect} transitions", component="Collection")

            # Brief sleep to avoid busy-waiting
            threading.Event().wait(0.1)

        log(f"Stopping actors and waiting for them to finish...", component="Collection")

        # Shutdown the batched model to unblock any waiting threads
        model.shutdown()

        # Wait for all threads to finish
        for i, future in enumerate(futures):
            try:
                future.result(timeout=10.0)
                log(f"Actor {i} finished cleanly", component="Collection")
            except FuturesTimeoutError:
                log(f"Actor {i} timed out on shutdown", component="Collection")
            except ConnectionResetError:
                # Connection reset is expected during shutdown - the Lean server may close connections
                log(f"Actor {i} connection reset during shutdown (this is OK)", component="Collection")
            except Exception as e:
                log_error(f"Actor {i} exception on join", exception=e, component="Collection")

    log(f"Collection complete: {collected[0]} local transitions", component="Collection")


@torch.inference_mode()
def run_actor_single(total_to_collect: int, config: Config, model: TacticModel, replay_buffer: ReplayBuffer,
                     theorems_sampler: TheoremsSampler):
    """
    Sequential actor for collecting proofs (single-threaded).
    
    Use this when you have a regular TacticModel without batching.
    """
    ddp, _, _, world_size = get_dist_info()
    device = model.network.get_device()
    collected = 0

    while True:
        # Check if we have collected enough proofs globally
        if ddp:
            collected_tensor = torch.tensor([collected], dtype=torch.long, device=device)
            dist.all_reduce(collected_tensor, op=dist.ReduceOp.SUM)
            global_collected = collected_tensor.item()
        else:
            global_collected = collected

        if global_collected >= total_to_collect:
            break

        game = play_game(config, model, theorems_sampler)
        if game is None:
            continue
        if game.root.is_solved:
            collected += replay_buffer.save_game(game)


def play_game_with_client(
        config: Config,
        model: TacticModel | BatchedTacticModel,
        theorems_sampler: TheoremsSampler,
        client: LeanClient
) -> Game | None:
    """
    Play a single game using the provided LeanClient.
    This version is used by parallel actors to avoid creating a new client each game.
    """
    theorem = theorems_sampler.sample_theorem()

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

        run_mcts(config, game, model)
        if game.root.is_solved:
            log(f"Proof found: {theorem[:80]}...", component="MCTS")

        return game


# Each game is produced by starting from the initial Lean state, and executing
# BFS/MCTS to find a proof. If one is found, we extract from the search tree the
# state-tactic-value transitions in the proof, which are added to a replay
# buffer for training.
def play_game(config: Config, model: TacticModel | BatchedTacticModel,
              theorems_sampler: TheoremsSampler) -> Game | None:
    """Play a single game, creating a new LeanClient for this game."""
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

        run_mcts(config, game, model)
        if game.root.is_solved:
            # TODO: Perform final check to ensure the proof is valid.
            # game.root.is_solved = final_check(game)

            # TODO: try to remove each tactic from the proof and check if the proof is still valid (maybe even more iterations of this)

            # TODO: Compute value targets for the proof.
            # compute_value_target(game.root)
            log(f"Proof found: {theorem[:80]}...", component="MCTS")
            pass

        return game


def _main():
    config = Config()
    model = TacticModel.create(num_samples=config.num_sampled_tactics)
    replay_buffer = ReplayBuffer(config)
    theorems_sampler = TheoremsSampler()
    run_actor_single(100, config, model, replay_buffer, theorems_sampler)


if __name__ == "__main__":
    _main()
