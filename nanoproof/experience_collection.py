"""
Experience collection for RL training.

When distributed=False, this module uses local actors to run proofs.
When distributed=True, the rl_server.py handles remote prover coordination.
"""

import asyncio
import random
import threading
import time
from typing import Callable, Optional
import traceback

import torch
import torch.distributed as dist
from leantree.repl_adapter.server import LeanClient

from nanoproof.common import get_dist_info
from nanoproof.search import Node, Player, Config, Game, run_mcts, extract_transitions, compute_value_target, verify_node
from nanoproof.inference import BlockingTacticModel, TacticModel
from nanoproof.cli import get_monitor, log
from nanoproof.data.leanworkbook import list_theorems
from nanoproof.data.leantree_dataloader import STATE_MAX_LEN, TACTIC_MAX_LEN


class TheoremsSampler:
    """Samples theorems for local experience collection."""
    
    def __init__(self, seed: int | None = 0):
        self.theorems = list_theorems(split="train")
        self.rng = random.Random(seed)
    
    def sample_theorem(self) -> str:
        return self.rng.choice(self.theorems)


class ReplayBuffer:
    """
    Replay buffer for storing proof transitions.
    
    Supports DDP synchronization across multiple ranks.
    """
    def __init__(self, config: Config, seed: int = 0):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.local_buffer = []
        self.buffer = []
        self.rng = random.Random(seed)
        self._lock = threading.Lock()  # Thread-safe access to local_buffer

    def add_transitions(self, transitions: list[tuple[str, str, float]]):
        with self._lock:
            received_count = len(transitions)
            transitions = [
                (context.strip(), tactic.strip(), value_target)
                for context, tactic, value_target in transitions
                if len(context.strip()) <= STATE_MAX_LEN and len(tactic.strip()) <= TACTIC_MAX_LEN
            ]
            # log(f"Adding {len(transitions)}/{received_count} transitions to replay buffer:" + "\n".join(f"  {context} {tactic} {value_target}" for context, tactic, value_target in transitions), component="Collection")
            for context, tactic, value_target in transitions:
                assert len(context) != 0, f"Empty context in transition: tactic={tactic}, value_target={value_target}"
                assert len(tactic) != 0, f"Empty tactic in transition: context={context}, value_target={value_target}"
                assert value_target is not None, f"None value_target in transition: context={context}, tactic={tactic}"
            self.local_buffer.extend(transitions)

    def synchronize(self):
        """Synchronize local buffers across all DDP ranks."""
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

    def sample_transition(self) -> tuple[str, str, float]:
        return self.rng.choice(self.buffer)


class ProverWorker:
    """
    Runs prover threads that play proof games.
    
    Configurable with callbacks for:
    - get_theorem: returns (id, theorem) or None
    - on_result: called with (id, theorem, game_or_none, error_or_none)
    
    Used by both local mode (with TheoremsSampler) and distributed mode (with coordinator).
    """
    
    def __init__(
        self,
        config: Config,
        tactic_model,  # TacticModel or RemoteTacticModel
        lean_address: str,
        lean_port: int,
        get_theorem: Callable[[], Optional[tuple[str, str]]],
        on_result: Callable[[str, str, Optional[Game], Optional[str]], None],
        num_actors: Optional[int] = None,
        paused: bool = False,
    ):
        self.config = config
        self.tactic_model = tactic_model
        self.lean_address = lean_address
        self.lean_port = lean_port
        self.get_theorem = get_theorem
        self.on_result = on_result
        self.num_actors = num_actors or config.num_actors
        
        self._running = False
        self._paused = paused
        self._stop_flag = threading.Event()
        self._threads: list[threading.Thread] = []
        self._thread_states: dict[int, str] = {}
        self._thread_states_lock = threading.Lock()
        
        self.games_played = 0
        self.games_solved = 0
        self.expansions = 0
        self._stats_lock = threading.Lock()
    
    def start(self):
        """Start the actor threads."""
        if self._running:
            return
        
        self._running = True
        self._paused = False
        self._stop_flag.clear()
        
        for i in range(self.num_actors):
            t = threading.Thread(target=self._actor_loop, args=(i,), daemon=True)
            t.start()
            self._threads.append(t)
    
    def stop(self):
        """Stop all actor threads."""
        self._stop_flag.set()
        self._running = False
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads = []
    
    def pause(self):
        """Pause collection (actors will finish current game then wait)."""
        self._paused = True
    
    def resume(self):
        """Resume collection."""
        self._paused = False
    
    def _set_thread_state(self, actor_id: int, state: str):
        with self._thread_states_lock:
            self._thread_states[actor_id] = state
    
    def get_thread_states(self) -> list[str]:
        with self._thread_states_lock:
            return [self._thread_states.get(i, "idle") for i in range(self.num_actors)]
    
    def get_expansions(self) -> int:
        with self._stats_lock:
            return self.expansions
    
    def has_started_actors(self) -> bool:
        return len(self._threads) > 0 and self._running
    
    def all_actors_exited(self) -> bool:
        return all(not t.is_alive() for t in self._threads)
    
    def _actor_loop(self, actor_id: int):
        """Main loop for a single actor thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Connect to Lean server
        self._set_thread_state(actor_id, "blocked")
        client = LeanClient(self.lean_address, self.lean_port)
        self._set_thread_state(actor_id, "idle")
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while not self._stop_flag.is_set():
            # Check if paused
            if self._paused:
                self._set_thread_state(actor_id, "idle")
                time.sleep(0.5)
                continue
            
            # Get theorem
            theorem_data = self.get_theorem()
            if theorem_data is None:
                self._set_thread_state(actor_id, "idle")
                time.sleep(0.5)
                continue
            
            theorem_id, theorem = theorem_data
            self._set_thread_state(actor_id, "running")
            
            # Try to prove it
            try:
                game = self._play_game(client, theorem)
                consecutive_errors = 0
                
                if game is not None and game.root is not None:
                    with self._stats_lock:
                        self.games_played += 1
                        if game.root.is_solved:
                            self.games_solved += 1
                
                # Report result
                self.on_result(theorem_id, theorem, game, None)
                
            except ConnectionResetError:
                consecutive_errors += 1
                self._set_thread_state(actor_id, "error")
                self.on_result(theorem_id, theorem, None, "Connection reset")
                if consecutive_errors >= max_consecutive_errors:
                    break
                try:
                    self._set_thread_state(actor_id, "blocked")
                    client = LeanClient(self.lean_address, self.lean_port)
                except Exception:
                    pass
            except Exception as e:
                # The inference model returns this when paused, even if the # local _paused flag isn't set yet.
                if "Model paused for training" in str(e):
                    self._set_thread_state(actor_id, "idle")
                    continue
                consecutive_errors += 1
                self._set_thread_state(actor_id, "error")
                self.on_result(theorem_id, theorem, None, str(e))
                log(f"[Actor {actor_id}] Error (lean={self.lean_address}:{self.lean_port}): {e}", component="Collection")
                traceback.print_exc()
                if consecutive_errors >= max_consecutive_errors:
                    break
        
        self._set_thread_state(actor_id, "idle")
    
    def _play_game(self, client, theorem: str):
        """Play a single proof game."""
        process = client.get_process()
        if process is None:
            return None
        
        with process as env:
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
            
            game = Game(theorem, self.config.num_simulations)
            game.root = Node(
                parent=None,
                action=None,
                prior=None,
                state=[init_branch],
                to_play=Player.OR,
                reward=None,
            )
            
            def on_expansion():
                with self._stats_lock:
                    self.expansions += 1
            
            run_mcts(
                self.config, game, self.tactic_model,
                expansion_callback=on_expansion,
            )
            if game.root.is_solved:
                compute_value_target(game.root)

                try:
                    verify_node(game.root)
                except AssertionError as e:
                    message = f"Verification failed: '{e}'\nTheorem: '{theorem}'\nProof tree:\n{game.root.pp_tree()}"
                    log(message, component="Collection")
                    game.root.is_solved = False

            return game

# TODO: this should utilize more closely the code used when distributed=True
@torch.no_grad()
def run_actor(
    total_to_collect: int,
    config: Config,
    model: BlockingTacticModel,
    replay_buffer: ReplayBuffer,
    theorems_sampler: TheoremsSampler
):
    """
    Run parallel actors to collect proofs locally.
    
    Uses ProverWorker to manage actor threads. LLM calls are automatically
    batched via the BlockingTacticModel.
    """
    ddp, rank, _, world_size = get_dist_info()
    master_process = rank == 0
    device = model.network.get_device()
    num_actors = config.num_actors
    
    # Counter for generating theorem IDs
    theorem_counter = [0]

    if master_process:
        log(f"Starting collection with {num_actors} actors/rank, target={total_to_collect} transitions", 
            component="Collection")

    def get_theorem():
        """Sample a theorem from the local sampler."""
        theorem = theorems_sampler.sample_theorem()
        theorem_counter[0] += 1
        return (f"local_{theorem_counter[0]}", theorem)
    
    def on_result(theorem_id: str, theorem: str, game, error: str | None):
        """Handle proof result by extracting transitions."""
        if error is not None or game is None or game.root is None:
            return
        if not game.root.is_solved:
            return
        # Extract transitions and add to buffer
        transitions = extract_transitions(game.root)
        replay_buffer.add_transitions(transitions)
        # Update monitor
        monitor = get_monitor()
        if monitor is not None:
            monitor.record_transitions(transitions)

    # Create the prover worker
    worker = ProverWorker(
        config=config,
        tactic_model=model,
        lean_address=config.server_address,
        lean_port=config.server_port,
        get_theorem=get_theorem,
        on_result=on_result,
        num_actors=num_actors,
    )
    
    # Start collection
    worker.start()
    
    monitor = get_monitor()
    
    def check_global_collected() -> int:
        """Check how many transitions have been collected globally (across all DDP ranks)."""
        with replay_buffer._lock:
            local_collected = len(replay_buffer.local_buffer)
        if ddp:
            collected_tensor = torch.tensor([local_collected], dtype=torch.long, device=device)
            dist.all_reduce(collected_tensor, op=dist.ReduceOp.SUM)
            return collected_tensor.item()
        return local_collected

    # Wait until target is reached
    loop_count = 0
    try:
        while True:
            global_collected = check_global_collected()
            if global_collected >= total_to_collect:
                if master_process:
                    log(f"Target reached: {global_collected}/{total_to_collect} transitions collected",
                        component="Collection")
                break
            
            # Update monitor with actor status
            if monitor is not None:
                states = worker.get_thread_states()
                for i, state in enumerate(states):
                    monitor.update_local_actor(i, state=state)
            
            # Check if all actors have exited (error condition)
            if worker.has_started_actors() and worker.all_actors_exited():
                log("WARNING: All actors have exited unexpectedly", component="Collection")
                break
            
            # Periodic status update (master only)
            loop_count += 1
            if loop_count % 50 == 0 and master_process:  # Every 5 seconds
                log(f"Progress: {global_collected}/{total_to_collect} transitions", 
                    component="Collection")
            
            time.sleep(0.1)
    finally:
        # Shutdown
        if master_process:
            log(f"Stopping actors...", component="Collection")
        model.shutdown()  # Unblock any waiting threads
        worker.stop()
        
        # Clear local actor status in monitor
        if monitor is not None:
            monitor.clear_local_actors()

    if master_process:
        log(f"Collection complete: {len(replay_buffer.local_buffer)} local transitions", 
            component="Collection")


def _main():
    config = Config()
    model = TacticModel.create(num_samples=config.num_sampled_tactics)
    replay_buffer = ReplayBuffer(config)
    theorems_sampler = TheoremsSampler()
    
    # For testing, use BlockingTacticModel
    batched_model = BlockingTacticModel(
        inner_model=model,
        batch_size=config.num_actors,
        timeout_seconds=0.1
    )
    run_actor(100, config, batched_model, replay_buffer, theorems_sampler)


if __name__ == "__main__":
    _main()
