"""
Experience collection for RL training.

When distributed=False, this module uses ProverWorker from prover_server.py
to run actors locally. When distributed=True, the rl_server.py handles 
remote prover coordination.
"""

import random
import threading
import time

import torch
import torch.distributed as dist

from nanoproof.common import get_dist_info
from nanoproof.prover_server import ProverWorker, TheoremsSampler
from nanoproof.search import Node, Player, Config, BatchedTacticModel
from nanoproof.cli import get_monitor, log

# Re-export for backwards compatibility
__all__ = ['ReplayBuffer', 'TheoremsSampler', 'Config', 'run_actor']


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


class _ReplayBufferAdapter:
    """
    Adapts ReplayBuffer to the LocalBuffer interface expected by ProverWorker.
    
    This allows ProverWorker to write transitions directly into a ReplayBuffer.
    """
    def __init__(self, replay_buffer: ReplayBuffer):
        self._replay_buffer = replay_buffer
        self._lock = threading.Lock()
        self.games_played = 0
        self.games_solved = 0
        self.expansions = 0
    
    def add_transitions(self, transitions: list, solved: bool):
        """Add transitions to the replay buffer."""
        with self._lock:
            if transitions:
                self._replay_buffer.local_buffer.extend(transitions)
                # Record to monitor for real-time display
                monitor = get_monitor()
                if monitor is not None:
                    monitor.record_transitions(transitions)
            self.games_played += 1
            if solved:
                self.games_solved += 1
    
    def add_tactic(self, state: str, tactic: str, success: bool):
        """Record a tactic application (for web UI display)."""
        pass  # Not needed for RL training
    
    def add_expansion(self):
        """Record an MCTS expansion."""
        with self._lock:
            self.expansions += 1
    
    def get_stats(self) -> dict:
        with self._lock:
            return {
                "transitions_pending": len(self._replay_buffer.local_buffer),
                "games_played": self.games_played,
                "games_solved": self.games_solved,
                "expansions": self.expansions,
            }


@torch.no_grad()
def run_actor(
    total_to_collect: int,
    config: Config,
    model: BatchedTacticModel,
    replay_buffer: ReplayBuffer,
    theorems_sampler: TheoremsSampler
):
    """
    Run parallel actors to collect proofs locally.
    
    Uses ProverWorker to manage actor threads. LLM calls are automatically
    batched via the BatchedTacticModel.
    """
    ddp, _, _, world_size = get_dist_info()
    device = model.network.get_device()
    num_actors = config.num_actors

    log(f"Starting collection with {num_actors} actors, target={total_to_collect} transitions", 
        component="Collection")

    # Create adapter for the replay buffer
    buffer_adapter = _ReplayBufferAdapter(replay_buffer)
    
    # Create the prover worker
    worker = ProverWorker(
        config=config,
        tactic_model=model,
        lean_server_address=config.server_address,
        lean_server_port=config.server_port,
        buffer=buffer_adapter,
        num_actors=num_actors,
        theorems_seed=theorems_sampler.rng.randint(0, 2**31),
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
            
            # Periodic status update
            loop_count += 1
            if loop_count % 50 == 0:  # Every 5 seconds
                log(f"Progress: {global_collected}/{total_to_collect} transitions", 
                    component="Collection")
            
            time.sleep(0.1)
    finally:
        # Shutdown
        log(f"Stopping actors...", component="Collection")
        model.shutdown()  # Unblock any waiting threads
        worker.stop()
        
        # Clear local actor status in monitor
        if monitor is not None:
            monitor.clear_local_actors()

    log(f"Collection complete: {len(replay_buffer.local_buffer)} local transitions", 
        component="Collection")


def _main():
    config = Config()
    from nanoproof.search import TacticModel
    model = TacticModel.create(num_samples=config.num_sampled_tactics)
    replay_buffer = ReplayBuffer(config)
    theorems_sampler = TheoremsSampler()
    
    # For testing, use BatchedTacticModel
    batched_model = BatchedTacticModel(
        inner_model=model,
        batch_size=config.num_actors,
        timeout_seconds=0.1
    )
    run_actor(100, config, batched_model, replay_buffer, theorems_sampler)


if __name__ == "__main__":
    _main()
