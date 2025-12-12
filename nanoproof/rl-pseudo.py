import enum
import random
from dataclasses import dataclass
from typing import Callable

from nanoproof.core import Observation, Theorem, Action
from nanoproof.model import Network


class Player(enum.Enum):
  OR = 1
  AND = 2


@dataclass
class State:
  id: int
  reward: float
  observation: Observation
  terminal: bool
  num_goals: int


class Environment:
  """Lean environment."""

  def initial_state(self, theorem: Theorem) -> State:
    """Returns the initial tactic state."""
    raise NotImplementedError()

  def step(self, state_id: int, action: Action) -> State:
    """Applies the action in the given state, returns the new state."""
    raise NotImplementedError()


class Config:
  def __init__(
      self,
      num_simulations: int,
      batch_size: int,
      num_actors: int,
      lr: float,
      environment_ctor: Callable[[], Environment] = lambda: Environment(),
  ):
    ### Acting
    self.environment_ctor = environment_ctor
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

    ### Training
    self.training_steps = int(1000e3)
    self.checkpoint_interval = int(1e3)
    self.window_size = int(1e6)
    self.batch_size = batch_size
    self.sequence_length = 32
    self.lr = lr
    self.value_weight = 0.001

    # Matchmaker
    self.mm_disprove_rate = 0.5
    self.mm_trust_count = 8
    self.mm_fully_decided_trust_count = 12
    self.mm_proved_weight = 1e-3
    self.mm_undecided_weight = 0.1


class Node:
  """Node in the search tree."""

  def __init__(
      self,
      action: Action | None,
      observation: Observation,
      prior: float,
      state_id: int,
      to_play: Player,
      reward: float,
      is_optimal: bool = False,
      is_terminal: bool = False,
  ):
    # Action that was taken to reach this node.
    self.action = action
    # Observation after the action has been applied.
    self.observation = observation
    # Environment state ID after the action has been applied.
    self.state_id = state_id
    # Whether the node is an OR or AND node.
    self.to_play = to_play
    # Whether the action closed the proof of the previous goal.
    self.is_terminal = is_terminal
    # Whether the node is part of an optimal path.
    self.is_optimal = is_optimal
    # Per-step reward obtained after applying the action.
    self.reward = reward
    # Prior probability of the node according to the policy.
    self.prior = prior

    self.visit_count = 0
    self.evaluations = 0
    self.value_sum = 0
    self.children: dict[Action, Node] = {}

    # Not used in search, but used as a regression target in RL.
    self.value_target = 0

  def expanded(self) -> bool:
    return len(self.children) > 0

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

  def prior_sum(self) -> float:
    return sum(child.prior for child in self.children.values())


class Game:
  """A single episode of interaction with the environment."""

  def __init__(self, theorem: Theorem, disprove: bool, num_simulations: int):
    self.theorem = theorem
    # Whether to try to prove or disprove the theorem.
    self.disprove = disprove
    # Number of simulations to run. Providehd by the matchmaker.
    self.num_simulations = num_simulations
    # Dummy node for the type checker.
    self.root = Node(
        action=None,
        observation='',
        prior=1.0,
        state_id=0,
        to_play=Player.OR,
        reward=0.0,
    )


def compute_value_target(node: Node) -> float:
  """Computes the actual value for a node, to be used as a target in learning."""
  if node.is_terminal:
    node.value_target = 0
    return 0
  elif node.to_play == Player.OR:
    action = select_optimal_action(node)
    child_value = compute_value_target(node.children[action])
    value = -1 + child_value
    node.value_target = value
    return value
  elif node.to_play == Player.AND:
    value = min(compute_value_target(child) for child in node.children.values())
    node.value_target = value
    return value
  else:
    raise ValueError(f'Unknown to_play: {node.to_play}')


def extract_transitions(node: Node) -> list[tuple[Observation, Action, float]]:
  """Extracts transitions from the game."""
  if not node.is_optimal:
    return []
  assert node.to_play == Player.OR
  transitions = []
  while node.to_play == Player.OR and not node.is_terminal:
    action = select_optimal_action(node)
    transitions.append((node.observation, action, node.value_target))
    node = node.children[action]
  if node.to_play == Player.AND:
    for _, child in node.children.items():
      transitions.extend(extract_transitions(child))
  return transitions


def select_optimal_action(node: Node) -> Action:
  """Selects the optimal action from the node."""
  assert node.to_play == Player.OR
  [(action, _)] = [
      (action, child)
      for action, child in node.children.items()
      if child.is_optimal
  ]
  return action


def final_check(game: Game) -> bool:
  """Checks that the proof found is actually valid."""
  # Extract tactics from the tree, write the statement and its proof to a file,
  # add a footer checking the axioms, and then run the `lean` binary.
  # Properly handle the case where we attempt to disprove a theorem.
  return True


class ReplayBuffer:

  def __init__(self, config: Config):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.sequence_length = config.sequence_length
    self.buffer = []

  def save_game(self, game):
    transitions = extract_transitions(game.root)
    self.buffer.extend(transitions)
    self.buffer = self.buffer[-self.window_size:]

  def sample_batch(self) -> list[tuple[jax.Array, jax.Array, float]]:
    return [self.sample_transition() for _ in range(self.batch_size)]

  def sample_transition(self) -> tuple[jax.Array, jax.Array, float]:
    # Sample transition from buffer either uniformly or according to some
    # priority.
    observation, action, value = self.buffer[0]
    tokenized_observation = self.tokenize(observation)
    tokenized_action = self.tokenize(action)
    return (tokenized_observation, tokenized_action, value)

  def tokenize(self, input_string: str) -> jax.Array:
    return jnp.zeros((self.batch_size, self.sequence_length), dtype=jnp.int32)



class SharedStorage:

  def __init__(self):
    self._params = {}

  def latest_params(self) -> Params:
    return self._params[max(self._params.keys())]

  def save_params(self, step: int, params: Params):
    self._params[step] = params


class Matchmaker:
  @dataclass
  class Stats:
    """Statistics for a theorem."""
    # List of (disprove, result) tuples:
    # Disprove is True iff this was an attempt to disprove the theorem.
    # Result is True iff the attempt was successful.
    attempts: list[tuple[bool, bool]]

    def update(self, game: Game):
      """Update statistics with the results of a game."""
      self.attempts.append((game.disprove, game.root.is_optimal))

    def weight(self, config: Config) -> float:
      """Compute weight of this theorem."""
      if not self.attempts:
        return 1.0
      disproved = any(
          disprove and success for (disprove, success) in self.attempts
      )
      proved = any(
          (not disprove) and success for (disprove, success) in self.attempts
      )
      if disproved:
        return 0.0
      elif len(self.attempts) < config.mm_trust_count:
        return 1.0
      elif not disproved and not proved:
        # Never managed to prove or disprove.
        return config.mm_undecided_weight
      else:
        latest = self.attempts[-config.mm_fully_decided_trust_count :]
        if all((not disprove) and success for (disprove, success) in latest):
          # Consistently proved.
          return config.mm_proved_weight
      return 1.0

  def __init__(self, config: Config):
    self.config = config
    # Load theorems and their stats from the database.
    self.theorem_stats: dict[Theorem, Matchmaker.Stats] = {}

  def compute_num_simulations(self, theorem: Theorem, stats: Stats) -> int:
    """Compute number of simulations to run for a theorem."""
    return 1000

  def get_start_position(self) -> Game:
    """Get a start position for a new game to be played."""
    # Get a theorem to be proved or disproved based on the per-theorem stats.
    # Prefer interesting theorems.
    weights = [
        stats.weight(self.config) for stats in self.theorem_stats.values()
    ]
    [(theorem, stats)] = random.choices(
        list(self.theorem_stats.items()), weights, k=1
    )
    disprove = random.random() < self.config.mm_disprove_rate
    num_simulations = self.compute_num_simulations(theorem, stats)
    return Game(
        theorem=theorem, disprove=disprove, num_simulations=num_simulations
    )

  def send_game(self, game: Game):
    """Send completed game to matchmaker."""
    self.theorem_stats[game.theorem].update(game)


def make_config() -> Config:
  return Config(
      num_simulations=800,
      batch_size=2048,
      num_actors=3000,
      lr=1.0,
  )


def launch_job(f, *args):
  f(*args)



##### End Helpers #####


# AlphaProof training is split into two independent parts: A learner which
# updates the network, and actors which play games to generate data.
# These two parts only communicate by transferring the latest network checkpoint
# from the learner to the actor, and the finished games from the actor
# to the learner.
def alphaproof_train(config: Config) -> Network:
  storage = SharedStorage()
  replay_buffer = ReplayBuffer(config)
  matchmaker = Matchmaker(config)

  for _ in range(config.num_actors):
    launch_job(run_actor, config, storage, replay_buffer, matchmaker)

  train_network(config, storage, replay_buffer)

  return storage.latest_params()


##### RL part 1: Actors #####


# Each acting job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the learner by
# writing it to a shared replay buffer.
def run_actor(config: Config, storage: SharedStorage,
              replay_buffer: ReplayBuffer, matchmaker: Matchmaker):
  network = Network(config)
  while True:
    network.params = storage.latest_params()
    game = play_game(config, network, matchmaker)
    if game.root.is_optimal:
      replay_buffer.save_game(game)
    matchmaker.send_game(game)


# Each game is produced by starting from the initial Lean state, and executing
# Monte Carlo tree search to find a proof. If one is found, we extract from the
# search tree the state-tactic-value transitions in the proof, which are added
# to a replay buffer for training.
def play_game(config: Config, network: Network, matchmaker: Matchmaker) -> Game:
  game = matchmaker.get_start_position()
  environment = config.environment_ctor()

  state = environment.initial_state(game.theorem)
  if game.disprove:
    state = environment.step(state.id, 'disprove')
  game.root = Node(
      action=None,
      observation=state.observation,
      prior=1.0,
      to_play=Player.OR,
      state_id=state.id,
      is_optimal=state.terminal,
      is_terminal=state.terminal,
      reward=state.reward,
  )
  assert game.root.to_play == Player.OR

  # Run Monte Carlo tree search to find a proof.
  run_mcts(config, game, network, environment)
  if game.root.is_optimal:
    # Perform final check to ensure the proof is valid.
    game.root.is_optimal = final_check(game)
    # Compute value targets for the proof.
    compute_value_target(game.root)

  return game


def train_network(config: Config, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):

  network = Network(config)

  for i in range(config.training_steps):
    if i % config.checkpoint_interval == 0:
      storage.save_params(i, network.params)
    batch = replay_buffer.sample_batch()
    network.update(batch)
  storage.save_params(config.training_steps, network.params)