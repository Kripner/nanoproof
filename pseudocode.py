"""Pseudocode for AlphaProof: RL, autoformalization, and variant generation."""

# pylint: disable=all

import collections
import dataclasses
import enum
import math
import random
import typing
from typing import Any, Callable, List, Dict

import jax
import jax.numpy as jnp
import optax

Counter = collections.Counter

################################################################################
### RL
################################################################################

##### Helpers #####

# Observations in AlphaProof are the tactic state.
Observation = str

# Actions in AlphaProof are Lean tactics (except for special actions, to start a
# disproof, or to focus on a goal).
Action = str

# Network parameters.
Params = Any


class Player(enum.Enum):
  OR = 1
  AND = 2


class State(typing.NamedTuple):
  id: int
  reward: float
  observation: Observation
  terminal: bool
  num_goals: int


class Theorem(typing.NamedTuple):
  """A theorem to be proved."""
  header: str
  statement: str


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
    # Number of simulations to run. Provided by the matchmaker.
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


class NetworkTrainingOutput(typing.NamedTuple):
  """Output of the network during training."""
  value_logits: jax.Array
  policy_logits: jax.Array


class NetworkSamplingOutput(typing.NamedTuple):
  """Output of the network when sampling actions."""
  action_logprobs: Dict[Action, float]
  value: float


class Network:
  def __init__(self, config: Config):
    self.params = {'weights': jnp.array([0])}

    self.num_value_bins = config.num_value_bins
    self.value_weight = config.value_weight
    self.optimizer = optax.adam(config.lr)
    self.opt_state = self.optimizer.init(self.params)

    def _loss_fn(params, batch):
      loss = 0
      for observations, actions, value_targets in batch:
        network_output = self.forward(params, observations, actions)
        # Policy loss
        loss += jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(
                network_output.policy_logits, actions
            )
        )
        # Value loss
        loss += self.value_weight * value_loss(network_output.value_logits,
                                               value_targets)

      return loss

    self._loss_grad = jax.grad(_loss_fn)

  def forward(
      self, params: Params, observation: jax.Array, action: jax.Array
  ) -> NetworkTrainingOutput:
    # Predict value logits and policy logits from given observation and action.
    # observation and action are passed to the network.
    value_logits = jnp.zeros(self.num_value_bins)
    policy_logits = jnp.array([0])
    return NetworkTrainingOutput(
        value_logits=value_logits, policy_logits=policy_logits
    )

  def sample(self, observation: str) -> NetworkSamplingOutput:
    # Predict value and sample actions from a given observation.
    # observation is tokenized and passed to the network to produce value
    # logits. The value is then calcualated from value logits and bin locations.
    value = 0.
    return NetworkSamplingOutput(action_logprobs={'placeholder_action': -2.},
                                 value=value)

  def update(self, batch: list[tuple[jax.Array, jax.Array, float]]):
    # Update the network weights.
    grads = self._loss_grad(self.params, batch)
    updates, self.opt_state = self.optimizer.update(
        grads, self.opt_state, self.params
    )
    self.params = optax.apply_updates(self.params, updates)


def value_loss(value_logits: jax.Array, value_targets: float) -> float:
  # Calculate the categorical cross-entropy loss.
  return 0.0


class SharedStorage:

  def __init__(self):
    self._params = {}

  def latest_params(self) -> Params:
    return self._params[max(self._params.keys())]

  def save_params(self, step: int, params: Params):
    self._params[step] = params


class Matchmaker:

  @dataclasses.dataclass
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


# Core Monte Carlo tree search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(
    config: Config,
    game: Game,
    network: Network,
    environment: Environment,
):
  root = game.root
  for i in range(game.num_simulations):
    node = root
    search_path = [node]

    while node.expanded() and not progressive_sample(node, config):
      _, node = select_child(config, node)
      search_path.append(node)

    assert node.observation is not None
    network_sample_output = network.sample(node.observation)
    expand_node(node, network_sample_output.action_logprobs,
                environment, config.prior_temperature)
    backpropagate(
        search_path,
        network_sample_output.value,
        config,
    )
    if root.is_optimal:
      break


def progressive_sample(node: Node, config: Config) -> bool:
  """Whether to expand a node in the search tree again (progressive sampling)."""
  return (
      node.to_play == Player.OR
      and node.evaluations <= config.ps_c * node.visit_count**config.ps_alpha
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
    value_score = 0

  if parent.to_play == Player.AND:
    # Invert value score for AND nodes.
    value_score = 1 - value_score
    if child.is_optimal:
      # Avoid re-selecting proven subgoals.
      value_score = -1e9
  return prior_score + value_score


# We expand a node using the value and sampled actions obtained from
# the neural network. Immediately attempt the actions in the environment.
def expand_node(
    node: Node,
    network_action_logprobs: Dict[Action, float],
    environment: Environment,
    temperature: float,
):
  node.evaluations += 1
  policy = {
      a: math.exp(network_action_logprobs[a] / temperature)
      for a in network_action_logprobs
  }
  for action, p in policy.items():
    # Check if action is duplicate.
    if action in node.children:
      node.children[action].prior += p
      continue
    # Immediately apply the actions in the environment.
    try:
      state = environment.step(node.state_id, action)
    except ValueError:
      # Invalid action encountered.
      continue
    else:
      node.children[action] = Node(
          observation=state.observation,
          action=action,
          prior=p,
          state_id=state.id,
          to_play=Player.AND if state.num_goals > 1 else Player.OR,
          is_optimal=state.terminal,
          is_terminal=state.terminal,
          reward=state.reward,
      )
      node.is_optimal |= state.terminal
      if state.num_goals > 1:
        # For AND nodes, immediately add children with pseudo-actions to focus
        # on each goal.
        expand_node(
            node.children[action],
            {f'focus_goal {i}': math.log(1./state.num_goals)
             for i in range(state.num_goals)},
            environment,
            temperature,
        )


def backprop_value_towards_min(node):
  """Computes the value for an AND node by propagating the min value from children."""
  value = 1
  for child in node.children.values():
    if not child.is_optimal and child.visit_count > 0:
      value = min(value, child.value())
  return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(
    search_path: List[Node],
    value: float,
    config: Config,
):
  if not search_path[-1].expanded():
    value = config.no_legal_actions_value
  is_optimal = False
  for ix, node in reversed(list(enumerate(search_path))):
    node.value_sum += value
    node.visit_count += 1
    if node.to_play == Player.AND:
      is_optimal = all(child.is_optimal for child in node.children.values())
    else:
      is_optimal |= node.is_optimal
    node.is_optimal = is_optimal
    if ix > 0 and search_path[ix - 1].to_play == Player.AND:
      value = backprop_value_towards_min(search_path[ix - 1])
    else:
      value = node.reward + value


##### End Actors #####

##### RL part 2: Learning #####


def train_network(config: Config, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):

  network = Network(config)

  for i in range(config.training_steps):
    if i % config.checkpoint_interval == 0:
      storage.save_params(i, network.params)
    batch = replay_buffer.sample_batch()
    network.update(batch)
  storage.save_params(config.training_steps, network.params)


################################################################################
### Autoformalization
################################################################################


class ComputeBudget(enum.Enum):
  LOW = "low"
  HIGH = "high"


def sample_auto_formalization(nl_problem: str) -> str:
  """Samples a Lean formalization using a finetuned version of Gemini."""
  raise NotImplementedError()


def extract_lean_code(sample: str) -> str:
  """Extracts the Lean code from a sample."""
  raise NotImplementedError()


def lean_is_valid_syntax(lean_statement: str) -> bool:
  """Validates Lean code for syntax and common linting errors."""
  raise NotImplementedError()


def lean_is_complete_proof(lean_code: str) -> bool:
  """Checks if Lean accepts the code as a full proof."""
  raise NotImplementedError()


def lean_replace_goal_with_false(lean_code: str) -> str:
  """Creates a new statement where the goal is to prove a contradiction is among the hypotheses."""
  raise NotImplementedError()


def lean_negate_statement(lean_code: str) -> str:
  """Creates a new statement where the goal is to disprove the original statement."""
  raise NotImplementedError()


def is_provable(lean_statement: str, budget: ComputeBudget) -> bool:
  """Runs Alphaproof to check if the Lean statement is provable."""
  raise NotImplementedError()


def has_trivial_counterexample(lean_statement: str) -> bool:
  """Run a modified version of Lean's `plausible` tactic with extra support for real numbers."""
  raise NotImplementedError()


def is_easily_provable(lean_statement: str) -> bool:
  """Checks if the statement can be easily decided by an ad-hoc set of simple tactics."""
  # try to prove the statement
  for tactic in [
      "simp",
      "norm_num",
      "abel",
      "nlinarith",
      "linarith",
      "ring",
      "aesop",
      "trivial",
  ]:
    if lean_is_complete_proof(lean_statement + " := by " + tactic):
      return True

  return False


def deformalize_lean(lean_statement: str) -> str:
  """Deformalizes a Lean statement into a natural language statement."""
  # Uses an off-the-shelf, publicly available model.
  raise NotImplementedError()


def check_cycle_consistency(
    original_statement: str,
    deformalized_statement: str,
) -> bool:
  """Checks if the original and deformalized statements are equivalent."""
  # Uses an off-the-shelf, publicly available model.
  raise NotImplementedError()


def auto_formalize_problem(nl_problem: str, n_samples: int) -> str | None:
  """Translates for a natural language statement into a Lean statement."""

  samples = [sample_auto_formalization(nl_problem) for _ in range(n_samples)]
  lean_problems = [extract_lean_code(sample) for sample in samples]
  vote_counter = Counter(lean_problems)  # Deduplicate and count votes.

  problems_with_votes = [
      (votes, problem) for problem, votes in vote_counter.items()
  ]
  problems_with_votes.sort(reverse=True)  # Order by votes (most to least).

  # Find the most-voted candidate that passes sanity checking.
  for _, lean_problem in problems_with_votes:
    # Remove samples that do not have a valid Lean syntax.
    if not lean_is_valid_syntax(lean_problem):
      continue

    # Create two new Lean statements: one where the goal is to disprove the
    # original statement, and one where the goal is to prove the hypotheses
    # are contradictory.
    lean_negated = lean_negate_statement(lean_problem)
    lean_exfalso = lean_replace_goal_with_false(lean_problem)

    # Discard statements that have a single-tactic proof.
    if (
        is_easily_provable(lean_problem)
        or is_easily_provable(lean_negated)
        or is_easily_provable(lean_exfalso)
    ):
      continue

    # Discard statements that have a trivial counterexamples.
    if has_trivial_counterexample(lean_problem):
      continue

    # Check cycle consistency: ask a public model to deformalize a statement,
    # then ask if the original and deformalized statements are equivalent.
    deformalized_stmt = deformalize_lean(lean_problem)
    if not check_cycle_consistency(nl_problem, deformalized_stmt):
      continue

    # Use small-budget Alphaproof to check if the statement is disprovable or
    # the hypotheses are contradictory.
    if is_provable(lean_negated, ComputeBudget.LOW) or is_provable(
        lean_exfalso, ComputeBudget.LOW
    ):
      continue

    return lean_problem

  # All samples failed.
  return None


################################################################################
### Variant generation
################################################################################


def llm_sample(
    lean_problem: str, example_variants: list[tuple[str, str]],
    temperature: float, persona: str, prompting_strategy: str,
) -> str:
  """Samples formal Lean variants (as a string) via LLM for a Lean problem."""
  raise NotImplementedError()


def extract_lean_problems(sample: str) -> list[str]:
  """Extracts formal Lean problems from an LLM sample."""
  raise NotImplementedError()


def is_valid_syntax(lean_problem: str) -> bool:
  """Checks if a formal Lean problem is syntactically valid."""
  raise NotImplementedError()


def programmatic_augmentation(variants: list[str]) -> list[str]:
  """Generates programmatic variants for a set of formal Lean variants."""
  raise NotImplementedError()


def deduplicate_variants(variants: list[str]) -> list[str]:
  """Deduplicates a set of formal Lean variants."""
  return list(set(variants))


def get_most_interesting_variant(variants: list[str]) -> str:
  """Returns the most interesting variant."""
  raise NotImplementedError()


def example_variants_all() -> list[tuple[str, str]]:
  """Returns a set of example formal Lean problems and their corresponding variants."""
  raise NotImplementedError()


def vary_prompting_strategy() -> bool:
  """Returns whether to vary the prompting strategy."""
  raise NotImplementedError()


def sample_prompting_strategy() -> str:
  """Returns the problem strategy for sampling variants.

  Prompting strategy is one of the following:
  - REFORMULATION: semantically equivalent statement.
  - SIMPLIFICATION: simpler version of the statement.
  - GENERALIZATION: generalization of the statement.
  - SPECIALIZATION: special case of the statement.
  - LEMMA: statement that is useful to solve the original statement.
  - PROOFSTEP: statement that simulates a proof step.
  - PROOF_SIMULATION: variants that simulate a proof.
  - DEFINITION: statement that contains a new definition.
  - ANALOGY: statement that is analogous to the original statement.
  - PARTIALPOINTS: statement that is worth partial points.
  - HINDSIGHT: statement that empirically worked well.
  - PROBLEM_DECOMPOSITION: decomposing a statement into parts.
  - PART2PART: different part of the same underlying statement.
  - PROBLEM2PART: decomposing a statement into a part.
  """
  raise NotImplementedError()


def sample_variants(lean_problem: str) -> list[str]:
  """Samples a set of formal Lean variants for a formal Lean problem.

  For each variant, we call this function multiple times (possibly in parallel)
  to generate enough variants.

  Args:
    lean_problem: The Lean problem to sample variants for.
  Returns:
    A list of sampled Lean variants.
  """
  example_variants = example_variants_all()
  variants = []
  current_problem = lean_problem
  num_evolutions = random.choice([1, 3, 6, 10, 15])
  vary_prompting = vary_prompting_strategy()
  prompt_strategy = sample_prompting_strategy()
  for _ in range(num_evolutions):
    temperature = random.choice([0.5, 1.0, 1.5])
    persona = random.choice(["IMO winner", "Putnam winner"])
    if vary_prompting:
      prompt_strategy = sample_prompting_strategy()
    variant_sample = llm_sample(
        lean_problem=current_problem,
        example_variants=example_variants,
        temperature=temperature,
        persona=persona,
        prompting_strategy=prompt_strategy,
    )
    extracted_lean_problems = extract_lean_problems(variant_sample)
    current_lean_variants = [
        extracted_lean_problem
        for extracted_lean_problem in extracted_lean_problems
        if is_valid_syntax(extracted_lean_problem)
    ]
    if not current_lean_variants:
      break
    variants.extend(current_lean_variants)
    current_problem = get_most_interesting_variant(
        variants=current_lean_variants
    )
    programmatic_variants = programmatic_augmentation(current_lean_variants)
    variants.extend(
        programmatic_variant
        for programmatic_variant in programmatic_variants
        if is_valid_syntax(programmatic_variant)
    )
  variants = deduplicate_variants(variants)
  return variants
