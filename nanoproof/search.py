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