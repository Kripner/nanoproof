# RL TODO

Weaknesses identified during the pre-RL code review. File:line anchors are pre-refactor; update once the refactor commits land.

## Critical

- [ ] **Action log-probs unused in UCB** - all sampled tactics get prior 1.0; the model's policy distribution is ignored in tree traversal. The "policy" head only influences search through *which* tactics get sampled, not how the tree exploits them. Author flagged: `# TODO (!): use the actual action logprobs`. [nanoproof/search.py:502](nanoproof/search.py#L502)
- [ ] **Value target semantics are confused** - `compute_value_target` returns `-1 + max_child` (an MCTS-best-path depth biased by simulation budget), then `rl.py` does `proof_depth = -value_target` and bins it. Works in practice only because depths are small. The value head is being trained on a target that isn't the true minimax depth. [nanoproof/search.py:236-252](nanoproof/search.py#L236-L252), [nanoproof/rl.py:217-218](nanoproof/rl.py#L217-L218)
- [ ] **Duplicate-action handling in MCTS expansion is almost certainly wrong** - when a sampled tactic already exists as a child, priors are summed and the second expansion is silently dropped. Author flagged: `# TODO: wtf is this?`. [nanoproof/search.py:604-606](nanoproof/search.py#L604-L606)
- [ ] **No gradient clipping** before `optimizer.step()`. With Muon + high embedding LR (0.2), this is risky. [nanoproof/rl.py:554](nanoproof/rl.py#L554)
- [ ] **`--value-weight` default 0.01** - value head trained at ~1% the effective rate of the policy head. Either bump default to ~0.1-0.5 or document the justification. [nanoproof/rl.py:75](nanoproof/rl.py#L75)
- [ ] **Only positive signal - solved-only learning** - failed proofs are entirely discarded. No negative gradient on bad tactics, no exposure to "tactic that locally seemed good but didn't pan out". RL gains will plateau once the model can't bootstrap new solves on its own.
- [ ] **Replay buffer effectively unbounded** - `window_size: int = 60_000_000` tokens. Old stale transitions accumulate forever; off-policy drift unchecked. [nanoproof/experience_collection.py:68](nanoproof/experience_collection.py#L68)
- [ ] **UCB value term can blow up** - `value_discount ** (-1 - value)` is unbounded when `value < -1` (i.e. depth ≥ 2 along the best subtree). Currently masked by short proofs but not bounded. [nanoproof/search.py:555-576](nanoproof/search.py#L555-L576)
- [ ] **`Node.value` defaults to 0 when `visit_count == 0`** - affects early UCB tie-breaking. Author flagged: `# TODO: isn't this also weird?`. [nanoproof/search.py:73-75](nanoproof/search.py#L73-L75)

## Smaller

- [ ] **Replay-buffer assertion blocks step 0** - `assert len(replay_buffer.buffer) >= args.collect_transitions` fires if collection on the first step is slow. [nanoproof/rl.py:209](nanoproof/rl.py#L209)
- [ ] **No semantic validation of extracted transitions** - only length filtering; no NaN/Inf checks, no per-proof dedup. [nanoproof/experience_collection.py:75-88](nanoproof/experience_collection.py#L75-L88)
- [ ] **Prompt-token masking is fragile** - done in the dataloader and *again* via `targets >= 0` at loss time. A dataloader bug would silently train on prompt tokens. [nanoproof/rl.py:541](nanoproof/rl.py#L541), [nanoproof/data/sft/leantree_dataloader.py:76-82](nanoproof/data/sft/leantree_dataloader.py#L76-L82)
- [ ] **No-progress timeouts default to 5 min (collect) / 2 min (eval)** - long enough to waste meaningful compute when provers hang. [nanoproof/rl_server.py:505-621](nanoproof/rl_server.py#L505-L621), [nanoproof/rl_server.py:646-831](nanoproof/rl_server.py#L646-L831)
