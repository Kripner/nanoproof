from dataclasses import dataclass

# Observations in AlphaProof are the tactic state.
Observation = str

# Actions in AlphaProof are Lean tactics (except for special actions, to start a
# disproof, or to focus on a goal).
Action = str


@dataclass
class Theorem:
    """A theorem to be proved."""
    header: str
    statement: str
