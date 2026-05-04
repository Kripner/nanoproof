"""
Tests for the --disable-solvers feature. Run with:

    python -m pytest tests/test_disable_solvers.py -v
"""

import pytest

from nanoproof.common import Player, ValueOrError
from nanoproof.experience_collection import extract_transitions
from nanoproof.search import (
    Node,
    SOLVER_TACTIC_NAMES,
    close_leaves_with_grind,
    expand_node,
    is_solver_tactic,
)


# -----------------------------------------------------------------------------
# Test fakes


class FakeBranch:
    """Stand-in for leantree's LeanProofBranch.

    ``mapping`` maps a tactic string to a ``ValueOrError[list[FakeBranch]]``
    that ``try_apply_tactic`` returns. The ``state`` attribute is mirrored on
    the same object so the production code path that does ``str(branch.state)``
    still works (``__str__`` returns the descriptor we set at construction).
    """

    def __init__(self, descriptor: str, mapping: dict | None = None):
        self.descriptor = descriptor
        self.state = self  # str(branch.state) -> str(self) -> descriptor
        self.mapping = mapping or {}
        self.calls: list[str] = []

    def __str__(self):
        return self.descriptor

    def try_apply_tactic(self, tactic: str, timeout=None):
        self.calls.append(tactic)
        if tactic in self.mapping:
            return self.mapping[tactic]
        return ValueOrError.from_error(f"unknown tactic: {tactic}")


# -----------------------------------------------------------------------------
# is_solver_tactic


def test_is_solver_tactic_exact_match():
    for name in SOLVER_TACTIC_NAMES:
        assert is_solver_tactic(name)
        assert is_solver_tactic(f"  {name}  ")


def test_is_solver_tactic_with_arguments():
    assert is_solver_tactic("grind +arith")
    assert is_solver_tactic("grind?")
    assert is_solver_tactic("aesop (config := ...)")
    assert is_solver_tactic("lia,")


def test_is_solver_tactic_rejects_lookalikes():
    assert not is_solver_tactic("grindy_lemma")
    assert not is_solver_tactic("grinder")
    assert not is_solver_tactic("grind_extra")
    assert not is_solver_tactic("liability")
    assert not is_solver_tactic("aesopian")


def test_is_solver_tactic_non_string():
    assert not is_solver_tactic(None)
    assert not is_solver_tactic(7)


# -----------------------------------------------------------------------------
# extract_transitions filter_grind


def _make_or_node(parent, action, descriptor: str, value_target: float | None = None):
    branch = FakeBranch(descriptor)
    n = Node(
        parent=parent,
        action=action,
        prior=None if action is None else 1.0,
        state=[branch],
        reward=None if action is None else -1.0,
        to_play=Player.OR,
    )
    n.value_target = value_target
    return n


def _make_terminal(parent, action):
    n = Node(
        parent=parent,
        action=action,
        prior=1.0,
        state=[],
        reward=-1.0,
        to_play=Player.OR,
    )
    n.is_solved = True
    n.value_target = 0
    return n


def test_extract_transitions_filters_grind():
    # root --intro x--> mid --grind--> terminal
    root = _make_or_node(None, None, "root_state", value_target=-2)
    mid = _make_or_node(root, "intro x", "mid_state", value_target=-1)
    term = _make_terminal(mid, "grind")
    mid.children = {"grind": term}
    mid.is_solved = True
    root.children = {"intro x": mid}
    root.is_solved = True

    # Without filter we keep both transitions.
    raw = extract_transitions(root, filter_grind=False)
    assert [t[1] for t in raw] == ["intro x", "grind"]

    # With filter we drop grind.
    filtered = extract_transitions(root, filter_grind=True)
    assert [t[1] for t in filtered] == ["intro x"]
    assert filtered[0][0] == "root_state"
    assert filtered[0][2] == -2


def test_extract_transitions_drops_only_solver_tactics():
    # root --aesop--> terminal: filter drops the only transition.
    root = _make_or_node(None, None, "root_state", value_target=-1)
    term = _make_terminal(root, "aesop")
    root.children = {"aesop": term}
    root.is_solved = True

    assert extract_transitions(root, filter_grind=True) == []
    assert len(extract_transitions(root, filter_grind=False)) == 1


# -----------------------------------------------------------------------------
# expand_node injects grind


def test_expand_node_injects_grind():
    branch = FakeBranch(
        "root",
        {
            "intro x": ValueOrError.from_success([FakeBranch("after_intro")]),
            "grind": ValueOrError.from_success([]),  # grind closes the goal
        },
    )
    root = Node(
        parent=None,
        action=None,
        prior=None,
        state=[branch],
        reward=None,
        to_play=Player.OR,
    )

    expand_node(
        root,
        actions=["intro x"],
        action_logprobs=[0.0],
        temperature=1.0,
        inject_grind=True,
    )

    assert "grind" in root.children
    # grind closed the goal, so the child is terminal and is_solved.
    assert root.children["grind"].is_terminal
    assert root.children["grind"].is_solved
    # intro x produced a residual branch and is therefore not solved.
    assert "intro x" in root.children
    assert not root.children["intro x"].is_solved


def test_expand_node_no_inject_when_disabled():
    branch = FakeBranch(
        "root",
        {"intro x": ValueOrError.from_success([FakeBranch("after_intro")])},
    )
    root = Node(
        parent=None,
        action=None,
        prior=None,
        state=[branch],
        reward=None,
        to_play=Player.OR,
    )

    expand_node(
        root,
        actions=["intro x"],
        action_logprobs=[0.0],
        temperature=1.0,
        inject_grind=False,
    )

    assert "grind" not in root.children
    # We never asked grind to be tried.
    assert "grind" not in branch.calls


# -----------------------------------------------------------------------------
# close_leaves_with_grind


def test_close_leaves_with_grind_solves_in_bfs_order():
    # Build:
    #     root (OR)
    #     +-- a -> shallow_leaf (OR, unsolved)
    #     +-- b -> mid (OR, expanded, unsolved)
    #              +-- c -> deep_leaf (OR, unsolved)
    # Each FakeBranch maps "grind" -> success([]).
    shallow_branch = FakeBranch(
        "shallow", {"grind": ValueOrError.from_success([])}
    )
    deep_branch = FakeBranch(
        "deep", {"grind": ValueOrError.from_success([])}
    )

    root = Node(
        parent=None, action=None, prior=None,
        state=[FakeBranch("root_state")], reward=None, to_play=Player.OR,
    )
    shallow_leaf = Node(
        parent=root, action="a", prior=1.0,
        state=[shallow_branch], reward=-1.0, to_play=Player.OR,
    )
    mid = Node(
        parent=root, action="b", prior=1.0,
        state=[FakeBranch("mid_state")], reward=-1.0, to_play=Player.OR,
    )
    root.children = {"a": shallow_leaf, "b": mid}
    deep_leaf = Node(
        parent=mid, action="c", prior=1.0,
        state=[deep_branch], reward=-1.0, to_play=Player.OR,
    )
    mid.children = {"c": deep_leaf}

    closed = close_leaves_with_grind(root, timeout=1000)

    # The shallow leaf gets closed first; that does NOT solve root (root
    # needs both children solved at its OR? No — OR only needs one.) Hmm:
    # for an OR root, any solved child means root is solved → close=1 and
    # we early-exit. Verify behavior matches the early-exit contract.
    assert closed == 1
    assert root.is_solved
    assert shallow_leaf.is_solved
    # Shallow tried first (BFS); deep should not have been touched.
    assert shallow_branch.calls == ["grind"]
    assert deep_branch.calls == []


def test_close_leaves_with_grind_handles_failure():
    # grind fails on the only leaf -> nothing closed, root stays unsolved.
    failing_branch = FakeBranch(
        "leaf", {"grind": ValueOrError.from_error("grind failed")}
    )
    root = Node(
        parent=None, action=None, prior=None,
        state=[failing_branch], reward=None, to_play=Player.OR,
    )
    closed = close_leaves_with_grind(root, timeout=1000)
    assert closed == 0
    assert not root.is_solved
    assert failing_branch.calls == ["grind"]


def test_close_leaves_with_grind_rejects_partial_close():
    # grind "succeeds" but leaves a residual branch; we don't count it as a
    # solve since it didn't close the goal entirely.
    partial_branch = FakeBranch(
        "leaf", {"grind": ValueOrError.from_success([FakeBranch("residual")])}
    )
    root = Node(
        parent=None, action=None, prior=None,
        state=[partial_branch], reward=None, to_play=Player.OR,
    )
    closed = close_leaves_with_grind(root, timeout=1000)
    assert closed == 0
    assert not root.is_solved
