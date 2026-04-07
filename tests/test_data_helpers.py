"""
Tests for nanoproof.data helpers. Run with:

    python -m pytest tests/test_data_helpers.py -v
"""

# -----------------------------------------------------------------------------
# shuffle_train_valid_split

from nanoproof.data.rl.common import shuffle_train_valid_split


def test_shuffle_train_valid_split_sizes():
    items = list(range(2000))
    split = shuffle_train_valid_split(items, valid_size=500, seed=0)
    assert set(split.keys()) == {"train", "valid"}
    assert len(split["valid"]) == 500
    assert len(split["train"]) == 1500
    # Disjoint and full coverage
    assert set(split["train"]) | set(split["valid"]) == set(items)
    assert not (set(split["train"]) & set(split["valid"]))


def test_shuffle_train_valid_split_is_deterministic():
    items = list(range(100))
    a = shuffle_train_valid_split(items, valid_size=10, seed=0)
    b = shuffle_train_valid_split(items, valid_size=10, seed=0)
    assert a == b


def test_shuffle_train_valid_split_does_not_mutate_input():
    items = list(range(100))
    snapshot = items[:]
    shuffle_train_valid_split(items, valid_size=10, seed=0)
    assert items == snapshot


def test_shuffle_train_valid_split_actually_shuffles():
    # With seed=0 and 100 items, the valid split should NOT just be the last 10
    items = list(range(100))
    split = shuffle_train_valid_split(items, valid_size=10, seed=0)
    assert split["valid"] != list(range(90, 100))


def test_shuffle_train_valid_split_smaller_than_valid_size():
    items = list(range(5))
    split = shuffle_train_valid_split(items, valid_size=10, seed=0)
    # When valid_size > len(items), valid gets everything and train is empty
    assert len(split["valid"]) == 5
    assert len(split["train"]) == 0


# -----------------------------------------------------------------------------
# deepseek_prover._statement_only

from nanoproof.data.rl.deepseek_prover import _statement_only


def test_statement_only_strips_tactic_proof():
    s = "theorem foo (n : Nat) : n + 0 = n := by\n  simp"
    assert _statement_only(s) == "theorem foo (n : Nat) : n + 0 = n := by sorry"


def test_statement_only_strips_term_proof():
    s = "theorem foo : True := trivial"
    assert _statement_only(s) == "theorem foo : True := by sorry"


def test_statement_only_strips_multiline_tactic_proof():
    s = "theorem foo : 1 + 1 = 2 := by\n  rfl\n  -- another tactic\n  sorry"
    assert _statement_only(s) == "theorem foo : 1 + 1 = 2 := by sorry"


def test_statement_only_strips_already_sorry():
    s = "theorem foo : True := by sorry"
    assert _statement_only(s) == "theorem foo : True := by sorry"


def test_statement_only_returns_none_when_no_assignment():
    # Statement with no `:=` is unparseable - return None and let caller skip
    assert _statement_only("theorem foo : True") is None
    assert _statement_only("") is None


def test_statement_only_strips_whitespace_around_assignment():
    s = "  theorem foo : True   :=   by trivial  "
    assert _statement_only(s) == "theorem foo : True := by sorry"
