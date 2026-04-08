"""
Tests for nanoproof.data and checkpoint helpers. Run with:

    python -m pytest tests/test_data_helpers.py -v
"""

import os

# -----------------------------------------------------------------------------
# parse_checkpoint_path

from nanoproof.checkpoints import parse_checkpoint_path
from nanoproof.common import get_base_dir


def test_parse_checkpoint_path_absolute():
    p = "/abs/path/to/pretrain/run/model_005000.pt"
    d, s = parse_checkpoint_path(p)
    assert d == "/abs/path/to/pretrain/run"
    assert s == 5000


def test_parse_checkpoint_path_relative():
    d, s = parse_checkpoint_path("pretrain/run/model_000123.pt")
    assert d == os.path.join(get_base_dir(), "models", "pretrain", "run")
    assert s == 123


def test_parse_checkpoint_path_zero_step():
    _, s = parse_checkpoint_path("pretrain/run/model_000000.pt")
    assert s == 0


def test_parse_checkpoint_path_high_step():
    _, s = parse_checkpoint_path("pretrain/run/model_999999.pt")
    assert s == 999999


def test_parse_checkpoint_path_rejects_directory():
    try:
        parse_checkpoint_path("pretrain/run")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for directory path")


def test_parse_checkpoint_path_rejects_non_model_file():
    try:
        parse_checkpoint_path("pretrain/run/meta_005000.json")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-model file")


def test_parse_checkpoint_path_rejects_non_numeric_step():
    try:
        parse_checkpoint_path("pretrain/run/model_abcdef.pt")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-numeric step")


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


def test_statement_only_appends_sorry_after_by():
    s = "theorem foo (n : Nat) : n + 0 = n := by"
    assert _statement_only(s) == "theorem foo (n : Nat) : n + 0 = n := by sorry"


def test_statement_only_appends_by_sorry_after_assign():
    s = "theorem foo : True :="
    assert _statement_only(s) == "theorem foo : True := by sorry"


def test_statement_only_strips_trailing_whitespace():
    s = "theorem foo : True := by\n\n   "
    assert _statement_only(s) == "theorem foo : True := by sorry"


def test_statement_only_returns_none_when_no_clean_ending():
    # Anything not ending in `:=` or `:= by` is unparseable - return None
    assert _statement_only("theorem foo : True") is None
    assert _statement_only("theorem foo : True := by trivial") is None
    assert _statement_only("") is None


def test_statement_only_preserves_let_bindings_inside_statement():
    """Regression test: a theorem header may contain `let x := ...` bindings
    *inside* the statement body. We must not split on the first ``:=`` and
    truncate the let-binding away. The whole header (including let bindings)
    must be preserved up to the trailing ``:= by``.
    """
    s = (
        "theorem thm_0 :\n"
        "  let h := (3 : ℝ) / 2;\n"
        "  let n := 5;\n"
        "  h^n ≤ 0.5 → false := by"
    )
    expected = (
        "theorem thm_0 :\n"
        "  let h := (3 : ℝ) / 2;\n"
        "  let n := 5;\n"
        "  h^n ≤ 0.5 → false := by sorry"
    )
    assert _statement_only(s) == expected


def test_statement_only_preserves_multiple_let_bindings():
    s = (
        "theorem thm_2 (PQ PR : ℝ) (h₀ : PQ = 4) :\n"
        "  let PL := PR / 2;\n"
        "  let RM := PQ / 2;\n"
        "  let QR := PQ;\n"
        "  QR = 9 := by"
    )
    expected = s + " sorry"
    assert _statement_only(s) == expected
