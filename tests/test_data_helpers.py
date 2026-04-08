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
