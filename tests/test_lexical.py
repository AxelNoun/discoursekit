from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Sequence
from fractions import Fraction
from itertools import accumulate

import numpy as np
import pytest

from discoursekit import DEFAULT_WINDOW, LexicalDiversity, lexical_diversity, mattr


def brute_force_mattr(tokens: Sequence[str], window: int) -> Fraction:
    """Definition of MATTR written naively, in exact arithmetic."""
    ratios = [
        Fraction(len(set(tokens[i : i + window])), window)
        for i in range(len(tokens) - window + 1)
    ]
    return sum(ratios, Fraction(0)) / len(ratios)


def zipfian_tokens(n: int, seed: int, vocabulary: int = 5000) -> list[str]:
    """Draw n words from a fixed vocabulary with Zipf-like frequencies.

    The generating process is the same at every length, so any change of a
    measure with n is an artefact of length, not a change in the "speaker".
    """
    words = [f"w{rank}" for rank in range(1, vocabulary + 1)]
    cumulative = list(accumulate(1 / rank for rank in range(1, vocabulary + 1)))
    return random.Random(seed).choices(words, cum_weights=cumulative, k=n)


# Values worked out by hand


def test_hand_computed_value() -> None:
    # windows: (a b a) 2/3, (b a b) 2/3, (a b c) 3/3
    assert mattr(["a", "b", "a", "b", "c"], window=3) == pytest.approx(7 / 9)


def test_hand_computed_value_with_repeat_inside_window() -> None:
    # windows: (a a) 1/2, (a b) 2/2, (b a) 2/2
    assert mattr(["a", "a", "b", "a"], window=2) == pytest.approx(2.5 / 3)


def test_all_tokens_distinct_gives_one() -> None:
    assert mattr([str(i) for i in range(200)], window=50) == 1.0


def test_single_repeated_token_gives_one_over_window() -> None:
    assert mattr(["the"] * 200, window=50) == pytest.approx(1 / 50)


def test_window_of_one_is_always_one() -> None:
    assert mattr(["a", "a", "b"], window=1) == 1.0


def test_window_equal_to_length_is_the_plain_ratio() -> None:
    tokens = ["a", "b", "a", "c", "a", "b"]
    assert mattr(tokens, window=len(tokens)) == pytest.approx(3 / 6)


# Agreement with the definition


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("window", [1, 2, 7, 50, 300])
def test_matches_brute_force_definition(seed: int, window: int) -> None:
    rng = random.Random(seed)
    tokens = [rng.choice("abcdefghijklmnopqrst") for _ in range(300)]
    expected = brute_force_mattr(tokens, window)
    assert mattr(tokens, window) == pytest.approx(float(expected), abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_value_is_within_bounds(seed: int) -> None:
    tokens = zipfian_tokens(1000, seed)
    assert 1 / DEFAULT_WINDOW <= mattr(tokens) <= 1.0


def test_accepts_any_sequence_and_does_not_modify_it() -> None:
    as_list = ["a", "b", "a", "b", "c"]
    as_tuple = tuple(as_list)
    assert mattr(as_tuple, window=3) == mattr(as_list, window=3)
    assert as_list == ["a", "b", "a", "b", "c"]


# The reason this measure was chosen


def test_stable_across_length_where_raw_ttr_is_not() -> None:
    # With independent draws every window has the same distribution whatever
    # the text length, so the expected MATTR is identical for short and long
    # texts. Averaging over many pairs brings sampling noise (standard error
    # of the mean difference about 0.0014 here) well below the tolerance.
    short_n, long_n, n_pairs = 300, 3000, 300
    ttr_drop = []
    mattr_shift = []
    for seed in range(n_pairs):
        short = zipfian_tokens(short_n, seed)
        long = zipfian_tokens(long_n, seed + 10_000)
        ttr_drop.append(len(set(short)) / short_n - len(set(long)) / long_n)
        mattr_shift.append(mattr(short) - mattr(long))

    mean_ttr_drop = sum(ttr_drop) / n_pairs
    mean_mattr_shift = sum(mattr_shift) / n_pairs

    # Same generating process, 10 times more text: raw TTR falls by about
    # 0.25, MATTR does not move.
    assert mean_ttr_drop > 0.2
    assert abs(mean_mattr_shift) < 0.01


# Texts shorter than the window


def test_shorter_than_window_is_nan_not_ttr() -> None:
    assert math.isnan(mattr(["a", "b", "c"], window=4))
    assert math.isnan(mattr([], window=1))


def test_exactly_window_length_is_defined() -> None:
    assert mattr(["a", "b", "c", "a"], window=4) == pytest.approx(3 / 4)


# Invalid input


def test_rejects_untokenised_string() -> None:
    with pytest.raises(TypeError, match="tokenize"):
        mattr("this is not a list of tokens")


@pytest.mark.parametrize("window", [0, -1])
def test_rejects_non_positive_window(window: int) -> None:
    with pytest.raises(ValueError, match="window must be >= 1"):
        mattr(["a", "b"], window=window)
    with pytest.raises(ValueError, match="window must be >= 1"):
        lexical_diversity("a b", window=window)


@pytest.mark.parametrize("window", [2.0, "2", True, None, np.float64(2.0)])
def test_rejects_non_integer_window(window: object) -> None:
    with pytest.raises(TypeError, match="window must be an int"):
        mattr(["a", "b"], window=window)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="window must be an int"):
        lexical_diversity("a b", window=window)  # type: ignore[arg-type]


@pytest.mark.parametrize("window", [np.int64(3), np.int32(3), np.uint8(3)])
def test_accepts_the_integers_of_numpy(window: int) -> None:
    tokens = ["a", "b", "a", "b", "c"]
    assert mattr(tokens, window=window) == mattr(tokens, window=3)
    result = lexical_diversity("a b a b c", window=window)
    assert result == lexical_diversity("a b a b c", window=3)
    assert type(result.window) is int


# Text-level interface


def test_lexical_diversity_reports_value_and_context() -> None:
    result = lexical_diversity("A b a. B, c!", window=3)
    assert isinstance(result, LexicalDiversity)
    assert result.mattr == pytest.approx(7 / 9)
    assert (result.window, result.n_tokens, result.n_types, result.n_windows) == (
        3,
        5,
        3,
        3,
    )


def test_lexical_diversity_agrees_with_mattr_on_tokens() -> None:
    tokens = zipfian_tokens(400, seed=3)
    result = lexical_diversity(" ".join(tokens))
    assert result.mattr == mattr(tokens)
    assert result.window == DEFAULT_WINDOW
    assert result.n_tokens == 400
    assert result.n_types == len(set(tokens))
    assert result.n_windows == 400 - DEFAULT_WINDOW + 1


def test_lexical_diversity_of_short_text_keeps_counts() -> None:
    result = lexical_diversity("too short to measure", window=50)
    assert math.isnan(result.mattr)
    assert (result.n_tokens, result.n_types, result.n_windows) == (4, 4, 0)


def test_lexical_diversity_of_empty_text() -> None:
    result = lexical_diversity("")
    assert math.isnan(result.mattr)
    assert (result.n_tokens, result.n_types, result.n_windows) == (0, 0, 0)


def test_case_handling_changes_types() -> None:
    folded = lexical_diversity("The the THE cat", window=4)
    kept = lexical_diversity("The the THE cat", window=4, casefold=False)
    assert folded.n_types == 2
    assert kept.n_types == 4


def test_excluded_tokens_do_not_count() -> None:
    text = "so um the cat um sat uh on the mat"
    kept = lexical_diversity(text, window=3)
    removed = lexical_diversity(text, window=3, exclude={"um", "uh"})
    assert kept.n_tokens == 10
    assert removed.n_tokens == 7
    assert removed.n_types == 6
    assert removed.mattr == mattr(["so", "the", "cat", "sat", "on", "the", "mat"], 3)


def test_result_is_immutable() -> None:
    result = lexical_diversity("a b c", window=2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.mattr = 0.0  # type: ignore[misc]
