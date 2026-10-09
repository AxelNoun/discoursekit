from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
import pytest
from numpy.typing import NDArray

from discoursekit import (
    DEFAULT_ALPHA,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_PERMUTATIONS,
    Coherence,
    TopicSegmentation,
    adjacent_similarities,
    coherence,
    segment_topics,
)
from discoursekit.semantic import (
    _area_table,
    _block_similarities,
    _block_sizes,
    _valley,
)

Array = NDArray[np.float64]

# Number of reorderings used wherever the test for boundaries is not the object
# of the test, to keep the suite fast. It allows p-values down to 0.005.
FAST = 199


def normalise(vectors: Array) -> Array:
    return np.asarray(vectors / np.linalg.norm(vectors, axis=-1, keepdims=True))


def rng(seed: int) -> np.random.RandomState:
    # The legacy generator is used for test data because its stream is
    # guaranteed never to change between NumPy versions.
    return np.random.RandomState(seed)


def integer(state: np.random.RandomState, low: int, high: int) -> int:
    # Without an explicit type the legacy generator draws C long integers,
    # which have 32 bits on Windows and 64 bits elsewhere.
    return int(state.randint(low, high, dtype=np.int64))


def topic(centre: Array, n: int, noise: float, state: np.random.RandomState) -> Array:
    """Draw ``n`` unit vectors scattered around a direction."""
    scatter = state.standard_normal((n, centre.size)) / math.sqrt(centre.size)
    return normalise(centre + noise * scatter)


def directions(n: int, dimensions: int, state: np.random.RandomState) -> list[Array]:
    return [normalise(state.standard_normal(dimensions)) for _ in range(n)]


# Values worked out by hand

EAST, NORTH, NORTH_EAST = [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]


def test_hand_computed_values() -> None:
    result = coherence([EAST, NORTH, NORTH_EAST], boundaries=())
    half = math.sqrt(0.5)
    # Consecutive pairs: east/north are orthogonal, north/north-east at 45 degrees.
    assert result.local_coherence == pytest.approx((0.0 + half) / 2)
    # All three pairs: 0, cos 45, cos 45.
    assert result.local_coherence_shuffled == pytest.approx((0.0 + half + half) / 3)
    # East and north are each at 67.5 degrees from the sum of the other two;
    # north-east is aligned with the sum of east and north.
    to_rest = math.cos(3 * math.pi / 8)
    assert result.global_coherence == pytest.approx((to_rest + to_rest + 1.0) / 3)
    assert result.n_units == 3


def test_adjacent_similarities_series() -> None:
    series = adjacent_similarities([EAST, NORTH, NORTH_EAST])
    assert series == pytest.approx([0.0, math.sqrt(0.5)])
    assert adjacent_similarities([EAST]).size == 0
    assert adjacent_similarities([]).size == 0


def test_identical_units_are_perfectly_coherent() -> None:
    result = coherence([[2.0, 1.0, 3.0]] * 5, boundaries=())
    assert result.local_coherence == pytest.approx(1.0)
    assert result.local_coherence_shuffled == pytest.approx(1.0)
    assert result.global_coherence == pytest.approx(1.0)


def test_mutually_orthogonal_units_have_zero_coherence() -> None:
    result = coherence(np.eye(4), boundaries=())
    assert result.local_coherence == pytest.approx(0.0)
    assert result.local_coherence_shuffled == pytest.approx(0.0)
    assert result.global_coherence == pytest.approx(0.0)


def test_opposite_units_have_negative_coherence() -> None:
    result = coherence([[1.0, 0.0], [-1.0, 0.0]], boundaries=())
    assert result.local_coherence == pytest.approx(-1.0)
    assert result.global_coherence == pytest.approx(-1.0)


# Too few units


@pytest.mark.parametrize("embeddings", [[], np.empty((0, 5)), [[1.0, 2.0]]])
def test_fewer_than_two_units_give_nan(embeddings: object) -> None:
    result = coherence(embeddings)  # type: ignore[arg-type]
    assert result.n_units in (0, 1)
    assert math.isnan(result.local_coherence)
    assert math.isnan(result.local_coherence_shuffled)
    assert math.isnan(result.global_coherence)
    assert math.isnan(result.local_coherence_within_segments)
    assert math.isnan(result.global_coherence_within_segments)
    assert result.boundaries == ()


def test_two_units() -> None:
    result = coherence([EAST, NORTH_EAST], boundaries=())
    half = math.sqrt(0.5)
    assert result.local_coherence == pytest.approx(half)
    assert result.local_coherence_shuffled == pytest.approx(half)
    assert result.global_coherence == pytest.approx(half)


# Invariances


@pytest.fixture
def units() -> Array:
    return rng(1).standard_normal((9, 6))


def summary(result: Coherence) -> tuple[float, float, float]:
    return (
        result.local_coherence,
        result.local_coherence_shuffled,
        result.global_coherence,
    )


def test_length_of_the_vectors_does_not_matter(units: Array) -> None:
    scales = rng(2).uniform(0.1, 50.0, size=(units.shape[0], 1))
    assert summary(coherence(units * scales, boundaries=())) == pytest.approx(
        summary(coherence(units, boundaries=()))
    )


def test_rotation_of_the_space_does_not_matter(units: Array) -> None:
    rotation, _ = np.linalg.qr(rng(3).standard_normal((6, 6)))
    assert summary(coherence(units @ rotation, boundaries=())) == pytest.approx(
        summary(coherence(units, boundaries=()))
    )


def test_global_coherence_does_not_depend_on_order(units: Array) -> None:
    reference = coherence(units, boundaries=())
    for seed in range(5):
        shuffled = coherence(units[rng(seed).permutation(len(units))], boundaries=())
        assert shuffled.global_coherence == pytest.approx(reference.global_coherence)
        assert shuffled.local_coherence_shuffled == pytest.approx(
            reference.local_coherence_shuffled
        )


def test_local_coherence_depends_on_order_but_not_on_direction(units: Array) -> None:
    forward = coherence(units, boundaries=()).local_coherence
    backward = coherence(units[::-1], boundaries=()).local_coherence
    assert backward == pytest.approx(forward)
    other_order = coherence(units[rng(0).permutation(len(units))], boundaries=())
    assert other_order.local_coherence != pytest.approx(forward)


def test_shuffled_level_is_the_mean_over_all_orders() -> None:
    six = rng(4).standard_normal((6, 4))
    over_all_orders = [
        coherence(six[list(order)], boundaries=()).local_coherence
        for order in itertools.permutations(range(6))
    ]
    assert len(over_all_orders) == 720
    expected = coherence(six, boundaries=()).local_coherence_shuffled
    assert sum(over_all_orders) / 720 == pytest.approx(expected, abs=1e-12)


def test_leaving_the_unit_out_removes_a_built_in_bias() -> None:
    # Five unrelated directions in a space of 2000 dimensions.
    unrelated = normalise(rng(5).standard_normal((5, 2000)))
    assert abs(coherence(unrelated, boundaries=()).global_coherence) < 0.05
    # Had each unit been compared with a centroid that includes itself, the
    # result would be near 1 / sqrt(5) although the units share nothing.
    centroid = unrelated.sum(axis=0)
    with_self = float(np.mean(unrelated @ centroid / np.linalg.norm(centroid)))
    assert with_self == pytest.approx(1 / math.sqrt(5), abs=0.05)


# Agreement of the summary with a direct implementation


def cosine(first: Array, second: Array) -> float:
    return float(first @ second / (np.linalg.norm(first) * np.linalg.norm(second)))


def mean_or_nan(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def summarise_directly(
    vectors: Array, boundaries: tuple[int, ...], reference: Array
) -> tuple[float, ...]:
    """Each quantity of the summary written out from its definition."""
    unit = [vector / np.linalg.norm(vector) for vector in vectors]
    indices = range(len(unit))
    segment = [sum(index >= boundary for boundary in boundaries) for index in indices]
    adjacent = [cosine(unit[i], unit[i + 1]) for i in indices[:-1]]
    pairs = [cosine(unit[i], unit[j]) for i in indices for j in indices if i < j]
    to_rest = [
        cosine(unit[i], np.sum([unit[j] for j in indices if j != i], axis=0))
        for i in indices
    ]
    to_reference = [cosine(unit[i], reference) for i in indices]
    adjacent_within = [
        adjacent[i] for i in indices[:-1] if segment[i] == segment[i + 1]
    ]
    to_rest_within = []
    for i in indices:
        others = [unit[j] for j in indices if j != i and segment[j] == segment[i]]
        if others:
            to_rest_within.append(cosine(unit[i], np.sum(others, axis=0)))
    return tuple(
        mean_or_nan(values)
        for values in (
            adjacent,
            pairs,
            to_rest,
            to_reference,
            adjacent_within,
            to_rest_within,
        )
    )


@pytest.mark.parametrize("seed", range(10))
def test_summary_matches_a_direct_implementation(seed: int) -> None:
    state = rng(40 + seed)
    n_units = integer(state, 2, 15)
    vectors = state.standard_normal((n_units, 5)) * state.uniform(
        0.2, 9.0, (n_units, 1)
    )
    reference = state.standard_normal(5)
    n_boundaries = integer(state, 0, n_units)
    chosen = state.choice(np.arange(1, n_units), size=n_boundaries, replace=False)
    boundaries = tuple(sorted(int(boundary) for boundary in chosen))
    result = coherence(vectors, reference=reference, boundaries=boundaries)
    computed = (
        result.local_coherence,
        result.local_coherence_shuffled,
        result.global_coherence,
        result.reference_coherence,
        result.local_coherence_within_segments,
        result.global_coherence_within_segments,
    )
    expected = summarise_directly(vectors, boundaries, reference)
    assert computed == pytest.approx(expected, abs=1e-12, nan_ok=True)
    assert result.boundaries == boundaries


# Reference coherence


def test_reference_coherence_by_hand() -> None:
    result = coherence([EAST, NORTH, NORTH_EAST], reference=EAST, boundaries=())
    assert result.reference_coherence == pytest.approx((1.0 + 0.0 + math.sqrt(0.5)) / 3)


def test_reference_coherence_is_nan_without_a_reference() -> None:
    assert math.isnan(coherence([EAST, NORTH], boundaries=()).reference_coherence)


def test_length_of_the_reference_does_not_matter() -> None:
    short = coherence([EAST, NORTH_EAST], reference=[0.5, 0.2], boundaries=())
    long = coherence([EAST, NORTH_EAST], reference=[50.0, 20.0], boundaries=())
    assert short.reference_coherence == pytest.approx(long.reference_coherence)


def test_reference_coherence_does_not_depend_on_order(units: Array) -> None:
    reference = rng(8).standard_normal(units.shape[1])
    expected = coherence(units, reference=reference, boundaries=())
    for seed in range(5):
        order = rng(seed).permutation(len(units))
        shuffled = coherence(units[order], reference=reference, boundaries=())
        assert shuffled.reference_coherence == pytest.approx(
            expected.reference_coherence
        )


def test_reference_coherence_is_unchanged_by_rotation_and_rescaling(
    units: Array,
) -> None:
    reference = rng(8).standard_normal(units.shape[1])
    rotation, _ = np.linalg.qr(rng(3).standard_normal((6, 6)))
    scales = rng(2).uniform(0.1, 50.0, size=(units.shape[0], 1))
    expected = coherence(units, reference=reference, boundaries=())
    changed = coherence(
        (units * scales) @ rotation, reference=reference @ rotation, boundaries=()
    )
    assert changed.reference_coherence == pytest.approx(expected.reference_coherence)


def test_reference_without_units_gives_nan() -> None:
    # An empty list does not say how many dimensions the units would have.
    assert math.isnan(coherence([], reference=[1.0, 2.0]).reference_coherence)
    empty = np.empty((0, 2))
    assert math.isnan(coherence(empty, reference=[1.0, 2.0]).reference_coherence)
    with pytest.raises(ValueError, match="reference has 3 dimensions"):
        coherence(empty, reference=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="finite, non-zero"):
        coherence([], reference=[0.0, 0.0])


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ([1.0, 0.0, 0.0], "reference has 3 dimensions, embeddings have 2"),
        ([[1.0, 0.0]], "reference must be a single vector"),
        ([0.0, 0.0], "finite, non-zero"),
        ([math.nan, 1.0], "finite, non-zero"),
        ([math.inf, 1.0], "finite, non-zero"),
    ],
)
def test_invalid_reference(reference: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        coherence([EAST, NORTH], reference=reference, boundaries=())


# Supplied boundaries

A, B = [1.0, 0.1], [0.1, 1.0]


def test_within_segment_values_with_a_known_boundary() -> None:
    result = coherence([A, A, A, B, B, B], boundaries=[3])
    across = 0.2 / 1.01  # cosine between A and B
    assert result.local_coherence == pytest.approx((4 + across) / 5)
    assert result.local_coherence_within_segments == pytest.approx(1.0)
    assert result.global_coherence_within_segments == pytest.approx(1.0)
    assert result.global_coherence < 0.7
    assert result.boundaries == (3,)
    assert result.segmentation is None


def test_no_boundary_means_one_segment() -> None:
    result = coherence([A, B, A, A, B], boundaries=())
    assert result.local_coherence_within_segments == result.local_coherence
    assert result.global_coherence_within_segments == result.global_coherence
    assert result.boundaries == ()


def test_segments_of_one_unit_do_not_enter_global_coherence() -> None:
    # Segments: [east] and [north, north-east].
    result = coherence([EAST, NORTH, NORTH_EAST], boundaries=[1])
    half = math.sqrt(0.5)
    assert result.local_coherence_within_segments == pytest.approx(half)
    assert result.global_coherence_within_segments == pytest.approx(half)


def test_every_unit_its_own_segment_leaves_nothing_to_average() -> None:
    result = coherence([EAST, NORTH, NORTH_EAST], boundaries=[1, 2])
    assert math.isnan(result.local_coherence_within_segments)
    assert math.isnan(result.global_coherence_within_segments)


def test_boundaries_accept_numpy_integers_and_tuples() -> None:
    as_array = coherence([A, A, B, B], boundaries=np.array([2]))  # type: ignore[arg-type]
    as_tuple = coherence([A, A, B, B], boundaries=(2,))
    assert as_array.boundaries == as_tuple.boundaries == (2,)


@pytest.mark.parametrize(
    ("boundaries", "error", "message"),
    [
        ([0], ValueError, "between 1 and n_units - 1"),
        ([4], ValueError, "between 1 and n_units - 1"),
        ([-1], ValueError, "between 1 and n_units - 1"),
        ([2, 2], ValueError, "strictly increasing"),
        ([3, 1], ValueError, "strictly increasing"),
        ([1.0], TypeError, "boundaries must be integers"),
        ([True], TypeError, "boundaries must be integers"),
        ("12", TypeError, "sequence of unit indices, got str"),
        (2, TypeError, "sequence of unit indices, got int"),
        (np.array(2), TypeError, "sequence of unit indices, got ndarray"),
    ],
)
def test_invalid_boundaries(
    boundaries: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        coherence([A, A, B, B], boundaries=boundaries)  # type: ignore[arg-type]


# Invalid embeddings


@pytest.mark.parametrize(
    ("embeddings", "message"),
    [
        ([1.0, 2.0, 3.0], "two-dimensional array"),
        (np.zeros((2, 3, 4)), "two-dimensional array"),
        ([[1.0, math.nan], [1.0, 0.0]], "NaN or infinite"),
        ([[1.0, math.inf], [1.0, 0.0]], "NaN or infinite"),
        ([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], "unit 1 has a zero embedding"),
    ],
)
def test_invalid_embeddings(embeddings: object, message: str) -> None:
    for function in (coherence, segment_topics, adjacent_similarities):
        with pytest.raises(ValueError, match=message):
            function(embeddings)  # type: ignore[arg-type]


def test_accepts_lists_and_single_precision(units: Array) -> None:
    from_list = coherence(units.tolist(), boundaries=())
    from_float32 = coherence(units.astype(np.float32), boundaries=())
    assert summary(from_list) == pytest.approx(summary(coherence(units, boundaries=())))
    assert summary(from_float32) == pytest.approx(summary(from_list), abs=1e-6)


@pytest.mark.parametrize("exponent", [-1000, -600, 600, 1000])
def test_very_small_and_very_large_vectors(units: Array, exponent: int) -> None:
    # The squares of these components are beyond the range of floating-point
    # numbers. Multiplying by a power of two changes no digit of a vector, so
    # the results must be exactly those of the original vectors.
    reference = rng(8).standard_normal(units.shape[1])
    expected = coherence(units, reference=reference, n_permutations=FAST)
    scaled = coherence(
        np.ldexp(units, exponent),
        reference=np.ldexp(reference, exponent),
        n_permutations=FAST,
    )
    assert scaled == expected


def test_units_that_cancel_out_have_no_centroid() -> None:
    # The first two units are opposite and the third is at right angles to
    # both. For the third unit the other two sum to zero: their centroid has
    # no direction, and the unit is left out of the mean. Each of the first
    # two is at 135 degrees from the sum of the other two.
    expected = math.cos(3 * math.pi / 4)
    exact = coherence([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], boundaries=())
    assert exact.global_coherence == pytest.approx(expected)
    # The same when the two opposite units cancel out only to within
    # rounding error, as happens when they are computed separately.
    inexact = 0
    for angle in np.linspace(0.1, 3.0, 30):
        first = [math.cos(angle), math.sin(angle), 0.0]
        second = [math.cos(angle + math.pi), math.sin(angle + math.pi), 0.0]
        inexact += first[0] + second[0] != 0.0 or first[1] + second[1] != 0.0
        result = coherence([first, second, [0.0, 0.0, 1.0]], boundaries=())
        assert result.global_coherence == pytest.approx(expected)
    assert inexact > 0


def test_input_is_not_modified(units: Array) -> None:
    before = units.copy()
    coherence(units, n_permutations=FAST)
    assert np.array_equal(units, before)


# Building blocks of the test for boundaries


@pytest.mark.parametrize("block_size", [1, 2, 3, 5, 7])
def test_block_similarities_match_the_definition(block_size: int) -> None:
    unit = normalise(rng(6).standard_normal((20, 7)))
    fast = _block_similarities(_area_table(unit @ unit.T), block_size)
    slow = []
    for gap in range(block_size, 20 - block_size + 1):
        before = unit[gap - block_size : gap].sum(axis=0)
        after = unit[gap : gap + block_size].sum(axis=0)
        slow.append(before @ after / (np.linalg.norm(before) * np.linalg.norm(after)))
    assert fast == pytest.approx(slow, abs=1e-10)
    assert fast.size == 20 - 2 * block_size + 1


def test_block_similarities_need_a_block_on_each_side() -> None:
    unit = normalise(rng(7).standard_normal((5, 3)))
    table = _area_table(unit @ unit.T)
    assert _block_similarities(table, 3).size == 0
    assert _block_similarities(table, 2).size == 2


@pytest.mark.parametrize(
    ("scores", "depth", "position"),
    [
        ([5.0, 1.0, 4.0], 3.0, 1),  # the lower side rises by 3
        ([4.0, 1.0, 5.0], 3.0, 1),
        ([3.0, 1.0, 2.0, 0.0, 5.0], 3.0, 3),  # to the highest point of each side
        ([5.0, 2.0, 3.0, 2.5, 6.0], 3.0, 1),  # a small bump does not stop the rise
        ([1.0, 2.0, 3.0], 0.0, 0),  # lowest point at the start: no valley
        ([3.0, 2.0, 1.0], 0.0, 2),  # lowest point at the end: no valley
        ([2.0, 2.0, 2.0], 0.0, 0),  # flat
        ([4.0, 1.0, 1.0, 4.0], 3.0, 1),  # of two equal lowest points, the first
        ([7.0], 0.0, 0),
    ],
)
def test_valley(scores: list[float], depth: float, position: int) -> None:
    found_depth, found_position = _valley(np.array(scores))
    assert found_depth == pytest.approx(depth)
    assert found_position == position


def test_block_sizes() -> None:
    assert _block_sizes(7, 3) == []  # fewer than three candidate positions
    assert _block_sizes(8, 3) == [3]
    assert _block_sizes(24, 3) == [3, 4, 5, 6]
    assert _block_sizes(24, 2) == [2, 3, 4, 5, 6]
    assert _block_sizes(6, 1) == [1]
    # Steps of one, then of a tenth of the current size.
    assert _block_sizes(100, 3) == [*range(3, 21), 22, 24]
    for n in (8, 30, 77, 400):
        sizes = _block_sizes(n, 3)
        assert sizes == sorted(set(sizes))
        assert all(n - 2 * size + 1 >= 3 for size in sizes)
        assert sizes[-1] <= max(3, n // 4)


# The test for boundaries on simulated sequences


def two_topics(seed: int, n_each: int = 12, noise: float = 1.0) -> Array:
    state = rng(seed)
    first, second = directions(2, 48, state)
    return np.vstack(
        [topic(first, n_each, noise, state), topic(second, n_each, noise, state)]
    )


def test_two_topics_give_one_boundary_between_them() -> None:
    result = segment_topics(two_topics(0))
    assert result.boundaries == (12,)
    assert len(result.p_values) == 1
    assert result.p_values[0] <= 0.01
    assert result.segments == ((0, 12), (12, 24))
    assert result.tested
    assert result.n_units == 24


def test_three_topics_give_two_boundaries() -> None:
    result = segment_topics(three_topics(), n_permutations=FAST)
    assert result.boundaries == (12, 24)
    assert result.segments == ((0, 12), (12, 24), (24, 36))
    assert all(p <= DEFAULT_ALPHA for p in result.p_values)


def count_reported(make: Callable[[np.random.RandomState], Array]) -> int:
    """Number of simulated sequences, out of 60, in which a boundary is reported.

    If boundaries were reported at the nominal rate of 5%, more than 9 out
    of 60 would occur with probability below 0.001.
    """
    reported = 0
    for seed in range(60):
        sequence = make(rng(1000 + seed))
        if segment_topics(sequence, n_permutations=99, seed=seed).boundaries:
            reported += 1
    return reported


def test_one_topic_is_rarely_split() -> None:
    def one_topic(state: np.random.RandomState) -> Array:
        (centre,) = directions(1, 48, state)
        return topic(centre, 24, 1.0, state)

    assert count_reported(one_topic) <= 9


def test_gradual_drift_is_rarely_split() -> None:
    def drifting(state: np.random.RandomState) -> Array:
        first, second = directions(2, 48, state)
        weights = np.linspace(0.0, 1.0, 24)[:, np.newaxis]
        centres = normalise((1 - weights) * first + weights * second)
        scatter = state.standard_normal((24, 48)) / math.sqrt(48)
        return normalise(centres + 0.5 * scatter)

    # The content does change: the two ends have little in common, while
    # neighbouring stretches stay close to each other.
    example = drifting(rng(0))
    blocks = [
        normalise(example[start : start + 6].sum(axis=0)) for start in (0, 6, 12, 18)
    ]
    assert float(blocks[0] @ blocks[3]) < 0.5
    assert min(float(blocks[i] @ blocks[i + 1]) for i in range(3)) > 0.8
    # But there is no point at which it changes abruptly.
    assert count_reported(drifting) <= 9


def test_one_unrelated_unit_is_rarely_a_boundary() -> None:
    def with_intruder(state: np.random.RandomState) -> Array:
        centre, elsewhere = directions(2, 48, state)
        sequence = topic(centre, 24, 1.0, state)
        sequence[integer(state, 0, 24)] = elsewhere
        return sequence

    assert count_reported(with_intruder) <= 9


def test_the_units_of_two_topics_in_random_order_are_rarely_split() -> None:
    def mixed(state: np.random.RandomState) -> Array:
        first, second = directions(2, 48, state)
        sequence = np.vstack(
            [topic(first, 12, 1.0, state), topic(second, 12, 1.0, state)]
        )
        return np.asarray(sequence[state.permutation(24)])

    assert count_reported(mixed) <= 9


@pytest.mark.parametrize("n_units", [0, 1, 2, 7])
def test_text_too_short_to_be_tested(n_units: int) -> None:
    state = rng(6)
    (centre,) = directions(1, 8, state)
    result = segment_topics(topic(centre, n_units, 1.0, state))
    assert result.boundaries == ()
    assert not result.tested
    assert result.n_units == n_units


def test_identical_units_leave_nothing_to_test() -> None:
    # Every block similarity is 1 in every order: there is no valley at all.
    result = segment_topics([[1.0, 2.0, 3.0]] * 12, n_permutations=FAST)
    assert result.boundaries == ()
    assert result.tested


def test_shortest_testable_text() -> None:
    # Eight units leave three candidate positions for blocks of three.
    state = rng(7)
    (centre,) = directions(1, 8, state)
    assert segment_topics(topic(centre, 8, 1.0, state), n_permutations=FAST).tested


@pytest.mark.parametrize("block_size", [1, 2, 3, 4])
def test_every_segment_has_more_units_than_a_block(block_size: int) -> None:
    # At a level of 0.9 most parts that can be examined are divided, which
    # gives many short segments on which to check the lower limit.
    lengths: list[int] = []
    for seed in range(12):
        sequence = normalise(rng(500 + seed).standard_normal((40, 8)))
        result = segment_topics(
            sequence, block_size=block_size, alpha=0.9, n_permutations=99, seed=seed
        )
        if result.boundaries:
            lengths.extend(stop - start for start, stop in result.segments)
    assert len(lengths) >= 30
    assert min(lengths) > block_size


def three_topics() -> Array:
    state = rng(1)
    centres = directions(3, 48, state)
    return np.vstack([topic(centre, 12, 0.7, state) for centre in centres])


def test_the_level_of_a_part_is_its_share_of_alpha_in_exact_arithmetic() -> None:
    sequence = three_topics()
    # With 19 reorderings the smallest p-value is 1/20, the level of the test
    # of the whole text. The two parts on either side of the first boundary
    # have levels of 0.05 x 24/36 = 1/30 and 0.05 x 12/36 = 1/60, which no
    # p-value can reach, and are not examined.
    coarse = segment_topics(sequence, n_permutations=19)
    assert coarse.boundaries == (24,)
    assert coarse.p_values == (1 / 20,)
    # With 29 reorderings the smallest p-value is 1/30, exactly the level of
    # the part made of the first two topics. A p-value equal to the level is
    # significant, and rounding error must not decide otherwise.
    finer = segment_topics(sequence, n_permutations=29)
    assert finer.boundaries == (12, 24)
    assert finer.p_values == (1 / 30, 1 / 30)


def test_units_alike_to_within_rounding_have_no_boundary() -> None:
    # The same vector 24 times, with the relative error of single precision.
    for seed in range(10):
        state = rng(300 + seed)
        base = state.standard_normal(48)
        nearly_identical = base * (1 + 1e-7 * state.standard_normal((24, 48)))
        result = segment_topics(nearly_identical, alpha=0.5, n_permutations=99)
        assert result.boundaries == ()
        assert result.tested


def test_result_records_the_parameters() -> None:
    result = segment_topics(
        two_topics(0), block_size=4, alpha=0.01, seed=7, n_permutations=FAST
    )
    assert (result.block_size, result.alpha, result.n_permutations, result.seed) == (
        4,
        0.01,
        199,
        7,
    )
    default = segment_topics(two_topics(0))
    assert (default.block_size, default.alpha, default.n_permutations) == (
        DEFAULT_BLOCK_SIZE,
        DEFAULT_ALPHA,
        DEFAULT_PERMUTATIONS,
    )
    assert default.seed == 0


def test_parameters_accept_numpy_numbers_and_are_recorded_as_plain_ones() -> None:
    parameters: dict[str, object] = {
        "block_size": np.int64(4),
        "alpha": np.float32(0.25),
        "n_permutations": np.int32(59),
        "seed": np.uint8(7),
    }
    result = segment_topics(two_topics(0), **parameters)  # type: ignore[arg-type]
    assert result == segment_topics(
        two_topics(0), block_size=4, alpha=0.25, n_permutations=59, seed=7
    )
    recorded = (result.block_size, result.alpha, result.n_permutations, result.seed)
    assert [type(value) for value in recorded] == [int, float, int, int]


def test_same_seed_gives_same_result() -> None:
    sequence = two_topics(8, noise=1.6)
    assert segment_topics(sequence, seed=3, n_permutations=FAST) == segment_topics(
        sequence, seed=3, n_permutations=FAST
    )


def test_p_values_are_multiples_of_one_over_permutations_plus_one() -> None:
    for seed in range(10):
        result = segment_topics(
            two_topics(seed, noise=1.6), alpha=0.5, n_permutations=FAST
        )
        for p_value in result.p_values:
            assert 1 / 200 <= p_value <= 0.5
            assert p_value * 200 == pytest.approx(round(p_value * 200))


def test_permutations_are_the_same_everywhere() -> None:
    # The reorderings come from the raw stream of the PCG64 bit generator.
    # These two values pin that stream: if they change, results obtained
    # with a given seed are no longer reproducible.
    sequence = two_topics(11, n_each=6, noise=1.8)
    assert segment_topics(
        sequence, alpha=0.9, seed=0, n_permutations=FAST
    ).p_values == (0.03,)
    assert segment_topics(
        sequence, alpha=0.9, seed=1, n_permutations=FAST
    ).p_values == (0.04,)


def test_false_positive_rate_under_exchangeability() -> None:
    # 200 sequences of independent random directions, in which every order
    # is equally likely. At alpha = 0.05 a boundary should be reported in
    # about 10 of them at most. If the rate were exactly 5%, more than 21
    # would occur with probability 0.0005. This check can only reveal a
    # gross excess; the simulation reported in docs/validation.md rests on
    # 70,000 sequences.
    state = rng(9)
    reported = 0
    for seed in range(200):
        sequence = normalise(state.standard_normal((16, 8)))
        if segment_topics(sequence, n_permutations=99, seed=seed).boundaries:
            reported += 1
    assert reported <= 21


def test_power_for_two_clearly_distinct_topics() -> None:
    found = 0
    for seed in range(50):
        result = segment_topics(two_topics(100 + seed, n_each=10), n_permutations=99)
        if len(result.boundaries) == 1 and abs(result.boundaries[0] - 10) <= 1:
            found += 1
    assert found >= 40


# Agreement of the whole procedure with a direct implementation


def block_curve(unit: Array, size: int) -> list[float]:
    """Cosine between the sums of the ``size`` units before and after each gap."""
    curve = []
    for gap in range(size, len(unit) - size + 1):
        before = unit[gap - size : gap].sum(axis=0)
        after = unit[gap : gap + size].sum(axis=0)
        cosine = before @ after / (np.linalg.norm(before) * np.linalg.norm(after))
        curve.append(float(cosine))
    return curve


def deepest_valley(curve: list[float]) -> tuple[float, int]:
    """Depth and position of the lowest point; no depth at either end."""
    lowest = curve.index(min(curve))
    if lowest in (0, len(curve) - 1):
        return 0.0, lowest
    return min(max(curve[:lowest]), max(curve[lowest + 1 :])) - curve[lowest], lowest


def segment_directly(
    vectors: Array, block_size: int, alpha: Fraction, n_permutations: int, seed: int
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """The procedure of ``segment_topics`` written out step by step.

    Every block similarity is computed from the vectors, for every
    reordering, without the table of sums that the library uses.
    """
    unit = normalise(vectors)
    generator = np.random.PCG64(seed)
    found: dict[int, float] = {}
    pending = [(0, len(unit), alpha)]
    while pending:
        start, stop, level = pending.pop(0)
        part = unit[start:stop]
        n = len(part)
        sizes = []
        size = block_size
        while size <= max(block_size, n // 4) and n - 2 * size + 1 >= 3:
            sizes.append(size)
            size += max(1, size // 10)
        if not sizes or level < Fraction(1, 1 + n_permutations):
            continue
        orders = [np.arange(n)]
        for _ in range(n_permutations):
            orders.append(np.argsort(generator.random_raw(n), kind="stable"))
        depths = np.array(
            [
                [deepest_valley(block_curve(part[order], size))[0] for order in orders]
                for size in sizes
            ]
        )
        # Depths closer than this are equal to within rounding error.
        rounding = 1e-12 * n
        spread = depths.std(axis=1)
        keep = spread > rounding
        if not keep.any():
            continue
        centred = depths[keep] - depths[keep].mean(axis=1, keepdims=True)
        standard = centred / spread[keep, np.newaxis]
        statistic = standard.max(axis=0)
        ties = rounding / spread[keep].min()
        reached = int((statistic[1:] >= statistic[0] - ties).sum())
        if Fraction(1 + reached, 1 + n_permutations) > level:
            continue
        best = int(np.array(sizes)[keep][int(np.argmax(standard[:, 0]))])
        middle = start + best + deepest_valley(block_curve(part, best))[1]
        found[middle] = (1 + reached) / (1 + n_permutations)
        pending.append((start, middle, level * Fraction(middle - start, n)))
        pending.append((middle, stop, level * Fraction(stop - middle, n)))
    boundaries = tuple(sorted(found))
    return boundaries, tuple(found[boundary] for boundary in boundaries)


def test_segmentation_matches_a_direct_implementation() -> None:
    numbers_found = []
    for seed in range(24):
        state = rng(900 + seed)
        n_topics = integer(state, 1, 4)
        lengths = [integer(state, 4, 12) for _ in range(n_topics)]
        noise = float(state.uniform(0.6, 1.6))
        sequence = np.vstack(
            [
                topic(centre, length, noise, state)
                for centre, length in zip(
                    directions(n_topics, 16, state), lengths, strict=True
                )
            ]
        )
        block_size = integer(state, 1, 4)
        alpha = Fraction(integer(state, 1, 7), 20)
        result = segment_topics(
            sequence,
            block_size=block_size,
            alpha=float(alpha),
            n_permutations=59,
            seed=seed,
        )
        expected = segment_directly(sequence, block_size, alpha, 59, seed)
        assert (result.boundaries, result.p_values) == expected, seed
        numbers_found.append(len(result.boundaries))
    # The comparison covers texts without a boundary, with one, and with
    # several, found by successive subdivision.
    assert numbers_found.count(0) >= 5
    assert numbers_found.count(1) >= 5
    assert max(numbers_found) >= 2


# Parameters of the test


@pytest.mark.parametrize(
    ("parameters", "error", "message"),
    [
        ({"block_size": 0}, ValueError, "block_size must be >= 1"),
        ({"block_size": 2.0}, TypeError, "block_size must be an int"),
        ({"block_size": True}, TypeError, "block_size must be an int"),
        ({"n_permutations": 0}, ValueError, "n_permutations must be >= 1"),
        ({"n_permutations": 99.0}, TypeError, "n_permutations must be an int"),
        ({"seed": -1}, ValueError, "seed must be >= 0"),
        ({"seed": None}, TypeError, "seed must be an int"),
        ({"alpha": 0.0}, ValueError, "strictly between 0 and 1"),
        ({"alpha": 1.0}, ValueError, "strictly between 0 and 1"),
        ({"alpha": "0.05"}, TypeError, "alpha must be a float"),
        ({"alpha": True}, TypeError, "alpha must be a float"),
        ({"alpha": 0.05, "n_permutations": 9}, ValueError, "smallest attainable"),
    ],
)
def test_invalid_test_parameters(
    parameters: dict[str, object], error: type[Exception], message: str
) -> None:
    sequence = two_topics(0)
    with pytest.raises(error, match=message):
        segment_topics(sequence, **parameters)  # type: ignore[arg-type]
    with pytest.raises(error, match=message):
        coherence(sequence, **parameters)  # type: ignore[arg-type]


def test_test_parameters_are_ignored_when_boundaries_are_supplied() -> None:
    result = coherence(two_topics(0), boundaries=[12], n_permutations=0)
    assert result.boundaries == (12,)


# Estimated boundaries in the summary


def test_summary_uses_the_estimated_boundaries() -> None:
    sequence = two_topics(0)
    result = coherence(sequence)
    assert isinstance(result.segmentation, TopicSegmentation)
    assert result.segmentation == segment_topics(sequence)
    assert result.boundaries == (12,)
    supplied = coherence(sequence, boundaries=[12])
    assert result.local_coherence_within_segments == pytest.approx(
        supplied.local_coherence_within_segments
    )
    assert result.global_coherence_within_segments == pytest.approx(
        supplied.global_coherence_within_segments
    )


def test_a_change_of_topic_lowers_plain_but_not_within_segment_coherence() -> None:
    state = rng(12)
    first, second = directions(2, 48, state)
    one_topic = topic(first, 24, 1.0, state)
    changed = np.vstack([topic(first, 12, 1.0, state), topic(second, 12, 1.0, state)])
    single = coherence(one_topic, n_permutations=FAST)
    double = coherence(changed, n_permutations=FAST)
    assert double.boundaries == (12,)
    assert double.global_coherence < single.global_coherence - 0.1
    assert double.global_coherence_within_segments == pytest.approx(
        single.global_coherence, abs=0.1
    )


def test_shuffling_lowers_local_but_not_global_coherence() -> None:
    ordered = two_topics(13)
    shuffled = ordered[rng(13).permutation(24)]
    before = coherence(ordered, boundaries=())
    after = coherence(shuffled, boundaries=())
    assert after.local_coherence < before.local_coherence
    assert after.global_coherence == pytest.approx(before.global_coherence)
    # In random order, local coherence falls to its expected level.
    assert after.local_coherence == pytest.approx(
        after.local_coherence_shuffled, abs=0.1
    )
    assert before.local_coherence > before.local_coherence_shuffled + 0.1


def test_results_are_immutable() -> None:
    result = coherence(two_topics(0), n_permutations=FAST)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.local_coherence = 0.0  # type: ignore[misc]
    assert result.segmentation is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.segmentation.boundaries = ()  # type: ignore[misc]
