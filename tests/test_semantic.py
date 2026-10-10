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
    _rises,
    _scores,
    _similarities,
    _similarity_bits,
    _similarity_counts,
    _size_pairs,
    _test_segment,
    _valley_depths,
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
    # Of the six orders, the four that begin or end with north-east have a
    # local coherence of half / 2 and the two others of half. The variance of
    # these six values is half ** 2 / 18, so their standard deviation is 1/6.
    assert result.local_coherence_shuffled_sd == pytest.approx(1 / 6)
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
    assert result.local_coherence_shuffled_sd == 0.0
    assert result.global_coherence == pytest.approx(1.0)


def test_mutually_orthogonal_units_have_zero_coherence() -> None:
    result = coherence(np.eye(4), boundaries=())
    assert result.local_coherence == pytest.approx(0.0)
    assert result.local_coherence_shuffled == pytest.approx(0.0)
    assert result.local_coherence_shuffled_sd == 0.0
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
    assert math.isnan(result.local_coherence_shuffled_sd)
    assert math.isnan(result.global_coherence)
    assert math.isnan(result.local_coherence_within_segments)
    assert math.isnan(result.global_coherence_within_segments)
    assert result.boundaries == ()


def test_two_units() -> None:
    result = coherence([EAST, NORTH_EAST], boundaries=())
    half = math.sqrt(0.5)
    assert result.local_coherence == pytest.approx(half)
    assert result.local_coherence_shuffled == pytest.approx(half)
    # The two orders of two units have the same pair of consecutive units.
    assert result.local_coherence_shuffled_sd == 0.0
    assert result.global_coherence == pytest.approx(half)


# Invariances


@pytest.fixture
def units() -> Array:
    return rng(1).standard_normal((9, 6))


def summary(result: Coherence) -> tuple[float, float, float, float]:
    return (
        result.local_coherence,
        result.local_coherence_shuffled,
        result.local_coherence_shuffled_sd,
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
        assert shuffled.local_coherence_shuffled_sd == pytest.approx(
            reference.local_coherence_shuffled_sd
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


@pytest.mark.parametrize("n_units", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("dimensions", [2, 5])
def test_shuffled_spread_is_the_standard_deviation_over_all_orders(
    n_units: int, dimensions: int
) -> None:
    state = rng(60 + 10 * n_units + dimensions)
    vectors = state.standard_normal((n_units, dimensions))
    over_all_orders = [
        coherence(vectors[list(order)], boundaries=()).local_coherence
        for order in itertools.permutations(range(n_units))
    ]
    assert len(over_all_orders) == math.factorial(n_units)
    result = coherence(vectors, boundaries=())
    assert result.local_coherence_shuffled == pytest.approx(
        float(np.mean(over_all_orders)), abs=1e-12
    )
    assert result.local_coherence_shuffled_sd == pytest.approx(
        float(np.std(over_all_orders)), abs=1e-12
    )


def test_shuffled_spread_does_not_depend_on_how_the_pairs_are_accumulated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # In a long text the similarities are computed a slice of units at a
    # time. With 50 units, the values below give slices of 1, 3, 7, 12, 33,
    # 49 and 50 units: most leave a shorter slice at the end.
    vectors = rng(70).standard_normal((50, 7))
    expected = coherence(vectors, boundaries=()).local_coherence_shuffled_sd
    assert expected > 0.01
    for chunk in (1, 150, 350, 600, 1650, 2450, 2500):
        monkeypatch.setattr("discoursekit.semantic._CHUNK", chunk)
        result = coherence(vectors, boundaries=())
        assert result.local_coherence_shuffled_sd == pytest.approx(expected, abs=1e-14)


def test_shuffled_spread_matches_random_orders() -> None:
    # Thirty units on two topics. Over 4000 random orders, the standard error
    # of the mean of local coherence is the standard deviation divided by 63,
    # and that of the standard deviation about 1.1% of its value.
    vectors = two_topics(14, n_each=15)
    state = rng(71)
    sampled = [
        coherence(vectors[state.permutation(30)], boundaries=()).local_coherence
        for _ in range(4000)
    ]
    result = coherence(vectors, boundaries=())
    assert result.local_coherence_shuffled == pytest.approx(
        float(np.mean(sampled)), abs=4 * result.local_coherence_shuffled_sd / 60
    )
    assert result.local_coherence_shuffled_sd == pytest.approx(
        float(np.std(sampled)), rel=0.08
    )


def test_the_difference_in_units_of_its_spread_grows_with_the_text() -> None:
    # Texts on two topics, half the units on each, at two lengths. Local
    # coherence exceeds its reference level by about the same amount at both
    # lengths. Divided by the standard deviation over orders, which shrinks
    # as the text grows, the difference is larger for the longer texts: it
    # measures the evidence that the order is not random, not its strength.
    differences, in_units_of_spread = [], []
    for n_each in (4, 16):
        raw, standardised = [], []
        for seed in range(40):
            result = coherence(two_topics(700 + seed, n_each=n_each), boundaries=())
            difference = result.local_coherence - result.local_coherence_shuffled
            raw.append(difference)
            standardised.append(difference / result.local_coherence_shuffled_sd)
        differences.append(float(np.mean(raw)))
        in_units_of_spread.append(float(np.mean(standardised)))
    assert differences[0] == pytest.approx(differences[1], abs=0.05)
    assert differences[0] > 0.15
    assert in_units_of_spread[1] > 1.5 * in_units_of_spread[0]


@pytest.mark.parametrize("n_units", [2, 5, 20, 80])
def test_leaving_the_unit_out_removes_a_built_in_bias(n_units: int) -> None:
    # Unrelated directions in a space of 2000 dimensions.
    unrelated = normalise(rng(5).standard_normal((n_units, 2000)))
    assert abs(coherence(unrelated, boundaries=()).global_coherence) < 0.05
    # Had each unit been compared with a centroid that includes itself, the
    # result would be near one over the square root of the number of units,
    # although the units share nothing.
    centroid = unrelated.sum(axis=0)
    with_self = float(np.mean(unrelated @ centroid / np.linalg.norm(centroid)))
    assert with_self == pytest.approx(1 / math.sqrt(n_units), abs=0.05)


def equally_similar(n_units: int, similarity: float) -> Array:
    """Unit vectors with the same cosine similarity between all pairs."""
    common = np.zeros(n_units + 1)
    common[0] = 1.0
    own = np.eye(n_units + 1)[1:]
    return np.asarray(math.sqrt(similarity) * common + math.sqrt(1 - similarity) * own)


@pytest.mark.parametrize("similarity", [0.2, 0.5])
def test_global_coherence_grows_with_the_number_of_units(similarity: float) -> None:
    # When every pair of units has the same similarity r, a unit and the sum
    # of the n - 1 others have a similarity of
    # sqrt(n - 1) r / sqrt(1 + (n - 2) r), which increases with n from r
    # towards sqrt(r). Local coherence and its reference level stay at r.
    previous = 0.0
    for n_units in (2, 4, 8, 32, 128):
        result = coherence(equally_similar(n_units, similarity), boundaries=())
        expected = (
            math.sqrt(n_units - 1)
            * similarity
            / math.sqrt(1 + (n_units - 2) * similarity)
        )
        assert result.global_coherence == pytest.approx(expected)
        assert result.global_coherence > previous
        assert result.local_coherence == pytest.approx(similarity)
        assert result.local_coherence_shuffled == pytest.approx(similarity)
        assert result.local_coherence_shuffled_sd == 0.0
        previous = result.global_coherence
    assert previous == pytest.approx(math.sqrt(similarity), abs=0.01)
    assert coherence(
        equally_similar(2, similarity), boundaries=()
    ).global_coherence == pytest.approx(similarity)


# Agreement of the summary with a direct implementation


def cosine(first: Array, second: Array) -> float:
    return float(first @ second / (np.linalg.norm(first) * np.linalg.norm(second)))


def mean_or_nan(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def spread_over_orders_directly(unit: list[Array]) -> float:
    """Standard deviation of local coherence over all orders, pair by pair.

    In a random order of ``n`` units, two given units are consecutive with
    probability ``2 / n``. Two pairs that share a unit are both consecutive
    with probability ``2 / (n (n - 1))``, and two pairs without a common
    unit with probability ``4 / (n (n - 1))``. The variance of the sum of the
    similarities between consecutive units follows from the covariances of
    these events, summed over every pair of pairs.
    """
    n = len(unit)
    pairs = list(itertools.combinations(range(n), 2))
    variance = 0.0
    for first in pairs:
        for second in pairs:
            shared = len(set(first) & set(second))
            both = (4 / (n * (n - 1)), 2 / (n * (n - 1)), 2 / n)[shared]
            covariance = both - (2 / n) ** 2
            variance += (
                cosine(unit[first[0]], unit[first[1]])
                * cosine(unit[second[0]], unit[second[1]])
                * covariance
            )
    return math.sqrt(max(variance, 0.0)) / (n - 1)


def summarise_directly(
    vectors: Array, boundaries: tuple[int, ...], reference: Array
) -> tuple[float, ...]:
    """Each quantity of the summary written out from its definition."""
    unit = [vector / np.linalg.norm(vector) for vector in vectors]
    indices = range(len(unit))
    segment = [sum(index >= boundary for boundary in boundaries) for index in indices]
    adjacent = [cosine(unit[i], unit[i + 1]) for i in indices[:-1]]
    pairs = [cosine(unit[i], unit[j]) for i in indices for j in indices if i < j]
    spread = spread_over_orders_directly(unit)
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
    means = tuple(
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
    return (*means, spread)


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
        result.local_coherence_shuffled_sd,
    )
    expected = summarise_directly(vectors, boundaries, reference)
    assert computed == pytest.approx(expected, abs=1e-12, nan_ok=True)
    assert result.boundaries == boundaries


# Reference coherence


def test_reference_coherence_by_hand() -> None:
    result = coherence([EAST, NORTH, NORTH_EAST], reference=EAST, boundaries=())
    assert result.reference_coherence == pytest.approx((1.0 + 0.0 + math.sqrt(0.5)) / 3)


def test_reference_coherence_is_defined_for_a_single_unit() -> None:
    result = coherence([EAST], reference=NORTH_EAST)
    assert result.reference_coherence == pytest.approx(math.sqrt(0.5))
    assert math.isnan(result.local_coherence)
    assert math.isnan(result.global_coherence)


def test_reference_coherence_is_nan_without_a_reference() -> None:
    assert math.isnan(coherence([EAST, NORTH], boundaries=()).reference_coherence)


def test_length_of_the_reference_does_not_matter(units: Array) -> None:
    reference = rng(8).standard_normal(units.shape[1])
    expected = coherence(units, reference=reference, boundaries=())
    for factor in (0.004, 100.0):
        scaled = coherence(units, reference=factor * reference, boundaries=())
        assert scaled.reference_coherence == pytest.approx(expected.reference_coherence)


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
    expected = coherence(units, reference=reference, boundaries=[3])
    scaled = coherence(
        np.ldexp(units, exponent),
        reference=np.ldexp(reference, exponent),
        boundaries=[3],
    )
    assert scaled == expected
    found = segment_topics(units, n_permutations=FAST)
    assert segment_topics(np.ldexp(units, exponent), n_permutations=FAST) == found


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


def test_single_precision_gives_the_result_of_its_double_precision_values() -> None:
    sequence = two_topics(0).astype(np.float32)
    assert segment_topics(sequence, n_permutations=FAST) == segment_topics(
        sequence.astype(np.float64), n_permutations=FAST
    )


def test_complex_numbers_are_refused() -> None:
    complex_units = np.array([[1.0 + 1.0j, 0.0], [0.0, 1.0]])
    for function in (coherence, segment_topics, adjacent_similarities):
        with pytest.raises(TypeError, match="real numbers"):
            function(complex_units)
    with pytest.raises(TypeError, match="real numbers"):
        coherence([[1.0, 0.0]], reference=[1.0j, 1.0])


def test_input_is_not_modified(units: Array) -> None:
    before = units.copy()
    coherence(units, reference=units[0], boundaries=[2])
    segment_topics(units, n_permutations=FAST)
    assert np.array_equal(units, before)


# Building blocks of the test for boundaries


def table_of(unit: Array) -> tuple[NDArray[np.int64], float]:
    """Summed-area table of the similarities between units, and its unit."""
    counts, quantum = _similarity_counts(_similarities(unit))
    return _area_table(counts), quantum


def test_equal_units_have_equal_similarities() -> None:
    # Forty units, of which only seven are distinct. Whatever the product of
    # matrices returns, two equal units have the same similarity to every
    # unit, to the last digit, and so have their counts.
    state = rng(33)
    distinct = normalise(state.standard_normal((7, 19)))
    which = np.array([integer(state, 0, 7) for _ in range(40)])
    unit = distinct[which]
    gram = _similarities(unit)
    counts, _ = _similarity_counts(gram)
    for table in (gram, counts):
        assert np.array_equal(table, table.T)
        for first in range(40):
            for second in range(first + 1, 40):
                if which[first] == which[second]:
                    assert np.array_equal(table[first], table[second])
    assert gram == pytest.approx(unit @ unit.T, abs=1e-15)
    # Without equal units the similarities are those of the product itself.
    assert np.array_equal(_similarities(distinct), distinct @ distinct.T)
    assert _similarities(distinct[:1]).shape == (1, 1)
    assert _similarities(distinct[:0]).shape == (0, 0)


class NoisyProduct(np.ndarray):
    """Vectors whose product has noise in its last digit, as BLAS may give."""

    def __matmul__(self, other: object) -> Array:  # type: ignore[override]
        product = np.asarray(self) @ np.asarray(other)
        noise = rng(3).choice([-1.0, 0.0, 1.0], size=product.shape)
        return np.asarray(product + noise * 2.0**-52)


def test_last_digits_of_the_product_do_not_reach_the_similarities() -> None:
    # The product is neither symmetric nor equal between equal units. The
    # similarities kept are symmetric, and equal units have equal rows.
    state = rng(33)
    distinct = normalise(state.standard_normal((7, 19)))
    which = np.array([integer(state, 0, 7) for _ in range(40)])
    noisy = distinct[which].view(NoisyProduct)
    product = noisy @ noisy.T
    assert not np.array_equal(product, product.T)
    gram = _similarities(noisy)
    assert np.array_equal(gram, gram.T)
    unequal_in_the_product = 0
    for first in range(40):
        for second in range(first + 1, 40):
            if which[first] == which[second]:
                assert np.array_equal(gram[first], gram[second])
                unequal_in_the_product += not np.array_equal(
                    product[first], product[second]
                )
    assert unequal_in_the_product > 0
    assert gram == pytest.approx(distinct[which] @ distinct[which].T, abs=1e-15)


def test_similarities_are_counted_in_a_small_unit() -> None:
    unit = normalise(rng(6).standard_normal((20, 7)))
    gram = unit @ unit.T
    counts, quantum = _similarity_counts(gram)
    assert counts.dtype == np.int64
    assert quantum == 2.0**-40
    # Each similarity is the nearest whole number of units.
    assert np.abs(counts * quantum - gram).max() <= quantum / 2
    assert counts.diagonal().tolist() == [2**40] * 20
    # In a longer text the unit is larger, so that the sum of all the
    # similarities, at most the square of the number of units, fits in 64 bits.
    for n_units, bits in (
        (1, 40),
        (2048, 40),
        (2049, 38),
        (4096, 38),
        (4097, 36),
        (5000, 36),
        (100_000, 28),
    ):
        assert _similarity_bits(n_units) == bits
    for n_units in range(1, 200_000, 997):
        assert n_units**2 * 2 ** _similarity_bits(n_units) <= 2**62
    long_text = np.ones((2049, 1))
    counts, quantum = _similarity_counts(long_text @ long_text.T)
    assert quantum == 2.0**-38
    assert int(_area_table(counts)[-1, -1]) == 2049**2 * 2**38


def test_the_largest_sums_fit_in_64_bits() -> None:
    # 2,048 identical units: every similarity is 1, and their sum is two to
    # the power 62. The block similarities are exactly 1 up to the last gap.
    identical = np.ones((2048, 1))
    counts, quantum = _similarity_counts(identical @ identical.T)
    table = _area_table(counts)
    assert int(table[-1, -1]) == 2**62
    for block_size in (1, 12, 500):
        of_sums, between_units = _block_similarities(table, quantum, block_size)
        assert of_sums.min() == of_sums.max() == 1.0
        assert between_units.min() == between_units.max() == 1.0


@pytest.mark.parametrize("block_size", [1, 2, 3, 5, 7])
def test_block_similarities_match_the_definition(block_size: int) -> None:
    unit = normalise(rng(6).standard_normal((20, 7)))
    of_sums, between_units = _block_similarities(*table_of(unit), block_size)
    slow_sums, slow_units = [], []
    for gap in range(block_size, 20 - block_size + 1):
        before, after = unit[gap - block_size : gap], unit[gap : gap + block_size]
        slow_sums.append(cosine(before.sum(axis=0), after.sum(axis=0)))
        slow_units.append(
            np.mean([first @ second for first in before for second in after])
        )
    assert of_sums == pytest.approx(slow_sums, abs=1e-10)
    assert between_units == pytest.approx(slow_units, abs=1e-10)
    assert of_sums.size == between_units.size == 20 - 2 * block_size + 1


def test_block_similarities_need_a_block_on_each_side() -> None:
    table, quantum = table_of(normalise(rng(7).standard_normal((5, 3))))
    assert [values.size for values in _block_similarities(table, quantum, 3)] == [0, 0]
    assert [values.size for values in _block_similarities(table, quantum, 2)] == [2, 2]


def test_blocks_that_mix_two_contents_alike_point_the_same_way() -> None:
    # East and north, then north and east: the two blocks have the same sum,
    # and half of the pairs made of a unit of each are at right angles.
    mixed = np.array([EAST, NORTH, NORTH, EAST])
    of_sums, between_units = _block_similarities(*table_of(mixed), 2)
    assert of_sums.tolist() == [1.0]
    assert between_units.tolist() == [0.5]
    # Four units on one content are alike on both counts.
    same = np.array([EAST] * 4)
    of_sums, between_units = _block_similarities(*table_of(same), 2)
    assert of_sums.tolist() == [1.0]
    assert between_units.tolist() == [1.0]


def test_blocks_whose_units_cancel_out_have_no_direction() -> None:
    # East then west: the sum of the first block is zero and has no
    # direction. The cosine between the sums is then taken to be zero.
    west = [-1.0, 0.0]
    cancelling = np.array([EAST, west, NORTH, NORTH])
    of_sums, between_units = _block_similarities(*table_of(cancelling), 2)
    assert of_sums.tolist() == [0.0]
    assert between_units.tolist() == [0.0]
    # The same when the two units cancel out only to within rounding error.
    angle = 0.7
    first = [math.cos(angle), math.sin(angle)]
    second = [math.cos(angle + math.pi), math.sin(angle + math.pi)]
    assert first[0] + second[0] != 0.0 or first[1] + second[1] != 0.0
    of_sums, _ = _block_similarities(
        *table_of(np.array([first, second, EAST, EAST])), 2
    )
    assert of_sums.tolist() == [0.0]
    # Four units a quarter of a turn apart cancel out as well, whichever way
    # they face and whatever block follows them.
    for turn in np.linspace(0.05, 3.0, 40):
        quarter_turns = [
            [math.cos(turn + step * math.pi / 2), math.sin(turn + step * math.pi / 2)]
            for step in range(4)
        ]
        for following in ([EAST] * 4, [EAST, EAST, NORTH, NORTH], [NORTH] * 4):
            blocks = np.array(quarter_turns + following)
            of_sums, _ = _block_similarities(*table_of(blocks), 4)
            assert of_sums.tolist() == [0.0]


def test_a_block_cancels_out_when_its_mean_is_shorter_than_the_bound() -> None:
    # The four corners of a regular tetrahedron sum to zero; their sixteen
    # similarities are 1 and -1/3, which are not exact in binary, and their
    # counts sum to a few units, not to zero. The block has no direction.
    corners = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / 3**0.5
    for turn in (0.0, 0.3, 1.1):
        rotation = np.array(
            [
                [math.cos(turn), -math.sin(turn), 0.0],
                [math.sin(turn), math.cos(turn), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        blocks = np.vstack([corners @ rotation.T, [[1.0, 0.0, 0.0]] * 4])
        table, quantum = table_of(blocks)
        assert int(table[4, 4]) != 0
        of_sums, _ = _block_similarities(table, quantum, 4)
        assert of_sums.tolist() == [0.0]
    # Two nearly opposite units: the mean of the block has the length of the
    # small tilt. Just above the bound of two to the power -10 the cosine is
    # accurate to about one part in a million; just below it is zero.
    after = normalise(np.array([[1.0, 2.0, 0.5], [1.5, 1.0, 0.2]]))
    for mean_length, expected_zero in ((2.0**-9.9, False), (2.0**-10.1, True)):
        tilt = np.array([0.0, 0.0, mean_length])
        first = normalise(np.array([1.0, 0.0, 0.0]) + tilt)
        second = normalise(np.array([-1.0, 0.0, 0.0]) + tilt)
        unit = np.vstack([first, second, after])
        of_sums, _ = _block_similarities(*table_of(unit), 2)
        if expected_zero:
            assert of_sums.tolist() == [0.0]
        else:
            exact = cosine(unit[:2].sum(axis=0), unit[2:].sum(axis=0))
            assert of_sums[0] == pytest.approx(exact, abs=2e-6)


def test_the_same_blocks_give_exactly_the_same_similarities() -> None:
    # The sums of similarities are exact. Reordering the units inside the two
    # blocks compared, or anywhere outside them, leaves the two similarities
    # at that gap unchanged to the last digit; reading the text backwards
    # gives the same curves in reverse.
    state = rng(31)
    unit = normalise(state.standard_normal((40, 9)) + state.standard_normal(9))
    for block_size in (2, 5):
        expected = _block_similarities(*table_of(unit), block_size)
        backwards = _block_similarities(*table_of(unit[::-1]), block_size)
        for forward, reverse in zip(expected, backwards, strict=True):
            assert forward.tolist() == reverse[::-1].tolist()
        for _ in range(20):
            gap = integer(state, block_size, 40 - block_size + 1)
            order = np.concatenate(
                [
                    state.permutation(np.arange(0, gap - block_size)),
                    state.permutation(np.arange(gap - block_size, gap)),
                    state.permutation(np.arange(gap, gap + block_size)),
                    state.permutation(np.arange(gap + block_size, 40)),
                ]
            )
            reordered = _block_similarities(*table_of(unit[order]), block_size)
            for found, wanted in zip(reordered, expected, strict=True):
                assert found[gap - block_size] == wanted[gap - block_size]


def test_block_sums_keep_their_precision_in_a_long_text() -> None:
    # Three thousand units that are all much alike: the sums of similarities
    # reach millions. A few gaps are checked against sums computed without
    # rounding error from the similarities themselves.
    state = rng(32)
    unit = normalise(
        state.standard_normal(12) + 0.3 * state.standard_normal((3000, 12))
    )
    gram = unit @ unit.T
    table, quantum = table_of(unit)
    for block_size in (3, 12):
        of_sums, between_units = _block_similarities(table, quantum, block_size)
        for gap in (block_size, 700, 1500, 2900, 3000 - block_size):
            before = slice(gap - block_size, gap)
            after = slice(gap, gap + block_size)
            cross = math.fsum(gram[before, after].ravel())
            within = math.fsum(gram[before, before].ravel()) * math.fsum(
                gram[after, after].ravel()
            )
            index = gap - block_size
            assert of_sums[index] == pytest.approx(cross / math.sqrt(within), abs=1e-11)
            assert between_units[index] == pytest.approx(
                cross / block_size**2, abs=1e-11
            )


INF = math.inf


@pytest.mark.parametrize(
    ("scores", "left", "right"),
    [
        ([5.0, 1.0, 4.0], [-INF, 4.0, 1.0], [-1.0, 3.0, -INF]),
        # Each rise goes to the highest point of its side, not to the nearest.
        (
            [3.0, 1.0, 2.0, 0.0, 5.0],
            [-INF, 2.0, 1.0, 3.0, -2.0],
            [2.0, 4.0, 3.0, 5.0, -INF],
        ),
        ([1.0, 2.0, 3.0], [-INF, -1.0, -1.0], [2.0, 1.0, -INF]),  # no valley
        ([2.0, 2.0, 2.0], [-INF, 0.0, 0.0], [0.0, 0.0, -INF]),  # flat
        ([7.0], [-INF], [-INF]),
        ([], [], []),
    ],
)
def test_rises(scores: list[float], left: list[float], right: list[float]) -> None:
    found_left, found_right = _rises(np.array(scores))
    assert found_left.tolist() == left
    assert found_right.tolist() == right


def pairs_of(sizes: list[int]) -> list[tuple[int, int]]:
    on_the_left, on_the_right = _size_pairs(sizes)
    return [
        (sizes[i], sizes[j]) for i, j in zip(on_the_left, on_the_right, strict=True)
    ]


def test_size_pairs() -> None:
    assert pairs_of([3]) == [(3, 3)]
    assert pairs_of([3, 4]) == [(3, 3), (3, 4), (4, 3), (4, 4)]
    # Neither size is more than twice the other.
    assert pairs_of([3, 6, 7]) == [
        (3, 3),
        (3, 6),
        (6, 3),
        (6, 6),
        (6, 7),
        (7, 6),
        (7, 7),
    ]
    for n in (8, 24, 40, 100, 400):
        found = pairs_of(_block_sizes(n, 3))
        assert len(found) == len(set(found))
        assert all((right, left) in found for left, right in found)
        assert all(left <= 2 * right and right <= 2 * left for left, right in found)


def test_a_pair_of_block_sizes_that_fits_both_sides_is_examined() -> None:
    # With a units of one topic before a boundary and b units of another
    # after it, the blocks compared on the left of the boundary each lie
    # within one topic up to a size of a / 2 or b, whichever is smaller, and
    # those compared on its right up to b / 2 or a. In whole units, one of
    # these two sizes exceeds twice the other by one unit at most. The pair
    # examined then has a block one unit smaller on that side.
    examined = set(pairs_of(list(range(1, 60))))
    for before in range(2, 60):
        for after in range(2, 60):
            left = min(before // 2, after)
            right = min(after // 2, before)
            assert left <= 2 * right + 1
            assert right <= 2 * left + 1
            assert (min(left, 2 * right), min(right, 2 * left)) in examined
    # Seven units before the boundary and fourteen after: 3 and 7.
    assert (3, 7) not in examined
    assert (3, 6) in examined


def depths_by_pair(unit: Array, sizes: list[int]) -> dict[tuple[int, int], Array]:
    """Depth of every gap on the two similarities, by pair of block sizes."""
    depths = _valley_depths(*table_of(unit), sizes, *_size_pairs(sizes))
    return {pair: depths[:, index] for index, pair in enumerate(pairs_of(sizes))}


def deepest(depths: Array) -> tuple[float, float, int]:
    """Largest depth on each similarity, and the gap where their sum is largest."""
    return (
        float(depths[0].max()),
        float(depths[1].max()),
        int(depths.sum(axis=0).argmax()),
    )


def test_valley_depths_by_hand() -> None:
    # Twelve units in two dimensions: six along one axis, then six along the
    # other. With blocks of three both similarities are 1 at gaps 3 and 9,
    # where the two blocks lie on the same axis, and fall to 0 at gap 6.
    unit = np.array([EAST] * 6 + [NORTH] * 6)
    (depths,) = depths_by_pair(unit, [3]).values()
    assert depths.shape == (2, 13)
    # In between, the block after gap 4 holds two units of the first axis
    # and one of the second: the sums are at an angle whose cosine is
    # 2 / sqrt(5), and six of the nine pairs of units are alike. After gap 5
    # it holds one and two: 1 / sqrt(5), and three pairs of nine. The curves
    # are the same read from the other end. Both sides of every gap rise to
    # 1, so that the depth of a gap is 1 minus its similarity, and the first
    # and the last gap of the curves, which have one side only, have none.
    slope = [1 - 2 / math.sqrt(5), 1 - 1 / math.sqrt(5)]
    assert depths[0] == pytest.approx([0, 0, 0, 0, *slope, 1, *slope[::-1], 0, 0, 0, 0])
    assert depths[1] == pytest.approx(
        [0, 0, 0, 0, 1 / 3, 2 / 3, 1, 2 / 3, 1 / 3, 0, 0, 0, 0]
    )
    assert deepest(depths) == (1.0, 1.0, 6)
    # Blocks of two and of three, in the four combinations: the deepest
    # valley is at the same place and has the same depth whichever sizes
    # measure its sides.
    found = depths_by_pair(unit, [2, 3])
    assert len(found) == 4
    for depths in found.values():
        assert deepest(depths) == (1.0, 1.0, 6)


def test_a_valley_needs_two_sides() -> None:
    # Three units on one axis and nine on the other. With blocks of three the
    # similarities are lowest at the first gap of the curves, which has
    # nothing on its left, and they rise from there on: a rise on one side
    # only is not a valley, and no gap has any depth.
    unit = np.array([EAST] * 3 + [NORTH] * 9)
    (depths,) = depths_by_pair(unit, [3]).values()
    assert not depths.any()
    # The test reports nothing, at any level, although many reorderings of
    # these units do have a valley.
    result = segment_topics(unit, alpha=0.99, n_permutations=FAST)
    assert result.boundaries == ()
    assert result.tested


def test_a_low_point_at_the_edge_does_not_hide_a_valley() -> None:
    # Three units on one axis, six on the other, six on the first again. The
    # similarities are as low at gap 3, the first of the curves, as at gap 9.
    # Gap 3 has nothing on its left; gap 9 is a valley with two full sides.
    unit = np.array([EAST] * 3 + [NORTH] * 6 + [EAST] * 6)
    (depths,) = depths_by_pair(unit, [3]).values()
    assert depths[:, 3].tolist() == [0.0, 0.0]
    assert deepest(depths) == (1.0, 1.0, 9)
    # Read backwards, the text has its valley at the mirror position.
    (depths,) = depths_by_pair(unit[::-1], [3]).values()
    assert deepest(depths) == (1.0, 1.0, 6)


def test_the_deepest_valley_is_not_always_at_the_lowest_point() -> None:
    # Eleven units in two dimensions, given by their angles. Unit 4 stands
    # apart from its neighbours, and the direction turns for good, by one
    # radian, before unit 7.
    angles = np.array([0.0, 0.0, 0.0, 0.0, 1.2, 0.3, 0.3, 1.3, 1.3, 1.3, 1.3])
    unit = np.column_stack([np.cos(angles), np.sin(angles)])
    found = depths_by_pair(unit, [1, 2])
    # Between single units the two similarities are the same thing. They are
    # lowest before unit 4, at cos 1.2, and both sides rise to 1.
    apart = 1.0 - math.cos(1.2)
    assert deepest(found[1, 1]) == pytest.approx((apart, apart, 4))
    # With blocks of two, the block after gap 4 holds unit 4 and the next
    # unit. Its sum points halfway between them, at 0.75 radian from the
    # block before, and its two units are at 1.2 and 0.3 radian from the
    # units before. Both similarities are lower at gap 7, at cos 1.
    turn = 1.0 - math.cos(1.0)
    of_sums = 1.0 - math.cos(0.75)
    between_units = 1.0 - (math.cos(1.2) + math.cos(0.3)) / 2
    assert found[2, 2][:, 4] == pytest.approx([of_sums, between_units])
    assert deepest(found[2, 2]) == pytest.approx((turn, turn, 7))
    # For the two mixed pairs, one side of a gap rises from the similarities
    # of single units and the other from those of blocks of two, and the
    # smaller rise is the depth. At gap 4 the smaller rise is that of the
    # blocks of two; at gap 7 the two rises are equal. Gap 4 is where the
    # four curves are lowest together, and gap 7 is the deeper valley.
    assert 2 * math.cos(1.2) + (1 - of_sums) + (1 - between_units) < 4 * math.cos(1.0)
    for pair in ((1, 2), (2, 1)):
        assert found[pair][:, 4] == pytest.approx([of_sums, between_units])
        assert deepest(found[pair]) == pytest.approx((turn, turn, 7))


def test_the_two_similarities_can_have_their_deepest_valleys_at_different_gaps() -> (
    None
):
    # Nine units in the plane, at 0, 120 and 60 degrees from the first axis:
    # A B B A B C C A A. The cosine is -1/2 between A and B, and 1/2 between
    # C and either of them. Blocks of two.
    angles = np.radians([0.0, 120.0, 120.0, 0.0, 120.0, 60.0, 60.0, 0.0, 0.0])
    unit = np.column_stack([np.cos(angles), np.sin(angles)])
    (depths,) = depths_by_pair(unit, [2]).values()
    # From gap 2 to gap 7 the cosine between the sums of the two blocks is
    # 1, 1/2, sqrt(3)/2, 1, 1/2, 1/2. It has a valley of depth 1/2 at gap 3
    # and a shallow one at gap 4; gaps 6 and 7 have no rise on their right.
    shallow = 1 - math.sqrt(3) / 2
    assert depths[0] == pytest.approx([0, 0, 0, 0.5, shallow, 0, 0, 0, 0, 0])
    # The mean similarity between units is 1/4, 1/4, 3/8, 1/2, 3/8, 1/2. Its
    # only valley is at gap 6, with a depth of 1/8.
    assert depths[1] == pytest.approx([0, 0, 0, 0, 0, 0, 0.125, 0, 0, 0])
    # The statistic adds the two depths, each on its scale, although no gap
    # has a valley on both similarities. The candidate boundary is the gap
    # where the two depths together are largest, which is gap 3.
    found = _test_segment(_similarities(unit), 2, FAST, np.random.PCG64(0))
    assert found is not None
    assert found[0] == 3


def test_the_two_sides_of_a_valley_can_use_different_block_sizes() -> None:
    # Six units on one axis and eighteen on the other. On the left of the
    # boundary there is room for a rise with blocks of three only; on the
    # right, blocks of six give a rise as well.
    found = depths_by_pair(np.array([EAST] * 6 + [NORTH] * 18), [3, 6])
    assert deepest(found[3, 3]) == (1.0, 1.0, 6)
    assert deepest(found[3, 6]) == (1.0, 1.0, 6)
    # With blocks of six on the left, the first gap of the curves is the
    # boundary itself, which then has nothing on its left, and the curves
    # rise from there on.
    assert not found[6, 6].any()
    assert not found[6, 3].any()


def test_depths_that_do_not_vary_count_for_nothing() -> None:
    # One pair of block sizes and a thousand arrangements. On the first
    # similarity the depth is 0.1 everywhere, a number whose square is not
    # exact in floating point; on the second it varies.
    mirror = np.array([0])
    depths = np.empty((2, 1, 1000))
    depths[0] = 0.1
    depths[1] = rng(34).uniform(0.0, 0.5, 1000)
    scored = _scores(depths, mirror)
    assert scored is not None
    scores, kept, weights, margin = scored
    assert kept.tolist() == [0]
    assert weights[0].tolist() == [0.0]
    spread = float(depths[1].std())
    assert weights[1, 0] == pytest.approx(1 / spread)
    assert scores[0] == pytest.approx((depths[1, 0] - depths[1, 0].mean()) / spread)
    # The margin within which two scores are equal stays at rounding level.
    assert margin == pytest.approx(2e-12 / spread)
    # When nothing varies there is nothing to compare.
    depths[1] = 0.3
    assert _scores(depths, mirror) is None
    # Two pairs that are mirror images of each other share their scale.
    mirrored = np.array([1, 0])
    depths = rng(35).uniform(0.0, 1.0, (2, 2, 500))
    scored = _scores(depths, mirrored)
    assert scored is not None
    _, kept, weights, _ = scored
    assert kept.tolist() == [0, 1]
    assert np.array_equal(weights[:, 0], weights[:, 1])
    pooled = np.concatenate([depths[0, 0], depths[0, 1]])
    assert weights[0, 0] == pytest.approx(1 / pooled.std())


def statistics_of_all_arrangements(
    first: int, second: int, block_size: int
) -> tuple[list[tuple[int, ...]], Array, float]:
    """Statistic of every arrangement of two groups of identical units.

    An arrangement is a choice of places for the first group; the first one
    listed has the first group in the first places. Returns the
    arrangements, their statistics, and the margin within which two
    statistics are equal. The statistics are those that the test would
    compare if it examined every reordering of the text.
    """
    n = first + second
    sizes = _block_sizes(n, block_size)
    on_the_left, on_the_right = _size_pairs(sizes)
    mirror = np.lexsort((on_the_left, on_the_right))
    arrangements = list(itertools.combinations(range(n), first))
    depths = np.empty((2, on_the_left.size, len(arrangements)))
    for index, places in enumerate(arrangements):
        unit = np.array([EAST if place in places else NORTH for place in range(n)])
        depths[:, :, index] = _valley_depths(
            *table_of(unit), sizes, on_the_left, on_the_right
        ).max(axis=2)
    scored = _scores(depths, mirror)
    assert scored is not None
    scores, _, _, margin = scored
    return arrangements, scores.max(axis=0), margin


def reaching_the_ordered_text(first: int, second: int, block_size: int) -> Fraction:
    """Share of the arrangements of two groups that reach the ordered one.

    The proportion of arrangements whose statistic is at least that of the
    arrangement in which the first group comes first: the p-value that the
    test would give to that text if it examined every reordering.
    """
    arrangements, statistics, margin = statistics_of_all_arrangements(
        first, second, block_size
    )
    reaching = int((statistics >= statistics[0] - margin).sum())
    return Fraction(reaching, len(arrangements))


@pytest.mark.parametrize(
    ("first", "second", "block_size", "reaching"),
    [
        # Groups of exactly twice the largest block: the ordered text and its
        # mirror image are alone.
        (4, 4, 2, 2),
        (6, 6, 3, 2),
        (8, 8, 3, 2),
        # Longer groups leave room for other arrangements with as sharp a
        # change: those that keep, somewhere, a run of twice the largest
        # block of each group side by side, as in AAAAAABBBBBBAB.
        (5, 5, 2, 12),
        (4, 6, 2, 6),
        (4, 8, 2, 6),
        (5, 5, 3, 2),
        (6, 7, 3, 4),
        (6, 8, 3, 6),
        (7, 7, 3, 12),
    ],
)
def test_no_arrangement_of_two_groups_beats_the_ordered_text(
    first: int, second: int, block_size: int, reaching: int
) -> None:
    # Two groups of identical units, each with at least twice block_size
    # units, except five and five with blocks of three.
    arrangements, statistics, margin = statistics_of_all_arrangements(
        first, second, block_size
    )
    assert int((statistics > statistics[0] + margin).sum()) == 0
    assert int((statistics >= statistics[0] - margin).sum()) == reaching
    # With every reordering examined, the p-value of the ordered text would
    # be the share of the arrangements that reach it.
    assert reaching_the_ordered_text(first, second, block_size) == Fraction(
        reaching, len(arrangements)
    )


def test_an_arrangement_and_its_mirror_image_have_the_same_statistic() -> None:
    # Exactly the same, to the last digit: the sums of similarities are
    # exact, and a pair of block sizes shares its scale with its mirror
    # image.
    for first, second, block_size in ((5, 7, 2), (6, 6, 3), (3, 6, 1)):
        n = first + second
        arrangements, statistics, _ = statistics_of_all_arrangements(
            first, second, block_size
        )
        index = {places: number for number, places in enumerate(arrangements)}
        for number, places in enumerate(arrangements):
            mirrored = tuple(sorted(n - 1 - place for place in places))
            assert statistics[number] == statistics[index[mirrored]]
        assert len(set(statistics.tolist())) > 5


def test_each_similarity_alone_is_matched_by_other_arrangements() -> None:
    # Six units and six units, all those of a group identical, with blocks
    # of three. In the ordered text both similarities go from 1 to 0 and
    # back to 1: a depth of 1, the largest possible.
    places = list(itertools.combinations(range(12), 6))
    depths = np.empty((2, len(places)))
    for index, chosen in enumerate(places):
        unit = np.array([EAST if place in chosen else NORTH for place in range(12)])
        (found,) = depths_by_pair(unit, [3]).values()
        depths[:, index] = found.max(axis=1)
    assert depths[:, 0].tolist() == [1.0, 1.0]
    # The cosine between the sums of two blocks is 1 whenever they hold the
    # two groups in the same proportions, and 58 of the 924 arrangements
    # have a valley as deep as the ordered text by this similarity alone.
    assert int((depths[0] == 1.0).sum()) == 58
    # By the mean similarity between units, only the ordered text and its
    # mirror image do.
    assert int((depths[1] == 1.0).sum()) == 2
    assert int((depths == 1.0).all(axis=0).sum()) == 2


@pytest.mark.parametrize(
    ("first", "second", "block_size", "larger", "reaching"),
    [(3, 3, 2, 2, 4), (4, 4, 3, 2, 4), (4, 8, 3, 46, 53)],
)
def test_a_group_shorter_than_two_blocks_can_be_out_of_reach(
    first: int, second: int, block_size: int, larger: int, reaching: int
) -> None:
    # When a group has fewer than twice block_size units, the blocks on
    # either side of the boundary cannot both lie within a group, and other
    # arrangements have a larger statistic than the ordered one. With four
    # units and four units and blocks of three, 4 of the 70 arrangements
    # reach the ordered text: with every reordering examined its p-value
    # would be 4/70, which is above 0.05, however distinct the groups.
    arrangements, statistics, margin = statistics_of_all_arrangements(
        first, second, block_size
    )
    assert int((statistics > statistics[0] + margin).sum()) == larger
    assert reaching_the_ordered_text(first, second, block_size) == Fraction(
        reaching, len(arrangements)
    )


@pytest.mark.parametrize(
    ("first", "second", "block_size", "larger", "reaching"),
    [(2, 10, 1, 9, 11), (3, 9, 1, 8, 10), (4, 16, 2, 25, 29)],
)
def test_a_short_group_inside_a_long_one_can_beat_the_ordered_text(
    first: int, second: int, block_size: int, larger: int, reaching: int
) -> None:
    # Each group has at least twice block_size units, but the second is much
    # longer. Placing the first group in the middle of the second, as in
    # BBBBAABBBBBB, gives a larger statistic than keeping the groups apart:
    # H-SEG-6 holds for the configurations enumerated above, not in general.
    arrangements, statistics, margin = statistics_of_all_arrangements(
        first, second, block_size
    )
    assert int((statistics > statistics[0] + margin).sum()) == larger
    assert int((statistics >= statistics[0] - margin).sum()) == reaching
    best = arrangements[int(statistics.argmax())]
    assert 0 not in best and first + second - 1 not in best


def test_more_units_can_raise_the_p_value_of_a_clean_boundary() -> None:
    # With block size 2, five and five identical units reach a p-value of
    # 12/252, below 0.05, if every reordering is examined; five and six do
    # not: 26 of the 462 arrangements reach the ordered text, none beats it.
    assert reaching_the_ordered_text(5, 5, 2) == Fraction(12, 252)
    _, statistics, margin = statistics_of_all_arrangements(5, 6, 2)
    assert int((statistics > statistics[0] + margin).sum()) == 0
    assert reaching_the_ordered_text(5, 6, 2) == Fraction(26, 462)
    assert Fraction(26, 462) > Fraction(1, 20)


def test_random_reorderings_scatter_the_p_value_around_its_exact_value() -> None:
    # Four units and four units, identical within a group, with blocks of
    # three: with every reordering examined the p-value would be 4/70, about
    # 0.057. With 999 random reorderings it varies from seed to seed around
    # that value, and falls at or below 0.05 for some seeds.
    text = np.array([EAST] * 4 + [NORTH] * 4)
    p_values: list[float] = []
    for seed in range(40):
        result = segment_topics(text, alpha=0.9, seed=seed)
        assert result.boundaries == (4,)
        p_values.extend(result.p_values)
    assert float(np.mean(p_values)) == pytest.approx(4 / 70, abs=0.005)
    assert min(p_values) < 0.05 < max(p_values)
    # The figures quoted in docs/validation.md.
    assert float(np.mean(p_values)) == pytest.approx(0.059, abs=0.0005)
    assert sum(p_value <= 0.05 for p_value in p_values) == 5


def test_block_sizes() -> None:
    assert _block_sizes(7, 3) == []  # fewer than three candidate positions
    assert _block_sizes(8, 3) == [3]
    assert _block_sizes(24, 3) == [3, 4, 5, 6]
    assert _block_sizes(24, 2) == [2, 3, 4, 5, 6]
    assert _block_sizes(6, 1) == [1]
    assert _block_sizes(3, 1) == []
    # The smallest block is used even where it exceeds a quarter of the text.
    assert _block_sizes(11, 3) == [3]
    assert _block_sizes(12, 3) == [3]
    assert _block_sizes(16, 3) == [3, 4]
    # No block is larger than four times the smallest, however long the text.
    assert _block_sizes(48, 3) == list(range(3, 13))
    assert _block_sizes(100, 3) == list(range(3, 13))
    assert _block_sizes(10_000, 3) == list(range(3, 13))
    # Steps of one, then of a tenth of the current size, and the largest
    # size at the end whether or not a step falls on it.
    assert _block_sizes(400, 12) == [
        *range(12, 21),
        *(22, 24, 26, 28),
        *(30, 33, 36, 39),
        *(42, 46, 48),
    ]
    assert _block_sizes(1000, 8)[-3:] == [28, 30, 32]
    for n in (8, 30, 77, 400):
        for smallest in (1, 2, 3, 5, 9):
            sizes = _block_sizes(n, smallest)
            if not sizes:
                assert n < 2 * smallest + 2
                continue
            assert sizes == sorted(set(sizes))
            assert sizes[0] == smallest
            assert sizes[-1] == max(smallest, min(n // 4, 4 * smallest))
            assert all(n - 2 * size + 1 >= 3 for size in sizes)


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


def test_at_a_change_of_topic_the_two_similarities_have_their_valley_at_one_gap() -> (
    None
):
    # The deepest valley is sought on each similarity separately. In texts
    # with two topics the two valleys are at the same gap, for every pair of
    # block sizes, and at the change of topic for blocks of three.
    sizes = _block_sizes(24, DEFAULT_BLOCK_SIZE)
    for seed in range(30):
        found = depths_by_pair(two_topics(seed), sizes)
        for pair, depths in found.items():
            assert depths[0].argmax() == depths[1].argmax(), (seed, pair)
        assert found[3, 3][0].argmax() == 12
    # In texts on one topic they are at the same gap in 18 texts of 30.
    same = 0
    for seed in range(30):
        state = rng(seed)
        (centre,) = directions(1, 48, state)
        (depths,) = depths_by_pair(topic(centre, 24, 1.0, state), [3]).values()
        same += depths[0].argmax() == depths[1].argmax()
    assert same == 18


def test_a_boundary_near_the_start_or_the_end_is_found() -> None:
    def unequal(seed: int, lengths: tuple[int, int]) -> Array:
        state = rng(seed)
        first, second = directions(2, 48, state)
        return np.vstack(
            [
                topic(first, lengths[0], 1.0, state),
                topic(second, lengths[1], 1.0, state),
            ]
        )

    for lengths in ((6, 18), (18, 6)):
        found = 0
        for seed in range(30):
            result = segment_topics(unequal(200 + seed, lengths), n_permutations=99)
            if len(result.boundaries) == 1:
                found += abs(result.boundaries[0] - lengths[0]) <= 1
        assert found >= 24, lengths


def test_a_short_passage_does_not_hide_the_boundary_that_follows() -> None:
    # Three units on a first topic, then six and six on two others. The
    # similarities are often lowest at the end of the first passage, which is
    # too short to have a valley; the valley that counts is the next one.
    found = 0
    for seed in range(30):
        state = rng(1600 + seed)
        sequence = np.vstack(
            [
                topic(centre, length, 1.0, state)
                for centre, length in zip(
                    directions(3, 48, state), (3, 6, 6), strict=True
                )
            ]
        )
        result = segment_topics(sequence, n_permutations=99, seed=seed)
        found += len(result.boundaries) == 1 and abs(result.boundaries[0] - 9) <= 1
    assert found >= 24


def test_reading_direction_does_not_matter_when_units_repeat() -> None:
    # Three units on one axis, six on the other, six on the first again. The
    # boundary between the last two runs is found, and found at the mirror
    # position when the text is read backwards.
    repeated = np.array([EAST] * 3 + [NORTH] * 6 + [EAST] * 6)
    assert segment_topics(repeated, n_permutations=FAST).boundaries == (9,)
    assert segment_topics(repeated[::-1], n_permutations=FAST).boundaries == (6,)
    # Two runs of identical units, of 6 to 11 units each.
    for seed in range(20):
        state = rng(1500 + seed)
        first, second = directions(2, 48, state)
        lengths = (integer(state, 6, 12), integer(state, 6, 12))
        runs = np.array([first] * lengths[0] + [second] * lengths[1])
        forward = segment_topics(runs, n_permutations=FAST, seed=seed)
        backward = segment_topics(runs[::-1], n_permutations=FAST, seed=seed)
        assert forward.boundaries == (lengths[0],)
        assert backward.boundaries == (lengths[1],)


def test_rounding_does_not_decide_when_units_repeat() -> None:
    # A text made of runs of identical units has gaps, pairs of block sizes
    # and reorderings that are exactly as good as one another. Multiplying
    # the embeddings by a constant or reversing their columns changes the
    # last digits of some similarities. The result must not change.
    last_digits_differ = 0
    for seed in range(20):
        repeated = repeated_units(1300 + seed)
        expected = segment_topics(repeated, alpha=0.3, n_permutations=99, seed=seed)
        gram = normalise(repeated) @ normalise(repeated).T
        for other in (3.0 * repeated, repeated[:, ::-1]):
            found = segment_topics(other, alpha=0.3, n_permutations=99, seed=seed)
            assert found == expected, seed
            unit = normalise(other)
            last_digits_differ += not np.array_equal(unit @ unit.T, gram)
    assert last_digits_differ >= 10


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


def test_a_slow_transition_between_two_topics_is_rarely_split() -> None:
    def slow_transition(state: np.random.RandomState) -> Array:
        # Twenty-four units on one topic, forty-eight during which the
        # subject moves by equal steps to another, twenty-four on the other.
        first, second = directions(2, 48, state)
        steps = (np.arange(48) + 0.5) / 48
        weights = np.concatenate([np.zeros(24), steps, np.ones(24)])[:, np.newaxis]
        centres = normalise((1 - weights) * first + weights * second)
        scatter = state.standard_normal((96, 48)) / math.sqrt(48)
        return normalise(centres + 1.0 * scatter)

    # The transition lasts four times as long as the largest block compared.
    assert _block_sizes(96, DEFAULT_BLOCK_SIZE)[-1] == 12
    reported = 0
    for seed in range(30):
        sequence = slow_transition(rng(2000 + seed))
        if segment_topics(sequence, n_permutations=99, seed=seed).boundaries:
            reported += 1
    assert reported <= 4


def test_one_unrelated_unit_at_a_random_position_is_rarely_a_boundary() -> None:
    # The position is drawn at random, so that every order of the units is
    # equally likely: this is one more sequence to which the level applies.
    # In the middle of the text such a unit is reported more often than the
    # level (see docs/validation.md).
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


def test_related_consecutive_units_are_often_split_without_a_change_of_topic() -> None:
    # The limit of the test, documented in docs/metrics.md: 48 units on one
    # topic, each sharing half of its scatter with the next. Nothing changes
    # along the text, but its order is not exchangeable, and boundaries are
    # reported far more often than at the level of the test. With the same
    # scatter drawn independently for each unit, they are not.
    reported = {}
    for share in (0.0, 0.5):
        count = 0
        for seed in range(40):
            state = rng(2500 + seed)
            (centre,) = directions(1, 48, state)
            draws = state.standard_normal((49, 48)) / math.sqrt(48)
            scatter = math.sqrt(1 - share) * draws[:-1] + math.sqrt(share) * draws[1:]
            sequence = normalise(centre + scatter)
            count += bool(
                segment_topics(sequence, n_permutations=99, seed=seed).boundaries
            )
        reported[share] = count
    # 3 and 17 on the platform where the test was written; the bounds leave
    # room for differences in the last digits of the products.
    assert reported[0.0] <= 6
    assert reported[0.5] >= 12


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


def test_parts_whose_level_cannot_be_reached_are_not_examined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With 19 reorderings the parts left after the first boundary have
    # levels of 1/30 and 1/60, below the smallest p-value of 1/20: the test
    # of the whole text is the only one carried out.
    import discoursekit.semantic as semantic

    lengths: list[int] = []
    original = semantic._test_segment

    def recording(gram: Array, *arguments: object) -> object:
        lengths.append(gram.shape[0])
        return original(gram, *arguments)  # type: ignore[arg-type]

    monkeypatch.setattr(semantic, "_test_segment", recording)
    assert segment_topics(three_topics(), n_permutations=19).boundaries == (24,)
    assert lengths == [36]
    # With 29 reorderings the smallest p-value is 1/30: the part of 24 units,
    # at the level 1/30, is examined; the part of 12, at 1/60, is not, nor
    # are the two parts of 12 units into which the first is then divided.
    lengths.clear()
    assert segment_topics(three_topics(), n_permutations=29).boundaries == (12, 24)
    assert lengths == [36, 24]


def test_alpha_is_read_as_written() -> None:
    # 0.3 in binary is slightly less than three tenths. A p-value of exactly
    # 3/10, with 9 reorderings, is at the level 0.3 and is significant.
    state = rng(9010)
    sequence = state.standard_normal((integer(state, 8, 20), 5))
    result = segment_topics(sequence, alpha=0.3, n_permutations=9, seed=10)
    assert result.p_values == (0.3,)
    assert result.boundaries == (7,)


def test_the_first_of_equally_good_pairs_of_block_sizes_gives_the_candidate() -> None:
    # Runs of identical units in which two pairs of block sizes give the
    # observed statistic exactly; they point to different gaps, and the
    # first pair decides. The step-by-step implementation does the same.
    for case, block_size, expected in (
        (255, 2, ((5, 13), (0.1, 0.05))),
        (495, 1, ((5, 15), (0.05, 0.15))),
    ):
        state = rng(7000 + case)
        centres = state.standard_normal((3, 4))
        rows: list[Array] = []
        for _ in range(int(state.randint(2, 7))):
            rows.extend([centres[state.randint(0, 3)]] * int(state.randint(1, 9)))
        state.randint(1, 4)  # the draw of the block size in the search
        sequence = np.array(rows)
        result = segment_topics(
            sequence, block_size=block_size, alpha=0.3, n_permutations=19, seed=case
        )
        assert (result.boundaries, result.p_values) == expected
        assert (
            segment_directly(sequence, block_size, Fraction(3, 10), 19, case)
            == expected
        )


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
    examined: list[float] = []
    for seed in range(10):
        result = segment_topics(
            two_topics(seed, noise=1.6), alpha=0.5, n_permutations=FAST
        )
        for p_value in result.p_values:
            assert 1 / 200 <= p_value <= 0.5
            assert p_value * 200 == pytest.approx(round(p_value * 200))
        examined.extend(result.p_values)
    # Enough of them, and not all at the smallest value.
    assert len(examined) >= 10
    assert len(set(examined)) >= 3


def test_permutations_are_the_same_everywhere() -> None:
    # The reorderings come from the raw stream of the PCG64 bit generator.
    # These two values pin that stream: if they change, results obtained
    # with a given seed are no longer reproducible.
    sequence = two_topics(14, n_each=6, noise=1.8)
    assert segment_topics(
        sequence, alpha=0.9, seed=0, n_permutations=FAST
    ).p_values == (0.105,)
    assert segment_topics(
        sequence, alpha=0.9, seed=1, n_permutations=FAST
    ).p_values == (0.16,)


def test_false_positive_rate_under_exchangeability() -> None:
    # 200 sequences of independent random directions, in which every order
    # is equally likely. At alpha = 0.05 a boundary should be reported in
    # about 10 of them at most. If the rate were exactly 5%, more than 21
    # would occur with probability 0.0005. This check can only reveal a
    # gross excess; the simulation reported in docs/validation.md rests on
    # 80,000 sequences.
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

# Two numbers closer than this are taken to be equal by the direct
# implementation. It is far above the rounding error of either
# implementation and far below any difference between two arrangements.
ALIKE = 1e-9


def block_curves(unit: Array, size: int) -> tuple[dict[int, float], dict[int, float]]:
    """The two block similarities at each gap, computed from the vectors.

    The cosine between the sums of the ``size`` units before and after the
    gap, and the mean similarity between a unit before and a unit after.
    """
    of_sums, between_units = {}, {}
    for gap in range(size, len(unit) - size + 1):
        before, after = unit[gap - size : gap], unit[gap : gap + size]
        cross = float(before.sum(axis=0) @ after.sum(axis=0))
        lengths = (
            float(np.linalg.norm(before.sum(axis=0))),
            float(np.linalg.norm(after.sum(axis=0))),
        )
        # A block whose units cancel one another out has no direction: the
        # mean of its units is shorter than two to the power -10.
        cancelled = min(lengths) <= size * 2.0**-10
        of_sums[gap] = 0.0 if cancelled else cross / (lengths[0] * lengths[1])
        pairs = [float(first @ second) for first in before for second in after]
        between_units[gap] = sum(pairs) / len(pairs)
    return of_sums, between_units


def depth_of_each_gap(
    on_the_left: dict[int, float], on_the_right: dict[int, float]
) -> dict[int, float]:
    """Depth of the valley at each gap that two curves both cover.

    The rise on the left of a gap is the highest value of the first curve at
    an earlier gap minus its value at the gap; the rise on its right is the
    same on the second curve, with later gaps. The depth is the smaller of
    the two, and zero if one is missing or negative.
    """
    depths = {}
    for gap in on_the_left:
        if gap not in on_the_right:
            continue
        earlier = [value for other, value in on_the_left.items() if other < gap]
        later = [value for other, value in on_the_right.items() if other > gap]
        depth = 0.0
        if earlier and later:
            depth = min(max(earlier) - on_the_left[gap], max(later) - on_the_right[gap])
        depths[gap] = max(depth, 0.0)
    return depths


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
        largest = max(block_size, min(n // 4, 4 * block_size))
        if n - 2 * largest + 1 < 3 or level < Fraction(1, 1 + n_permutations):
            continue
        sizes = []
        size = block_size
        while size < largest:
            sizes.append(size)
            size += max(1, size // 10)
        sizes.append(largest)
        pairs = [
            (before, after)
            for before in sizes
            for after in sizes
            if before <= 2 * after and after <= 2 * before
        ]
        orders = [np.arange(n)]
        for _ in range(n_permutations):
            orders.append(np.argsort(generator.random_raw(n), kind="stable"))
        # For each arrangement, each pair of sizes and each similarity, the
        # depth of every gap.
        valleys = []
        for order in orders:
            curves = {size: block_curves(part[order], size) for size in sizes}
            valleys.append(
                [
                    [
                        depth_of_each_gap(
                            curves[before][measure], curves[after][measure]
                        )
                        for measure in (0, 1)
                    ]
                    for before, after in pairs
                ]
            )
        # The deepest valley: one row for each similarity, one column for
        # each pair of sizes, one layer for each arrangement.
        deepest = np.array(
            [
                [
                    [max(valley[row][measure].values()) for valley in valleys]
                    for row in range(len(pairs))
                ]
                for measure in (0, 1)
            ]
        )
        # A pair of sizes and its mirror image share one mean and one
        # standard deviation. Depths that do not vary count for zero.
        weights = np.zeros((2, len(pairs)))
        standard = np.zeros(deepest.shape)
        for measure in (0, 1):
            for row, (before, after) in enumerate(pairs):
                mirror = pairs.index((after, before))
                both = np.concatenate([deepest[measure, row], deepest[measure, mirror]])
                if both.std() > ALIKE:
                    weights[measure, row] = 1 / both.std()
                    centred = deepest[measure, row] - both.mean()
                    standard[measure, row] = centred * weights[measure, row]
        keep = [row for row in range(len(pairs)) if weights[:, row].any()]
        if not keep:
            continue
        scores = standard.sum(axis=0)[keep]
        statistic = scores.max(axis=0)
        alike = ALIKE * float(weights.max())
        reached = int((statistic[1:] >= statistic[0] - alike).sum())
        if Fraction(1 + reached, 1 + n_permutations) > level:
            continue
        # The first pair that gives the observed statistic, and for that
        # pair the first gap whose depths add up to the most on the common
        # scale.
        best = next(
            row
            for row, score in zip(keep, scores[:, 0], strict=True)
            if score >= statistic[0] - alike
        )
        total = {
            gap: sum(
                valleys[0][best][measure][gap] * weights[measure, best]
                for measure in (0, 1)
            )
            for gap in valleys[0][best][0]
        }
        candidate = next(
            gap for gap, value in total.items() if value >= max(total.values()) - alike
        )
        middle = start + candidate
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


def repeated_units(seed: int) -> Array:
    """A text made of runs of identical units, on two or three directions."""
    state = rng(seed)
    centres = directions(3, 16, state)
    n_runs = integer(state, 2, 6)
    rows = []
    for run in range(n_runs):
        centre = centres[integer(state, 0, 3)] if run else centres[0]
        rows.extend([centre] * integer(state, 2, 9))
    return np.array(rows)


def test_segmentation_matches_a_direct_implementation_when_units_cancel() -> None:
    # Runs of identical units in which a direction and its opposite both
    # occur: many blocks have a sum of zero, and no direction.
    numbers_found = []
    for seed in range(12):
        state = rng(1400 + seed)
        centres = directions(2, 16, state)
        choices = [centres[0], -centres[0], centres[1], -centres[1]]
        rows: list[Array] = []
        for _ in range(integer(state, 3, 7)):
            rows.extend([choices[integer(state, 0, 4)]] * integer(state, 2, 7))
        sequence = np.array(rows)
        block_size = integer(state, 1, 4)
        alpha = Fraction(integer(state, 2, 9), 20)
        result = segment_topics(
            sequence,
            block_size=block_size,
            alpha=float(alpha),
            n_permutations=39,
            seed=seed,
        )
        expected = segment_directly(sequence, block_size, alpha, 39, seed)
        assert (result.boundaries, result.p_values) == expected, seed
        numbers_found.append(len(result.boundaries))
    assert max(numbers_found) >= 1


def test_segmentation_matches_a_direct_implementation_when_units_repeat() -> None:
    # Identical units give reorderings that are exactly as deep as one
    # another, and several gaps or pairs of block sizes that are equally
    # good: the count of ties and the choice among equals are compared.
    numbers_found = []
    tied = 0
    for seed in range(30):
        sequence = repeated_units(1200 + seed)
        state = rng(seed)
        block_size = integer(state, 1, 4)
        alpha = Fraction(integer(state, 2, 9), 20)
        result = segment_topics(
            sequence,
            block_size=block_size,
            alpha=float(alpha),
            n_permutations=39,
            seed=seed,
        )
        expected = segment_directly(sequence, block_size, alpha, 39, seed)
        assert (result.boundaries, result.p_values) == expected, seed
        numbers_found.append(len(result.boundaries))
        tied += any(p_value > 1 / 40 for p_value in result.p_values)
    assert numbers_found.count(0) >= 3
    assert max(numbers_found) >= 2
    # In some of the texts a reordering reaches the observed statistic.
    assert tied >= 3


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
    with pytest.raises(error, match=message):
        segment_topics(two_topics(0), **parameters)  # type: ignore[arg-type]


def test_coherence_does_not_estimate_boundaries() -> None:
    # A text with two topics is one segment unless boundaries are supplied,
    # and the parameters of the test are not accepted.
    sequence = two_topics(0)
    result = coherence(sequence)
    assert result.boundaries == ()
    assert result.local_coherence_within_segments == result.local_coherence
    assert result.global_coherence_within_segments == result.global_coherence
    with pytest.raises(TypeError):
        coherence(sequence, n_permutations=FAST)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="sequence of unit indices"):
        coherence(sequence, boundaries=None)  # type: ignore[arg-type]


# Estimated boundaries in the summary


def test_estimated_boundaries_can_be_passed_to_the_summary() -> None:
    sequence = two_topics(0)
    found = segment_topics(sequence)
    assert isinstance(found, TopicSegmentation)
    assert found.boundaries == (12,)
    result = coherence(sequence, boundaries=found.boundaries)
    assert result.boundaries == (12,)
    supplied = coherence(sequence, boundaries=[12])
    assert summary(result) == summary(supplied)
    assert result.local_coherence_within_segments == (
        supplied.local_coherence_within_segments
    )
    assert result.global_coherence_within_segments == (
        supplied.global_coherence_within_segments
    )


def test_a_change_of_topic_lowers_plain_coherence_far_more_than_within_segments() -> (
    None
):
    state = rng(12)
    first, second = directions(2, 48, state)
    one_topic = topic(first, 24, 1.0, state)
    changed = np.vstack([topic(first, 12, 1.0, state), topic(second, 12, 1.0, state)])
    assert segment_topics(one_topic, n_permutations=FAST).boundaries == ()
    found = segment_topics(changed, n_permutations=FAST).boundaries
    assert found == (12,)
    single = coherence(one_topic)
    double = coherence(changed, boundaries=found)
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
    result = coherence(two_topics(0), boundaries=[12])
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.local_coherence = 0.0  # type: ignore[misc]
    found = segment_topics(two_topics(0), n_permutations=FAST)
    with pytest.raises(dataclasses.FrozenInstanceError):
        found.boundaries = ()  # type: ignore[misc]
