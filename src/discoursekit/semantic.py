"""Semantic coherence of a sequence of text units.

The functions of this module take one embedding vector per unit of text
(sentence, utterance or other unit chosen by the study), in order of
production, and return a small number of quantities.

Local coherence
    Mean cosine similarity between consecutive units: the degree to which
    adjoining units relate to one another. This is the first-order coherence
    of Bedi et al. (2015), computed here on any embedding.

Local coherence expected under random order
    The value local coherence takes, on average, when the same units are put
    in random order. It is the mean cosine similarity over all pairs of units
    and has a closed form. Local coherence is only informative relative to
    this level, which depends on the text and on the embedding model.

Global coherence
    Mean cosine similarity between each unit and the centroid of all the
    other units: the degree to which each unit relates to the overall content
    of the text. It does not depend on the order of the units.

Reference coherence
    Mean cosine similarity between each unit and a reference vector supplied
    by the study, for example the embedding of the question that was asked
    (Elvevag et al., 2007) or the average response of other speakers to the
    same prompt (Hoffman et al., 2018).

Topic boundaries
    Positions where the content changes abruptly and durably. They can be
    supplied, when the elicitation procedure defines them, or estimated with
    a permutation test built on the block comparison of TextTiling (Hearst,
    1997). See :func:`segment_topics`.

Coherence within segments
    Local and global coherence computed inside the segments delimited by the
    boundaries, so that a change of topic is not counted as a loss of
    coherence.

None of these quantities has been validated as a measure of any clinical
construct. ``docs/metrics.md`` gives the definitions, hypotheses and known
confounds; ``docs/validation.md`` states what has been checked and how.

References
----------
Bedi, G., Carrillo, F., Cecchi, G. A., Slezak, D. F., Sigman, M., Mota, N. B.,
Ribeiro, S., Javitt, D. C., Copelli, M., & Corcoran, C. M. (2015). Automated
analysis of free speech predicts psychosis onset in high-risk youths. npj
Schizophrenia, 1, 15030. https://doi.org/10.1038/npjschz.2015.30

Elvevag, B., Foltz, P. W., Weinberger, D. R., & Goldberg, T. E. (2007).
Quantifying incoherence in speech: An automated methodology and novel
application to schizophrenia. Schizophrenia Research, 93(1-3), 304-316.
https://doi.org/10.1016/j.schres.2007.03.001

Hearst, M. A. (1997). TextTiling: Segmenting text into multi-paragraph
subtopic passages. Computational Linguistics, 23(1), 33-64.
https://aclanthology.org/J97-1003/

Hoffman, P., Loginova, E., & Russell, A. (2018). Poor coherence in older
people's speech is explained by impaired semantic and executive processes.
eLife, 7, e38907. https://doi.org/10.7554/eLife.38907

Matteson, D. S., & James, N. A. (2014). A nonparametric approach for multiple
change point analysis of multivariate data. Journal of the American
Statistical Association, 109(505), 334-345.
https://doi.org/10.1080/01621459.2013.849605

Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be
zero: Calculating exact P-values when permutations are randomly drawn.
Statistical Applications in Genetics and Molecular Biology, 9(1), Article 39.
https://doi.org/10.2202/1544-6115.1585
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_PERMUTATIONS",
    "Coherence",
    "TopicSegmentation",
    "adjacent_similarities",
    "coherence",
    "segment_topics",
]

#: Default smallest number of units compared on each side of a candidate
#: boundary. A convention of this library: a topic is taken to last at least
#: this many units. Always report it.
DEFAULT_BLOCK_SIZE = 3

#: Default significance level of the test for topic boundaries, for the text
#: as a whole.
DEFAULT_ALPHA = 0.05

#: Default number of random permutations in each test.
DEFAULT_PERMUTATIONS = 999

Vectors = NDArray[np.float64]

# Unit vectors and their cosine similarities are of the order of one. A
# quantity that should be zero and is computed from them is treated as zero
# below this value, which is far above rounding error and far below any
# difference between two texts.
_ROUNDING = 1e-12


# Input


def _to_unit_length(vectors: Vectors) -> Vectors:
    """Scale each row of finite, non-zero vectors to unit length.

    Each row is first multiplied by a power of two that brings its largest
    component between one half and one. This step is exact, and it keeps the
    squares that enter the norm away from overflow and underflow, whatever
    the magnitude of the vectors.
    """
    largest = np.abs(vectors).max(axis=-1, keepdims=True, initial=0.0)
    _, exponents = np.frexp(largest)
    scaled = np.ldexp(vectors, -exponents)
    norms = np.linalg.norm(scaled, axis=-1, keepdims=True)
    return np.asarray(scaled / norms, dtype=np.float64)


def _unit_vectors(embeddings: ArrayLike) -> Vectors:
    """Validate the embeddings and scale each row to unit length."""
    vectors = np.asarray(embeddings, dtype=np.float64)
    if vectors.ndim == 1 and vectors.size == 0:
        return vectors.reshape(0, 0)
    if vectors.ndim != 2:
        raise ValueError(
            "embeddings must be a two-dimensional array with one row per unit, "
            f"got {vectors.ndim} dimension(s)"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings contain NaN or infinite values")
    empty = ~vectors.any(axis=1)
    if empty.any():
        index = int(np.flatnonzero(empty)[0])
        raise ValueError(
            f"unit {index} has a zero embedding, for which cosine similarity is "
            "undefined; remove empty units or check the embedding model"
        )
    return _to_unit_length(vectors)


def _reference_vector(reference: ArrayLike, dimensions: int | None) -> Vectors:
    """Validate the reference and scale it to unit length.

    ``dimensions`` is the number of dimensions of the embeddings, or ``None``
    when there are no units from which it could be known.
    """
    vector = np.asarray(reference, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(
            f"reference must be a single vector, got {vector.ndim} dimension(s)"
        )
    if dimensions is not None and vector.size != dimensions:
        raise ValueError(
            f"reference has {vector.size} dimensions, embeddings have {dimensions}"
        )
    if not np.isfinite(vector).all() or not vector.any():
        raise ValueError("reference must be a finite, non-zero vector")
    return _to_unit_length(vector)


def _check_integer(name: str, value: int, minimum: int) -> int:
    """Validate an integer parameter and return it as a plain ``int``."""
    # bool is a subclass of int and is never a meaningful value here.
    # Integer types of other libraries, such as NumPy, are accepted.
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


@dataclass(frozen=True)
class _TestParameters:
    """Validated parameters of the test for topic boundaries."""

    block_size: int
    alpha: float
    n_permutations: int
    seed: int

    @property
    def level(self) -> Fraction:
        """The level as an exact number: ``0.05`` stands for one twentieth.

        The level of each part of a text is a fraction of ``alpha``, and a
        p-value is a ratio of two integers. Comparing them exactly keeps
        rounding error out of the decision.
        """
        return Fraction(str(self.alpha))

    @property
    def smallest_p_value(self) -> Fraction:
        """The smallest p-value that ``n_permutations`` reorderings can give."""
        return Fraction(1, 1 + self.n_permutations)


def _check_test_parameters(
    block_size: int, alpha: float, n_permutations: int, seed: int
) -> _TestParameters:
    block_size = _check_integer("block_size", block_size, 1)
    n_permutations = _check_integer("n_permutations", n_permutations, 1)
    seed = _check_integer("seed", seed, 0)
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError(f"alpha must be a float, got {type(alpha).__name__}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}")
    parameters = _TestParameters(block_size, float(alpha), n_permutations, seed)
    if parameters.level < parameters.smallest_p_value:
        raise ValueError(
            f"with {n_permutations} permutations the smallest attainable p-value "
            f"is 1/{n_permutations + 1}, which is above alpha={alpha}; "
            "increase n_permutations"
        )
    return parameters


# Local, global and reference coherence


def _adjacent(unit: Vectors) -> Vectors:
    if unit.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    similarities = np.einsum("ij,ij->i", unit[:-1], unit[1:])
    return np.asarray(np.clip(similarities, -1.0, 1.0), dtype=np.float64)


def adjacent_similarities(embeddings: ArrayLike) -> Vectors:
    """Cosine similarity between each unit and the next.

    Parameters
    ----------
    embeddings
        Array of shape ``(n_units, n_dimensions)``, one row per unit, in
        order.

    Returns
    -------
    numpy.ndarray
        ``n_units - 1`` values in ``[-1, 1]``; empty if there are fewer than
        two units. Their mean is the local coherence.

    Notes
    -----
    Summaries other than the mean can be computed from this series. Extreme
    values such as the minimum depend on the number of units, because a
    longer text gives more opportunities for a low value.
    """
    return _adjacent(_unit_vectors(embeddings))


def _mean(values: Vectors) -> float:
    return float(values.mean()) if values.size else math.nan


def _mean_pairwise(unit: Vectors) -> float:
    """Mean cosine similarity over all pairs of distinct units."""
    n = unit.shape[0]
    if n < 2:
        return math.nan
    total = unit.sum(axis=0)
    # The squared norm of the sum counts every ordered pair once, plus the n
    # self-similarities, each equal to 1.
    mean = float((total @ total - n) / (n * (n - 1)))
    return min(1.0, max(-1.0, mean))


def _to_rest(unit: Vectors) -> Vectors:
    """Cosine similarity between each unit and the centroid of the other units.

    The result is NaN for a unit when the other units cancel one another out,
    which leaves the direction of their centroid undefined. They are taken to
    cancel out when the length of the centroid is within rounding error of
    zero.
    """
    n = unit.shape[0]
    if n < 2:
        return np.full(n, np.nan)
    rest = unit.sum(axis=0) - unit
    norms = np.linalg.norm(rest, axis=1)
    dots = np.einsum("ij,ij->i", unit, rest)
    similarities = np.full(n, np.nan)
    np.divide(dots, norms, out=similarities, where=norms > _ROUNDING * (n - 1))
    return np.asarray(np.clip(similarities, -1.0, 1.0), dtype=np.float64)


def _mean_defined(values: Vectors) -> float:
    return _mean(values[~np.isnan(values)])


# Topic boundaries


def _area_table(gram: Vectors) -> Vectors:
    """Summed-area table of a matrix of similarities.

    ``table[a, b]`` is the sum of ``gram[:a, :b]``, so that the sum of any
    rectangular block of ``gram`` is obtained from four entries.
    """
    n = gram.shape[0]
    table = np.zeros((n + 1, n + 1))
    table[1:, 1:] = gram.cumsum(axis=0).cumsum(axis=1)
    return table


def _block_similarities(table: Vectors, block_size: int) -> Vectors:
    """Cosine similarity between the blocks before and after each candidate gap.

    ``table`` is the summed-area table of the similarities between units.
    Candidate gap ``j`` lies before unit ``block_size + j``. Only gaps with a
    full block of ``block_size`` units on each side are candidates, so there
    are ``n_units - 2 * block_size + 1`` of them. The cosine between the sums
    of two blocks of unit vectors is the sum of their cross similarities
    divided by the geometric mean of the sums of their internal similarities.
    """
    k = block_size
    n_gaps = table.shape[0] - 2 * k
    if n_gaps < 1:
        return np.empty(0, dtype=np.float64)
    # The sums needed lie on three diagonals of the table: main[i] is the sum
    # over the first i units, near[i] over the first i units by the first
    # i + k, far[i] over the first i units by the first i + 2k. Gap j has
    # units j to j + k - 1 before it and units j + k to j + 2k - 1 after it.
    main = np.diagonal(table)
    near = np.diagonal(table, k)
    far = np.diagonal(table, 2 * k)
    first = slice(0, n_gaps)
    second = slice(k, k + n_gaps)
    third = slice(2 * k, 2 * k + n_gaps)
    cross = near[second] - far[first] - main[second] + near[first]
    before = main[second] - 2.0 * near[first] + main[first]
    after = main[third] - 2.0 * near[second] + main[second]
    norms = np.sqrt(np.maximum(before * after, 0.0))
    similarities = np.zeros(n_gaps)
    np.divide(cross, norms, out=similarities, where=norms > 0)
    return similarities


def _valley(scores: Vectors) -> tuple[float, int]:
    """Depth and position of the lowest point of a series of scores.

    The depth is the smaller of the two rises from the lowest point to the
    highest point on its left and to the highest point on its right. It is
    zero when the lowest point is at either end of the series, where it
    cannot be told from the side of a valley that lies outside.
    """
    lowest = int(np.argmin(scores))
    if lowest == 0 or lowest == scores.size - 1:
        return 0.0, lowest
    rim = min(float(scores[:lowest].max()), float(scores[lowest + 1 :].max()))
    return rim - float(scores[lowest]), lowest


def _block_sizes(n_units: int, block_size: int) -> list[int]:
    """Block sizes at which a segment of ``n_units`` units is examined.

    From ``block_size`` up to a quarter of the segment, in steps of one tenth
    of the current size (at least one). A size is kept only if it leaves at
    least three candidate gaps, the minimum for a valley to have two sides.
    """
    largest = max(block_size, n_units // 4)
    sizes = []
    size = block_size
    while size <= largest and n_units - 2 * size + 1 >= 3:
        sizes.append(size)
        size += max(1, size // 10)
    return sizes


def _test_segment(
    gram: Vectors,
    block_size: int,
    n_permutations: int,
    bit_generator: np.random.PCG64,
) -> tuple[int, int] | None:
    """Compare the deepest valley of one segment with random reorderings.

    ``gram`` holds the similarities between the units of the segment. Returns
    ``(position, reached)``: the index, within the segment, of the first unit
    after the candidate boundary, and the number of reorderings whose
    statistic is at least the observed one. Returns ``None`` if the segment
    is too short to be examined or if no reordering changes the depths.
    """
    n = gram.shape[0]
    sizes = _block_sizes(n, block_size)
    if not sizes:
        return None

    # Column 0 is the observed order; the others are random reorderings.
    depths = np.empty((len(sizes), n_permutations + 1))
    positions = []
    for arrangement in range(n_permutations + 1):
        if arrangement == 0:
            table = _area_table(gram)
        else:
            # The raw output of the bit generator is stable across NumPy
            # versions, so a seed gives the same permutations everywhere.
            order = np.argsort(bit_generator.random_raw(n), kind="stable")
            table = _area_table(gram[np.ix_(order, order)])
        for row, size in enumerate(sizes):
            depth, lowest = _valley(_block_similarities(table, size))
            depths[row, arrangement] = depth
            if arrangement == 0:
                positions.append(size + lowest)

    # Two depths closer than this are equal to within rounding error, which
    # grows with the number of similarities summed.
    rounding = _ROUNDING * n

    # Valley depths at different block sizes have different spreads. They are
    # put on a common scale with the mean and standard deviation, at each
    # block size, over all arrangements including the observed one, which
    # keeps the observed order exchangeable with the others under the null.
    # A block size is left out when its depths are the same in every
    # arrangement: units that are all alike have no boundary.
    spreads = depths.std(axis=1)
    usable = np.flatnonzero(spreads > rounding)
    if usable.size == 0:
        return None
    standard = (depths[usable] - depths[usable].mean(axis=1, keepdims=True)) / (
        spreads[usable, np.newaxis]
    )
    strongest = standard.max(axis=0)

    # A reordering can have exactly the depths of the observed order: its
    # mirror image does, and so does one that leaves the blocks around the
    # valley in place. Such ties count as reaching the observed statistic.
    # They are frequent in short segments, and rounding error must not
    # break them.
    tolerance = rounding / spreads[usable].min()
    reached = int((strongest[1:] >= strongest[0] - tolerance).sum())
    best = usable[int(np.argmax(standard[:, 0]))]
    return positions[best], reached


@dataclass(frozen=True)
class TopicSegmentation:
    """Result of the test for topic boundaries.

    Attributes
    ----------
    boundaries
        Index of the first unit of each new segment, in increasing order.
        Empty if no boundary was found.
    p_values
        P-value of each boundary, in the same order, in the test in which it
        was found: the proportion ``(1 + b) / (1 + B)``, where ``b`` counts
        the reorderings, among ``B``, whose statistic is at least as large as
        the observed one. The level of that test is ``alpha`` multiplied by
        the share of the text that was under test, which is below ``alpha``
        for every boundary after the first.
    n_units
        Number of units in the text.
    tested
        Whether the text was long enough to be tested, that is, had at least
        ``2 * block_size + 2`` units. If false, the absence of boundaries
        says nothing about the text.
    block_size, alpha, n_permutations, seed
        Parameters of the test. Report them with the result.
    """

    boundaries: tuple[int, ...]
    p_values: tuple[float, ...]
    n_units: int
    tested: bool
    block_size: int
    alpha: float
    n_permutations: int
    seed: int

    @property
    def segments(self) -> tuple[tuple[int, int], ...]:
        """Segments as ``(start, stop)`` pairs of unit indices, ``stop`` excluded."""
        return _segments(self.n_units, self.boundaries)


def _segments(n_units: int, boundaries: Sequence[int]) -> tuple[tuple[int, int], ...]:
    if n_units == 0:
        return ()
    edges = [0, *boundaries, n_units]
    return tuple(pairwise(edges))


def _segment(unit: Vectors, parameters: _TestParameters) -> TopicSegmentation:
    n = unit.shape[0]
    n_permutations = parameters.n_permutations
    gram = unit @ unit.T
    bit_generator = np.random.PCG64(parameters.seed)
    p_values: dict[int, float] = {}
    # Parts of the text still to be examined, in the order in which they were
    # created, each with the level at which it is tested.
    pending = [(0, n, parameters.level)]
    while pending:
        start, stop, level = pending.pop(0)
        if level < parameters.smallest_p_value:
            continue  # no p-value can be as small as the level of this part
        found = _test_segment(
            gram[start:stop, start:stop],
            parameters.block_size,
            n_permutations,
            bit_generator,
        )
        if found is None:
            continue
        position, reached = found
        if Fraction(1 + reached, 1 + n_permutations) > level:
            continue
        boundary = start + position
        p_values[boundary] = (1 + reached) / (1 + n_permutations)
        # The level is shared between the two parts in proportion to their
        # lengths, in exact arithmetic. The level of a part is thus alpha
        # times its share of the text, and the levels of the parts under
        # test at any stage add up to alpha at most.
        length = stop - start
        pending.append((start, boundary, level * (boundary - start) / length))
        pending.append((boundary, stop, level * (stop - boundary) / length))

    boundaries = tuple(sorted(p_values))
    return TopicSegmentation(
        boundaries=boundaries,
        p_values=tuple(p_values[boundary] for boundary in boundaries),
        n_units=n,
        tested=bool(_block_sizes(n, parameters.block_size)),
        block_size=parameters.block_size,
        alpha=parameters.alpha,
        n_permutations=n_permutations,
        seed=parameters.seed,
    )


def segment_topics(
    embeddings: ArrayLike,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    alpha: float = DEFAULT_ALPHA,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> TopicSegmentation:
    """Test for abrupt and durable changes of topic in a sequence of units.

    Statistic. For a block size ``k`` and each position with ``k`` units on
    both sides, the cosine similarity between the sum of the ``k`` units
    before and the sum of the ``k`` units after is computed, as in the block
    comparison of TextTiling (Hearst, 1997). The lowest point of this curve
    is the candidate boundary. Its depth is the smaller of the two rises from
    that point to the highest point of the curve on its left and on its
    right: a boundary is a valley with two sides, where two blocks differ
    from each other more than the blocks on either side do. Depths are
    computed for block sizes from ``block_size`` up to a quarter of the
    number of units and put on a common scale; the largest is the statistic.

    Test. The statistic is recomputed on ``n_permutations`` random
    reorderings of the units. The p-value is ``(1 + b) / (1 + B)``, where
    ``b`` is the number of reorderings, out of ``B``, whose statistic is at
    least the observed one (Phipson & Smyth, 2010). Statistics that are
    equal to within rounding error count as equal. A boundary is reported
    when the p-value is at most the level of the test. If every order of the
    units is equally likely (exchangeability), the probability of reporting a
    boundary is at most ``alpha``.

    Subdivision. When a boundary is found, the two parts are examined in the
    same way, with reorderings confined to each part, until no further
    boundary is found. This hierarchical procedure follows Matteson and James
    (2014). Each part is tested at the level ``alpha`` multiplied by its
    share of the units of the text, so that the levels of the parts under
    test at any stage add up to ``alpha`` at most. A part is not examined
    when its level is below the smallest p-value that the reorderings can
    give, ``1 / (1 + n_permutations)``.

    Parameters
    ----------
    embeddings
        Array of shape ``(n_units, n_dimensions)``, one row per unit, in
        order.
    block_size
        Smallest number of units compared on each side of a position. A
        boundary is never placed within ``block_size`` units of either end of
        the part under test, so every segment has more than ``block_size``
        units.
    alpha
        Significance level of the test of the whole text.
    n_permutations
        Number of random reorderings in each test. With the default values
        of ``alpha`` and ``n_permutations``, a part shorter than a fiftieth
        of the text is not examined.
    seed
        Seed of the random reorderings. The result is deterministic for a
        given seed.

    Returns
    -------
    TopicSegmentation

    Notes
    -----
    The depth used here differs from the depth score of Hearst (1997) in two
    ways: it takes the smaller of the two rises instead of their sum, and it
    measures each rise to the highest point on that side instead of the
    nearest peak. A rise on one side only is therefore not a valley, and the
    smoothing step of TextTiling is not needed.

    What the test does not report. A gradual drift of content produces no
    valley and no boundary. A single unit unrelated to its neighbours does
    not separate two blocks. Low similarity scattered through the text is
    what random reorderings also produce.

    Power. The test compares blocks of units and needs the segments on both
    sides of a boundary to be long enough. Its power is limited in short
    texts, for segments shorter than about twice ``block_size``, and when the
    units of a topic are only weakly similar to one another.
    ``docs/validation.md`` gives detection rates on simulated data.

    Memory. The computation holds the matrix of similarities between all
    pairs of units, that is, ``n_units`` squared values.

    Examples
    --------
    Sixteen units in two dimensions: eight scattered around one direction,
    then eight around another.

    >>> import numpy as np
    >>> state = np.random.RandomState(0)
    >>> first, second = np.array([1.0, 0.1]), np.array([0.1, 1.0])
    >>> scatter = 0.2 * state.standard_normal((16, 2))
    >>> result = segment_topics(np.array([first] * 8 + [second] * 8) + scatter)
    >>> result.boundaries
    (8,)
    >>> result.p_values
    (0.018,)
    >>> result.segments
    ((0, 8), (8, 16))
    """
    parameters = _check_test_parameters(block_size, alpha, n_permutations, seed)
    return _segment(_unit_vectors(embeddings), parameters)


# Summary


@dataclass(frozen=True)
class Coherence:
    """Semantic coherence of one text, with what is needed to interpret it.

    Attributes
    ----------
    local_coherence
        Mean cosine similarity between consecutive units; NaN with fewer than
        two units.
    local_coherence_shuffled
        Expected value of ``local_coherence`` if the same units were in random
        order (mean similarity over all pairs of units). The difference
        between the two is the part of local coherence that is due to order.
    global_coherence
        Mean cosine similarity between each unit and the centroid of the other
        units; NaN with fewer than two units. Independent of order.
    reference_coherence
        Mean cosine similarity between each unit and the reference vector;
        NaN if no reference was given.
    local_coherence_within_segments
        ``local_coherence`` restricted to pairs of units in the same segment.
    global_coherence_within_segments
        ``global_coherence`` computed inside each segment (each unit against
        the centroid of the other units of its segment), averaged over all
        units in segments of at least two units.
    boundaries
        Index of the first unit of each new segment used for the two
        quantities above: those supplied by the caller, or those found by
        :func:`segment_topics`.
    n_units
        Number of units. Report it and use it as a covariate.
    segmentation
        Details of the test when the boundaries were estimated; ``None`` when
        they were supplied.
    """

    local_coherence: float
    local_coherence_shuffled: float
    global_coherence: float
    reference_coherence: float
    local_coherence_within_segments: float
    global_coherence_within_segments: float
    boundaries: tuple[int, ...]
    n_units: int
    segmentation: TopicSegmentation | None


def _check_boundaries(boundaries: Sequence[int], n_units: int) -> tuple[int, ...]:
    kind = type(boundaries).__name__
    if isinstance(boundaries, (str, bytes)):
        raise TypeError(f"boundaries must be a sequence of unit indices, got {kind}")
    try:
        items = list(boundaries)
    except TypeError:
        raise TypeError(
            f"boundaries must be a sequence of unit indices, got {kind}"
        ) from None
    checked: list[int] = []
    for boundary in items:
        if isinstance(boundary, bool) or not isinstance(boundary, Integral):
            raise TypeError(
                f"boundaries must be integers, got {type(boundary).__name__}"
            )
        checked.append(int(boundary))
    if any(later <= earlier for earlier, later in pairwise(checked)):
        raise ValueError("boundaries must be strictly increasing")
    if checked and not (1 <= checked[0] and checked[-1] <= n_units - 1):
        raise ValueError(
            f"boundaries must lie between 1 and n_units - 1 = {n_units - 1}; "
            "each is the index of the first unit of a new segment"
        )
    return tuple(checked)


def coherence(
    embeddings: ArrayLike,
    *,
    reference: ArrayLike | None = None,
    boundaries: Sequence[int] | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    alpha: float = DEFAULT_ALPHA,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> Coherence:
    """Compute the semantic coherence of a sequence of units.

    Parameters
    ----------
    embeddings
        Array of shape ``(n_units, n_dimensions)``, one row per unit, in
        order of production. Any embedding can be used; rows are scaled to
        unit length internally.
    reference
        Vector of ``n_dimensions`` values against which every unit is
        compared, obtained with the same embedding as the units: for example
        the question that was asked, or the average of the responses of
        other speakers to the same prompt.
    boundaries
        Known changes of topic, as the index of the first unit of each new
        segment, for example where a new question was asked. If ``None``
        (default) they are estimated with :func:`segment_topics`. Pass an
        empty sequence to treat the text as a single segment.
    block_size, alpha, n_permutations, seed
        Parameters of :func:`segment_topics`; ignored when ``boundaries`` is
        given.

    Returns
    -------
    Coherence

    Examples
    --------
    Six units: three about one subject, then three about another.

    >>> import numpy as np
    >>> a, b = np.array([1.0, 0.1]), np.array([0.1, 1.0])
    >>> result = coherence([a, a, a, b, b, b], boundaries=[3])
    >>> round(result.local_coherence, 2)
    0.84
    >>> round(result.local_coherence_within_segments, 2)
    1.0
    >>> round(result.global_coherence, 2)
    0.66
    >>> round(result.global_coherence_within_segments, 2)
    1.0
    """
    unit = _unit_vectors(embeddings)
    n = unit.shape[0]

    segmentation = None
    if boundaries is None:
        parameters = _check_test_parameters(block_size, alpha, n_permutations, seed)
        segmentation = _segment(unit, parameters)
        used = segmentation.boundaries
    else:
        used = _check_boundaries(boundaries, n)

    adjacent = _adjacent(unit)
    same_segment = np.ones(adjacent.size, dtype=bool)
    same_segment[np.asarray(used, dtype=np.intp) - 1] = False
    to_rest_within = [_to_rest(unit[start:stop]) for start, stop in _segments(n, used)]

    reference_coherence = math.nan
    if reference is not None:
        # An empty list of units does not say how many dimensions they have.
        direction = _reference_vector(reference, unit.shape[1] or None)
        if n:
            reference_coherence = _mean(np.clip(unit @ direction, -1.0, 1.0))

    return Coherence(
        local_coherence=_mean(adjacent),
        local_coherence_shuffled=_mean_pairwise(unit),
        global_coherence=_mean_defined(_to_rest(unit)),
        reference_coherence=reference_coherence,
        local_coherence_within_segments=_mean(adjacent[same_segment]),
        global_coherence_within_segments=_mean_defined(
            np.concatenate(to_rest_within) if to_rest_within else np.empty(0)
        ),
        boundaries=used,
        n_units=n,
        segmentation=segmentation,
    )
