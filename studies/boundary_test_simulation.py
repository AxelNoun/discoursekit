"""Operating characteristics of the test for topic boundaries on simulated data.

This script measures how often :func:`discoursekit.segment_topics` reports a
boundary in sequences of simulated unit vectors. It answers four questions
about the test itself.

1. Level. When every order of the units is equally likely, how often is a
   boundary reported? The construction of the test puts this rate at
   ``alpha`` or below.
2. Unrelated units. How often is a boundary reported when one or a few
   unrelated units interrupt a single topic?
3. Gradual change. How often is a boundary reported when the content drifts
   steadily, and how fast must a transition between two topics be for the
   test to report it?
4. Power. When abrupt changes of topic are planted, how often are they found,
   and at the right place?

It also measures what becomes of the level when all the sequences are tested
with the same seed, and therefore with the same reorderings, and how often a
boundary is reported when consecutive units are related, as they are in
coherent text, without any change of topic. The level of the test does not
apply to that case.

The sequences are not text. A topic is a direction in a space of 48
dimensions; a unit on that topic is the direction plus isotropic scatter,
scaled to unit length. The results describe the behaviour of the statistical
procedure. They say nothing about its behaviour on embeddings of natural
language, which the planned benchmark is to establish.

Usage, from the root of the repository::

    python studies/boundary_test_simulation.py --jobs 2 > results.md

The output does not depend on the number of jobs: every sequence has its own
seed. The run takes about five hours on one core.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import discoursekit
from discoursekit import segment_topics

Array = NDArray[np.float64]
Sequence = tuple[Array, tuple[int, ...]]
DIMENSIONS = 48
BLOCK_SIZE = 3
ALPHA = 0.05
N_PERMUTATIONS = 199


@dataclass(frozen=True)
class Setting:
    """How distinct the topics are and how scattered the units of a topic are.

    ``overlap`` is the cosine similarity between the directions of any two
    topics; ``scatter`` is the standard deviation of the scatter relative to
    the length of the direction. The mean cosine similarity between two units
    is close to ``1 / (1 + scatter**2)`` within a topic and to ``overlap``
    times that between topics.
    """

    letter: str
    label: str
    overlap: float
    scatter: float

    @property
    def within(self) -> float:
        """Approximate cosine similarity between two units of the same topic."""
        return 1 / (1 + self.scatter**2)

    @property
    def between(self) -> float:
        """Approximate cosine similarity between two units of different topics."""
        return self.overlap * self.within

    def consecutive(self, correlation: float) -> float:
        """Approximate cosine similarity between two related consecutive units.

        The units are on the same topic and their scatters have the given
        correlation (see :func:`related_neighbours`).
        """
        return (1 + correlation * self.scatter**2) * self.within


SETTINGS = (
    Setting("A", "Distinct topics, moderate scatter", overlap=0.0, scatter=1.0),
    Setting("B", "Related topics, larger scatter", overlap=0.25, scatter=1.3),
    Setting("C", "Closely related topics, large scatter", overlap=0.55, scatter=1.6),
)


def normalise(vectors: Array) -> Array:
    """Scale vectors to unit length."""
    return np.asarray(vectors / np.linalg.norm(vectors, axis=-1, keepdims=True))


class Simulator:
    """Source of simulated sequences for one setting and one random state."""

    def __init__(self, setting: Setting, state: np.random.RandomState) -> None:
        self.setting = setting
        self.state = state
        # What all the topics of a sequence have in common, followed by what
        # is specific to each topic drawn so far.
        self._orthogonal = [self.direction()]
        self.home = self.topic()

    def direction(self) -> Array:
        """Draw a direction uniformly at random."""
        return normalise(self.state.standard_normal(DIMENSIONS))

    def topic(self) -> Array:
        """Draw the direction of a new topic.

        A topic is the sum of a component common to all the topics of the
        sequence and of a component of its own, orthogonal to the common one
        and to those of the other topics. The cosine similarity between any
        two topics is then exactly the overlap of the setting.
        """
        own = self.state.standard_normal(DIMENSIONS)
        for other in self._orthogonal:
            own = own - (own @ other) * other
        own = normalise(own)
        self._orthogonal.append(own)
        overlap = self.setting.overlap
        common = self._orthogonal[0]
        return np.asarray(math.sqrt(overlap) * common + math.sqrt(1 - overlap) * own)

    def units(self, centres: Array, n: int | None = None) -> Array:
        """Draw units scattered around one direction or around one each.

        ``centres`` is a single direction, around which ``n`` units are
        drawn, or one direction per unit.
        """
        count = len(centres) if n is None else n
        scatter = self.state.standard_normal((count, DIMENSIONS))
        scatter = scatter / math.sqrt(DIMENSIONS)
        return normalise(centres + self.setting.scatter * scatter)

    def blocks(self, *lengths: int) -> Array:
        """Draw consecutive blocks: the home topic first, then a new topic each."""
        centres = [self.home, *(self.topic() for _ in lengths[1:])]
        return np.vstack(
            [self.units(c, n) for c, n in zip(centres, lengths, strict=True)]
        )


Scenario = Callable[[Simulator], Sequence]

# Sequences in which every order is equally likely


def independent_directions(n: int) -> Scenario:
    """Return ``n`` directions drawn independently and uniformly."""

    def scenario(sim: Simulator) -> Sequence:
        return np.vstack([sim.direction() for _ in range(n)]), ()

    return scenario


def one_topic(n: int) -> Scenario:
    """Return ``n`` units that are all on the same topic."""

    def scenario(sim: Simulator) -> Sequence:
        return sim.units(sim.home, n), ()

    return scenario


def two_topics_in_random_order(n: int) -> Scenario:
    """Return the units of two topics, half each, thoroughly mixed."""

    def scenario(sim: Simulator) -> Sequence:
        sequence = sim.blocks(n // 2, n - n // 2)
        return np.asarray(sequence[sim.state.permutation(n)]), ()

    return scenario


def intruder_anywhere(n: int) -> Scenario:
    """Return one topic with one unrelated unit at a random position."""

    def scenario(sim: Simulator) -> Sequence:
        sequence = sim.units(sim.home, n)
        # The type is given because the default has 32 bits on Windows only.
        position = int(sim.state.randint(0, n, dtype=np.int64))
        sequence[position] = sim.units(sim.direction(), 1)[0]
        return sequence, ()

    return scenario


# One topic whose units are related to their neighbours


def related_neighbours(n: int, correlation: float) -> Scenario:
    """Return ``n`` units on one topic, each related to the next.

    The scatter of a unit is a mixture of a draw of its own and of the draw
    of the next unit, so that the scatters of two consecutive units have the
    given correlation, 0.5 at most, and those of units further apart are
    independent. Nothing changes along the sequence, but its order is not
    exchangeable: consecutive units are more alike than other pairs.
    """
    share = (1 - math.sqrt(1 - 4 * correlation**2)) / 2
    own, next_one = math.sqrt(1 - share), math.sqrt(share)

    def scenario(sim: Simulator) -> Sequence:
        draws = sim.state.standard_normal((n + 1, DIMENSIONS))
        draws = draws / math.sqrt(DIMENSIONS)
        scatter = own * draws[:-1] + next_one * draws[1:]
        return normalise(sim.home + sim.setting.scatter * scatter), ()

    return scenario


# One topic interrupted by unrelated units


def intruders_in_the_middle(n: int, count: int) -> Scenario:
    """Return one topic with adjacent unrelated units in the middle.

    The unrelated units are all on one other subject.
    """

    def scenario(sim: Simulator) -> Sequence:
        sequence = sim.units(sim.home, n)
        first = n // 2 - count // 2
        sequence[first : first + count] = sim.units(sim.direction(), count)
        return sequence, ()

    return scenario


def scattered_intruders(n: int, positions: tuple[int, ...]) -> Scenario:
    """Return one topic with unrelated units, each on a subject of its own."""

    def scenario(sim: Simulator) -> Sequence:
        sequence = sim.units(sim.home, n)
        for position in positions:
            sequence[position] = sim.units(sim.direction(), 1)[0]
        return sequence, ()

    return scenario


# Gradual change


def transition(before: int, length: int, after: int, *, unrelated: bool) -> Scenario:
    """Return a steady move from the home topic to another direction.

    The sequence has ``before`` units on the home topic, ``length`` units
    whose subject moves by equal steps from the home topic to the other
    direction, and ``after`` units on that direction. The other direction is
    drawn uniformly if ``unrelated``; otherwise it is a topic of the setting.
    """

    def scenario(sim: Simulator) -> Sequence:
        other = sim.direction() if unrelated else sim.topic()
        steps = (np.arange(length) + 0.5) / length
        weights = np.concatenate([np.zeros(before), steps, np.ones(after)])
        weights = weights[:, np.newaxis]
        centres = normalise((1 - weights) * sim.home + weights * other)
        return sim.units(centres), ()

    return scenario


# Sequences with planted changes of topic


def planted(*lengths: int) -> Scenario:
    """Return blocks of the given lengths, each on a different topic."""
    boundaries = tuple(int(edge) for edge in np.cumsum(lengths)[:-1])

    def scenario(sim: Simulator) -> Sequence:
        return sim.blocks(*lengths), boundaries

    return scenario


def excursion(before: int, away: int, after: int) -> Scenario:
    """Return a topic, a passage on another topic, then the first topic again."""

    def scenario(sim: Simulator) -> Sequence:
        elsewhere = sim.topic()
        sequence = np.vstack(
            [
                sim.units(sim.home, before),
                sim.units(elsewhere, away),
                sim.units(sim.home, after),
            ]
        )
        return sequence, (before, before + away)

    return scenario


def after_a_short_passage(passage: int, first: int, second: int) -> Scenario:
    """Return a short passage on one topic, then two segments on two others.

    The boundary counted as planted is the one between the two segments. The
    passage is meant to be too short for the test to delimit it.
    """

    def scenario(sim: Simulator) -> Sequence:
        return sim.blocks(passage, first, second), (passage + first,)

    return scenario


SCENARIOS: dict[str, Scenario] = {
    # Every order equally likely
    "8 independent directions": independent_directions(8),
    "10 independent directions": independent_directions(10),
    "12 independent directions": independent_directions(12),
    "16 independent directions": independent_directions(16),
    "24 independent directions": independent_directions(24),
    "40 independent directions": independent_directions(40),
    "64 independent directions": independent_directions(64),
    "One topic": one_topic(24),
    "Two topics, 12 + 12 units, in random order": two_topics_in_random_order(24),
    "One topic, one unrelated unit at a random position": intruder_anywhere(24),
    # Unrelated units
    "24 units, one unrelated unit in the middle": intruders_in_the_middle(24, 1),
    "24 units, two adjacent unrelated units in the middle": (
        intruders_in_the_middle(24, 2)
    ),
    "24 units, three scattered unrelated units": (scattered_intruders(24, (5, 12, 19))),
    "12 units, one unrelated unit in the middle": intruders_in_the_middle(12, 1),
    "12 units, two adjacent unrelated units in the middle": (
        intruders_in_the_middle(12, 2)
    ),
    # Gradual change
    "Drift through the whole sequence, 12 units": (
        transition(0, 12, 0, unrelated=True)
    ),
    "Drift through the whole sequence, 24 units": (
        transition(0, 24, 0, unrelated=True)
    ),
    "Drift through the whole sequence, 60 units": (
        transition(0, 60, 0, unrelated=True)
    ),
    "Drift through the whole sequence, 160 units": (
        transition(0, 160, 0, unrelated=True)
    ),
    "One topic, then a drift over the second half, 24 units": (
        transition(12, 12, 0, unrelated=True)
    ),
    "One topic, then a drift over the second half, 60 units": (
        transition(30, 30, 0, unrelated=True)
    ),
    "One topic, then a drift over the second half, 160 units": (
        transition(80, 80, 0, unrelated=True)
    ),
    "Transition of 6 units": transition(24, 6, 24, unrelated=False),
    "Transition of 12 units": transition(24, 12, 24, unrelated=False),
    "Transition of 24 units": transition(24, 24, 24, unrelated=False),
    "Transition of 48 units": transition(24, 48, 24, unrelated=False),
    # Abrupt changes
    "Two topics, 4 + 4 units": planted(4, 4),
    "Two topics, 5 + 5 units": planted(5, 5),
    "Two topics, 6 + 6 units": planted(6, 6),
    "Two topics, 8 + 8 units": planted(8, 8),
    "Two topics, 12 + 12 units": planted(12, 12),
    "Two topics, 20 + 20 units": planted(20, 20),
    "Two topics, 24 + 24 units": planted(24, 24),
    "Two topics, 80 + 80 units": planted(80, 80),
    "Two topics, 4 + 8 units": planted(4, 8),
    "Two topics, 6 + 18 units": planted(6, 18),
    "Two topics, 6 + 34 units": planted(6, 34),
    "Three topics, 8 + 8 + 8 units": planted(8, 8, 8),
    "Three topics, 12 + 12 + 12 units": planted(12, 12, 12),
    "Excursion and return, 6 + 4 + 6 units": excursion(6, 4, 6),
    "Excursion and return, 10 + 4 + 10 units": excursion(10, 4, 10),
    "Excursion and return, 10 + 8 + 10 units": excursion(10, 8, 10),
    "Excursion and return, 16 + 8 + 16 units": excursion(16, 8, 16),
    "Two topics, 6 + 6 units, after a passage of 3 units": (
        after_a_short_passage(3, 6, 6)
    ),
    # Consecutive units related, without any change of topic
    "One topic, 24 units, consecutive units related (0.2)": (
        related_neighbours(24, 0.2)
    ),
    "One topic, 24 units, consecutive units related (0.5)": (
        related_neighbours(24, 0.5)
    ),
    "One topic, 48 units, consecutive units related (0.2)": (
        related_neighbours(48, 0.2)
    ),
    "One topic, 48 units, consecutive units related (0.5)": (
        related_neighbours(48, 0.5)
    ),
}

# Correlations between the scatters of consecutive units in the scenarios above.
RELATED = (0.2, 0.5)

# The rows of each table: scenario and block size of the test.
TABLES: dict[str, list[tuple[str, int]]] = {
    "independent": [
        ("8 independent directions", 3),
        ("16 independent directions", 3),
        ("24 independent directions", 3),
        ("40 independent directions", 3),
        ("64 independent directions", 3),
        ("12 independent directions", 1),
        ("10 independent directions", 2),
    ],
    "exchangeable": [
        ("One topic", 3),
        ("Two topics, 12 + 12 units, in random order", 3),
        ("One topic, one unrelated unit at a random position", 3),
    ],
    "related neighbours": [
        ("One topic, 24 units, consecutive units related (0.2)", 3),
        ("One topic, 24 units, consecutive units related (0.5)", 3),
        ("One topic, 48 units, consecutive units related (0.2)", 3),
        ("One topic, 48 units, consecutive units related (0.5)", 3),
    ],
    "unrelated units": [
        ("24 units, one unrelated unit in the middle", 3),
        ("24 units, two adjacent unrelated units in the middle", 3),
        ("24 units, three scattered unrelated units", 3),
    ],
    "drift": [
        ("Drift through the whole sequence, 24 units", 3),
        ("Drift through the whole sequence, 60 units", 3),
        ("Drift through the whole sequence, 160 units", 3),
        ("One topic, then a drift over the second half, 24 units", 3),
        ("One topic, then a drift over the second half, 60 units", 3),
        ("One topic, then a drift over the second half, 160 units", 3),
    ],
    "transition": [
        ("Two topics, 24 + 24 units", 3),
        ("Transition of 6 units", 3),
        ("Transition of 12 units", 3),
        ("Transition of 24 units", 3),
        ("Transition of 48 units", 3),
    ],
    "planted": [
        ("Two topics, 6 + 6 units", 3),
        ("Two topics, 8 + 8 units", 3),
        ("Two topics, 12 + 12 units", 3),
        ("Two topics, 20 + 20 units", 3),
        ("Two topics, 80 + 80 units", 3),
        ("Two topics, 6 + 18 units", 3),
        ("Two topics, 6 + 34 units", 3),
        ("Three topics, 8 + 8 + 8 units", 3),
        ("Three topics, 12 + 12 + 12 units", 3),
        ("Excursion and return, 10 + 4 + 10 units", 3),
        ("Excursion and return, 10 + 8 + 10 units", 3),
        ("Excursion and return, 16 + 8 + 16 units", 3),
    ],
    # Sequences of 8 to 16 units, examined with block sizes 3 and 2.
    "short, block size 3": [
        ("12 units, one unrelated unit in the middle", 3),
        ("12 units, two adjacent unrelated units in the middle", 3),
        ("Drift through the whole sequence, 12 units", 3),
        ("Two topics, 4 + 4 units", 3),
        ("Two topics, 5 + 5 units", 3),
        ("Two topics, 4 + 8 units", 3),
        ("Excursion and return, 6 + 4 + 6 units", 3),
        ("Two topics, 6 + 6 units, after a passage of 3 units", 3),
    ],
    "short, block size 2": [
        ("12 units, one unrelated unit in the middle", 2),
        ("12 units, two adjacent unrelated units in the middle", 2),
        ("Drift through the whole sequence, 12 units", 2),
        ("Two topics, 4 + 4 units", 2),
        ("Two topics, 5 + 5 units", 2),
        ("Two topics, 6 + 6 units", 2),
        ("Two topics, 4 + 8 units", 2),
        ("Excursion and return, 6 + 4 + 6 units", 2),
    ],
    # Longer sequences, examined with a block size of 6.
    "long, block size 6": [
        ("Two topics, 80 + 80 units", 6),
        ("Transition of 24 units", 6),
        ("Transition of 48 units", 6),
        ("One topic, then a drift over the second half, 160 units", 6),
    ],
}

# Tables whose rates are compared with the level of the test, on more
# sequences than the others and, for the first, in one setting only.
LEVEL_TABLES = ("independent", "exchangeable")

# Level when all the sequences are tested with one seed: number of
# reorderings and seed, None standing for a seed of its own for each
# sequence. The sequences are those of the scenario named below.
ONE_SEED = "one seed"
ONE_SEED_SCENARIO = "8 independent directions"
SEEDING: tuple[tuple[int, int | None], ...] = (
    (19, None),
    (19, 0),
    (19, 3),
    (199, None),
    (199, 0),
    (999, None),
    (999, 0),
)


Key = tuple[str, str, int, int, int, int | None]


@dataclass(frozen=True)
class Row:
    """One scenario under one setting, examined with one block size.

    ``stream`` identifies the scenario and the setting. With the index of
    the sequence within the row, it seeds the random state from which the
    sequence is drawn. Two rows that differ only in the parameters of the
    test thus examine the same sequences, and no two other sequences share a
    seed. ``seed`` is the seed of the reorderings, or ``None`` when each
    sequence has a seed of its own, its index within the row.
    """

    table: str
    name: str
    block_size: int
    setting: int
    replicates: int
    stream: int
    n_permutations: int = N_PERMUTATIONS
    seed: int | None = None

    @property
    def key(self) -> Key:
        """What identifies the row among those of a run."""
        return (
            self.table,
            self.name,
            self.block_size,
            self.setting,
            self.n_permutations,
            self.seed,
        )


@dataclass(frozen=True)
class Outcome:
    """Counts over the sequences of one row."""

    replicates: int
    any_boundary: int
    all_correct: int
    too_many: int


def rows(replicates: int, level_replicates: int, seed_replicates: int) -> list[Row]:
    """Return every row, in a fixed order."""
    names = list(SCENARIOS)
    listed = []
    for table, entries in TABLES.items():
        settings = [0] if table == "independent" else range(len(SETTINGS))
        count = level_replicates if table in LEVEL_TABLES else replicates
        for setting in settings:
            for name, block_size in entries:
                stream = names.index(name) * len(SETTINGS) + setting
                listed.append(Row(table, name, block_size, setting, count, stream))
    stream = names.index(ONE_SEED_SCENARIO) * len(SETTINGS)
    for n_permutations, seed in SEEDING:
        listed.append(
            Row(
                ONE_SEED,
                ONE_SEED_SCENARIO,
                BLOCK_SIZE,
                0,
                seed_replicates,
                stream,
                n_permutations,
                seed,
            )
        )
    return listed


def run(row: Row) -> Outcome:
    """Apply the test to the sequences of one row."""
    scenario = SCENARIOS[row.name]
    any_boundary = all_correct = too_many = 0
    for replicate in range(row.replicates):
        state = np.random.RandomState([row.stream, replicate])
        sequence, truth = scenario(Simulator(SETTINGS[row.setting], state))
        found = segment_topics(
            sequence,
            block_size=row.block_size,
            alpha=ALPHA,
            n_permutations=row.n_permutations,
            seed=replicate if row.seed is None else row.seed,
        ).boundaries
        any_boundary += bool(found)
        too_many += len(found) > len(truth)
        all_correct += len(found) == len(truth) and all(
            abs(a - b) <= 1 for a, b in zip(found, truth, strict=True)
        )
    return Outcome(row.replicates, any_boundary, all_correct, too_many)


Outcomes = dict[Key, Outcome]


def outcome_of(
    outcomes: Outcomes, table: str, name: str, block_size: int, setting: int
) -> Outcome:
    """Return the outcome of a row tested with the parameters of the study."""
    return outcomes[table, name, block_size, setting, N_PERMUTATIONS, None]


def percent(count: int, total: int, digits: int = 0) -> str:
    """Format a proportion as a percentage."""
    return f"{100 * count / total:.{digits}f}%"


def interval(count: int, total: int) -> str:
    """Return the 95% Wilson score interval of a proportion, in percent."""
    z = 1.959964
    p = count / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / (1 + z * z / total)
    )
    return f"{100 * (centre - half):.1f}% to {100 * (centre + half):.1f}%"


def level_table(outcomes: Outcomes, table: str) -> list[str]:
    """Format a table of rates with their intervals, one row per kind of sequence."""
    lines = [
        "| Sequence | Sequences | A boundary is reported | 95% interval |",
        "|---|---:|---:|---:|",
    ]
    settings = [0] if table == "independent" else range(len(SETTINGS))
    reported = total = 0
    for setting in settings:
        for name, block_size in TABLES[table]:
            outcome = outcome_of(outcomes, table, name, block_size, setting)
            reported += outcome.any_boundary
            total += outcome.replicates
            label = name
            if block_size != BLOCK_SIZE:
                label += f", block size {block_size}"
            if table != "independent":
                label += f" (setting {SETTINGS[setting].letter})"
            lines.append(
                f"| {label} | {outcome.replicates} "
                f"| {percent(outcome.any_boundary, outcome.replicates, 1)} "
                f"| {interval(outcome.any_boundary, outcome.replicates)} |"
            )
    lines.append(
        f"| All of the above | {total} | {percent(reported, total, 1)} "
        f"| {interval(reported, total)} |"
    )
    return lines


def rows_of(table: str) -> list[tuple[str, str, int]]:
    """Return the rows of a table as (table, scenario, block size)."""
    return [(table, name, block_size) for name, block_size in TABLES[table]]


def by_setting(
    outcomes: Outcomes, entries: list[tuple[str, str, int]], count: str
) -> list[str]:
    """Format a table with one row per scenario and one column per setting."""
    letters = " | ".join(setting.letter for setting in SETTINGS)
    lines = [f"| Sequence | {letters} |", "|---|" + "---:|" * len(SETTINGS)]
    for table, name, block_size in entries:
        cells = []
        for setting in range(len(SETTINGS)):
            outcome = outcome_of(outcomes, table, name, block_size, setting)
            cells.append(percent(getattr(outcome, count), outcome.replicates))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def by_block_size(
    outcomes: Outcomes, count: str, names: list[str], sizes: tuple[int, int]
) -> list[str]:
    """Format rates for the same sequences examined with two block sizes."""
    letters = [setting.letter for setting in SETTINGS]
    heads = [f"{letter}, block size {size}" for size in sizes for letter in letters]
    lines = [
        "| Sequence | " + " | ".join(heads) + " |",
        "|---|" + "---:|" * len(heads),
    ]
    for name in names:
        cells = []
        for size in sizes:
            # A row examined with the default block size may be in any table.
            table = next(
                table
                for table, entries in TABLES.items()
                if (name, size) in entries and table not in LEVEL_TABLES
            )
            for setting in range(len(SETTINGS)):
                outcome = outcome_of(outcomes, table, name, size, setting)
                cells.append(percent(getattr(outcome, count), outcome.replicates))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def by_seeding(outcomes: Outcomes) -> list[str]:
    """Format the rates obtained with one seed for all the sequences."""
    lines = [
        "| Reorderings | Seed | Sequences | A boundary is reported | 95% interval |",
        "|---:|---|---:|---:|---:|",
    ]
    for n_permutations, seed in SEEDING:
        key = (ONE_SEED, ONE_SEED_SCENARIO, BLOCK_SIZE, 0, n_permutations, seed)
        outcome = outcomes[key]
        label = "One for each sequence" if seed is None else f"{seed} for all"
        lines.append(
            f"| {n_permutations} | {label} | {outcome.replicates} "
            f"| {percent(outcome.any_boundary, outcome.replicates, 1)} "
            f"| {interval(outcome.any_boundary, outcome.replicates)} |"
        )
    return lines


def related_similarities() -> list[str]:
    """Format the similarities expected when consecutive units are related."""
    heads = " | ".join(f"Consecutive units ({value})" for value in RELATED)
    lines = [f"| Setting | {heads} | Other units |", "|---|" + "---:|" * 3]
    for setting in SETTINGS:
        cells = " | ".join(f"{setting.consecutive(value):.2f}" for value in RELATED)
        lines.append(f"| {setting.letter} | {cells} | {setting.within:.2f} |")
    return lines


def report(outcomes: Outcomes, replicates: int) -> list[str]:
    """Return the report as lines of Markdown."""
    margin = 100 * 1.96 * math.sqrt(0.25 / replicates)
    without_change = [
        "12 units, one unrelated unit in the middle",
        "12 units, two adjacent unrelated units in the middle",
        "Drift through the whole sequence, 12 units",
    ]
    with_change = [
        "Two topics, 4 + 4 units",
        "Two topics, 5 + 5 units",
        "Two topics, 6 + 6 units",
        "Two topics, 4 + 8 units",
        "Excursion and return, 6 + 4 + 6 units",
    ]
    after_a_passage = [
        ("planted", "Two topics, 6 + 6 units", BLOCK_SIZE),
        (
            "short, block size 3",
            "Two topics, 6 + 6 units, after a passage of 3 units",
            BLOCK_SIZE,
        ),
    ]
    scale = [
        "Two topics, 80 + 80 units",
        "Transition of 24 units",
        "Transition of 48 units",
        "One topic, then a drift over the second half, 160 units",
    ]
    lines = [
        "# Test for topic boundaries: results on simulated sequences",
        "",
        "Produced by `studies/boundary_test_simulation.py`. The sequences are "
        "simulated vectors, not text: see the documentation of the script.",
        "",
        f"Parameters: alpha {ALPHA}, {N_PERMUTATIONS} permutations, "
        f"{DIMENSIONS} dimensions, block size {BLOCK_SIZE}, and a seed of its "
        "own for each sequence, unless stated otherwise.",
        "",
        f"Software: discoursekit {discoursekit.__version__}, NumPy "
        f"{np.__version__}, Python {platform.python_version()}.",
        "",
        "## Settings",
        "",
        "Mean cosine similarity between two units, within a topic and between "
        "two topics.",
        "",
        "| Setting | Within a topic | Between topics |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {setting.letter}. {setting.label} | {setting.within:.2f} "
        f"| {setting.between:.2f} |"
        for setting in SETTINGS
    ]
    lines += [
        "",
        "## Level",
        "",
        "In the sequences of this section every order of the units is equally "
        f"likely. The rate of reported boundaries should not exceed "
        f"{100 * ALPHA:.0f}%. The intervals are Wilson score intervals.",
        "",
        "Directions drawn independently and uniformly.",
        "",
        *level_table(outcomes, "independent"),
        "",
        "Sequences of 24 units that have a structure but whose order is random.",
        "",
        *level_table(outcomes, "exchangeable"),
        "",
        "## One seed for all the sequences",
        "",
        "In the tables above each sequence is tested with a seed of its own. "
        "With one seed for all, sequences of the same length are compared "
        "with the same reorderings. The rate of reported boundaries over the "
        f"sequences is then {100 * ALPHA:.0f}% on average over the seeds, and "
        "can be above or below it for a given seed.",
        "",
        f"Sequences of {ONE_SEED_SCENARIO}:",
        "",
        *by_seeding(outcomes),
        "",
        "## Consecutive units related to each other",
        "",
        "One topic, without any change, in which each unit shares part of its "
        "scatter with the next: two consecutive units are more alike than two "
        "other units, as in coherent text. The orders of the units are no "
        "longer equally likely, and the level of the test does not apply. The "
        "figure in brackets is the correlation between the scatters of two "
        "consecutive units; units further apart have independent scatters.",
        "",
        "Mean cosine similarity between two consecutive units, and between two "
        "other units:",
        "",
        *related_similarities(),
        "",
        "Percentage of sequences in which a boundary is reported. In this "
        "section and the following ones, each percentage rests on "
        f"{replicates} sequences and is known to within about {margin:.0f} "
        "points (95% interval, at its widest).",
        "",
        *by_setting(outcomes, rows_of("related neighbours"), "any_boundary"),
        "",
        "## Unrelated units",
        "",
        "One topic interrupted by unrelated units. Two adjacent unrelated units "
        "are on one other subject; scattered ones are each on a subject of "
        "their own.",
        "",
        "Percentage of sequences in which a boundary is reported:",
        "",
        *by_setting(outcomes, rows_of("unrelated units"), "any_boundary"),
        "",
        "## Gradual change",
        "",
        "The subject moves by equal steps from one topic to an unrelated "
        "direction, through the whole sequence or through its second half.",
        "",
        "Percentage of sequences in which a boundary is reported:",
        "",
        *by_setting(outcomes, rows_of("drift"), "any_boundary"),
        "",
        "Two topics of 24 units each, joined by a transition during which the "
        "subject moves by equal steps from the first topic to the second. The "
        "first row has no transition.",
        "",
        "Percentage of sequences in which a boundary is reported:",
        "",
        *by_setting(outcomes, rows_of("transition"), "any_boundary"),
        "",
        "## Abrupt changes of topic",
        "",
        "An excursion and return is a passage on a second topic between two "
        "segments on the same first topic.",
        "",
        "Percentage of sequences in which the number of boundaries is right "
        "and each lies within one unit of its true position:",
        "",
        *by_setting(outcomes, rows_of("planted"), "all_correct"),
        "",
        "Percentage of sequences in which at least one boundary is reported:",
        "",
        *by_setting(outcomes, rows_of("planted"), "any_boundary"),
        "",
        "Percentage of sequences in which more boundaries are reported than "
        "were planted:",
        "",
        *by_setting(outcomes, rows_of("planted"), "too_many"),
        "",
        "## Short sequences and block size",
        "",
        "Sequences of 8 to 16 units, examined with block size 3 (the default) "
        "and with block size 2. The same sequences are used for both.",
        "",
        "Percentage of sequences in which a boundary is reported (none is planted):",
        "",
        *by_block_size(outcomes, "any_boundary", without_change, (3, 2)),
        "",
        "Percentage of sequences in which the number of boundaries is right "
        "and each lies within one unit of its true position:",
        "",
        *by_block_size(outcomes, "all_correct", with_change, (3, 2)),
        "",
        "Two topics of 6 units, alone and after a passage of 3 units on a "
        "third topic, which is too short to be delimited with block size 3. "
        "Percentage of sequences in which the boundary between the two topics "
        "is the only one reported, within one unit of its true position:",
        "",
        *by_setting(outcomes, after_a_passage, "all_correct"),
        "",
        "## Longer sequences and block size",
        "",
        "Sequences of 72 to 160 units, examined with block size 3 (the "
        "default) and with block size 6. The same sequences are used for both.",
        "",
        "Percentage of sequences in which a boundary is reported:",
        "",
        *by_block_size(outcomes, "any_boundary", scale, (3, 6)),
        "",
        "Two topics, 80 + 80 units: percentage of sequences in which one "
        "boundary is reported, within one unit of its true position:",
        "",
        *by_block_size(outcomes, "all_correct", scale[:1], (3, 6)),
    ]
    return lines


def main() -> None:
    """Run every scenario and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--level-replicates", type=int, default=5000)
    parser.add_argument("--seed-replicates", type=int, default=20000)
    parser.add_argument("--jobs", type=int, default=1)
    arguments = parser.parse_args()

    work = rows(
        arguments.replicates, arguments.level_replicates, arguments.seed_replicates
    )
    outcomes: Outcomes = {}
    with ProcessPoolExecutor(max_workers=arguments.jobs) as pool:
        for number, (row, outcome) in enumerate(
            zip(work, pool.map(run, work), strict=True), start=1
        ):
            outcomes[row.key] = outcome
            done = f"{number}/{len(work)}"
            print(done, row.table, row.name, file=sys.stderr, flush=True)
    print("\n".join(report(outcomes, arguments.replicates)))


if __name__ == "__main__":
    main()
