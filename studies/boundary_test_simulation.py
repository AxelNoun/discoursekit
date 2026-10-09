"""Operating characteristics of the test for topic boundaries on simulated data.

This script measures how often :func:`discoursekit.segment_topics` reports a
boundary in sequences of simulated unit vectors. It answers three questions
about the test itself.

1. Level. When every order of the units is equally likely, how often is a
   boundary reported? The construction of the test puts this rate at
   ``alpha`` or below.
2. Content that varies without a change of topic. How often is a boundary
   reported when unrelated units are scattered through the sequence, or when
   the content drifts gradually?
3. Power. When changes of topic are planted, how often are they found, and at
   the right place?

The sequences are not text. A topic is a direction in a space of 48
dimensions; a unit on that topic is the direction plus isotropic scatter,
scaled to unit length. The results describe the behaviour of the statistical
procedure. They say nothing about its behaviour on embeddings of natural
language, which the planned benchmark is to establish.

Usage, from the root of the repository::

    python studies/boundary_test_simulation.py --jobs 2 > results.md

The output does not depend on the number of jobs: every sequence has its own
seed. The run takes about forty minutes on one core.
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
LENGTH = 24
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

    def units(self, centre: Array, n: int) -> Array:
        """Draw ``n`` units scattered around a direction."""
        scatter = self.state.standard_normal((n, DIMENSIONS)) / math.sqrt(DIMENSIONS)
        return normalise(centre + self.setting.scatter * scatter)

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


def one_topic(sim: Simulator) -> Sequence:
    """Return units that are all on the same topic."""
    return sim.units(sim.home, LENGTH), ()


def two_topics_in_random_order(sim: Simulator) -> Sequence:
    """Return the units of two topics, half each, thoroughly mixed."""
    sequence = sim.blocks(LENGTH // 2, LENGTH // 2)
    return np.asarray(sequence[sim.state.permutation(LENGTH)]), ()


def intruder_anywhere(sim: Simulator) -> Sequence:
    """Return one topic with one unrelated unit at a random position."""
    sequence = sim.units(sim.home, LENGTH)
    # The type is given because the default has 32 bits on Windows only.
    position = int(sim.state.randint(0, LENGTH, dtype=np.int64))
    sequence[position] = sim.units(sim.direction(), 1)[0]
    return sequence, ()


# Sequences whose content varies without a change of topic


def intruder_in_the_middle(sim: Simulator) -> Sequence:
    """Return one topic with one unrelated unit in the middle of the sequence."""
    sequence = sim.units(sim.home, LENGTH)
    sequence[LENGTH // 2] = sim.units(sim.direction(), 1)[0]
    return sequence, ()


def three_scattered_intruders(sim: Simulator) -> Sequence:
    """Return one topic with three unrelated units far from one another."""
    sequence = sim.units(sim.home, LENGTH)
    for position in (5, 12, 19):
        sequence[position] = sim.units(sim.direction(), 1)[0]
    return sequence, ()


def gradual_drift(sim: Simulator) -> Sequence:
    """Return units whose subject moves steadily to an unrelated direction."""
    weights = np.linspace(0.0, 1.0, LENGTH)[:, np.newaxis]
    centres = normalise((1 - weights) * sim.home + weights * sim.direction())
    scatter = sim.state.standard_normal((LENGTH, DIMENSIONS)) / math.sqrt(DIMENSIONS)
    return normalise(centres + sim.setting.scatter * scatter), ()


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


# The rows of each table: name, scenario and block size of the test.
TABLES: dict[str, dict[str, tuple[Scenario, int]]] = {
    "independent": {
        "8 units, the fewest that can be tested": (independent_directions(8), 3),
        "16 units": (independent_directions(16), 3),
        "24 units": (independent_directions(24), 3),
        "40 units": (independent_directions(40), 3),
        "12 units, block size 1": (independent_directions(12), 1),
    },
    "exchangeable": {
        "One topic": (one_topic, BLOCK_SIZE),
        "Two topics, 12 + 12 units, in random order": (
            two_topics_in_random_order,
            BLOCK_SIZE,
        ),
        "One topic, one unrelated unit at a random position": (
            intruder_anywhere,
            BLOCK_SIZE,
        ),
    },
    "without change": {
        "One topic, one unrelated unit in the middle": (
            intruder_in_the_middle,
            BLOCK_SIZE,
        ),
        "One topic, three scattered unrelated units": (
            three_scattered_intruders,
            BLOCK_SIZE,
        ),
        "Gradual drift to an unrelated subject": (gradual_drift, BLOCK_SIZE),
    },
    "planted": {
        "Two topics, 6 + 6 units": (planted(6, 6), BLOCK_SIZE),
        "Two topics, 8 + 8 units": (planted(8, 8), BLOCK_SIZE),
        "Two topics, 12 + 12 units": (planted(12, 12), BLOCK_SIZE),
        "Two topics, 20 + 20 units": (planted(20, 20), BLOCK_SIZE),
        "Two topics, 6 + 18 units": (planted(6, 18), BLOCK_SIZE),
        "Three topics, 8 + 8 + 8 units": (planted(8, 8, 8), BLOCK_SIZE),
        "Three topics, 12 + 12 + 12 units": (planted(12, 12, 12), BLOCK_SIZE),
        "Excursion and return, 10 + 4 + 10 units": (excursion(10, 4, 10), BLOCK_SIZE),
        "Excursion and return, 10 + 8 + 10 units": (excursion(10, 8, 10), BLOCK_SIZE),
    },
}


@dataclass(frozen=True)
class Row:
    """One scenario under one setting.

    ``number`` is the position of the row in the list of all rows. With the
    index of the sequence within the row, it seeds the random state from
    which the sequence is drawn, so that no two sequences share a seed.
    """

    number: int
    table: str
    name: str
    setting: int
    replicates: int


@dataclass(frozen=True)
class Outcome:
    """Counts over the sequences of one row."""

    replicates: int
    any_boundary: int
    n_boundaries: int
    all_correct: int
    too_many: int


def rows(replicates: int, level_replicates: int) -> list[Row]:
    """Return every row, in a fixed order."""
    settings = range(len(SETTINGS))
    listed = [
        ("independent", name, 0, level_replicates) for name in TABLES["independent"]
    ]
    listed += [
        ("exchangeable", name, setting, level_replicates)
        for setting in settings
        for name in TABLES["exchangeable"]
    ]
    listed += [
        (table, name, setting, replicates)
        for setting in settings
        for table in ("without change", "planted")
        for name in TABLES[table]
    ]
    return [Row(number, *row) for number, row in enumerate(listed)]


def run(row: Row) -> Outcome:
    """Apply the test to the sequences of one row."""
    scenario, block_size = TABLES[row.table][row.name]
    any_boundary = n_boundaries = all_correct = too_many = 0
    for replicate in range(row.replicates):
        state = np.random.RandomState([row.number, replicate])
        sequence, truth = scenario(Simulator(SETTINGS[row.setting], state))
        found = segment_topics(
            sequence,
            block_size=block_size,
            alpha=ALPHA,
            n_permutations=N_PERMUTATIONS,
            seed=replicate,
        ).boundaries
        any_boundary += bool(found)
        n_boundaries += len(found)
        too_many += len(found) > len(truth)
        all_correct += len(found) == len(truth) and all(
            abs(a - b) <= 1 for a, b in zip(found, truth, strict=True)
        )
    return Outcome(row.replicates, any_boundary, n_boundaries, all_correct, too_many)


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


def level_line(label: str, reported: int, total: int) -> str:
    """Format one row of a table of rates with their intervals."""
    return (
        f"| {label} | {total} | {percent(reported, total, 1)} "
        f"| {interval(reported, total)} |"
    )


def by_setting(
    outcomes: dict[tuple[str, str, int], Outcome], table: str, count: str
) -> list[str]:
    """Format a table with one row per scenario and one column per setting."""
    letters = " | ".join(setting.letter for setting in SETTINGS)
    lines = [f"| Sequence | {letters} |", "|---|" + "---:|" * len(SETTINGS)]
    for name in TABLES[table]:
        cells = []
        for index in range(len(SETTINGS)):
            outcome = outcomes[table, name, index]
            cells.append(percent(getattr(outcome, count), outcome.replicates))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def report(outcomes: dict[tuple[str, str, int], Outcome], replicates: int) -> list[str]:
    """Return the report as lines of Markdown."""
    margin = 100 * 1.96 * math.sqrt(0.25 / replicates)
    lines = [
        "# Test for topic boundaries: results on simulated sequences",
        "",
        "Produced by `studies/boundary_test_simulation.py`. The sequences are "
        "simulated vectors, not text: see the documentation of the script.",
        "",
        f"Parameters: alpha {ALPHA}, {N_PERMUTATIONS} permutations, "
        f"{DIMENSIONS} dimensions, block size {BLOCK_SIZE} unless stated "
        "otherwise.",
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
        f"{100 * ALPHA:.0f}%.",
        "",
        "Directions drawn independently and uniformly.",
        "",
        "| Sequence | Sequences | A boundary is reported | 95% interval |",
        "|---|---:|---:|---:|",
    ]
    reported = total = 0
    for name in TABLES["independent"]:
        outcome = outcomes["independent", name, 0]
        reported += outcome.any_boundary
        total += outcome.replicates
        lines.append(level_line(name, outcome.any_boundary, outcome.replicates))
    lines.append(level_line("All of the above", reported, total))
    lines += [
        "",
        f"Sequences of {LENGTH} units that have a structure but whose order is random.",
        "",
        "| Sequence | Sequences | A boundary is reported | 95% interval |",
        "|---|---:|---:|---:|",
    ]
    reported = total = 0
    for index, setting in enumerate(SETTINGS):
        for name in TABLES["exchangeable"]:
            outcome = outcomes["exchangeable", name, index]
            reported += outcome.any_boundary
            total += outcome.replicates
            label = f"{name} (setting {setting.letter})"
            lines.append(level_line(label, outcome.any_boundary, outcome.replicates))
    lines.append(level_line("All of the above", reported, total))
    lines += [
        "",
        "## Content that varies without a change of topic",
        "",
        f"Sequences of {LENGTH} units. No boundary should be reported. In this "
        f"section and the next, each percentage rests on {replicates} sequences "
        f"and is known to within about {margin:.0f} points (95% interval, at its "
        "widest).",
        "",
        "Percentage of sequences in which a boundary is reported:",
        "",
        *by_setting(outcomes, "without change", "any_boundary"),
        "",
        "## Planted changes of topic",
        "",
        "Percentage of sequences in which the number of boundaries is right "
        "and each lies within one unit of its true position:",
        "",
        *by_setting(outcomes, "planted", "all_correct"),
        "",
        "Percentage of sequences in which at least one boundary is reported:",
        "",
        *by_setting(outcomes, "planted", "any_boundary"),
        "",
        "Percentage of sequences in which more boundaries are reported than "
        "were planted:",
        "",
        *by_setting(outcomes, "planted", "too_many"),
    ]
    return lines


def main() -> None:
    """Run every scenario and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--level-replicates", type=int, default=5000)
    parser.add_argument("--jobs", type=int, default=1)
    arguments = parser.parse_args()

    work = rows(arguments.replicates, arguments.level_replicates)
    outcomes = {}
    with ProcessPoolExecutor(max_workers=arguments.jobs) as pool:
        for row, outcome in zip(work, pool.map(run, work), strict=True):
            outcomes[row.table, row.name, row.setting] = outcome
            done = f"{row.number + 1}/{len(work)}"
            print(done, row.table, row.name, file=sys.stderr, flush=True)
    print("\n".join(report(outcomes, arguments.replicates)))


if __name__ == "__main__":
    main()
