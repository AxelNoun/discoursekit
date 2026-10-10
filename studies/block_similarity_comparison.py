"""What each of the two block similarities contributes to the test for boundaries.

The test for topic boundaries measures the depth of a valley on two
similarities between blocks of units, the cosine similarity between the sums
of the two blocks and the mean similarity between their units, and adds the
two depths once each is on a common scale (see ``docs/metrics.md``, section
6). This script applies three versions of the test to the same simulated
sequences: the test of the library, and the test with the depth measured on
one similarity only.

The two reduced versions are obtained by setting the depths on the other
similarity to zero before the depths are combined. A similarity whose depths
are all zero counts for nothing, in the statistic and in the choice of the
candidate boundary. The sequences are those of
``boundary_test_simulation.py``, with the same seeds: the rates of the test
of the library are those of that simulation.

Usage, from the root of the repository::

    python studies/block_similarity_comparison.py --jobs 2 > results.md

The run takes about an hour and a half on one core.
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
from boundary_test_simulation import (
    ALPHA,
    BLOCK_SIZE,
    DIMENSIONS,
    N_PERMUTATIONS,
    SCENARIOS,
    SETTINGS,
    Simulator,
    interval,
    percent,
)
from numpy.typing import NDArray

import discoursekit
from discoursekit import segment_topics, semantic

Scores = Callable[
    [NDArray[np.float64], NDArray[np.intp]],
    tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.float64], float] | None,
]
COMBINED: Scores = semantic._scores

# Version of the test: the similarity whose depth is kept, or None for both.
VERSIONS: dict[str, int | None] = {"Both": None, "Sums": 0, "Units": 1}

# Sequence in which every order of the units is equally likely: the rate of
# reported boundaries is compared with the level, on more sequences.
LEVEL = "One topic"

# Sequences without an abrupt change, in which no boundary should be reported.
WITHOUT_CHANGE = [
    "24 units, one unrelated unit in the middle",
    "Drift through the whole sequence, 60 units",
    "One topic, then a drift over the second half, 60 units",
    "One topic, 24 units, consecutive units related (0.5)",
]

# Sequences with planted changes of topic.
WITH_CHANGE = [
    "Two topics, 6 + 6 units",
    "Two topics, 6 + 18 units",
    "Three topics, 8 + 8 + 8 units",
    "Excursion and return, 10 + 8 + 10 units",
]


def keeping(measure: int | None) -> Scores:
    """Return the combination of depths that keeps one similarity, or both."""

    def scores(
        depths: NDArray[np.float64], mirror: NDArray[np.intp]
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.float64], float] | None
    ):
        if measure is not None:
            depths = depths.copy()
            depths[1 - measure] = 0.0
        return COMBINED(depths, mirror)

    return scores


@dataclass(frozen=True)
class Row:
    """One scenario under one setting, examined with one version of the test."""

    name: str
    setting: int
    version: str
    replicates: int


def run(row: Row) -> tuple[int, int]:
    """Apply one version of the test to the sequences of one row.

    Returns the number of sequences in which a boundary is reported, and the
    number in which the boundaries are the planted ones, each within one unit
    of its true position.
    """
    # The private function that combines the depths is replaced on purpose.
    semantic._scores = keeping(VERSIONS[row.version])  # type: ignore[assignment]
    scenario = SCENARIOS[row.name]
    stream = list(SCENARIOS).index(row.name) * len(SETTINGS) + row.setting
    any_boundary = all_correct = 0
    for replicate in range(row.replicates):
        state = np.random.RandomState([stream, replicate])
        sequence, truth = scenario(Simulator(SETTINGS[row.setting], state))
        found = segment_topics(
            sequence,
            block_size=BLOCK_SIZE,
            alpha=ALPHA,
            n_permutations=N_PERMUTATIONS,
            seed=replicate,
        ).boundaries
        any_boundary += bool(found)
        all_correct += len(found) == len(truth) and all(
            abs(a - b) <= 1 for a, b in zip(found, truth, strict=True)
        )
    return any_boundary, all_correct


def level_table(outcomes: dict[Row, tuple[int, int]], replicates: int) -> list[str]:
    """Format the rates of reported boundaries with their intervals."""
    letters = " | ".join(setting.letter for setting in SETTINGS)
    lines = [f"| Version | {letters} |", "|---|" + "---:|" * len(SETTINGS)]
    for version in VERSIONS:
        cells = []
        for setting in range(len(SETTINGS)):
            reported = outcomes[Row(LEVEL, setting, version, replicates)][0]
            cells.append(
                f"{percent(reported, replicates, 1)} ({interval(reported, replicates)})"
            )
        lines.append(f"| {version} | " + " | ".join(cells) + " |")
    return lines


def table(
    outcomes: dict[Row, tuple[int, int]], names: list[str], count: int, replicates: int
) -> list[str]:
    """Format one row per scenario and one column per version and setting."""
    heads = [
        f"{version}, {setting.letter}" for version in VERSIONS for setting in SETTINGS
    ]
    lines = ["| Sequence | " + " | ".join(heads) + " |", "|---|" + "---:|" * len(heads)]
    for name in names:
        cells = []
        for version in VERSIONS:
            for setting in range(len(SETTINGS)):
                reported = outcomes[Row(name, setting, version, replicates)][count]
                cells.append(f"{100 * reported / replicates:.0f}%")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def report(
    outcomes: dict[Row, tuple[int, int]], replicates: int, level_replicates: int
) -> list[str]:
    """Return the report as lines of Markdown."""
    margin = 100 * 1.96 * math.sqrt(0.25 / replicates)
    return [
        "# Test for topic boundaries: the two block similarities compared",
        "",
        "Produced by `studies/block_similarity_comparison.py`. The sequences "
        "are simulated vectors, not text, and are those of "
        "`studies/boundary_test_simulation.py`.",
        "",
        f"Parameters: alpha {ALPHA}, {N_PERMUTATIONS} permutations, "
        f"{DIMENSIONS} dimensions, block size {BLOCK_SIZE}. The settings A, B "
        "and C are those of the main simulation.",
        "",
        f"Software: discoursekit {discoursekit.__version__}, NumPy "
        f"{np.__version__}, Python {platform.python_version()}.",
        "",
        "Three versions of the test. Both: the depth of the valley is measured "
        "on the two block similarities, as in the library. Sums: on the cosine "
        "similarity between the sums of the two blocks only. Units: on the "
        "mean similarity between the units of the two blocks only.",
        "",
        "## Level",
        "",
        "One topic, 24 units: every order of the units is equally likely, and "
        "the rate of reported boundaries should not exceed "
        f"{100 * ALPHA:.0f}% in any version. Percentage of {level_replicates} "
        "sequences in which a boundary is reported, with its 95% interval.",
        "",
        *level_table(outcomes, level_replicates),
        "",
        "## Sequences without an abrupt change of topic",
        "",
        "In this section and the next, each percentage rests on "
        f"{replicates} sequences and is known to within about {margin:.0f} "
        "points (95% interval, at its widest).",
        "",
        "Percentage of sequences in which a boundary is reported. In the last "
        "row the units of one topic are related to their neighbours, without "
        "any change of topic; the level of the test does not apply to it.",
        "",
        *table(outcomes, WITHOUT_CHANGE, 0, replicates),
        "",
        "## Sequences with planted changes of topic",
        "",
        "Percentage of sequences in which the number of boundaries is right "
        "and each lies within one unit of its true position.",
        "",
        *table(outcomes, WITH_CHANGE, 1, replicates),
    ]


def main() -> None:
    """Run every row and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--level-replicates", type=int, default=5000)
    parser.add_argument("--jobs", type=int, default=1)
    arguments = parser.parse_args()

    counts = dict.fromkeys((*WITHOUT_CHANGE, *WITH_CHANGE), arguments.replicates)
    counts[LEVEL] = arguments.level_replicates
    work = [
        Row(name, setting, version, replicates)
        for name, replicates in counts.items()
        for setting in range(len(SETTINGS))
        for version in VERSIONS
    ]
    outcomes = {}
    with ProcessPoolExecutor(max_workers=arguments.jobs) as pool:
        for number, (row, outcome) in enumerate(
            zip(work, pool.map(run, work), strict=True), start=1
        ):
            outcomes[row] = outcome
            done = f"{number}/{len(work)}"
            print(done, row.name, row.version, file=sys.stderr, flush=True)
    lines = report(outcomes, arguments.replicates, arguments.level_replicates)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
