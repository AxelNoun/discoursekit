from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def table_rows(path: Path, start: str = "", stop: str = "") -> list[str]:
    """The rows of the tables of a Markdown file, or of one part of it."""
    text = path.read_text(encoding="utf-8")
    first = text.index(start) if start else 0
    last = text.index(stop) if stop else len(text)
    return [line for line in text[first:last].splitlines() if line.startswith("|")]


def test_the_simulation_tables_of_the_validation_page_are_the_stored_outputs() -> None:
    # The tables that docs/validation.md gives for the two simulation studies
    # are copied from the outputs stored in studies/results/. A table changed
    # by hand, or an output replaced without the page being updated, fails
    # here.
    copied = table_rows(
        ROOT / "docs" / "validation.md",
        start="### Simulation",
        stop="### Reading of the results",
    )
    results = ROOT / "studies" / "results"
    stored = {
        *table_rows(results / "boundary_test_simulation.md"),
        *table_rows(results / "block_similarity_comparison.md"),
    }
    assert copied
    assert [row for row in copied if row not in stored] == []
