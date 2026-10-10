from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tables(path: Path, start: str = "", stop: str = "") -> list[tuple[str, ...]]:
    """The tables of a Markdown file, or of one part of it, each as its rows."""
    text = path.read_text(encoding="utf-8")
    first = text.index(start) if start else 0
    last = text.index(stop) if stop else len(text)
    found: list[tuple[str, ...]] = []
    rows: list[str] = []
    for line in [*text[first:last].splitlines(), ""]:
        if line.startswith("|"):
            rows.append(line)
        elif rows:
            found.append(tuple(rows))
            rows = []
    return found


def test_the_simulation_tables_of_the_validation_page_are_the_stored_outputs() -> None:
    # The tables that docs/validation.md gives for the two simulation studies
    # are copied from the outputs stored in studies/results/. A table changed
    # by hand, or an output replaced without the page being updated, fails
    # here: every table of that part of the page must be, row for row, one
    # of the stored tables. The sentences around the tables are not checked.
    copied = tables(
        ROOT / "docs" / "validation.md",
        start="### Simulation",
        stop="### Reading of the results",
    )
    results = ROOT / "studies" / "results"
    stored = {
        *tables(results / "boundary_test_simulation.md"),
        *tables(results / "block_similarity_comparison.md"),
    }
    assert len(copied) >= 10
    assert [table for table in copied if table not in stored] == []
