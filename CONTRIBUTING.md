# Contributing

Bug reports, questions and pull requests are welcome through the issue
tracker of the repository.

## Setting up

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Most tests of the embedding adapter need the optional dependency
(`pip install -e ".[dev,embeddings]"`). Without it they are skipped.

## Checks

The continuous integration workflow runs the following commands, and all of
them must pass.

```bash
pytest
ruff check .
ruff format --check .
mypy
```

`pytest` also runs the examples of the docstrings, of the README and of
`docs/short-texts.md`. Three tests download a published model and are not
run by default; run them with `pytest -m network tests/test_embedding.py`.

## Adding or changing a measure

A measure is accepted with all of the following, in this order.

1. An entry in [docs/metrics.md](docs/metrics.md): definition, sources,
   measurement hypotheses, motivating hypothesis, known confounds, and what
   to report. Every reference is checked against the publication.
2. Tests: values computed by hand, agreement with a direct implementation of
   the definition, behaviour on empty and too-short input, invalid
   parameters.
3. The implementation. It is deterministic, returns `nan` when the measure
   is undefined, and returns the parameters that affect the value.
4. An entry in [docs/validation.md](docs/validation.md) stating what was
   checked and what was not.
5. A line in [CHANGELOG.md](CHANGELOG.md).

A change that alters the value returned by a measure for the same input is a
breaking change and is flagged as such in the changelog.

A change to the test for topic boundaries is followed by a new run of
`studies/boundary_test_simulation.py` and of
`studies/block_similarity_comparison.py`. Their outputs replace the files of
`studies/results/`, and the tables of
[docs/validation.md](docs/validation.md) are copied from them: a test fails
when a table of that part of the page is not, row for row, one of the stored
tables. The text that comments on the tables is revised by hand.
`studies/repeated_unit_embeddings.py` concerns the embedding adapter and
needs the optional dependency; its output is stored in the same folder.
`studies/real_text_boundaries.py` needs the optional dependency and NLTK,
and downloads public corpora and a model on its first run; its output is
stored there too, with the name of the embedding when it is not the default.
The workflow `real texts` runs it on GitHub on demand and keeps its output
as an artifact of the run; when the workflow is given the name of a branch,
it also commits the output there, which is of use where the artifacts cannot
be downloaded; the branch holds the output of the latest run only, and is
deleted once its files have been fetched.

## Updating the Unicode data

Word segmentation uses a table generated from the Unicode Character
Database. To move to a new version of Unicode, download
`auxiliary/WordBreakProperty.txt`, `auxiliary/WordBreakTest.txt` and
`emoji/emoji-data.txt` for that version from
`https://www.unicode.org/Public/`, run `tools/generate_wordbreak_data.py`,
replace the conformance file in `tests/data/`, and check whether the rules of
UAX #29 have changed. The conformance test must pass in full.

## Scope

Read [docs/scope-and-ethics.md](docs/scope-and-ethics.md) first.
Contributions that make clinical claims, or that analyse data in order to
identify individuals, are declined.
