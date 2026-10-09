# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/). A change that alters the
value returned by a measure for the same input is a breaking change.

## Unreleased

### Added

- `tokenize`: word tokenisation according to the word boundary rules of
  Unicode Standard Annex #29 (Unicode 18.0.0), with one documented tailoring
  (U+202F is treated as a space), normalisation, case folding and optional
  exclusion of tokens.
- `mattr`, `lexical_diversity`: moving-average type-token ratio (Covington &
  McFall, 2010).
- `coherence`, `adjacent_similarities`: local, global and reference
  coherence computed from embeddings, with the level of local coherence
  expected under random order.
- `segment_topics`: permutation test for topic boundaries, and coherence
  within the resulting segments.
- `SentenceTransformerEmbedder`: optional adapter for the
  `sentence-transformers` library, with a warning for truncated units.
- Registry of measures (`docs/metrics.md`), validation status
  (`docs/validation.md`), background (`docs/background.md`) and scope
  statement (`docs/scope-and-ethics.md`).
- Simulation study of the test for topic boundaries
  (`studies/boundary_test_simulation.py`).
