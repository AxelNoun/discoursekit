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
  McFall, 2010), returned with the options of the tokenisation.
- `coherence`, `adjacent_similarities`: local, global and reference
  coherence computed from embeddings, with the level of local coherence
  expected under random order and its standard deviation over orders.
- `segment_topics`: permutation test for topic boundaries. Its level holds
  when every order of the units is equally likely; in coherent text, whose
  consecutive units are related, it can report boundaries where the topic
  does not change, and this is documented.
- `coherence` reports local and global coherence within the segments
  delimited by the boundaries passed to it. It does not estimate
  boundaries: by default the text is one segment.
- Embeddings with complex values are refused rather than truncated.
- `SentenceTransformerEmbedder`: optional adapter for the
  `sentence-transformers` library, with a warning for truncated units. A
  unit that occurs several times is embedded once, so that equal units have
  equal embeddings.
- Registry of measures (`docs/metrics.md`), validation status
  (`docs/validation.md`), guidance on short texts (`docs/short-texts.md`),
  background (`docs/background.md`) and scope statement
  (`docs/scope-and-ethics.md`).
- Simulation studies of the test for topic boundaries
  (`studies/boundary_test_simulation.py`,
  `studies/block_similarity_comparison.py`), and a count of the unequal
  embeddings that a model gives to a repeated unit
  (`studies/repeated_unit_embeddings.py`).
- A study of the test for topic boundaries on real texts that keep to one
  subject, Reuters-21578 dispatches and film reviews, with the circular block
  permutation of Chen as a comparison (`studies/real_text_boundaries.py`).
