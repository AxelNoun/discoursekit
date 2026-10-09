# discoursekit

[![tests](https://github.com/AxelNoun/discoursekit/actions/workflows/tests.yml/badge.svg)](https://github.com/AxelNoun/discoursekit/actions/workflows/tests.yml)

`discoursekit` is a Python library of linguistic measures for discourse
research. It provides a small number of measures. Each one is defined by
reference to published work, tied to a stated hypothesis, documented with its
known confounds, and tested against its definition.

This version contains:

* word tokenisation built on the Unicode word boundary rules (UAX #29),
  with a segmentation that passes the official conformance test;
* lexical diversity, as the moving-average type-token ratio (Covington &
  McFall, 2010);
* semantic coherence computed from embeddings of the units of a text: local
  coherence, global coherence, and coherence with respect to a reference such
  as the question asked;
* a permutation test for changes of topic, so that a change of topic need not
  be counted as a loss of coherence.

The library describes texts. It is not a clinical tool: it does not detect,
predict or diagnose anything, and none of its measures has been validated
against clinical assessments. See [Scope and ethics](docs/scope-and-ethics.md).

Status: alpha. The interface may change before version 0.1.0.

## Installation

```bash
pip install git+https://github.com/AxelNoun/discoursekit
```

Python 3.10 or later and NumPy are required. To compute sentence embeddings
with the `sentence-transformers` library, install the optional dependency:

```bash
pip install "discoursekit[embeddings] @ git+https://github.com/AxelNoun/discoursekit"
```

## Usage

### Tokens and lexical diversity

```python
>>> from discoursekit import lexical_diversity, tokenize

>>> tokenize("Well, I don't know... It's a well-known problem!")
['well', 'i', "don't", 'know', "it's", 'a', 'well', 'known', 'problem']

>>> result = lexical_diversity(
...     "the cat saw the dog and the dog saw the cat", window=5
... )
>>> result.mattr
0.8
>>> result.n_tokens, result.n_types, result.n_windows
(11, 5, 7)

```

A text with fewer tokens than the window has no moving-average type-token
ratio at that window size, and the value is `nan`. The library does not
substitute the plain type-token ratio, which would make the shortest samples
appear the most diverse.

### Semantic coherence

The coherence functions take one embedding vector per unit of text, in order.
The unit (sentence, utterance, or other) and the embedding model are chosen by
the study. Any source of embeddings can be used.

```python
>>> import numpy as np
>>> from discoursekit import coherence

>>> garden, market = np.array([1.0, 0.1]), np.array([0.1, 1.0])
>>> vectors = [garden, garden, garden, market, market, market]

>>> result = coherence(vectors, boundaries=[3])
>>> round(result.local_coherence, 2)
0.84
>>> round(result.local_coherence_within_segments, 2)
1.0

```

Here the change of topic is known and given as the index of the first unit of
the new segment. When `boundaries` is omitted, changes of topic are estimated
by a permutation test, which can also be called on its own:

```python
>>> from discoursekit import segment_topics

>>> state = np.random.RandomState(0)
>>> scatter = 0.2 * state.standard_normal((16, 2))
>>> units = np.array([garden] * 8 + [market] * 8) + scatter

>>> found = segment_topics(units)
>>> found.boundaries
(8,)
>>> found.p_values
(0.018,)

```

One boundary is reported, before the ninth unit. A text of fewer than eight
units is too short to be tested with the default settings; the field `tested`
says whether the test was carried out.

With a sentence-embedding model:

```python
from discoursekit import SentenceTransformerEmbedder, coherence

embed = SentenceTransformerEmbedder(
    "sentence-transformers/all-MiniLM-L6-v2",
    revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",  # commit of the model
)
units = [...]  # the sentences or utterances of one text, in order
result = coherence(embed(units))
```

The result contains the following fields.

| Field | Meaning |
|---|---|
| `local_coherence` | Mean cosine similarity between consecutive units |
| `local_coherence_shuffled` | Value expected for the same units in random order |
| `global_coherence` | Mean similarity between each unit and the centroid of the others |
| `reference_coherence` | Mean similarity between each unit and a reference vector, if one is given |
| `local_coherence_within_segments`, `global_coherence_within_segments` | Local and global coherence computed inside the segments delimited by the topic boundaries |
| `boundaries`, `segmentation` | The boundaries used, and the details of the test if they were estimated |
| `n_units` | Number of units, to be reported and used as a covariate |

## Documentation

| Document | Content |
|---|---|
| [docs/metrics.md](docs/metrics.md) | Registry of measures: definition, sources, hypotheses, known confounds and what to report, for each quantity |
| [docs/validation.md](docs/validation.md) | What has been checked, how, with what result, and what has not been checked |
| [docs/background.md](docs/background.md) | Related work and position of the library |
| [docs/scope-and-ethics.md](docs/scope-and-ethics.md) | Intended use and limits |

The functions are documented in their docstrings.

## Principles

A measure is added only with a stated hypothesis; there is no option to
extract every available feature. The same input and parameters give the same
output, and the parameters that affect a value are returned with it. When a
measure is undefined for a text the result is `nan`, never a substitute. Each
measure is tested against values worked out by hand and against a direct
implementation of its definition, and the word segmentation against the
conformance file of the standard. Heavy dependencies are optional.

## Roadmap

1. Further measures for English: syntactic complexity and markers of
   epistemic stance (certainty, hedges, evidentials).
2. A benchmark of topic change against loss of coherence. Ordinary texts are
   perturbed in controlled ways (sentences shuffled, unrelated sentences
   inserted, genuine changes of topic with transitions) to establish which
   measures tell these cases apart. Telling an original text from a shuffled
   one is the most common test of coherence models, and its limits have been
   discussed (Laban et al., 2021). Such a benchmark tests the robustness of
   the measures. It is not a clinical validation, and a perturbed text is not
   a model of anyone's speech.
3. A robustness study on public, non-clinical corpora: effect of text length,
   genre and language, and stability over time within an author.

## Development

```bash
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

See [CITATION.cff](CITATION.cff). When reporting a value computed with this
library, also cite the source of the measure, given in
[docs/metrics.md](docs/metrics.md), and report the parameters returned with
the result.

## Licence

The code is distributed under the [MIT licence](LICENSE). The file
`src/discoursekit/_wordbreak_data.py` and the test file
`tests/data/WordBreakTest-18.0.0.txt` are derived from or part of the Unicode
Character Database, copyright Unicode, Inc., and are distributed under the
[Unicode License v3](LICENSES/Unicode-3.0.txt).

## References

Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: The
moving-average type-token ratio (MATTR). *Journal of Quantitative
Linguistics*, 17(2), 94-100. https://doi.org/10.1080/09296171003643098

Laban, P., Dai, L., Bandarkar, L., & Hearst, M. A. (2021). Can transformer
models measure coherence in text: Re-thinking the shuffle test. In
*Proceedings of the 59th Annual Meeting of the Association for Computational
Linguistics and the 11th International Joint Conference on Natural Language
Processing (Volume 2: Short Papers)* (pp. 1058-1064).
https://doi.org/10.18653/v1/2021.acl-short.134

Unicode Consortium. (2026). *Unicode Standard Annex #29: Unicode text
segmentation* (Revision 49, Unicode 18.0.0).
https://www.unicode.org/reports/tr29/
