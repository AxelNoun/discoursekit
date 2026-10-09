# Validation status

This document states what has been checked in this library, how, and with
what result, and what has not been checked. It is updated with each release.

Everything below concerns the behaviour of the measures as computations. No
measure in this library has been validated against any clinical assessment
or any human judgement of discourse quality.

The hypotheses (H-TOK-1 and so on) are stated in [metrics.md](metrics.md).

## 1. Tokenisation

| Claim | Evidence |
|---|---|
| H-TOK-1: the word segmentation conforms to UAX #29, Unicode 18.0.0 | All 1,944 cases of the official conformance file `WordBreakTest-18.0.0.txt` pass (2,700 boundaries and 1,899 non-boundaries). `tests/test_word_boundaries.py` |
| H-TOK-2: tokens are well formed | On 20,000 random strings built from 33 characters that cover every class of the word boundary rules, in both casing modes, every token contains a letter or digit, contains no white space and no format character, and is in Normalization Form C. `tests/test_tokenization.py` |
| H-TOK-3: white space separates tokens | On 10,000 pairs of random strings joined by one of eleven white space characters or sequences, in both casing modes, the tokens of the whole are the tokens of the first string followed by those of the second. `tests/test_tokenization.py` |
| Behaviour on English text is as documented | 35 tests of specific cases: apostrophes, hyphens, full stops, numbers, case, normalisation, white space, exclusion. `tests/test_tokenization.py` |

The character property table is generated from the Unicode Character
Database by `tools/generate_wordbreak_data.py`; the generated module records
the names, dates and SHA-256 digests of the source files.

Not checked: agreement with the transcription conventions of any corpus;
languages other than English.

## 2. Lexical diversity

| Claim | Evidence |
|---|---|
| The implementation computes the moving-average type-token ratio as defined | Values worked out by hand; agreement to 12 decimal places with a direct implementation in exact rational arithmetic, on 5 random sequences at 5 window sizes each. `tests/test_lexical.py` |
| H-LEX-1: stability across text length where the plain ratio is not stable | On 300 pairs of synthetic texts of 300 and 3,000 tokens drawn from the same Zipf-like distribution, the plain type-token ratio falls by about 0.25 on average, while the mean difference in the moving-average ratio is below 0.01 (standard error about 0.0014). `tests/test_lexical.py` |
| A text shorter than the window gives NaN, not the plain ratio | `tests/test_lexical.py` |

Not checked: H-LEX-1 on natural text.

## 3. Local, global and reference coherence

All the evidence of this section is in `tests/test_semantic.py`.

| Claim | Evidence |
|---|---|
| The quantities are computed as defined | Values worked out by hand for small configurations; agreement to 12 decimal places with a direct implementation of each definition, on 10 random sets of vectors with random boundaries. |
| H-COH-1: the reference level is the mean of local coherence over all orderings | Enumeration of the 720 orderings of six units; agreement to 12 decimal places. |
| H-COH-2: invariance of local coherence to reversal, rescaling and rotation | Tested on random vectors. |
| H-COH-3: global coherence does not depend on order | Tested on random reorderings. |
| H-COH-4: leaving the unit out of the centroid removes a bias | Five random directions in 2,000 dimensions: global coherence within 0.05 of zero; the same quantity with the unit included within 0.05 of 0.447. |
| H-COH-5: invariance of reference coherence to order, to the length of the reference and to rotation | Tested on random vectors. |
| The magnitude of the vectors does not matter | Vectors multiplied by 2 to the power -1000, -600, 600 or 1000, whose squares cannot be represented, give exactly the results of the original vectors. |

Not checked: any property of these quantities on embeddings of natural
language. In particular, their dependence on the length of the units, on the
number of units and on the embedding model is the subject of the robustness
study of the roadmap.

## 4. Test for topic boundaries

### Agreement with a direct implementation

The block similarities computed from the table of pairwise similarities
agree with a direct computation from the vectors to 10 decimal places, for
block sizes 1, 2, 3, 5 and 7. The depth of a valley is checked on nine
hand-made series.

The whole procedure is compared with a second implementation, written out
step by step, which recomputes every block similarity from the vectors for
every reordering. On 24 simulated texts with one to three topics, block
sizes from 1 to 3 and levels from 0.05 to 0.30, the two return the same
boundaries and the same p-values. `tests/test_semantic.py`

### Simulation

`studies/boundary_test_simulation.py` applies the test to simulated
sequences of unit vectors in 48 dimensions. A topic is a direction; a unit
on that topic is the direction plus isotropic scatter. Parameters: block
size 3, alpha 0.05, 199 permutations (the default of the library is 999;
the simulation uses fewer to limit its running time). The full output is in
[`studies/results/boundary_test_simulation.md`](../studies/results/boundary_test_simulation.md).

Three settings are used, from distinct topics with moderate scatter to
closely related topics with large scatter. Mean cosine similarity between
two units:

| Setting | Within a topic | Between topics |
|---|---:|---:|
| A. Distinct topics, moderate scatter | 0.50 | 0.00 |
| B. Related topics, larger scatter | 0.37 | 0.09 |
| C. Closely related topics, large scatter | 0.28 | 0.15 |

**Level.** Sequences in which every order of the units is equally likely.
Percentage of sequences in which a boundary is reported, with its 95%
interval; the nominal level is 5%.

Directions drawn independently and uniformly:

| Sequence | Sequences | A boundary is reported | 95% interval |
|---|---:|---:|---:|
| 8 units, the fewest that can be tested | 5000 | 5.0% | 4.5% to 5.7% |
| 16 units | 5000 | 5.2% | 4.6% to 5.9% |
| 24 units | 5000 | 5.1% | 4.6% to 5.8% |
| 40 units | 5000 | 5.3% | 4.7% to 6.0% |
| 12 units, block size 1 | 5000 | 4.9% | 4.4% to 5.6% |
| All of the above | 25000 | 5.1% | 4.9% to 5.4% |

Sequences of 24 units that have a structure but whose order is random:

| Sequence | Sequences | A boundary is reported | 95% interval |
|---|---:|---:|---:|
| One topic (setting A) | 5000 | 5.0% | 4.5% to 5.7% |
| Two topics, 12 + 12 units, in random order (setting A) | 5000 | 5.3% | 4.7% to 6.0% |
| One topic, one unrelated unit at a random position (setting A) | 5000 | 4.9% | 4.3% to 5.5% |
| One topic (setting B) | 5000 | 5.4% | 4.8% to 6.0% |
| Two topics, 12 + 12 units, in random order (setting B) | 5000 | 4.9% | 4.4% to 5.6% |
| One topic, one unrelated unit at a random position (setting B) | 5000 | 5.3% | 4.7% to 5.9% |
| One topic (setting C) | 5000 | 5.5% | 4.9% to 6.2% |
| Two topics, 12 + 12 units, in random order (setting C) | 5000 | 5.0% | 4.4% to 5.7% |
| One topic, one unrelated unit at a random position (setting C) | 5000 | 5.2% | 4.6% to 5.8% |
| All of the above | 45000 | 5.2% | 5.0% to 5.4% |

**Content that varies without a change of topic.** Sequences of 24 units;
200 sequences per cell, so that a percentage is known to within about 7
points. Percentage of sequences in which a boundary is reported (none should
be):

| Sequence | A | B | C |
|---|---:|---:|---:|
| One topic, one unrelated unit in the middle | 17% | 12% | 9% |
| One topic, three scattered unrelated units | 0% | 0% | 2% |
| Gradual drift to an unrelated subject | 0% | 1% | 2% |

**Planted changes of topic.** 200 sequences per cell. Percentage of
sequences in which the number of boundaries is right and each lies within
one unit of its true position:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Two topics, 6 + 6 units | 87% | 62% | 32% |
| Two topics, 8 + 8 units | 96% | 81% | 44% |
| Two topics, 12 + 12 units | 96% | 94% | 66% |
| Two topics, 20 + 20 units | 97% | 94% | 88% |
| Two topics, 6 + 18 units | 16% | 11% | 6% |
| Three topics, 8 + 8 + 8 units | 72% | 44% | 8% |
| Three topics, 12 + 12 + 12 units | 94% | 84% | 28% |
| Excursion and return, 10 + 4 + 10 units | 19% | 10% | 2% |
| Excursion and return, 10 + 8 + 10 units | 81% | 55% | 14% |

### Reading of the results

* **Level.** In the fourteen kinds of sequence in which every order is
  equally likely, a boundary is reported in 4.9% to 5.5% of cases, and each
  95% interval contains the nominal 5%. Over all sequences the rate is 5.1%
  for independent directions and 5.2% for structured sequences in random
  order. This is consistent with H-SEG-1, which follows from the
  construction of the test. A simulation of this size would not reveal an
  excess of one or two tenths of a point.
* **Drift and scattered units.** A gradual drift is reported in at most 2%
  of cases, and three scattered unrelated units in at most 2%. This is
  consistent with H-SEG-2.
* **Long segments, distinct topics.** In settings A and B, two segments of
  12 or 20 units are separated at the right place in 94% to 97% of cases,
  and three segments of 12 units in 84% to 94%. This is consistent with
  H-SEG-3.
* **Weak contrast.** In setting C, two segments are separated at the right
  place in 66% of cases when each has 12 units and in 88% when each has 20.
  Three segments of 12 units are all delimited in 28% of cases.
* **Short segments.** Two segments of six units are separated in 87%, 62%
  and 32% of cases in settings A, B and C. A passage of four units between
  two longer ones is delimited on both sides in at most 19% of cases,
  although at least one boundary is reported in 60% to 100% of them.
* **Edges.** A boundary six units from the start of a text of 24 units is
  found in at most 16% of cases.
* **An unrelated unit in the middle.** A single unrelated unit in the middle
  of a text of 24 units leads to a reported boundary in 9% to 17% of cases,
  against a nominal 5%. Placed at a random position, where every order is
  equally likely, it does so in 4.9% to 5.3% of cases. In the middle of a
  text the unit lowers the similarity of the blocks on both sides of it,
  which gives a valley with two sides; near an end it gives only one.
* **Further boundaries.** More boundaries than were planted are reported in
  0% to 6% of cases, depending on the sequence.

These results describe the statistical procedure on synthetic vectors. They
do not establish how the test behaves on embeddings of natural language.

### Checks in the test suite

The test suite repeats reduced versions of these simulations at each run:
the rate of reported boundaries on 200 sequences of independent directions
and on 60 sequences of each of four kinds without a change of topic, and the
detection rate on 50 sequences with two distinct topics. With so few
sequences these checks can reveal a gross error, not a small excess over the
nominal level. Two p-values are pinned, so that a change in the stream of
permutations would very probably be noticed. `tests/test_semantic.py`

### Coherence within segments

H-SEG-4 is checked on two simulated texts of 24 units, one on a single topic
and one that deals with two topics in turn. The global coherence of the
second is lower than that of the first by more than 0.1, while its global
coherence within segments, computed with the estimated boundary, is within
0.1 of the global coherence of the first. With a boundary that is supplied,
the values within segments are checked against values worked out by hand.
`tests/test_semantic.py`

## 5. Embedding adapter

| Claim | Evidence |
|---|---|
| The adapter returns one vector per unit, is deterministic, is unaffected by batch size, and counts truncated units, the prompt of the model included | 12 tests run against the `sentence-transformers` library, versions 3.0.0 and 6.1.0, with a small model built locally with random weights. `tests/test_embedding.py` |
| The checks of the arguments, the conversion of the output and the count of truncated units behave as documented | 11 tests that run without the library, with a stand-in for the model where one is needed. `tests/test_embedding.py` |

Three further tests load a published model,
`sentence-transformers/all-MiniLM-L6-v2` at a fixed revision. They check the
revision, the dimensions and the token limit, the ordering of the
similarities of related and unrelated sentences, and the count of truncated
units. They need network access to download the model and are not part of
the default run (`pytest -m network tests/test_embedding.py`); the
continuous integration workflow runs them.

## 6. Environments

The test suite passes on Linux (x86-64) with Python 3.10, 3.11, 3.12, 3.13
and 3.14, and with NumPy from 1.26 to 2.5. The continuous integration
workflow runs it on Linux, Windows and macOS with the same versions of
Python, and runs the tests of the embedding adapter on the three systems.

## 7. Reproducibility

* Tokenisation and lexical diversity are deterministic and use integer
  arithmetic up to the final division.
* The permutations of the test for boundaries are drawn from the raw stream
  of NumPy's PCG64 bit generator, which NumPy keeps stable across versions,
  so that a seed gives the same permutations everywhere.
* Levels and p-values are compared in exact arithmetic. Two statistics are
  counted as equal when they differ by less than a tolerance set far above
  rounding error, so that the last digit of a similarity, which can differ
  between platforms and linear algebra libraries, does not decide whether a
  reordering equivalent to the observed order is counted.
* Embeddings produced by a neural model can differ in their last digits
  between devices and library versions. For an exactly repeatable analysis,
  compute the embeddings once and store them.

## 8. What has not been validated

* The behaviour of the coherence measures and of the test for boundaries on
  natural language. This is the purpose of the benchmark and of the
  robustness study in the roadmap.
* The division of texts into units, which is left to the study.
* Any relation between these measures and clinical constructs, symptoms,
  diagnoses or outcomes.
* Use in languages other than English.
