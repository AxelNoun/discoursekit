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
| H-COH-6: the spread of the reference level is the standard deviation of local coherence over all orderings | Enumeration of all orderings of two to seven units, in two and in five dimensions; agreement to 12 decimal places. Agreement to 12 decimal places with a computation pair by pair, from the probabilities that two pairs of units are consecutive, on 10 random sets of 2 to 14 vectors. Agreement within sampling error with 4,000 random orders of 30 units. |
| H-COH-3: global coherence does not depend on order | Tested on random reorderings. |
| H-COH-4: leaving the unit out of the centroid removes a bias | Five random directions in 2,000 dimensions: global coherence within 0.05 of zero; the same quantity with the unit included within 0.05 of 0.447. |
| H-COH-5: invariance of reference coherence to order, to the length of the reference and to rotation | Tested on random vectors. |
| H-COH-7: global coherence increases with the number of units | For 2, 4, 8, 32 and 128 vectors with a similarity of 0.2 or 0.5 between all pairs, global coherence equals the closed-form value and increases with the number of vectors, while local coherence and its reference level stay at 0.2 or 0.5. For texts simulated on one topic, the example of [short-texts.md](short-texts.md), which is run as a test, gives 0.62 with four units and 0.70 with 32. |
| The magnitude of the vectors does not matter | Vectors multiplied by 2 to the power -1000, -600, 600 or 1000, whose squares cannot be represented, give exactly the results of the original vectors. |

Not checked: any property of these quantities on embeddings of natural
language. In particular, their dependence on the length of the units, on the
number of units and on the embedding model is the subject of the robustness
study of the roadmap.

## 4. Test for topic boundaries

### Agreement with a direct implementation

The two block similarities computed from the table of pairwise similarities
agree with a direct computation from the vectors to 10 decimal places, for
block sizes 1, 2, 3, 5 and 7. The rises of a curve are checked on six
hand-made series, and the candidate boundaries and their depths on four
hand-made texts, with equal and with unequal block sizes.

The whole procedure is compared with a second implementation, written out
step by step, which recomputes every block similarity from the vectors for
every reordering. On 24 simulated texts with one to three topics, block
sizes from 1 to 3 and levels from 0.05 to 0.30, the two return the same
boundaries and the same p-values. `tests/test_semantic.py`

### Every arrangement of two groups

For two groups of identical units, the statistic can be computed for every
arrangement of the units, which gives the p-value that the test would return
if it examined all the reorderings. `tests/test_semantic.py`

| Text | Block size | Arrangements that reach the statistic of the ordered text | Smallest possible p-value |
|---|---:|---:|---:|
| 3 + 3 units | 2 | 4 of 20 | 0.200 |
| 4 + 4 units | 2 | 2 of 70 | 0.029 |
| 4 + 4 units | 3 | 4 of 70 | 0.057 |
| 5 + 5 units | 3 | 2 of 252 | 0.008 |
| 6 + 6 units | 3 | 2 of 924 | 0.002 |
| 8 + 8 units | 3 | 2 of 12,870 | 0.0002 |

From five units a side with blocks of three, and from four with blocks of
two, the ordered text and its mirror image are alone at the top: no other
arrangement reaches their statistic. With four units a side and blocks of
three, two further arrangements do, and the p-value cannot go below 0.057
however distinct the two groups are.

The same count shows what each block similarity contributes. For 6 + 6 units
with blocks of three, 58 of the 924 arrangements have a valley as deep as
the ordered text when the depth is measured on the cosine between the sums
of the blocks alone, against 2 when it is measured on the mean similarity
between units, or on both.

### Simulation

`studies/boundary_test_simulation.py` applies the test to simulated
sequences of unit vectors in 48 dimensions. A topic is a direction; a unit
on that topic is the direction plus isotropic scatter. Parameters: block
size 3 unless stated otherwise, alpha 0.05, 199 permutations (the default of
the library is 999; the simulation uses fewer to limit its running time).
The full output is in
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
| 8 independent directions | 5000 | 5.0% | 4.5% to 5.7% |
| 16 independent directions | 5000 | 4.9% | 4.3% to 5.5% |
| 24 independent directions | 5000 | 4.6% | 4.1% to 5.2% |
| 40 independent directions | 5000 | 4.8% | 4.2% to 5.4% |
| 64 independent directions | 5000 | 4.8% | 4.3% to 5.4% |
| 12 independent directions, block size 1 | 5000 | 5.0% | 4.4% to 5.7% |
| 12 independent directions, block size 2 | 5000 | 4.9% | 4.3% to 5.5% |
| All of the above | 35000 | 4.9% | 4.6% to 5.1% |

Sequences of 24 units that have a structure but whose order is random:

| Sequence | Sequences | A boundary is reported | 95% interval |
|---|---:|---:|---:|
| One topic (setting A) | 5000 | 5.0% | 4.4% to 5.7% |
| Two topics, 12 + 12 units, in random order (setting A) | 5000 | 4.8% | 4.2% to 5.4% |
| One topic, one unrelated unit at a random position (setting A) | 5000 | 5.1% | 4.6% to 5.8% |
| One topic (setting B) | 5000 | 5.3% | 4.7% to 6.0% |
| Two topics, 12 + 12 units, in random order (setting B) | 5000 | 4.9% | 4.4% to 5.6% |
| One topic, one unrelated unit at a random position (setting B) | 5000 | 4.9% | 4.3% to 5.5% |
| One topic (setting C) | 5000 | 5.4% | 4.8% to 6.0% |
| Two topics, 12 + 12 units, in random order (setting C) | 5000 | 4.8% | 4.3% to 5.5% |
| One topic, one unrelated unit at a random position (setting C) | 5000 | 4.7% | 4.2% to 5.4% |
| All of the above | 45000 | 5.0% | 4.8% to 5.2% |

**Unrelated units.** One topic interrupted by unrelated units. Two adjacent
unrelated units are on one other subject; scattered ones are each on a
subject of their own. In this table and the following ones, each percentage
rests on 500 sequences and is known to within about 4 points. Percentage of
sequences in which a boundary is reported:

| Sequence | A | B | C |
|---|---:|---:|---:|
| 24 units, one unrelated unit in the middle | 16% | 14% | 13% |
| 24 units, two adjacent unrelated units in the middle | 57% | 53% | 41% |
| 24 units, three scattered unrelated units | 0% | 0% | 0% |

**Gradual change.** The subject moves by equal steps from one topic to an
unrelated direction, through the whole sequence or through its second half.
Percentage of sequences in which a boundary is reported:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Drift through the whole sequence, 24 units | 1% | 1% | 4% |
| Drift through the whole sequence, 60 units | 0% | 1% | 1% |
| Drift through the whole sequence, 160 units | 0% | 0% | 0% |
| One topic, then a drift over the second half, 24 units | 1% | 4% | 6% |
| One topic, then a drift over the second half, 60 units | 0% | 3% | 6% |
| One topic, then a drift over the second half, 160 units | 0% | 0% | 0% |

Two topics of 24 units each, joined by a transition during which the subject
moves by equal steps from the first topic to the second; the first row has
no transition. The largest block has 12 units. Percentage of sequences in
which a boundary is reported:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Two topics, 24 + 24 units | 100% | 100% | 99% |
| Transition of 6 units | 100% | 100% | 95% |
| Transition of 12 units | 100% | 100% | 79% |
| Transition of 24 units | 74% | 41% | 14% |
| Transition of 48 units | 0% | 0% | 2% |

**Abrupt changes of topic.** Percentage of sequences in which the number of
boundaries is right and each lies within one unit of its true position:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Two topics, 6 + 6 units | 98% | 82% | 40% |
| Two topics, 8 + 8 units | 96% | 88% | 50% |
| Two topics, 12 + 12 units | 96% | 97% | 74% |
| Two topics, 20 + 20 units | 96% | 97% | 89% |
| Two topics, 80 + 80 units | 96% | 95% | 88% |
| Two topics, 6 + 18 units | 96% | 87% | 36% |
| Two topics, 6 + 34 units | 93% | 64% | 17% |
| Three topics, 8 + 8 + 8 units | 97% | 77% | 15% |
| Three topics, 12 + 12 + 12 units | 96% | 95% | 43% |
| Excursion and return, 10 + 4 + 10 units | 19% | 10% | 2% |
| Excursion and return, 10 + 8 + 10 units | 96% | 73% | 20% |
| Excursion and return, 16 + 8 + 16 units | 95% | 91% | 38% |

**Short sequences and block size.** Sequences of 8 to 16 units, examined
with blocks of three units, the default, and of two. Percentage of sequences
in which a boundary is reported, none being planted:

| Sequence | A, blocks of 3 | B, blocks of 3 | C, blocks of 3 | A, blocks of 2 | B, blocks of 2 | C, blocks of 2 |
|---|---:|---:|---:|---:|---:|---:|
| 12 units, one unrelated unit in the middle | 11% | 10% | 9% | 14% | 13% | 11% |
| 12 units, two adjacent unrelated units in the middle | 46% | 39% | 29% | 37% | 36% | 35% |
| Drift through the whole sequence, 12 units | 0% | 2% | 3% | 0% | 1% | 3% |

Percentage of sequences in which the number of boundaries is right and each
lies within one unit of its true position:

| Sequence | A, blocks of 3 | B, blocks of 3 | C, blocks of 3 | A, blocks of 2 | B, blocks of 2 | C, blocks of 2 |
|---|---:|---:|---:|---:|---:|---:|
| Two topics, 4 + 4 units | 53% | 42% | 22% | 88% | 51% | 22% |
| Two topics, 6 + 6 units | 98% | 82% | 40% | 93% | 71% | 31% |
| Two topics, 4 + 8 units | 2% | 7% | 5% | 86% | 53% | 17% |
| Excursion and return, 6 + 4 + 6 units | 0% | 5% | 3% | 54% | 18% | 4% |

**Longer sequences and block size.** Sequences of 72 to 160 units, examined
with the default block size, which gives blocks of 3 to 12 units, and with a
block size of 6, which gives blocks of 6 to 24 units (6 to 18 in the
sequence of 72 units). Percentage of sequences in which a boundary is
reported:

| Sequence | A, blocks of 3 | B, blocks of 3 | C, blocks of 3 | A, blocks of 6 | B, blocks of 6 | C, blocks of 6 |
|---|---:|---:|---:|---:|---:|---:|
| Two topics, 80 + 80 units | 100% | 100% | 95% | 100% | 100% | 100% |
| Transition of 24 units | 74% | 41% | 14% | 100% | 100% | 89% |
| Transition of 48 units | 0% | 0% | 2% | 100% | 96% | 56% |
| One topic, then a drift over the second half, 160 units | 0% | 0% | 0% | 1% | 4% | 7% |

Two topics of 80 units each: percentage of sequences in which one boundary
is reported, within one unit of its true position:

| Sequence | A, blocks of 3 | B, blocks of 3 | C, blocks of 3 | A, blocks of 6 | B, blocks of 6 | C, blocks of 6 |
|---|---:|---:|---:|---:|---:|---:|
| Two topics, 80 + 80 units | 96% | 95% | 88% | 96% | 95% | 92% |

### The two block similarities compared

`studies/block_similarity_comparison.py` applies three versions of the test
to sequences of the simulation above: the test of the library, in which the
depth of the valley is measured on both block similarities, and the test
with the depth measured on one of them only. The candidate boundaries are
the same in the three versions. The full output is in
[`studies/results/block_similarity_comparison.md`](../studies/results/block_similarity_comparison.md).

In the tables below, Both is the test of the library, Sums the test with
the depth measured on the cosine between the sums of the blocks only, and
Units the test with the depth measured on the mean similarity between units
only.

One topic of 24 units, in which every order of the units is equally likely.
Percentage of 5,000 sequences in which a boundary is reported, with its 95%
interval; the nominal level is 5%:

| Version | A | B | C |
|---|---:|---:|---:|
| Both | 5.0% (4.4% to 5.7%) | 5.3% (4.7% to 6.0%) | 5.4% (4.8% to 6.0%) |
| Sums | 4.9% (4.3% to 5.5%) | 5.4% (4.8% to 6.0%) | 5.3% (4.7% to 5.9%) |
| Units | 4.6% (4.1% to 5.2%) | 5.7% (5.1% to 6.3%) | 5.2% (4.6% to 5.9%) |

Percentage of sequences in which a boundary is reported, none being planted.
In this table and the next, each percentage rests on 500 sequences:

| Sequence | Both, A | Both, B | Both, C | Sums, A | Sums, B | Sums, C | Units, A | Units, B | Units, C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 units, one unrelated unit in the middle | 16% | 14% | 13% | 10% | 10% | 11% | 22% | 17% | 13% |
| Drift through the whole sequence, 60 units | 0% | 1% | 1% | 0% | 0% | 0% | 1% | 4% | 3% |
| One topic, then a drift over the second half, 60 units | 0% | 3% | 6% | 0% | 1% | 2% | 2% | 5% | 12% |

Percentage of sequences in which the number of boundaries is right and each
lies within one unit of its true position:

| Sequence | Both, A | Both, B | Both, C | Sums, A | Sums, B | Sums, C | Units, A | Units, B | Units, C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Two topics, 6 + 6 units | 98% | 82% | 40% | 88% | 66% | 33% | 100% | 90% | 43% |
| Two topics, 6 + 18 units | 96% | 87% | 36% | 96% | 81% | 26% | 97% | 89% | 37% |
| Three topics, 8 + 8 + 8 units | 97% | 77% | 15% | 75% | 49% | 10% | 97% | 86% | 21% |
| Excursion and return, 10 + 8 + 10 units | 96% | 73% | 20% | 82% | 46% | 12% | 97% | 80% | 23% |

### Reading of the results

* **Level.** In the sixteen kinds of sequence in which every order is
  equally likely, a boundary is reported in 4.6% to 5.4% of cases, and each
  95% interval contains the nominal 5%. Over all sequences the rate is 4.9%
  for independent directions and 5.0% for structured sequences in random
  order. This is consistent with H-SEG-1, which follows from the
  construction of the test. A simulation of this size would not reveal an
  excess of one or two tenths of a point.
* **Drift.** A drift that runs through the whole sequence is reported in at
  most 4% of cases at 24 units and in at most 1% at 60 and 160 units. A
  topic followed by a drift over the second half of the sequence is
  reported in at most 6% of cases. This is consistent with H-SEG-2.
* **Transitions.** Between two topics of 24 units, a transition of 6 units
  is reported like an abrupt change, in 95% to 100% of cases, and one of 12
  units, the size of the largest block, in 79% to 100%. A transition of 24
  units is reported in 14% to 74% of cases, the more often the more distinct
  the topics, and one of 48 units in at most 2%. This is the sense in which
  the test concerns abrupt changes (H-SEG-5): it does not tell a transition
  shorter than its largest block from a change at one point.
* **Abrupt changes between segments of at least six units.** In setting A,
  the boundaries are all found at the right place in 93% to 98% of cases,
  for two or three topics, whether the segments are equal or not and
  wherever the boundary lies. In setting B the rate is 64% to 97%: it is
  lowest for the shortest segments, 82% for 6 + 6 units, 77% for 8 + 8 + 8,
  73% for a passage of 8 units between two others of 10, and for a boundary
  six units from the start of a sequence of 40 units, 64%. This is
  consistent with H-SEG-3.
* **Weak contrast.** In setting C, two segments are separated at the right
  place in 40% of cases when each has 6 units, 74% when each has 12 and 89%
  when each has 20. Three segments of 12 units are all delimited in 43% of
  cases. The rate does not rise beyond a few blocks: it is 88% for two
  segments of 80 units.
* **Segments shorter than two blocks.** A passage of four units between two
  longer ones is delimited on both sides in at most 19% of cases, although
  at least one boundary is reported in 62% to 100% of them. Two segments of
  four units are separated in 22% to 53% of cases, and a boundary four
  units from the start of a sequence of 12 units in at most 7%.
* **Block size and short sequences.** With blocks of two units, two segments
  of four units are separated in 88%, 51% and 22% of cases in settings A, B
  and C, against 53%, 42% and 22% with blocks of three, and a boundary four
  units from the start of a sequence of 12 units in 86%, 53% and 17%,
  against at most 7%. Two segments of six units are separated a little less
  often than with blocks of three (93%, 71% and 31% against 98%, 82% and
  40%). For the sequences of 12 units without a boundary, the two block
  sizes give rates of the same order: 9% to 14% for an unrelated unit in
  the middle, 29% to 46% for two adjacent ones, at most 3% for a drift.
* **Block size and longer sequences.** With a block size of 6 the blocks
  are twice as large. Two segments of 80 units in setting C are then
  separated at the right place in 92% of cases instead of 88%, and a
  boundary is reported in every case instead of 95%. Transitions of 24 and
  48 units are reported in 56% to 100% of cases, against 0% to 74% with the
  default block size, and a topic followed by a drift over 80 units in up
  to 7% of cases, against none. The block size sets the scale at which a
  change counts as abrupt.
* **Unrelated units.** A single unrelated unit in the middle of a sequence
  of 24 units leads to a reported boundary in 13% to 16% of cases, against a
  nominal 5%. Placed at a random position, where every order is equally
  likely, it does so in 4.7% to 5.1% of cases, as the level of the test
  requires: the excess in the middle of the sequence is made up for at
  other positions. Two adjacent unrelated units on one other subject lead
  to a reported boundary in 41% to 57% of cases. The test does not tell a
  digression of two units from the beginning of a new topic. Three
  unrelated units scattered through the sequence lead to a reported
  boundary in less than 1% of cases.
* **Further boundaries.** More boundaries than were planted are reported in
  0% to 5% of cases, depending on the sequence.
* **The two similarities: level.** The three versions compared are
  permutation tests of the same construction, which keeps each of them to
  the level. For one topic of 24 units they report a boundary in 4.6% to
  5.7% of 5,000 sequences. One of the nine intervals does not contain 5%:
  5.1% to 6.3%, for the mean similarity alone in setting B, where the two
  other versions give 5.3% and 5.4% on the same sequences. Among nine
  intervals at 95%, one that misses the nominal level by a tenth of a point
  is within what sampling produces.
* **The two similarities: what each finds.** Measured on the cosine between
  the sums of the blocks alone, the valley is reported least often where no
  boundary should be: in at most 2% of the sequences that drift, and in 10%
  to 11% of those with an unrelated unit in the middle. It also misses many
  changes of topic: in settings A and B it finds 10 to 28 points fewer than
  the test of the library in three of the four kinds of sequence, and it
  cannot tell the ordered text from other arrangements when the units of a
  topic are identical (see above). Measured on the mean similarity between
  units alone, the valley finds 0 to 9 points more changes than the test of
  the library, and is reported more often where no boundary should be: in up
  to 12% of the sequences that keep to one topic and then drift, against 6%,
  and in up to 22% of those with an unrelated unit in the middle, against
  16%. The sum of the two depths was retained as the middle course.

These results describe the statistical procedure on synthetic vectors. They
do not establish how the test behaves on embeddings of natural language.

### How the statistic was chosen

The form of the statistic was settled by comparing variants on simulated
sequences of the kinds described above, before the results reported here
were produced. Four choices were examined: one block size for both sides of
a valley or one for each; the pairs of sizes admitted; the largest block;
and the block similarity. The last comparison is the one reported above. For
the others, with fewer sequences than in the tables above and with the
cosine between the sums of the blocks as the only similarity:

* with one block size for both sides, a boundary six units from the start
  of a sequence of 24 units was found in at most 16% of cases;
* with pairs of sizes of any ratio, a sequence of 60 units that keeps to one
  topic and then drifts was reported in up to 39% of cases;
* with blocks as large as a quarter of the sequence whatever its length, a
  transition spread over a third of a sequence of 100 units was reported
  every time, and a passage of 20 units inside a sequence of 160 was
  delimited correctly about half the time.

These three comparisons are not part of the repository. The tables above
were produced after the statistic was fixed, from sequences drawn afresh.
Since the kinds of sequence are the same, they may still flatter the test
somewhat.

### Checks in the test suite

The test suite repeats reduced versions of these simulations at each run:
the rate of reported boundaries on 200 sequences of independent directions,
on 60 sequences of each of four kinds without a change of topic and on 30
sequences with a slow transition between two topics; the detection rate on
50 sequences with two distinct topics of equal length, and on 30 sequences
each with a boundary six units from the start and six units from the end.
With so few sequences these checks can reveal a gross error, not a small
excess over the nominal level. Two p-values are pinned, so that a change in
the stream of permutations would very probably be noticed.
`tests/test_semantic.py`

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
continuous integration workflow runs them at each change.

## 6. Environments

The test suite passes on Linux (x86-64) with Python 3.10, 3.11, 3.12, 3.13
and 3.14, and with NumPy from 1.26 to 2.5. At each change, the continuous
integration workflow runs it on Linux, Windows and macOS with the same
versions of Python, and runs the tests of the embedding adapter on the three
systems. The badge at the top of the README shows the state of the latest
run.

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
