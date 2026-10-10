# Validation status

This document states what has been checked in this library, how, and with
what result, and what has not been checked. It describes the repository as
it stands.

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
| Behaviour on English text is as documented | Tests of specific cases: apostrophes, hyphens, full stops, numbers, case, normalisation, white space, exclusion. `tests/test_tokenization.py` |

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
| The options of the tokenisation are returned with the result | The result of `lexical_diversity` states whether tokens were case folded and which tokens were excluded, as normalised for the comparison. `tests/test_lexical.py` |

Not checked: H-LEX-1 on natural text.

## 3. Local, global and reference coherence

The evidence of this section is in `tests/test_semantic.py`, except where
another file is named.

| Claim | Evidence |
|---|---|
| The quantities are computed as defined | Values worked out by hand for small configurations; agreement to 12 decimal places with a direct implementation of each definition, on 10 random sets of vectors with random boundaries. |
| H-COH-1: the reference level is the mean of local coherence over all orderings | Enumeration of the 720 orderings of six units; agreement to 12 decimal places. |
| H-COH-2: invariance of local coherence to reversal, rescaling and rotation | Tested on random vectors. |
| H-COH-6: the value returned as the spread of local coherence is its standard deviation over all orderings | Enumeration of all orderings of two to seven units, in two and in five dimensions; agreement to 12 decimal places. Agreement to 12 decimal places with a computation pair by pair, from the probabilities that two pairs of units are consecutive, on 10 random sets of 2 to 14 vectors. Agreement to within 8% with the standard deviation over 4,000 random orders of 30 units. |
| The difference between local coherence and its reference level, in units of that spread, grows with the length of an ordered text | 40 simulated texts on two topics at each of two lengths, 8 and 32 units: the difference is about the same at both lengths (0.22 and 0.24), and in units of the spread it is more than twice as large for the longer texts (2.2 and 5.1). |
| H-COH-3: global coherence does not depend on order | Tested on random reorderings. |
| H-COH-4: leaving the unit out of the centroid removes a bias | For 2, 5, 20 and 80 random directions in 2,000 dimensions: global coherence within 0.05 of zero; the same quantity with the unit included within 0.05 of the inverse square root of the number of units. |
| H-COH-5: invariance of reference coherence to order, to the length of the reference and to rotation | Tested on random vectors. |
| H-COH-7: global coherence increases with the number of units | For 2, 4, 8, 32 and 128 vectors with a similarity of 0.2 or 0.5 between all pairs, global coherence equals the closed-form value and increases with the number of vectors, while local coherence and its reference level stay at 0.2 or 0.5. For texts simulated on one topic, the example of [short-texts.md](short-texts.md), which is run as a test, gives 0.62 with four units and 0.70 with 32. |
| The example of [short-texts.md](short-texts.md) gives the figures quoted there | The example is run as a test, with its printed output. `docs/short-texts.md` |
| The magnitude of the vectors does not matter | Vectors multiplied by 2 to the power -1000, -600, 600 or 1000, whose squares cannot be represented, give exactly the results of the original vectors. |

Not checked: any property of these quantities on embeddings of natural
language. In particular, their dependence on the length of the units, on the
number of units and on the embedding model is the subject of the robustness
study of the roadmap.

## 4. Test for topic boundaries

### Agreement with a direct implementation

The two block similarities computed from the table of sums agree with a
direct computation from the vectors to 10 decimal places, for block sizes 1,
2, 3, 5 and 7, and to 11 decimal places in a text of 3,000 units whose sums
reach millions. The sums themselves are exact: reordering the units inside
the two blocks compared, or outside them, leaves the two similarities
unchanged to the last digit, and a text read backwards gives the same curves
in reverse. When the product of matrices from which the similarities are
taken has noise in its last digit, so that it is neither symmetric nor equal
between equal units, the similarities kept are both. A block whose units
have a mean shorter than 2^-10 has no direction, the four corners of a
regular tetrahedron among them; just above that length, the cosine between
the sums agrees with a direct computation to 2 × 10^-6. The rises of a curve
are checked on six hand-made series, and the depths of the gaps on six
hand-made texts, with equal and with unequal block sizes.

The whole procedure is compared with a second implementation, written out
step by step, which recomputes every block similarity from the vectors for
every reordering. The two return the same boundaries and the same p-values
on three sets of texts, with block sizes from 1 to 3:

* 24 simulated texts of 4 to 27 units with one to three topics, at levels
  from 0.05 to 0.30. Two are too short to be tested; 11 receive no boundary,
  8 one, 4 two and 1 three.
* 30 texts of 6 to 36 units made of runs of identical units, at levels from
  0.10 to 0.40. In such texts several gaps, several pairs of block sizes and
  several reorderings are exactly as good as one another, so that the count
  of ties and the choice among equals are compared as well. 11 receive no
  boundary, 12 one, 5 two and 2 three.
* 12 texts of 7 to 29 units made of runs of identical units in which a
  direction and its opposite both occur, at levels from 0.10 to 0.40. Many
  of their blocks have a sum of zero and no direction. 5 receive no
  boundary, 4 one and 3 two.

Two texts made of runs of identical units, in which two pairs of block sizes
give the observed statistic exactly and point to different gaps, check that
the first pair decides, in both implementations. Levels and p-values are
compared in exact arithmetic: with 9 reorderings, a p-value of 3/10 is
significant at `alpha` = 0.3, although 0.3 in binary is slightly less than
three tenths. Recording the parts that are tested shows that a part whose
level is below 1 / (1 + *B*) is not examined. `tests/test_semantic.py`

### The two deepest valleys

The deepest valley is sought on each block similarity separately (see
[metrics.md](metrics.md), section 6). On 30 simulated texts with two topics
of 12 units, the two valleys are at the same gap for every pair of block
sizes, and at the change of topic for blocks of three. On 30 texts of 24
units on one topic, they are at the same gap in 18. A hand-made text of
nine units shows the case in which no gap has a valley on both
similarities: the statistic adds a depth of 1/2 found at one gap and a
depth of 1/8 found at another, and the candidate boundary is the first of
the two gaps. `tests/test_semantic.py`

### Every arrangement of two groups

For two groups of identical units, the statistic can be computed for every
arrangement of the units, which gives the p-value that the test would return
if it examined all the reorderings (H-SEG-6). `tests/test_semantic.py`

| Text | Block size | Arrangements with a larger statistic than the ordered text | Arrangements that reach it | p-value with every reordering examined |
|---|---:|---:|---:|---:|
| 4 + 4 units | 2 | 0 | 2 of 70 | 0.029 |
| 5 + 5 units | 2 | 0 | 12 of 252 | 0.048 |
| 4 + 6 units | 2 | 0 | 6 of 210 | 0.029 |
| 4 + 8 units | 2 | 0 | 6 of 495 | 0.012 |
| 6 + 6 units | 3 | 0 | 2 of 924 | 0.002 |
| 6 + 7 units | 3 | 0 | 4 of 1,716 | 0.002 |
| 6 + 8 units | 3 | 0 | 6 of 3,003 | 0.002 |
| 7 + 7 units | 3 | 0 | 12 of 3,432 | 0.003 |
| 8 + 8 units | 3 | 0 | 2 of 12,870 | 0.0002 |
| 5 + 5 units | 3 | 0 | 2 of 252 | 0.008 |
| 3 + 3 units | 2 | 2 | 4 of 20 | 0.200 |
| 4 + 4 units | 3 | 2 | 4 of 70 | 0.057 |
| 4 + 8 units | 3 | 46 | 53 of 495 | 0.107 |
| 5 + 6 units | 2 | 0 | 26 of 462 | 0.056 |
| 2 + 10 units | 1 | 9 | 11 of 66 | 0.167 |
| 3 + 9 units | 1 | 8 | 10 of 220 | 0.045 |
| 4 + 16 units | 2 | 25 | 29 of 4,845 | 0.006 |

In the first nine configurations, in which each group has at least twice
`block_size` units, no arrangement has a larger statistic than the ordered
text. Those that reach it are its mirror image and the arrangements that
hold an equally sharp change elsewhere, such as AAAAAABBBBBBAB for 7 + 7
units. They are 2 when each group has exactly twice as many units as the
largest block, and up to 12 otherwise. With 5 + 5 units and a block size
of 3 the groups are one unit short of two blocks, and the ordered text is
still unbeaten. In the next three rows a group is also shorter than two
blocks, and other arrangements do better than the ordered text.

The last four rows were added after the fourth independent review of the
code, which found that H-SEG-6 had been stated too broadly. With a block
size of 2, five and six units are not separated at the 5% level although
no arrangement beats the ordered text: 26 arrangements equal it, against
12 for five and five. In the last three rows each group has at least twice
`block_size` units, but one is much longer than the other, and placing the
short group in the middle of the long one, as in BBBBAABBBBBB, gives a
larger statistic than keeping the groups apart.

These p-values are those of a test that examines every reordering. The
library draws reorderings at random, and the p-value it returns varies with
the seed around the value above. For 4 + 4 units with a block size of 3, the
default, it has a mean of 0.059 over 40 seeds with 999 reorderings, and is
at or below 0.05 for 5 of them.

Every arrangement has exactly the same statistic as its mirror image, to the
last digit (H-SEG-7); checked for all the arrangements of 5 + 7 units with a
block size of 2, of 6 + 6 with 3 and of 3 + 6 with 1. On 20 texts made of
two runs of identical units, of 6 to 11 units each, the boundary is found
between the runs whether the text is read forwards or backwards. On 20
texts made of two to five runs of 2 to 8 identical units, the result is
unchanged when the embeddings are multiplied by a constant or their columns
reversed, which changes the last digits of the similarities.

The same enumeration shows what each block similarity contributes. For 6 + 6
units with a block size of 3, 58 of the 924 arrangements have a valley as
deep as the ordered text when the depth is measured on the cosine between
the sums of the blocks alone, against 2 when it is measured on the mean
similarity between units, or on both.

### Simulation

`studies/boundary_test_simulation.py` applies the test to simulated
sequences of unit vectors in 48 dimensions. A topic is a direction; a unit
on that topic is the direction plus isotropic scatter. Parameters: block
size 3 unless stated otherwise, alpha 0.05, 199 permutations (the default of
the library is 999; the simulation uses fewer to limit its running time),
and a seed of its own for each sequence. The full output is in
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
interval (Wilson score interval); the nominal level is 5%.

Directions drawn independently and uniformly:

| Sequence | Sequences | A boundary is reported | 95% interval |
|---|---:|---:|---:|
| 8 independent directions | 5000 | 5.0% | 4.5% to 5.7% |
| 16 independent directions | 5000 | 4.9% | 4.3% to 5.5% |
| 24 independent directions | 5000 | 4.3% | 3.8% to 4.9% |
| 40 independent directions | 5000 | 5.1% | 4.6% to 5.8% |
| 64 independent directions | 5000 | 5.4% | 4.8% to 6.1% |
| 12 independent directions, block size 1 | 5000 | 4.5% | 3.9% to 5.1% |
| 10 independent directions, block size 2 | 5000 | 4.9% | 4.4% to 5.6% |
| All of the above | 35000 | 4.9% | 4.7% to 5.1% |

Sequences of 24 units that have a structure but whose order is random:

| Sequence | Sequences | A boundary is reported | 95% interval |
|---|---:|---:|---:|
| One topic (setting A) | 5000 | 5.2% | 4.6% to 5.9% |
| Two topics, 12 + 12 units, in random order (setting A) | 5000 | 5.4% | 4.8% to 6.0% |
| One topic, one unrelated unit at a random position (setting A) | 5000 | 4.9% | 4.4% to 5.6% |
| One topic (setting B) | 5000 | 4.7% | 4.2% to 5.4% |
| Two topics, 12 + 12 units, in random order (setting B) | 5000 | 4.5% | 3.9% to 5.1% |
| One topic, one unrelated unit at a random position (setting B) | 5000 | 4.9% | 4.4% to 5.6% |
| One topic (setting C) | 5000 | 5.2% | 4.6% to 5.8% |
| Two topics, 12 + 12 units, in random order (setting C) | 5000 | 5.1% | 4.5% to 5.8% |
| One topic, one unrelated unit at a random position (setting C) | 5000 | 5.0% | 4.4% to 5.6% |
| All of the above | 45000 | 5.0% | 4.8% to 5.2% |

**One seed for all the sequences.** Sequences of 8 independent directions,
tested with a seed of its own for each sequence or with the same seed for
all, and with 19, 199 or 999 reorderings:

| Reorderings | Seed | Sequences | A boundary is reported | 95% interval |
|---:|---|---:|---:|---:|
| 19 | One for each sequence | 20000 | 4.7% | 4.4% to 5.0% |
| 19 | 0 for all | 20000 | 6.3% | 6.0% to 6.6% |
| 19 | 3 for all | 20000 | 4.3% | 4.1% to 4.6% |
| 199 | One for each sequence | 20000 | 4.9% | 4.6% to 5.2% |
| 199 | 0 for all | 20000 | 5.1% | 4.8% to 5.4% |
| 999 | One for each sequence | 20000 | 5.0% | 4.7% to 5.3% |
| 999 | 0 for all | 20000 | 4.9% | 4.6% to 5.2% |

**Consecutive units related to each other.** One topic, without any
change, in which each unit shares part of its scatter with the next, so
that two consecutive units are more alike than two other units, as in
coherent text. The orders of the units are then not equally likely, and
H-SEG-1 does not apply. The figure in brackets is the correlation between
the scatters of two consecutive units. Mean cosine similarity between two
consecutive units, and between two other units:

| Setting | Consecutive units (0.2) | Consecutive units (0.5) | Other units |
|---|---:|---:|---:|
| A | 0.60 | 0.75 | 0.50 |
| B | 0.50 | 0.69 | 0.37 |
| C | 0.42 | 0.64 | 0.28 |

Percentage of sequences in which a boundary is reported. In this table and
the following ones, each percentage rests on 500 sequences and is known to
within about 4 points:

| Sequence | A | B | C |
|---|---:|---:|---:|
| One topic, 24 units, consecutive units related (0.2) | 14% | 14% | 13% |
| One topic, 24 units, consecutive units related (0.5) | 11% | 12% | 10% |
| One topic, 48 units, consecutive units related (0.2) | 26% | 18% | 23% |
| One topic, 48 units, consecutive units related (0.5) | 48% | 42% | 35% |

**Unrelated units.** One topic interrupted by unrelated units. Two adjacent
unrelated units are on one other subject; scattered ones are each on a
subject of their own. Percentage of sequences in which a boundary is
reported:

| Sequence | A | B | C |
|---|---:|---:|---:|
| 24 units, one unrelated unit in the middle | 15% | 14% | 12% |
| 24 units, two adjacent unrelated units in the middle | 67% | 53% | 45% |
| 24 units, three scattered unrelated units | 0% | 1% | 2% |

**Gradual change.** The subject moves by equal steps from one topic to an
unrelated direction, through the whole sequence or through its second half.
Percentage of sequences in which a boundary is reported:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Drift through the whole sequence, 24 units | 0% | 1% | 2% |
| Drift through the whole sequence, 60 units | 0% | 0% | 2% |
| Drift through the whole sequence, 160 units | 0% | 0% | 0% |
| One topic, then a drift over the second half, 24 units | 1% | 2% | 5% |
| One topic, then a drift over the second half, 60 units | 1% | 3% | 5% |
| One topic, then a drift over the second half, 160 units | 0% | 0% | 0% |

Two topics of 24 units each, joined by a transition during which the subject
moves by equal steps from the first topic to the second; the first row has
no transition. The largest block has 12 units. Percentage of sequences in
which a boundary is reported:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Two topics, 24 + 24 units | 100% | 100% | 99% |
| Transition of 6 units | 100% | 100% | 97% |
| Transition of 12 units | 100% | 100% | 80% |
| Transition of 24 units | 72% | 37% | 15% |
| Transition of 48 units | 0% | 0% | 3% |

**Abrupt changes of topic.** An excursion and return is a passage on a
second topic between two segments on the same first topic. Percentage of
sequences in which the number of boundaries is right and each lies within
one unit of its true position:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Two topics, 6 + 6 units | 98% | 78% | 39% |
| Two topics, 8 + 8 units | 94% | 90% | 52% |
| Two topics, 12 + 12 units | 95% | 96% | 69% |
| Two topics, 20 + 20 units | 96% | 95% | 90% |
| Two topics, 80 + 80 units | 96% | 96% | 84% |
| Two topics, 6 + 18 units | 96% | 88% | 38% |
| Two topics, 6 + 34 units | 93% | 65% | 19% |
| Three topics, 8 + 8 + 8 units | 95% | 72% | 14% |
| Three topics, 12 + 12 + 12 units | 95% | 92% | 43% |
| Excursion and return, 10 + 4 + 10 units | 19% | 8% | 1% |
| Excursion and return, 10 + 8 + 10 units | 95% | 75% | 18% |
| Excursion and return, 16 + 8 + 16 units | 94% | 90% | 35% |

Percentage of sequences in which at least one boundary is reported:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Two topics, 6 + 6 units | 98% | 78% | 39% |
| Two topics, 8 + 8 units | 100% | 94% | 55% |
| Two topics, 12 + 12 units | 100% | 100% | 74% |
| Two topics, 20 + 20 units | 100% | 100% | 98% |
| Two topics, 80 + 80 units | 100% | 100% | 93% |
| Two topics, 6 + 18 units | 100% | 91% | 40% |
| Two topics, 6 + 34 units | 97% | 68% | 21% |
| Three topics, 8 + 8 + 8 units | 100% | 95% | 54% |
| Three topics, 12 + 12 + 12 units | 100% | 100% | 84% |
| Excursion and return, 10 + 4 + 10 units | 100% | 100% | 61% |
| Excursion and return, 10 + 8 + 10 units | 100% | 100% | 80% |
| Excursion and return, 16 + 8 + 16 units | 100% | 100% | 96% |

Percentage of sequences in which more boundaries are reported than were
planted:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Two topics, 6 + 6 units | 0% | 0% | 0% |
| Two topics, 8 + 8 units | 6% | 5% | 2% |
| Two topics, 12 + 12 units | 5% | 4% | 3% |
| Two topics, 20 + 20 units | 4% | 4% | 4% |
| Two topics, 80 + 80 units | 4% | 4% | 5% |
| Two topics, 6 + 18 units | 4% | 4% | 1% |
| Two topics, 6 + 34 units | 4% | 4% | 1% |
| Three topics, 8 + 8 + 8 units | 4% | 2% | 0% |
| Three topics, 12 + 12 + 12 units | 5% | 5% | 3% |
| Excursion and return, 10 + 4 + 10 units | 0% | 1% | 0% |
| Excursion and return, 10 + 8 + 10 units | 3% | 2% | 2% |
| Excursion and return, 16 + 8 + 16 units | 6% | 4% | 1% |

**Short sequences and block size.** Sequences of 8 to 16 units, examined
with block size 3, the default, and with block size 2. Percentage of
sequences in which a boundary is reported, none being planted:

| Sequence | A, block size 3 | B, block size 3 | C, block size 3 | A, block size 2 | B, block size 2 | C, block size 2 |
|---|---:|---:|---:|---:|---:|---:|
| 12 units, one unrelated unit in the middle | 14% | 10% | 11% | 11% | 13% | 14% |
| 12 units, two adjacent unrelated units in the middle | 44% | 39% | 27% | 41% | 39% | 36% |
| Drift through the whole sequence, 12 units | 1% | 2% | 3% | 0% | 1% | 3% |

Percentage of sequences in which the number of boundaries is right and each
lies within one unit of its true position:

| Sequence | A, block size 3 | B, block size 3 | C, block size 3 | A, block size 2 | B, block size 2 | C, block size 2 |
|---|---:|---:|---:|---:|---:|---:|
| Two topics, 4 + 4 units | 53% | 38% | 22% | 87% | 46% | 22% |
| Two topics, 5 + 5 units | 80% | 57% | 28% | 75% | 39% | 18% |
| Two topics, 6 + 6 units | 98% | 78% | 39% | 91% | 68% | 34% |
| Two topics, 4 + 8 units | 2% | 8% | 5% | 87% | 50% | 16% |
| Excursion and return, 6 + 4 + 6 units | 1% | 5% | 2% | 56% | 24% | 4% |

Two topics of 6 units, alone and after a passage of 3 units on a third
topic, which is too short to be delimited with block size 3. Percentage of
sequences in which the boundary between the two topics is the only one
reported, within one unit of its true position:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Two topics, 6 + 6 units | 98% | 78% | 39% |
| Two topics, 6 + 6 units, after a passage of 3 units | 98% | 72% | 24% |

**Longer sequences and block size.** Sequences of 72 to 160 units, examined
with block size 3, which gives blocks of 3 to 12 units, and with block size
6, which gives blocks of 6 to 24 units (6 to 18 in the sequence of 72
units). Percentage of sequences in which a boundary is reported:

| Sequence | A, block size 3 | B, block size 3 | C, block size 3 | A, block size 6 | B, block size 6 | C, block size 6 |
|---|---:|---:|---:|---:|---:|---:|
| Two topics, 80 + 80 units | 100% | 100% | 93% | 100% | 100% | 100% |
| Transition of 24 units | 72% | 37% | 15% | 100% | 100% | 90% |
| Transition of 48 units | 0% | 0% | 3% | 100% | 95% | 56% |
| One topic, then a drift over the second half, 160 units | 0% | 0% | 0% | 1% | 2% | 6% |

Two topics of 80 units each: percentage of sequences in which one boundary
is reported, within one unit of its true position:

| Sequence | A, block size 3 | B, block size 3 | C, block size 3 | A, block size 6 | B, block size 6 | C, block size 6 |
|---|---:|---:|---:|---:|---:|---:|
| Two topics, 80 + 80 units | 96% | 96% | 84% | 95% | 94% | 92% |

### The two block similarities compared

`studies/block_similarity_comparison.py` applies three versions of the test
to sequences of the simulation above: the test of the library, in which the
depth of a valley is measured on both block similarities, and the test with
the depth measured on one of them only. The full output is in
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
| Both | 5.2% (4.6% to 5.9%) | 4.7% (4.2% to 5.4%) | 5.2% (4.6% to 5.8%) |
| Sums | 4.7% (4.1% to 5.3%) | 4.6% (4.1% to 5.2%) | 5.1% (4.5% to 5.8%) |
| Units | 4.9% (4.3% to 5.5%) | 5.1% (4.5% to 5.7%) | 5.2% (4.6% to 5.9%) |

Percentage of sequences in which a boundary is reported, none being planted.
In this table and the next, each percentage rests on 500 sequences:

| Sequence | Both, A | Both, B | Both, C | Sums, A | Sums, B | Sums, C | Units, A | Units, B | Units, C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 units, one unrelated unit in the middle | 15% | 14% | 12% | 8% | 10% | 10% | 21% | 19% | 15% |
| Drift through the whole sequence, 60 units | 0% | 0% | 2% | 0% | 0% | 0% | 1% | 3% | 7% |
| One topic, then a drift over the second half, 60 units | 1% | 3% | 5% | 0% | 1% | 2% | 1% | 5% | 9% |
| One topic, 24 units, consecutive units related (0.5) | 11% | 12% | 10% | 5% | 3% | 2% | 19% | 18% | 19% |

Percentage of sequences in which the number of boundaries is right and each
lies within one unit of its true position:

| Sequence | Both, A | Both, B | Both, C | Sums, A | Sums, B | Sums, C | Units, A | Units, B | Units, C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Two topics, 6 + 6 units | 98% | 78% | 39% | 88% | 60% | 31% | 100% | 86% | 44% |
| Two topics, 6 + 18 units | 96% | 88% | 38% | 94% | 80% | 29% | 95% | 89% | 40% |
| Three topics, 8 + 8 + 8 units | 95% | 72% | 14% | 75% | 46% | 9% | 95% | 86% | 20% |
| Excursion and return, 10 + 8 + 10 units | 95% | 75% | 18% | 78% | 48% | 12% | 96% | 85% | 24% |

### Reading of the results

* **Level.** In the 16 kinds of sequence in which every order is equally
  likely, a boundary is reported in 4.3% to 5.4% of cases. One of the
  sixteen 95% intervals does not contain 5%: 3.8% to 4.9%, for 24
  independent directions, which lies below it; a rate below the level is
  allowed by H-SEG-1, and among sixteen intervals one that misses is within
  what sampling produces. Over all sequences the rate is 4.9% for
  independent directions and 5.0% for structured sequences in random order.
  This is consistent with H-SEG-1, which follows from the construction of
  the test. A simulation of this size would not reveal an excess of one or
  two tenths of a point.
* **One seed for all the sequences.** With a seed of its own for each
  sequence, the rate for 8 independent directions is 4.7%, 4.9% and 5.0%
  with 19, 199 and 999 reorderings. With seed 0 for all the sequences it is
  6.3%, 5.1% and 4.9%, and with seed 3 and 19 reorderings 4.3%. With few
  reorderings, the rate over sequences tested with one seed depends on that
  seed: with 19, seed 0, the default of the library, gives a rate whose
  interval, 6.0% to 6.6%, lies above 5%, and seed 3 one below it. With 199
  and 999 reorderings the departure is too small to be seen with 20,000
  sequences. A seed of its own for each sequence keeps the rate at the level
  whatever the number of reorderings.
* **Related consecutive units.** In sequences that keep to one topic but
  whose consecutive units are related, a boundary is reported in 10% to 48%
  of cases, against a nominal 5%: in 10% to 14% for 24 units and in 18% to
  48% for 48 units. The rate is higher for the longer sequences. In
  sequences of 48 units it is also higher when consecutive units are more
  closely related; in sequences of 24 units it is not. The test detects that
  the order is not random; in coherent text that does not require a change
  of topic. This is the main limit of the test, and the reason why
  `coherence` does not estimate boundaries.
* **Drift.** A drift that runs through the whole sequence is reported in at
  most 2% of cases, and a topic followed by a drift over the second half of
  the sequence in at most 5%. This is consistent with H-SEG-2.
* **Transitions.** Between two topics of 24 units, a transition of 6 units
  is reported like an abrupt change, in 97% to 100% of cases, and one of 12
  units, the size of the largest block, in 80% to 100%. A transition of 24
  units is reported in 15% to 72% of cases, the more often the more distinct
  the topics, and one of 48 units in at most 3%. This is the sense in which
  the test concerns abrupt changes (H-SEG-5): it does not tell a transition
  shorter than its largest block from a change at one point.
* **Abrupt changes between segments of at least six units.** In setting A,
  the boundaries are all found at the right place in 93% to 98% of cases,
  for two or three topics, whether the segments are equal or not and
  wherever the boundary lies. In setting B the rate is 65% to 96%. It is
  lowest for the shortest segments and for a boundary close to one end of a
  longer sequence: 78% for 6 + 6 units, 72% for 8 + 8 + 8, 75% for a passage
  of 8 units between two others of 10, and 65% for a boundary six units from
  the start of a sequence of 40 units. This is consistent with H-SEG-3.
* **Weak contrast.** In setting C, two segments are separated at the right
  place in 39% of cases when each has 6 units, 69% when each has 12 and 90%
  when each has 20. Three segments of 12 units are all delimited in 43% of
  cases. The rate does not go on rising with the length of the segments: it
  is 84% for two segments of 80 units.
* **Segments shorter than two blocks.** A passage of four units between two
  longer ones is delimited on both sides in at most 19% of cases, although
  at least one boundary is reported in 61% to 100% of them. Two segments of
  four units are separated in 22% to 53% of cases, and a boundary four units
  from the start of a sequence of 12 units in at most 8%.
* **A short passage before two topics.** Two topics of 6 units that follow a
  passage of 3 units on a third topic are separated in 98%, 72% and 24% of
  cases in settings A, B and C, against 98%, 78% and 39% when they stand
  alone. The passage, which is too short to be delimited, costs nothing when
  the topics are distinct and part of the detections when they are close.
* **Block size and short sequences.** With a block size of 2, two segments
  of four units are separated in 87%, 46% and 22% of cases in settings A, B
  and C, against 53%, 38% and 22% with a block size of 3, and a boundary
  four units from the start of a sequence of 12 units in 87%, 50% and 16%,
  against at most 8%. Segments of five and of six units are separated less
  often than with a block size of 3: 75%, 39% and 18% against 80%, 57% and
  28% for 5 + 5 units, and 91%, 68% and 34% against 98%, 78% and 39% for 6 +
  6. For the sequences of 12 units without a boundary, the two block sizes
  give rates of the same order: 10% to 14% for an unrelated unit in the
  middle, 27% to 44% for two adjacent ones, at most 3% for a drift.
* **Block size and longer sequences.** With a block size of 6 the blocks are
  twice as large. Two segments of 80 units in setting C are then separated
  at the right place in 92% of cases instead of 84%, and a boundary is
  reported in 100% of cases instead of 93%. Transitions of 24 and 48 units
  are reported in 56% to 100% of cases, against 0% to 72% with the default
  block size, and a topic followed by a drift over 80 units in up to 6% of
  cases, against 0%. The block size sets the scale at which a change counts
  as abrupt.
* **Unrelated units.** A single unrelated unit in the middle of a sequence
  of 24 units leads to a reported boundary in 12% to 15% of cases, against a
  nominal 5%. Placed at a random position, where every order is equally
  likely, it does so in 4.9% to 5.0% of cases, as the level of the test
  requires: the excess in the middle of the sequence is made up for at other
  positions. Two adjacent unrelated units on one other subject lead to a
  reported boundary in 45% to 67% of cases. The test does not tell a
  digression of two units from the beginning of a new topic. Three unrelated
  units scattered through the sequence lead to a reported boundary in at
  most 2% of cases.
* **Further boundaries.** More boundaries than were planted are reported in
  0% to 6% of cases, depending on the sequence.
* **The two similarities: level.** The three versions compared are
  permutation tests of the same construction, which keeps each of them to
  the level. For one topic of 24 units they report a boundary in 4.6% to
  5.2% of 5,000 sequences, and each of the nine 95% intervals contains 5%.
* **The two similarities: what each finds.** Measured on the cosine between
  the sums of the blocks alone, the valley is reported in at most 2% of the
  sequences that drift, and in 8% to 10% of those with an unrelated unit in
  the middle. It finds 2 to 27 points fewer changes of topic than the test
  of the library in settings A and B, and it cannot tell the ordered text
  from other arrangements when the units of a topic are identical (see
  above). Measured on the mean similarity between units alone, the valley
  finds up to 14 points more changes, and at most one fewer, and is reported
  in up to 9% of the sequences that keep to one topic and then drift,
  against 5%, and in up to 21% of those with an unrelated unit in the
  middle, against 15%. When consecutive units are related, the three
  versions report a boundary in 10% to 12%, 2% to 5% and 18% to 19% of
  cases. The sum of the two depths was chosen before related consecutive
  units were examined. On them, the cosine between the sums alone does
  markedly better, close to the nominal level; on planted changes it does
  markedly worse. Whether a statistic can keep the power of the sum without
  its sensitivity to related consecutive units is left for further work.

These results describe the statistical procedure on synthetic vectors. They
do not establish how the test behaves on embeddings of natural language.

### How the statistic was chosen

The form of the statistic was settled by comparing variants on simulated
sequences of the kinds described above, before the results reported here
were produced. Five choices were examined: one block size for both sides of
a valley or one for each; the pairs of sizes admitted; the largest block;
the block similarity; and the valley whose depth is taken, the one at the
lowest point of the curves or the deepest.

The comparison of the block similarities is the one reported above. Two of
the other choices can be followed in the history of the repository, where
the output of the simulation is stored with each version of the test.

* In the first version, with one block size for both sides of a valley and
  the cosine between the sums of the blocks as the only similarity, a
  boundary six units from the start of a sequence of 24 units was found in
  16%, 11% and 6% of cases in settings A, B and C, on 200 sequences each,
  against 96%, 88% and 38% here.
* In the second version, the depth was taken at the gap where the curves
  were lowest together. Its simulation was run on other sequences of the
  same kinds. Of the 239 percentages that both outputs give, 222 differ by 3
  points or less; the largest difference, 10 points, is for two adjacent
  unrelated units in setting A (57% then 67%), a kind of sequence on which
  the deepest valley is reported more often than the valley at the lowest
  point: by 4 to 5 points when the two were compared on the same sequences.
  It was given up for two reasons. A low point at the edge of the curves,
  which has one side only, could hide a valley elsewhere: for two topics of
  6 units after a passage of 3 units, the boundary was found in about half
  of the sequences of setting A, against 98% here. And when units repeat,
  the choice between equally low gaps depended on the rounding of the
  similarities and on the direction of reading.

The remaining comparisons, and the figure for the passage of 3 units in the
second version, come from runs that are not part of the repository: with
pairs of sizes of any ratio, a sequence of 60 units that keeps to one topic
and then drifts was reported in up to 39% of cases; with blocks as large as
a quarter of the sequence whatever its length, a transition spread over a
third of a sequence of 100 units was reported every time.

The tables above were produced after the statistic was fixed. Their
sequences are of the same kinds as those on which the variants were
compared, and in part the same sequences, so that they may flatter the test
somewhat.

### Checks in the test suite

The test suite repeats reduced versions of these simulations at each run:
the rate of reported boundaries on 200 sequences of independent directions,
on 60 sequences of each of four kinds in which no boundary should be
reported, three of them with every order equally likely, and on 30 sequences
with a slow transition between two topics; the detection rate on 50
sequences with two distinct topics of equal length, on 30 sequences each
with a boundary six units from the start and six units from the end, and on
30 sequences in which a passage of three units comes before two topics of
six. With so few sequences these checks can reveal a gross error, not a
small excess over the nominal level. One further check keeps the main limit
of the test in view: on 40 sequences of 48 units on one topic whose
consecutive units share half of their scatter, a boundary is reported in 17,
against 3 when the scatter of each unit is drawn on its own. Two p-values
are pinned, so that a change in the stream of permutations would very
probably be noticed.
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
| The adapter returns one vector per unit, is deterministic, is unaffected by batch size, and counts truncated units, the prompt of the model included | Tests run against the `sentence-transformers` library with a small model built locally with random weights. They pass with version 3.0.0, the oldest supported, and with version 6.1.0; the continuous integration workflow runs them with 3.0.0 and with the latest version. `tests/test_embedding.py` |
| Equal units have equal embeddings: each distinct unit is passed to the model once | With a stand-in that records what the model receives and numbers the rows it returns; with the library, the rows of a unit that occurs three times among others are equal to the last digit. `tests/test_embedding.py` |
| The checks of the arguments, the conversion of the output and the count of truncated units behave as documented | Tests that run without the library, with a stand-in for the model where one is needed. `tests/test_embedding.py` |

Passed to a model several times in one call, the same unit can receive
vectors that differ in their last digits, because it is not always batched
with the same units. `studies/repeated_unit_embeddings.py` counts how often
this happens with two small models with random weights, of the kind built
for the tests. A unit placed three times among 6 to 13 others of various
lengths, and encoded by the model in batches of four, received unequal
vectors in 27 and 25 of 200 inputs, with differences of up to 2.4 × 10^-7;
through the adapter, in none
([`studies/results/repeated_unit_embeddings.md`](../studies/results/repeated_unit_embeddings.md),
obtained with version 6.1.0 of the library; 30 and 25 of 200 with version
3.0.0). The counts depend on the platform. This is why the adapter passes
each distinct unit once: the test for topic boundaries treats units as the
same only when their embeddings are equal.

Three further tests load a published model,
`sentence-transformers/all-MiniLM-L6-v2`, at a fixed revision of its
repository. They check its dimensions and its token limit, the ordering of
the similarities of related and unrelated sentences, and the count of
truncated units. They need network access to download the model and are not
part of the default run (`pytest -m network tests/test_embedding.py`); the
continuous integration workflow runs them at each change.

## 6. Environments

The test suite passes on Linux (x86-64) with Python 3.10, 3.11, 3.12, 3.13
and 3.14, and with NumPy from 1.26 to 2.5. At each change, the continuous
integration workflow runs it on Linux, Windows and macOS with the same
versions of Python, and runs the tests of the embedding adapter on the three
systems. The badge at the top of the README shows the state of the latest
run.

## 7. Reproducibility

* Lexical diversity is deterministic and uses integer arithmetic up to the
  final division. Tokenisation is deterministic for a given version of
  Python: normalisation and case folding rely on the Unicode database of
  the interpreter, so that tokens can differ between Python versions in a
  text that contains recently added characters (see
  [metrics.md](metrics.md), section 1).
* The permutations of the test for boundaries are drawn from the raw stream
  of NumPy's PCG64 bit generator, which NumPy keeps stable across versions,
  so that a seed gives the same permutations everywhere.
* Levels and p-values are compared in exact arithmetic. The sums of
  similarities over blocks are exact as well: the similarities are made
  symmetric and equal for equal units, whatever the last digits of the
  product of matrices that gives them, then rounded to about 10^-12 and
  added as integers. A reordering that is equivalent to the observed order,
  for instance one that exchanges identical units, is therefore counted as
  such on every platform.
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
