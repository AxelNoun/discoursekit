# Test for topic boundaries: results on simulated sequences

Produced by `studies/boundary_test_simulation.py`. The sequences are simulated vectors, not text: see the documentation of the script.

Parameters: alpha 0.05, 199 permutations, 48 dimensions, block size 3 unless stated otherwise.

Software: discoursekit 0.1.0.dev0, NumPy 2.5.3, Python 3.13.16.

## Settings

Mean cosine similarity between two units, within a topic and between two topics.

| Setting | Within a topic | Between topics |
|---|---:|---:|
| A. Distinct topics, moderate scatter | 0.50 | 0.00 |
| B. Related topics, larger scatter | 0.37 | 0.09 |
| C. Closely related topics, large scatter | 0.28 | 0.15 |

## Level

In the sequences of this section every order of the units is equally likely. The rate of reported boundaries should not exceed 5%.

Directions drawn independently and uniformly.

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

Sequences of 24 units that have a structure but whose order is random.

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

## Unrelated units

One topic interrupted by unrelated units. Two adjacent unrelated units are on one other subject; scattered ones are each on a subject of their own. In this section and the following ones, each percentage rests on 500 sequences and is known to within about 4 points (95% interval, at its widest).

Percentage of sequences in which a boundary is reported:

| Sequence | A | B | C |
|---|---:|---:|---:|
| 24 units, one unrelated unit in the middle | 16% | 14% | 13% |
| 24 units, two adjacent unrelated units in the middle | 57% | 53% | 41% |
| 24 units, three scattered unrelated units | 0% | 0% | 0% |

## Gradual change

The subject moves by equal steps from one topic to an unrelated direction, through the whole sequence or through its second half.

Percentage of sequences in which a boundary is reported:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Drift through the whole sequence, 24 units | 1% | 1% | 4% |
| Drift through the whole sequence, 60 units | 0% | 1% | 1% |
| Drift through the whole sequence, 160 units | 0% | 0% | 0% |
| One topic, then a drift over the second half, 24 units | 1% | 4% | 6% |
| One topic, then a drift over the second half, 60 units | 0% | 3% | 6% |
| One topic, then a drift over the second half, 160 units | 0% | 0% | 0% |

Two topics of 24 units each, joined by a transition during which the subject moves by equal steps from the first topic to the second. The first row has no transition.

Percentage of sequences in which a boundary is reported:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Two topics, 24 + 24 units | 100% | 100% | 99% |
| Transition of 6 units | 100% | 100% | 95% |
| Transition of 12 units | 100% | 100% | 79% |
| Transition of 24 units | 74% | 41% | 14% |
| Transition of 48 units | 0% | 0% | 2% |

## Abrupt changes of topic

Percentage of sequences in which the number of boundaries is right and each lies within one unit of its true position:

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

Percentage of sequences in which at least one boundary is reported:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Two topics, 6 + 6 units | 98% | 82% | 41% |
| Two topics, 8 + 8 units | 100% | 93% | 54% |
| Two topics, 12 + 12 units | 100% | 100% | 79% |
| Two topics, 20 + 20 units | 100% | 100% | 99% |
| Two topics, 80 + 80 units | 100% | 100% | 95% |
| Two topics, 6 + 18 units | 100% | 90% | 37% |
| Two topics, 6 + 34 units | 98% | 68% | 18% |
| Three topics, 8 + 8 + 8 units | 100% | 97% | 56% |
| Three topics, 12 + 12 + 12 units | 100% | 100% | 84% |
| Excursion and return, 10 + 4 + 10 units | 100% | 100% | 62% |
| Excursion and return, 10 + 8 + 10 units | 100% | 100% | 75% |
| Excursion and return, 16 + 8 + 16 units | 100% | 100% | 97% |

Percentage of sequences in which more boundaries are reported than were planted:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Two topics, 6 + 6 units | 0% | 0% | 0% |
| Two topics, 8 + 8 units | 4% | 5% | 2% |
| Two topics, 12 + 12 units | 4% | 3% | 2% |
| Two topics, 20 + 20 units | 4% | 3% | 4% |
| Two topics, 80 + 80 units | 4% | 5% | 4% |
| Two topics, 6 + 18 units | 4% | 3% | 1% |
| Two topics, 6 + 34 units | 4% | 4% | 0% |
| Three topics, 8 + 8 + 8 units | 3% | 2% | 1% |
| Three topics, 12 + 12 + 12 units | 4% | 4% | 2% |
| Excursion and return, 10 + 4 + 10 units | 1% | 0% | 0% |
| Excursion and return, 10 + 8 + 10 units | 3% | 3% | 1% |
| Excursion and return, 16 + 8 + 16 units | 5% | 3% | 1% |

## Short sequences and block size

Sequences of 8 to 16 units, examined with blocks of three units (the default) and of two. The same sequences are used for both.

Percentage of sequences in which a boundary is reported (none is planted):

| Sequence | A, blocks of 3 | B, blocks of 3 | C, blocks of 3 | A, blocks of 2 | B, blocks of 2 | C, blocks of 2 |
|---|---:|---:|---:|---:|---:|---:|
| 12 units, one unrelated unit in the middle | 11% | 10% | 9% | 14% | 13% | 11% |
| 12 units, two adjacent unrelated units in the middle | 46% | 39% | 29% | 37% | 36% | 35% |
| Drift through the whole sequence, 12 units | 0% | 2% | 3% | 0% | 1% | 3% |

Percentage of sequences in which the number of boundaries is right and each lies within one unit of its true position:

| Sequence | A, blocks of 3 | B, blocks of 3 | C, blocks of 3 | A, blocks of 2 | B, blocks of 2 | C, blocks of 2 |
|---|---:|---:|---:|---:|---:|---:|
| Two topics, 4 + 4 units | 53% | 42% | 22% | 88% | 51% | 22% |
| Two topics, 6 + 6 units | 98% | 82% | 40% | 93% | 71% | 31% |
| Two topics, 4 + 8 units | 2% | 7% | 5% | 86% | 53% | 17% |
| Excursion and return, 6 + 4 + 6 units | 0% | 5% | 3% | 54% | 18% | 4% |

## Longer sequences and block size

Sequences of 72 to 160 units, examined with blocks of three units and more (the default) and with blocks of six units and more. The same sequences are used for both.

Percentage of sequences in which a boundary is reported:

| Sequence | A, blocks of 3 | B, blocks of 3 | C, blocks of 3 | A, blocks of 6 | B, blocks of 6 | C, blocks of 6 |
|---|---:|---:|---:|---:|---:|---:|
| Two topics, 80 + 80 units | 100% | 100% | 95% | 100% | 100% | 100% |
| Transition of 24 units | 74% | 41% | 14% | 100% | 100% | 89% |
| Transition of 48 units | 0% | 0% | 2% | 100% | 96% | 56% |
| One topic, then a drift over the second half, 160 units | 0% | 0% | 0% | 1% | 4% | 7% |

Two topics, 80 + 80 units: percentage of sequences in which one boundary is reported, within one unit of its true position:

| Sequence | A, blocks of 3 | B, blocks of 3 | C, blocks of 3 | A, blocks of 6 | B, blocks of 6 | C, blocks of 6 |
|---|---:|---:|---:|---:|---:|---:|
| Two topics, 80 + 80 units | 96% | 95% | 88% | 96% | 95% | 92% |
