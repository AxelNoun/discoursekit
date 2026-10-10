# Test for topic boundaries: results on simulated sequences

Produced by `studies/boundary_test_simulation.py`. The sequences are simulated vectors, not text: see the documentation of the script.

Parameters: alpha 0.05, 199 permutations, 48 dimensions, block size 3, and a seed of its own for each sequence, unless stated otherwise.

Software: discoursekit 0.1.0.dev0, NumPy 2.5.3, Python 3.13.16.

## Settings

Mean cosine similarity between two units, within a topic and between two topics.

| Setting | Within a topic | Between topics |
|---|---:|---:|
| A. Distinct topics, moderate scatter | 0.50 | 0.00 |
| B. Related topics, larger scatter | 0.37 | 0.09 |
| C. Closely related topics, large scatter | 0.28 | 0.15 |

## Level

In the sequences of this section every order of the units is equally likely. The rate of reported boundaries should not exceed 5%. The intervals are Wilson score intervals.

Directions drawn independently and uniformly.

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

Sequences of 24 units that have a structure but whose order is random.

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

## One seed for all the sequences

In the tables above each sequence is tested with a seed of its own. With one seed for all, sequences of the same length are compared with the same reorderings. The rate of reported boundaries over the sequences is then 5% on average over the seeds, and can be above or below it for a given seed.

Sequences of 8 independent directions:

| Reorderings | Seed | Sequences | A boundary is reported | 95% interval |
|---:|---|---:|---:|---:|
| 19 | One for each sequence | 20000 | 4.7% | 4.4% to 5.0% |
| 19 | 0 for all | 20000 | 6.3% | 6.0% to 6.6% |
| 19 | 3 for all | 20000 | 4.3% | 4.1% to 4.6% |
| 199 | One for each sequence | 20000 | 4.9% | 4.6% to 5.2% |
| 199 | 0 for all | 20000 | 5.1% | 4.8% to 5.4% |
| 999 | One for each sequence | 20000 | 5.0% | 4.7% to 5.3% |
| 999 | 0 for all | 20000 | 4.9% | 4.6% to 5.2% |

## Consecutive units related to each other

One topic, without any change, in which each unit shares part of its scatter with the next: two consecutive units are more alike than two other units, as in coherent text. The orders of the units are no longer equally likely, and the level of the test does not apply. The figure in brackets is the correlation between the scatters of two consecutive units; units further apart have independent scatters.

Mean cosine similarity between two consecutive units, and between two other units:

| Setting | Consecutive units (0.2) | Consecutive units (0.5) | Other units |
|---|---:|---:|---:|
| A | 0.60 | 0.75 | 0.50 |
| B | 0.50 | 0.69 | 0.37 |
| C | 0.42 | 0.64 | 0.28 |

Percentage of sequences in which a boundary is reported. In this section and the following ones, each percentage rests on 500 sequences and is known to within about 4 points (95% interval, at its widest).

| Sequence | A | B | C |
|---|---:|---:|---:|
| One topic, 24 units, consecutive units related (0.2) | 14% | 14% | 13% |
| One topic, 24 units, consecutive units related (0.5) | 11% | 12% | 10% |
| One topic, 48 units, consecutive units related (0.2) | 26% | 18% | 23% |
| One topic, 48 units, consecutive units related (0.5) | 48% | 42% | 35% |

## Unrelated units

One topic interrupted by unrelated units. Two adjacent unrelated units are on one other subject; scattered ones are each on a subject of their own.

Percentage of sequences in which a boundary is reported:

| Sequence | A | B | C |
|---|---:|---:|---:|
| 24 units, one unrelated unit in the middle | 15% | 14% | 12% |
| 24 units, two adjacent unrelated units in the middle | 67% | 53% | 45% |
| 24 units, three scattered unrelated units | 0% | 1% | 2% |

## Gradual change

The subject moves by equal steps from one topic to an unrelated direction, through the whole sequence or through its second half.

Percentage of sequences in which a boundary is reported:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Drift through the whole sequence, 24 units | 0% | 1% | 2% |
| Drift through the whole sequence, 60 units | 0% | 0% | 2% |
| Drift through the whole sequence, 160 units | 0% | 0% | 0% |
| One topic, then a drift over the second half, 24 units | 1% | 2% | 5% |
| One topic, then a drift over the second half, 60 units | 1% | 3% | 5% |
| One topic, then a drift over the second half, 160 units | 0% | 0% | 0% |

Two topics of 24 units each, joined by a transition during which the subject moves by equal steps from the first topic to the second. The first row has no transition.

Percentage of sequences in which a boundary is reported:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Two topics, 24 + 24 units | 100% | 100% | 99% |
| Transition of 6 units | 100% | 100% | 97% |
| Transition of 12 units | 100% | 100% | 80% |
| Transition of 24 units | 72% | 37% | 15% |
| Transition of 48 units | 0% | 0% | 3% |

## Abrupt changes of topic

An excursion and return is a passage on a second topic between two segments on the same first topic.

Percentage of sequences in which the number of boundaries is right and each lies within one unit of its true position:

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

Percentage of sequences in which more boundaries are reported than were planted:

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

## Short sequences and block size

Sequences of 8 to 16 units, examined with block size 3 (the default) and with block size 2. The same sequences are used for both.

Percentage of sequences in which a boundary is reported (none is planted):

| Sequence | A, block size 3 | B, block size 3 | C, block size 3 | A, block size 2 | B, block size 2 | C, block size 2 |
|---|---:|---:|---:|---:|---:|---:|
| 12 units, one unrelated unit in the middle | 14% | 10% | 11% | 11% | 13% | 14% |
| 12 units, two adjacent unrelated units in the middle | 44% | 39% | 27% | 41% | 39% | 36% |
| Drift through the whole sequence, 12 units | 1% | 2% | 3% | 0% | 1% | 3% |

Percentage of sequences in which the number of boundaries is right and each lies within one unit of its true position:

| Sequence | A, block size 3 | B, block size 3 | C, block size 3 | A, block size 2 | B, block size 2 | C, block size 2 |
|---|---:|---:|---:|---:|---:|---:|
| Two topics, 4 + 4 units | 53% | 38% | 22% | 87% | 46% | 22% |
| Two topics, 5 + 5 units | 80% | 57% | 28% | 75% | 39% | 18% |
| Two topics, 6 + 6 units | 98% | 78% | 39% | 91% | 68% | 34% |
| Two topics, 4 + 8 units | 2% | 8% | 5% | 87% | 50% | 16% |
| Excursion and return, 6 + 4 + 6 units | 1% | 5% | 2% | 56% | 24% | 4% |

Two topics of 6 units, alone and after a passage of 3 units on a third topic, which is too short to be delimited with block size 3. Percentage of sequences in which the boundary between the two topics is the only one reported, within one unit of its true position:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Two topics, 6 + 6 units | 98% | 78% | 39% |
| Two topics, 6 + 6 units, after a passage of 3 units | 98% | 72% | 24% |

## Longer sequences and block size

Sequences of 72 to 160 units, examined with block size 3 (the default) and with block size 6. The same sequences are used for both.

Percentage of sequences in which a boundary is reported:

| Sequence | A, block size 3 | B, block size 3 | C, block size 3 | A, block size 6 | B, block size 6 | C, block size 6 |
|---|---:|---:|---:|---:|---:|---:|
| Two topics, 80 + 80 units | 100% | 100% | 93% | 100% | 100% | 100% |
| Transition of 24 units | 72% | 37% | 15% | 100% | 100% | 90% |
| Transition of 48 units | 0% | 0% | 3% | 100% | 95% | 56% |
| One topic, then a drift over the second half, 160 units | 0% | 0% | 0% | 1% | 2% | 6% |

Two topics, 80 + 80 units: percentage of sequences in which one boundary is reported, within one unit of its true position:

| Sequence | A, block size 3 | B, block size 3 | C, block size 3 | A, block size 6 | B, block size 6 | C, block size 6 |
|---|---:|---:|---:|---:|---:|---:|
| Two topics, 80 + 80 units | 96% | 96% | 84% | 95% | 94% | 92% |
