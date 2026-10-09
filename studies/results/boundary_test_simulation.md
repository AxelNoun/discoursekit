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
| 8 units, the fewest that can be tested | 5000 | 5.0% | 4.5% to 5.7% |
| 16 units | 5000 | 5.2% | 4.6% to 5.9% |
| 24 units | 5000 | 5.1% | 4.6% to 5.8% |
| 40 units | 5000 | 5.3% | 4.7% to 6.0% |
| 12 units, block size 1 | 5000 | 4.9% | 4.4% to 5.6% |
| All of the above | 25000 | 5.1% | 4.9% to 5.4% |

Sequences of 24 units that have a structure but whose order is random.

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

## Content that varies without a change of topic

Sequences of 24 units. No boundary should be reported. In this section and the next, each percentage rests on 200 sequences and is known to within about 7 points (95% interval, at its widest).

Percentage of sequences in which a boundary is reported:

| Sequence | A | B | C |
|---|---:|---:|---:|
| One topic, one unrelated unit in the middle | 17% | 12% | 9% |
| One topic, three scattered unrelated units | 0% | 0% | 2% |
| Gradual drift to an unrelated subject | 0% | 1% | 2% |

## Planted changes of topic

Percentage of sequences in which the number of boundaries is right and each lies within one unit of its true position:

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

Percentage of sequences in which at least one boundary is reported:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Two topics, 6 + 6 units | 87% | 62% | 32% |
| Two topics, 8 + 8 units | 98% | 86% | 47% |
| Two topics, 12 + 12 units | 100% | 98% | 72% |
| Two topics, 20 + 20 units | 100% | 100% | 96% |
| Two topics, 6 + 18 units | 16% | 13% | 7% |
| Three topics, 8 + 8 + 8 units | 84% | 70% | 39% |
| Three topics, 12 + 12 + 12 units | 100% | 96% | 64% |
| Excursion and return, 10 + 4 + 10 units | 100% | 100% | 60% |
| Excursion and return, 10 + 8 + 10 units | 100% | 98% | 68% |

Percentage of sequences in which more boundaries are reported than were planted:

| Sequence | A | B | C |
|---|---:|---:|---:|
| Two topics, 6 + 6 units | 0% | 0% | 0% |
| Two topics, 8 + 8 units | 2% | 4% | 2% |
| Two topics, 12 + 12 units | 4% | 4% | 2% |
| Two topics, 20 + 20 units | 3% | 6% | 6% |
| Two topics, 6 + 18 units | 0% | 2% | 0% |
| Three topics, 8 + 8 + 8 units | 4% | 2% | 0% |
| Three topics, 12 + 12 + 12 units | 6% | 4% | 2% |
| Excursion and return, 10 + 4 + 10 units | 0% | 1% | 0% |
| Excursion and return, 10 + 8 + 10 units | 3% | 3% | 0% |
