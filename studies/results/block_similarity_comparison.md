# Test for topic boundaries: the two block similarities compared

Produced by `studies/block_similarity_comparison.py`. The sequences are simulated vectors, not text, and are those of `studies/boundary_test_simulation.py`.

Parameters: alpha 0.05, 199 permutations, 48 dimensions, block size 3. The settings A, B and C are those of the main simulation.

Software: discoursekit 0.1.0.dev0, NumPy 2.5.3, Python 3.13.16.

Three versions of the test. Both: the depth of the valley is measured on the two block similarities, as in the library. Sums: on the cosine similarity between the sums of the two blocks only. Units: on the mean similarity between the units of the two blocks only.

## Level

One topic, 24 units: every order of the units is equally likely, and the rate of reported boundaries should not exceed 5% in any version. Percentage of 5000 sequences in which a boundary is reported, with its 95% interval.

| Version | A | B | C |
|---|---:|---:|---:|
| Both | 5.0% (4.4% to 5.7%) | 5.3% (4.7% to 6.0%) | 5.4% (4.8% to 6.0%) |
| Sums | 4.9% (4.3% to 5.5%) | 5.4% (4.8% to 6.0%) | 5.3% (4.7% to 5.9%) |
| Units | 4.6% (4.1% to 5.2%) | 5.7% (5.1% to 6.3%) | 5.2% (4.6% to 5.9%) |

## Sequences without an abrupt change of topic

In this section and the next, each percentage rests on 500 sequences and is known to within about 4 points (95% interval, at its widest).

Percentage of sequences in which a boundary is reported:

| Sequence | Both, A | Both, B | Both, C | Sums, A | Sums, B | Sums, C | Units, A | Units, B | Units, C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 units, one unrelated unit in the middle | 16% | 14% | 13% | 10% | 10% | 11% | 22% | 17% | 13% |
| Drift through the whole sequence, 60 units | 0% | 1% | 1% | 0% | 0% | 0% | 1% | 4% | 3% |
| One topic, then a drift over the second half, 60 units | 0% | 3% | 6% | 0% | 1% | 2% | 2% | 5% | 12% |

## Sequences with planted changes of topic

Percentage of sequences in which the number of boundaries is right and each lies within one unit of its true position.

| Sequence | Both, A | Both, B | Both, C | Sums, A | Sums, B | Sums, C | Units, A | Units, B | Units, C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Two topics, 6 + 6 units | 98% | 82% | 40% | 88% | 66% | 33% | 100% | 90% | 43% |
| Two topics, 6 + 18 units | 96% | 87% | 36% | 96% | 81% | 26% | 97% | 89% | 37% |
| Three topics, 8 + 8 + 8 units | 97% | 77% | 15% | 75% | 49% | 10% | 97% | 86% | 21% |
| Excursion and return, 10 + 8 + 10 units | 96% | 73% | 20% | 82% | 46% | 12% | 97% | 80% | 23% |
