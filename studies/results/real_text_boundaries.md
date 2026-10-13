# Test for topic boundaries: results on real texts

Produced by `studies/real_text_boundaries.py`. The corpora, the selection of the texts and their limits are described in the documentation of the script.

Parameters: alpha 0.05, block size 3, 199 reorderings, and a seed of its own for each text.

Embedding: sentence-transformers/all-MiniLM-L6-v2 at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41, sentence-transformers 6.1.0.

Software: discoursekit 0.1.0.dev0, NumPy 2.5.4, Python 3.12.15, Linux x86_64.

Data: Reuters-21578, Distribution 1.0, ApteMod version, and the polarity dataset v2.0 of Pang and Lee (2004), both as distributed with NLTK. The texts are not reproduced here.

Percentages are given with their 95% Wilson interval, in points.

## Texts

Mean cosine similarity between consecutive sentences, and between sentences two or more apart.

| Corpus | Sentences | Texts | Consecutive | Further apart |
|---|---:|---:|---:|---:|
| Reuters | 12 | 400 | 0.385 | 0.334 |
| Reuters | 24 | 137 | 0.369 | 0.300 |
| Reviews | 12 | 400 | 0.292 | 0.229 |
| Reviews | 24 | 400 | 0.288 | 0.215 |
| Reviews | 48 | 228 | 0.289 | 0.198 |

## Texts that keep to one subject

Percentage of texts in which a boundary is reported, with each scheme of reordering. The level of the test is 5%. In the last column the sentences of each text are in a random order and reordered one by one; there the level applies.

| Corpus | Sentences | One by one | Circular blocks of 2 | Circular blocks of 3 | Sentences in random order |
|---|---:|---:|---:|---:|---:|
| Reuters | 12 | 11.8% (9 to 15) | 8.2% (6 to 11) | 5.0% (3 to 8) | 5.8% (4 to 8) |
| Reuters | 24 | 19.7% (14 to 27) | 12.4% (8 to 19) | 10.2% (6 to 16) | 3.6% (2 to 8) |
| Reviews | 12 | 12.0% (9 to 16) | 9.0% (7 to 12) | 5.5% (4 to 8) | 6.2% (4 to 9) |
| Reviews | 24 | 31.2% (27 to 36) | 16.2% (13 to 20) | 13.8% (11 to 17) | 5.5% (4 to 8) |
| Reviews | 48 | 52.6% (46 to 59) | 38.6% (33 to 45) | 25.0% (20 to 31) | 2.6% (1 to 6) |

The difference between the two mean similarities of a text, consecutive sentences less sentences further apart, varies from text to text. Percentage of texts in which a boundary is reported (One by one) in the half of the texts whose difference is smaller, and in the other half:

| Corpus | Sentences | Smaller difference | Larger difference |
|---|---:|---:|---:|
| Reuters | 12 | 9.5% (6 to 14) | 14.0% (10 to 19) |
| Reuters | 24 | 16.2% (9 to 27) | 23.2% (15 to 34) |
| Reviews | 12 | 11.5% (8 to 17) | 12.5% (9 to 18) |
| Reviews | 24 | 24.5% (19 to 31) | 38.0% (32 to 45) |
| Reviews | 48 | 38.6% (30 to 48) | 66.7% (58 to 75) |

## Two texts joined

The first half of a text followed by the first half of another: two dispatches of different categories, or reviews of two films. Percentage of joined texts in which one boundary is reported, within one sentence of the join:

| Corpus | Sentences | Texts | One by one | Circular blocks of 2 | Circular blocks of 3 |
|---|---:|---:|---:|---:|---:|
| Reuters | 12 | 181 | 65.7% (59 to 72) | 48.6% (41 to 56) | 19.3% (14 to 26) |
| Reuters | 24 | 170 | 77.6% (71 to 83) | 69.4% (62 to 76) | 54.1% (47 to 61) |
| Reuters | 48 | 62 | 69.4% (57 to 79) | 77.4% (66 to 86) | 79.0% (67 to 87) |
| Reviews | 12 | 200 | 28.5% (23 to 35) | 22.0% (17 to 28) | 10.5% (7 to 16) |
| Reviews | 24 | 200 | 44.0% (37 to 51) | 24.5% (19 to 31) | 18.0% (13 to 24) |
| Reviews | 48 | 200 | 30.0% (24 to 37) | 32.0% (26 to 39) | 31.5% (25 to 38) |

Percentage of joined texts in which a boundary is reported within one sentence of the join, whatever else is reported:

| Corpus | Sentences | One by one | Circular blocks of 2 | Circular blocks of 3 |
|---|---:|---:|---:|---:|
| Reuters | 12 | 65.7% (59 to 72) | 48.6% (41 to 56) | 19.3% (14 to 26) |
| Reuters | 24 | 90.0% (85 to 94) | 77.1% (70 to 83) | 55.3% (48 to 63) |
| Reuters | 48 | 93.5% (85 to 97) | 87.1% (77 to 93) | 82.3% (71 to 90) |
| Reviews | 12 | 28.5% (23 to 35) | 22.0% (17 to 28) | 10.5% (7 to 16) |
| Reviews | 24 | 48.5% (42 to 55) | 27.0% (21 to 34) | 19.0% (14 to 25) |
| Reviews | 48 | 50.0% (43 to 57) | 40.0% (33 to 47) | 34.5% (28 to 41) |

Percentage of joined texts in which at least one boundary is reported:

| Corpus | Sentences | One by one | Circular blocks of 2 | Circular blocks of 3 |
|---|---:|---:|---:|---:|
| Reuters | 12 | 66.3% (59 to 73) | 48.6% (41 to 56) | 19.3% (14 to 26) |
| Reuters | 24 | 94.7% (90 to 97) | 80.0% (73 to 85) | 57.1% (50 to 64) |
| Reuters | 48 | 100.0% (94 to 100) | 91.9% (82 to 97) | 88.7% (78 to 94) |
| Reviews | 12 | 29.0% (23 to 36) | 22.0% (17 to 28) | 10.5% (7 to 16) |
| Reviews | 24 | 64.0% (57 to 70) | 36.5% (30 to 43) | 25.5% (20 to 32) |
| Reviews | 48 | 83.5% (78 to 88) | 67.5% (61 to 74) | 54.0% (47 to 61) |
