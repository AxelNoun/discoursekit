# Test for topic boundaries: results on real texts

Produced by `studies/real_text_boundaries.py`. The corpora, the selection of the texts and their limits are described in the documentation of the script.

Parameters: alpha 0.05, block size 3, 199 reorderings, and a seed of its own for each text.

Embedding: mean of the en_core_web_md 3.8.0 vectors of the content words, spaCy 3.8.16.

Software: discoursekit 0.1.0.dev0, NumPy 2.5.4, Python 3.12.3, Linux x86_64.

Data: Reuters-21578, Distribution 1.0, ApteMod version, and the polarity dataset v2.0 of Pang and Lee (2004), both as distributed with NLTK. The texts are not reproduced here.

Percentages are given with their 95% Wilson interval, in points.

## Texts

Mean cosine similarity between consecutive sentences, and between sentences two or more apart.

| Corpus | Sentences | Texts | Consecutive | Further apart |
|---|---:|---:|---:|---:|
| Reuters | 12 | 400 | 0.736 | 0.719 |
| Reuters | 24 | 137 | 0.722 | 0.702 |
| Reviews | 12 | 398 | 0.647 | 0.630 |
| Reviews | 24 | 392 | 0.638 | 0.618 |
| Reviews | 48 | 216 | 0.627 | 0.602 |

## Texts that keep to one subject

Percentage of texts in which a boundary is reported, with each scheme of reordering. The level of the test is 5%. In the last column the sentences of each text are in a random order and reordered one by one; there the level applies.

| Corpus | Sentences | One by one | Circular blocks of 2 | Circular blocks of 3 | Sentences in random order |
|---|---:|---:|---:|---:|---:|
| Reuters | 12 | 11.8% (9 to 15) | 9.0% (7 to 12) | 6.5% (4 to 9) | 6.5% (4 to 9) |
| Reuters | 24 | 15.3% (10 to 22) | 8.8% (5 to 15) | 3.6% (2 to 8) | 2.2% (1 to 6) |
| Reviews | 12 | 7.3% (5 to 10) | 5.3% (3 to 8) | 3.8% (2 to 6) | 6.5% (4 to 9) |
| Reviews | 24 | 12.2% (9 to 16) | 7.9% (6 to 11) | 6.6% (5 to 10) | 5.1% (3 to 8) |
| Reviews | 48 | 28.7% (23 to 35) | 19.4% (15 to 25) | 13.4% (10 to 19) | 5.1% (3 to 9) |

## Two texts joined

The first half of a text followed by the first half of another: two dispatches of different categories, or reviews of two films. Percentage of joined texts in which one boundary is reported, within one sentence of the join:

| Corpus | Sentences | Texts | One by one | Circular blocks of 2 | Circular blocks of 3 |
|---|---:|---:|---:|---:|---:|
| Reuters | 12 | 181 | 34.8% (28 to 42) | 22.7% (17 to 29) | 9.9% (6 to 15) |
| Reuters | 24 | 170 | 28.8% (23 to 36) | 19.4% (14 to 26) | 11.8% (8 to 17) |
| Reuters | 48 | 62 | 32.3% (22 to 45) | 24.2% (15 to 36) | 21.0% (13 to 33) |
| Reviews | 12 | 199 | 10.1% (7 to 15) | 7.0% (4 to 11) | 5.0% (3 to 9) |
| Reviews | 24 | 199 | 8.5% (5 to 13) | 4.0% (2 to 8) | 5.0% (3 to 9) |
| Reviews | 48 | 190 | 12.1% (8 to 18) | 7.9% (5 to 13) | 7.4% (4 to 12) |

Percentage of joined texts in which at least one boundary is reported:

| Corpus | Sentences | One by one | Circular blocks of 2 | Circular blocks of 3 |
|---|---:|---:|---:|---:|
| Reuters | 12 | 35.4% (29 to 43) | 23.2% (18 to 30) | 9.9% (6 to 15) |
| Reuters | 24 | 40.6% (33 to 48) | 24.7% (19 to 32) | 12.9% (9 to 19) |
| Reuters | 48 | 71.0% (59 to 81) | 50.0% (38 to 62) | 38.7% (28 to 51) |
| Reviews | 12 | 12.1% (8 to 17) | 9.0% (6 to 14) | 6.0% (3 to 10) |
| Reviews | 24 | 22.6% (17 to 29) | 13.1% (9 to 18) | 14.6% (10 to 20) |
| Reviews | 48 | 28.9% (23 to 36) | 20.5% (15 to 27) | 18.9% (14 to 25) |
