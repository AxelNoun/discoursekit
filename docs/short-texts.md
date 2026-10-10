# Short texts

Samples of speech collected for research are often short: a few sentences in
answer to a question, or a minute or two of free speech. This page states
what each measure of the library needs, what precision can be expected when
a text has few units, and what the test for topic boundaries can and cannot
establish at that length. Every figure comes from a definition, from a test
of this repository or from the simulation reported in
[validation.md](validation.md).

## 1. Smallest input for each quantity

| Quantity | Smallest input | Below it |
|---|---|---|
| Tokens | None | An empty list |
| `mattr` | As many tokens as the window | NaN |
| `local_coherence`, `global_coherence` | 2 units | NaN |
| `local_coherence_shuffled_sd` | 2 units; it is zero with exactly 2, whose two orders have the same pair of consecutive units | NaN |
| `reference_coherence` | 1 unit and a reference vector | NaN |
| Estimated topic boundaries | 2 × `block_size` + 2 units, that is, 8 by default | No test; the field `tested` is false |
| Coherence within segments | One segment of 2 units | Segments of one unit are left out |

A value that exists is not thereby precise. The following sections say what
to expect.

## 2. Lexical diversity

The moving-average type-token ratio is defined for a text with at least as
many tokens as the window. To compare short texts, the window has to be no
longer than the shortest of them, and values obtained with different windows
are not comparable (see [metrics.md](metrics.md), section 2). Fergadiotis et
al. (2013) set the window to the length of their shortest sample, 17 tokens.
Texts shorter than the chosen window have no value and are reported as
excluded.

A text exactly as long as the window has a single window, and its value is
the plain type-token ratio of that text. The field `n_windows` says how many
windows were averaged.

## 3. Coherence with few units

The example below draws 200 texts on a single topic at each of four lengths.
A unit is the direction of the topic plus scatter, in 48 dimensions, so that
two units have a cosine similarity of about 0.5. For each length it prints
the number of units and the mean, over the texts, of five quantities: local
coherence, its reference level (the mean similarity over all pairs of
units), the standard deviation of local coherence over the orders of the
units, global coherence, and reference coherence with the direction of the
topic as reference.

```python
>>> import numpy as np
>>> from discoursekit import coherence

>>> state = np.random.RandomState(0)
>>> topic = state.standard_normal(48)
>>> topic /= np.linalg.norm(topic)
>>> for n_units in (4, 8, 16, 32):
...     rows = []
...     for _ in range(200):
...         scatter = state.standard_normal((n_units, 48)) / np.sqrt(48)
...         result = coherence(topic + scatter, reference=topic, boundaries=())
...         rows.append(
...             (
...                 result.local_coherence,
...                 result.local_coherence_shuffled,
...                 result.local_coherence_shuffled_sd,
...                 result.global_coherence,
...                 result.reference_coherence,
...             )
...         )
...     print(n_units, *(f"{value:.2f}" for value in np.mean(rows, axis=0)))
4 0.50 0.50 0.03 0.62 0.71
8 0.50 0.50 0.02 0.66 0.71
16 0.50 0.50 0.02 0.69 0.71
32 0.50 0.50 0.01 0.70 0.71

```

The texts are of the same kind at every length, and the quantities do not
behave alike.

* **Local coherence and its reference level** have the same mean at every
  length. The value of local coherence for one text is less stable when the
  text is short: it is the mean of the similarities between consecutive
  units, four of them for a text of five units, and the order of the units
  alone moves it. The standard deviation over orders,
  `local_coherence_shuffled_sd`, states by how much: 0.03 for four units in
  this example, 0.01 for 32. The figure depends on the embedding; it grows
  with the scatter of the similarities between pairs of units.
* **Global coherence increases with the number of units**, from 0.62 to
  0.70 here, although nothing else differs between the texts. Each unit is
  compared with the centroid of the others, which is a noisier estimate of
  the subject when there are fewer of them. When every pair of units has
  the same similarity *r*, global coherence equals
  √(*n* - 1) *r* / √(1 + (*n* - 2) *r*) for *n* units: it starts at *r* for
  two units and tends to √*r*. Texts of different lengths should not be
  compared on this quantity without taking their length into account. The
  mean similarity over all pairs of units, returned as
  `local_coherence_shuffled`, also says how much the units of a text have in
  common, and has the same mean at every length.
* **Reference coherence** has the same mean at every length. Each unit is
  compared with a vector that does not come from the text, here the
  direction of the topic and in practice, for example, the embedding of the
  question asked. It is defined from one unit on.

For a study of short texts, this suggests the following.

* Report local coherence with its reference level, the standard deviation
  over orders, and the number of units.
* The difference between local coherence and its reference level, divided
  by the standard deviation over orders, has mean zero and standard
  deviation one when the order is random, whatever the number of units. It
  puts texts of different lengths on one scale. It is undefined when the
  standard deviation is zero, and its distribution is not normal with few
  units.
* Where the elicitation procedure provides a reference, such as a question,
  use reference coherence.
* One short text supports no conclusion. The precision of a comparison
  between groups or conditions comes from the number of texts, and the
  number of units of each text belongs among the covariates.

## 4. Changes of topic

### Supply the boundaries when they are known

When the elicitation procedure defines where the subject changes, for
example a series of questions, pass these positions to `coherence` with the
argument `boundaries`. No test is needed, coherence within segments is then
defined for texts of any length, and the selection effect described in
[metrics.md](metrics.md), section 7, does not arise.

### What the order of a few units can establish

The test for topic boundaries decides from the order of the units alone.
This sets a limit that no such test can pass. Take a text of *n* units, *a*
on one topic followed by *n* - *a* on another, the units of a topic being
interchangeable: any order of them is as likely as any other. Among all the
orders of the *n* units, those that keep the two topics apart, in either
direction, are 2 in every C(*n*, *a*), where C(*n*, *a*) is the number of
ways of choosing *a* units among *n*. A test that reports a boundary in at
most a proportion `alpha` of the orders of any set of units, and that treats
a text and the same text read backwards alike, therefore reports this
boundary with a probability of at most `alpha` × C(*n*, *a*) / 2.

| Units | C(*n*, *a*) | Upper limit of the probability of detection at `alpha` = 0.05 |
|---|---:|---:|
| 3 + 3 | 20 | 50% |
| 3 + 4 | 35 | 87.5% |
| 4 + 4 | 70 | No limit below 100% |

With fewer than eight units the probability of detection is thus bounded
below one however distinct the two topics are: at most 87.5% with seven
units and 50% with six. This holds for this test and for any other of the
same kind.

The test has a further limit of its own, which comes from its blocks: one
side of a valley is made of two blocks, and both fit within a topic only if
the topic has twice as many units as a block. For two groups of identical
units the statistic can be computed for every arrangement of the units,
which gives the smallest p-value that the ordered text can receive (see
[validation.md](validation.md), section 4).

| Text | Block size | Smallest possible p-value |
|---|---:|---:|
| 3 + 3 units | 2 | 0.200 |
| 4 + 4 units | 2 | 0.029 |
| 4 + 4 units | 3 | 0.057 |
| 5 + 5 units | 3 | 0.008 |
| 6 + 6 units | 3 | 0.002 |

With the default blocks of three units, two groups of four identical units
cannot be told apart at the 5% level, because two other arrangements of the
same units reach the statistic of the ordered text. With blocks of two units
they can. When the units of a topic differ from one another, as they do in a
text, the comparison between arrangements is no longer the same, and the
ordered text is found in some cases, as the simulation shows.

### What the test finds in texts of 8 to 16 units

The figures below are those of the simulation reported in
[validation.md](validation.md), section 4, for three settings that go from
distinct topics to closely related ones. With the default blocks of three
units:

* two topics of six units each are separated at the right place in 98%, 82%
  and 40% of cases;
* two topics of four units each in 53%, 42% and 22%;
* a boundary four units from the start of a text of 12 units in at most 7%.

With blocks of two units (`block_size=2`), segments of four units are found
far more often: in 88%, 51% and 22% of cases for 4 + 4 units, and in 86%,
53% and 17% for 4 + 8 units. Two topics of six units are then separated a
little less often, in 93%, 71% and 31% of cases.

Where no boundary should be reported, the two block sizes give rates of the
same order. One unrelated unit in the middle of a text of 12 units is
reported as a boundary in 9% to 14% of cases, two adjacent unrelated units
in 29% to 46%, and a drift through the 12 units in at most 3%.

For texts of this length, then: the test finds a clear change between two
topics of at least six units; for topics of four or five units, blocks of
two units find more of the changes and still miss many unless the topics
are very distinct; and a digression of two units is often reported as a
boundary. When the positions of the changes are known, supply them.

## 5. References

Fergadiotis, G., Wright, H. H., & West, T. M. (2013). Measuring lexical
diversity in narrative discourse of people with aphasia. *American Journal
of Speech-Language Pathology*, 22(2), S397-S408.
https://doi.org/10.1044/1058-0360(2013/12-0083)
