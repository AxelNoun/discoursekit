# Registry of measures

Every measure computed by `discoursekit` has an entry here. An entry states
what is computed, why it is in the library, what is known to distort it, and
what must be reported alongside it. The counts and parameters returned with a
measure, such as the number of tokens or of units, are described in the
docstrings.

Two kinds of hypothesis are kept apart.

* A **measurement hypothesis** is a claim about the behaviour of the measure
  itself, for example that it does not depend on the length of the text.
  Measurement hypotheses are tested in this repository;
  [validation.md](validation.md) reports how and with what result.
* A **motivating hypothesis** is the substantive reason the measure may
  interest researchers. Motivating hypotheses are not tested in this
  repository, and nothing here should be read as evidence for them.

Contents:

1. [Tokens](#1-tokens)
2. [Lexical diversity: `mattr`](#2-lexical-diversity-mattr)
3. [Local coherence](#3-local-coherence)
4. [Global coherence](#4-global-coherence)
5. [Reference coherence](#5-reference-coherence)
6. [Topic boundaries](#6-topic-boundaries)
7. [Coherence within segments](#7-coherence-within-segments)
8. [Planned measures](#8-planned-measures)
9. [References](#9-references)

What the measures need and how precise they are when a text has few units
is the subject of [short-texts.md](short-texts.md).

---

## 1. Tokens

Function `tokenize`, module `discoursekit.tokenization`.

### Definition

A token is a word as delimited by the default word boundary rules of Unicode
Standard Annex #29 (UAX #29), provided that it contains at least one letter
or digit. The rules use the character properties of Unicode 18.0.0, which
are shipped with the library.

The rules are applied with one tailoring: U+202F NARROW NO-BREAK SPACE is
treated as U+00A0 NO-BREAK SPACE. UAX #29 presents its rules as defaults
that implementations may tailor to particular languages and environments.
It gives U+202F the class of the underscore (ExtendNumLet), which joins the
character to the letters and digits on either side. In running text this
character is a space. Since release 42, the Unicode Common Locale Data
Repository (CLDR) places it before AM and PM in English time formats, and
French number formats use it between groups of digits. Under the default
rules, `3:45 PM` written in that way contains the word `45 PM`. With the
tailoring, white space separates tokens in every case, and a text gives the
same tokens whichever space character it is written with. The same word
boundaries are obtained from any implementation of the default rules by
replacing U+202F with U+00A0 beforehand.

Before segmentation the text is converted to Normalization Form C. Each
token is then normalised: the single quotation marks U+2019 and U+2018 and
the fullwidth apostrophe U+FF07, which UAX #29 treats like the apostrophe
inside a word, are replaced by U+0027; format characters (general category
Cf) are removed; and, unless disabled, the token is case folded in the way
the Unicode Standard specifies for caseless comparison (decomposition, case
folding, normalisation).

For English, the rules mean that `don't`, `o'clock`, `e.g`, `3.14`, `1,000`
and `snake_case` are single tokens, and that a hyphen ends a word:
`well-known` gives `well` and `known`. UAX #29 leaves the hyphen out of the
default definition of a word on purpose.

### Measurement hypotheses

**H-TOK-1.** The segmentation conforms to UAX #29. Tested against the
conformance file published with Unicode 18.0.0: all 1,944 cases pass.

**H-TOK-2.** Every token contains a letter or digit, contains no white
space and no format character, and is in Normalization Form C. Tested on
random strings.

**H-TOK-3.** White space separates tokens: the tokens of two texts joined by
a white space character are the tokens of the first followed by the tokens
of the second. Tested on random strings, for eleven kinds of white space.

### Limits

* Tokens are word forms, not lemmas. `walk` and `walked` are two types.
* The default rules are language independent. In languages with elision
  they keep the clitic attached (`l'homme` is one token); UAX #29 notes that
  French and Italian may be better served by tailored rules. Token counts
  are not comparable across languages.
* A full stop or a colon typed between two letters without a space joins
  them (`end.Next`). Correct such errors before tokenisation.
* A number written with spaces between groups of digits (`10 000`) gives one
  token per group. UAX #29 mentions a tailoring that keeps such numbers
  whole. It is not applied here, because it would also join two separate
  numbers that follow each other.
* Nothing is removed by default. Filled pauses, false starts, transcription
  codes and speaker labels are tokens unless the study removes them, with
  the `exclude` argument or beforehand.
* Normalisation, case folding and the test for a letter or digit rely on the
  Unicode database of the Python interpreter (Unicode 13.0 for Python 3.10,
  16.0 for Python 3.14), which is older than the one used for the word
  boundaries. Tokens can differ between Python versions in a text that
  contains characters added to Unicode after the older of the two versions.

### What to report

Library version; whether case folding was applied; the list of excluded
tokens; any preprocessing applied to the text before tokenisation.

---

## 2. Lexical diversity: `mattr`

Functions `mattr` and `lexical_diversity`, module `discoursekit.lexical`.

### Definition

For a sequence of *N* tokens and a window of *W* tokens (*W* ≤ *N*), the
type-token ratio (number of distinct tokens divided by *W*) is computed in
each of the *N* - *W* + 1 windows obtained by sliding the window one token at
a time, and the values are averaged (Covington & McFall, 2010). The result
lies between 1/*W* and 1. It is undefined when *N* < *W*: the library then
returns NaN and never substitutes the plain type-token ratio.

### Measurement hypothesis

**H-LEX-1.** For text produced by a fixed generating process, the
moving-average type-token ratio does not change systematically with the
length of the text, whereas the plain type-token ratio decreases.

Tested on synthetic text drawn from a Zipf-like distribution. The hypothesis
remains to be tested on natural text, in the robustness study of the
roadmap: synthetic text has no topic structure and no repetition across
sentences.

### Motivating hypothesis

Reduced lexical variety is one of the ways in which discourse can be
impoverished or repetitive. Whether the measure tracks any construct of
clinical interest is a question for a study designed for that purpose.

### Known confounds and limits

* **Window size.** Values obtained with different windows are not
  comparable, and there is no standard window. Windows of 50 and 100 tokens
  are both in use; Fergadiotis et al. (2013) used 17, the length of their
  shortest sample. The default of this library, 50, is a convention.
* **Short texts.** A text shorter than the window has no value. With few
  windows the estimate rests on little data. Zenker and Kyle (2021) examine
  minimum text lengths for lexical diversity indices.
* **Tokenisation.** See section 1. In particular, inflected forms count as
  different types.
* **Transcription conventions.** Whether filled pauses, repetitions and
  cut-off words are kept changes the value.
* **Topic, genre and task.** A narrow topic or a constrained task lowers
  lexical diversity independently of the speaker.
* **Speaker background.** Palaniyappan (2021) reviews evidence that
  linguistic measures, lexical diversity among them, vary with socioeconomic
  factors, education and language background. A difference between groups
  or individuals cannot be attributed to a single cause without measuring
  these factors.

### What to report

Window size; tokenisation options; number of tokens of each text; number of
texts excluded for being shorter than the window; library version.

---

## 3. Local coherence

Field `local_coherence` of the result of `coherence`, with its reference
level `local_coherence_shuffled` and the spread of that level,
`local_coherence_shuffled_sd`; function `adjacent_similarities`. Module
`discoursekit.semantic`.

### Definition

Each unit of the text (sentence, utterance or other unit chosen by the
study) is represented by an embedding vector. All vectors are scaled to unit
length before anything is computed, so that every unit has the same weight
in the sums and centroids of sections 4 to 7. Local coherence is the mean
cosine similarity between each unit and the next. It corresponds to the
first-order coherence of Bedi et al. (2015), who computed it on latent
semantic analysis vectors of phrases, and to the local coherence of Hoffman
et al. (2018), who compared each window of 20 words with the previous one.

The reference level is the value that local coherence takes on average when
the same units are placed in random order. It equals the mean cosine
similarity over all pairs of distinct units and is computed in closed form.

The spread of the reference level is the standard deviation of local
coherence over all orders of the same units, also in closed form. Write
*d*(*i*, *j*) for the similarity between units *i* and *j* minus the mean
similarity over all pairs, *Q* for the sum of the squares of *d* over
ordered pairs of distinct units, and *R* for the sum, over units, of the
squared sum of *d* between a unit and all the others. Over the orders of *n*
units, the sum of the *n* - 1 similarities between consecutive units has
variance *Q* / *n* - 2 *R* / (*n* (*n* - 1)), and the standard deviation of
local coherence is the square root of this variance divided by *n* - 1.

### Why a reference level

The cosine similarity between two unrelated units is not zero in general,
and its typical value differs from one embedding model to another: embedding
spaces are not isotropic (Ethayarajh, 2019, for contextual word
representations). A local coherence of 0.4 is therefore uninterpretable in
isolation. The difference between local coherence and its reference level
isolates what is due to the order of the units.

That difference is not zero for units in random order: it varies from one
order to another, and the more so the fewer the units. The spread states
by how much. The difference divided by the spread has mean zero and standard
deviation one over the orders of the units, whatever their number. Its
distribution over orders is not normal in general, and with few units it
takes few distinct values.

### Measurement hypotheses

**H-COH-1.** The mean of local coherence over all orderings of a set of
units equals the reference level. Tested by enumeration of all 720 orderings
of six units.

**H-COH-2.** Local coherence is unchanged when the text is read backwards,
when the vectors are rescaled, and when the embedding space is rotated.
Tested on random vectors.

**H-COH-6.** The standard deviation of local coherence over all orderings of
a set of units equals the spread of the reference level. Tested by
enumeration of all orderings of two to seven units.

### Motivating hypothesis

In coherent discourse, each unit continues the previous one. Local coherence
has been proposed as a marker of disorganised speech (Elvevåg et al., 2007;
Bedi et al., 2015). This library makes no claim about that proposal.

### Known confounds and limits

* **Embedding model.** Values obtained with different models, or different
  revisions of a model, are not comparable.
* **Unit of analysis.** Values depend on how the text is divided into units.
  For spoken language the division into sentences is a decision of the
  transcriber.
* **Length of the units.** Iter et al. (2018) found that, in the models they
  examined, pairs of longer sentences received higher coherence scores.
  Whether this holds for a given sentence-embedding model has to be checked;
  record the length of the units and use it as a covariate.
* **Truncation.** Sentence-embedding models ignore the part of a unit beyond
  a maximum number of tokens. The adapter provided reports how many units
  were truncated.
* **Filled pauses and repetition.** Iter et al. (2018) report that filled
  pauses and repeated sentences distort coherence scores. A text that
  repeats itself has high local coherence.
* **Language.** Parola et al. (2023) examined how far automated measures of
  coherence generalise across languages. Do not compare values across
  languages.
* **Number of units.** With few units the mean rests on few pairs, and
  local coherence departs from its reference level by chance; the spread of
  the reference level gives the size of such departures. Summaries based on
  extreme values, such as the minimum used by Bedi et al. (2015), depend on
  the number of units.
* **Speaker background.** See section 2.

### What to report

Embedding model and revision; definition of the unit; number of units and
their mean length in tokens; number of truncated units; local coherence
together with its reference level and the spread of that level; library
version.

---

## 4. Global coherence

Field `global_coherence` of the result of `coherence`.

### Definition

Global coherence is the mean, over the units, of the cosine similarity
between a unit and the centroid of all the other units of the text. It
expresses the degree to which each unit relates to the overall content of
the text, following the distinction between local and global coherence used
in discourse research (see Hoffman et al., 2018, and the work they cite).

The unit is left out of the centroid it is compared with. Otherwise each
unit would be compared partly with itself, which raises the value by an
amount that depends on the number of units. A unit for which the other units
cancel one another out, so that their centroid has no direction, is left out
of the mean.

### Measurement hypotheses

**H-COH-3.** Global coherence does not depend on the order of the units.
This follows from the definition; tested on random reorderings.

**H-COH-4.** For unrelated units, global coherence is close to zero whatever
their number, whereas the same quantity computed with a centroid that
includes the unit is close to the inverse square root of the number of
units. Tested with five random directions in 2,000 dimensions.

**H-COH-7.** For related units, global coherence increases with their
number. When every pair of units has the same similarity *r*, it equals
√(*n* - 1) *r* / √(1 + (*n* - 2) *r*) for *n* units, which goes from *r*
for two units towards √*r*. Tested on vectors built to have the same
similarity between all pairs.

### Motivating hypothesis

A speaker who stays on topic produces units that resemble the rest of what
is said. Lower global coherence has been reported in the speech of older
adults (Hoffman et al., 2018). This library makes no claim about what the
measure indicates in any population.

### Known confounds and limits

Those of local coherence, and in addition:

* **Legitimate changes of topic.** A text that deals with two subjects in
  turn has lower global coherence than a text that deals with one, without
  being less coherent in the ordinary sense. See sections 6 and 7.
* **Progression.** A narrative whose content moves on steadily has lower
  global coherence than a description of a single scene.
* **Number of units.** The centroid is estimated from the text itself, and
  from fewer units in a shorter text. Global coherence therefore increases
  with the number of units for texts of the same kind (H-COH-7): in the
  simulated texts of [short-texts.md](short-texts.md) it goes from 0.62
  with four units to 0.70 with 32. Do not compare texts of different
  lengths on this quantity without taking the number of units into
  account. The mean similarity over all pairs of units, returned as
  `local_coherence_shuffled`, does not depend on the number of units in
  this way, and section 5 describes a reference that does not come from
  the text.

### What to report

As for local coherence.

---

## 5. Reference coherence

Field `reference_coherence` of the result of `coherence`, computed when a
`reference` vector is given.

### Definition

Reference coherence is the mean cosine similarity between each unit and a
reference vector supplied by the study and obtained with the same embedding.
Two choices of reference have precedents. Elvevåg et al. (2007) compared
windows of a response with the question that had been asked (as described by
Iter et al., 2018). Hoffman et al. (2018) compared each window of a response
with a composite vector of the responses of the other participants to the
same prompt.

### Measurement hypothesis

**H-COH-5.** Reference coherence does not depend on the order of the units
or on the length of the reference vector, and is unchanged when the
embedding space is rotated. Tested on random vectors.

### Motivating hypothesis

A response that stays on the subject of the question resembles the question,
or resembles what other speakers say in answer to it.

### Known confounds and limits

Those of local coherence, and in addition:

* **Choice of reference.** A question and an answer differ in form, which
  affects their similarity independently of content. A composite of other
  responses depends on who the other speakers are; if it is computed from
  the same sample, leave the speaker's own response out, as Hoffman et al.
  (2018) did.

### What to report

How the reference vector was obtained, in addition to what is reported for
local coherence.

---

## 6. Topic boundaries

Function `segment_topics`; fields `boundaries` and `segmentation` of the
result of `coherence`.

### Definition

A topic boundary is a position where the content changes abruptly and
durably. Boundaries can be supplied by the study when the elicitation
procedure defines them, for example where a new question was asked. This is
the preferred option whenever it is available. Otherwise they are estimated
as follows.

*Block similarities.* For a block size *s* and each gap between two units
that has *s* units on both sides, two similarities are computed between the
*s* units before the gap and the *s* units after it.

* The cosine similarity between the sum of the units before and the sum of
  the units after. This is the block comparison of TextTiling (Hearst,
  1997), applied to embeddings. It compares the directions of the two
  blocks.
* The mean cosine similarity between a unit before the gap and a unit after
  it: local coherence across the gap, at the scale of the block. It is low
  when the two blocks differ, and also when the units of a block differ
  among themselves.

A change of topic lowers both at the gap where it occurs and leaves them
high on either side, which gives each curve of similarities a valley with
two sides.

*Depth.* The two sides of a valley are measured with block sizes that may
differ. For a pair of sizes, one for each side, the candidate boundary is
the gap at which the similarities are lowest together: the gap that
minimises the sum of the two similarities at the two sizes, among the gaps
that the curves of both sizes cover. On each similarity, the rise on the
left of the candidate is the highest value found at an earlier gap minus the
value at the candidate, both computed with blocks of the first size. The
rise on its right is defined in the same way with later gaps and the second
size. The depth of the candidate is the smaller of its two rises, and zero
when one of them is missing or negative: a valley has two sides.

The two sides have their own block sizes because they do not have the same
room. With *a* units of one topic before a boundary and *b* units of another
after it, the rise on the left is complete, in the sense that the blocks it
compares each lie within one topic, for blocks of at most min(*a*/2, *b*)
units, and the rise on the right for blocks of at most min(*b*/2, *a*)
units. Neither of these two numbers is more than twice the other, whatever
*a* and *b*. Pairs of sizes in which one is more than twice the other are
therefore not examined.

*Statistic.* Block sizes go from `block_size` to four times `block_size`,
without exceeding a quarter of the number of units, in steps of one tenth of
the current size (at least one). For a pair of sizes, the depths on each
similarity are standardised with their mean and standard deviation over the
observed order and the reorderings described below, and the two standardised
depths are added. A pair and its mirror image, which exchanges left and
right, have the same distribution over reorderings and share one mean and
one standard deviation. The largest sum is the test statistic, and the
candidate of the pair that reaches it is the candidate boundary of the text.

*Test.* The statistic is recomputed on random reorderings of the units. The
p-value is (1 + *b*) / (1 + *B*), where *b* is the number of reorderings,
out of *B*, whose statistic is at least the observed one (Phipson & Smyth,
2010). A reordering can have exactly the statistic of the observed order,
for instance when it is its mirror image; two statistics that are equal to
within rounding error count as equal. A boundary is reported when the
p-value is at most the level of the test, which is `alpha` for the text as a
whole.

*Subdivision.* When a boundary is found, the two parts are tested in the
same way, with reorderings confined to each part, until no further boundary
is found. This divisive scheme follows Matteson and James (2014). The
allocation of the level is a choice of this library: it is shared between
the two parts in proportion to their lengths. A part is thus tested at the
level `alpha` multiplied by its share of the units of the text, and the
levels of all the parts under test at any stage add up to `alpha` at most.
Levels and p-values are compared in exact arithmetic. A part whose level is
below the smallest p-value that the reorderings can give, 1 / (1 + *B*), is
not examined.

### Why two similarities

Each similarity, used alone, has a defect that the other does not have.

Two blocks that hold two topics in the same proportions have the same
direction, and the cosine between their sums is 1, as it is for two blocks
on one topic. Random reorderings of a text that has two topics therefore
contain valleys as deep as a real boundary. With six units on one topic
followed by six on another, the units of a topic being identical, 58 of the
924 arrangements of the units have a valley as deep as that of the ordered
text by this similarity; its p-value could not go below 58/924, which is
above 0.05. By the mean similarity between units, only the ordered text and
its mirror image have it, and the same holds for the sum of the two
standardised depths.

The mean similarity between units falls wherever content varies, whether it
changes at one point or slides over many units, and a valley measured on it
alone is reported more often in texts whose content drifts.
[validation.md](validation.md) compares the three choices on simulated
sequences.

### Relation to TextTiling

TextTiling gives every gap a depth score, the sum of the rises from the gap
to the nearest peak on each side (Hearst, 1997, section 5.3). The depth used
here differs in four ways. It is computed at the lowest point only; it
takes the smaller of the two rises instead of their sum; it measures each
rise to the highest point on that side instead of the nearest peak; and it
lets the two rises be measured with blocks of different sizes. A rise on one
side only is thus not counted as a valley, and the smoothing step of
TextTiling is not needed.

TextTiling compares blocks of a single size, meant to approximate the
average length of a paragraph (Hearst, 1997, section 5.2.1). Here a range of
sizes is examined, and the range is set by `block_size`, not by the length
of the text, so that the same question is asked of a short text and of a
long one.

TextTiling places boundaries where the depth score exceeds a cutoff equal to
the mean of the depth scores of the text minus a fraction of their standard
deviation. Some position exceeds such a cutoff in any text whose depth
scores are not all equal, whether or not the topic changes. The permutation
test replaces the cutoff so that a text without a change of topic can be
recognised as such.

### What the definition includes and excludes

* A boundary is a valley with two sides: two blocks that differ from each
  other more than the blocks on either side of them do.
* A steady drift of content has no valley and gives no boundary.
* Abruptness is relative to the blocks. A transition between two topics
  that is completed within about the largest block gives a valley as a
  change at one point does; a transition four times as long does not.
* Low similarity scattered through a text is also what random reorderings
  produce, and gives no boundary.
* One or two units unrelated to their neighbours are not a topic. They
  lower the similarity of the blocks that contain them, however, and can
  give a valley.

This definition is a choice. It treats a change of topic as an ordered
structure and does not use the cues by which a listener recognises that a
speaker is changing the subject on purpose. Whether it separates changes of
topic from disorganised speech in natural language is the question of the
benchmark in the roadmap.

### Measurement hypotheses

**H-SEG-1.** If every order of the units is equally likely
(exchangeability), the probability of reporting a boundary is at most
`alpha`. This follows from the construction of the test; checked by
simulation.

**H-SEG-2.** A steady drift of content gives no boundary: it is reported at
a rate close to the level of the test or below it, whether it runs through
the whole text or begins after a stable passage. Checked by simulation.

**H-SEG-3.** An abrupt change of topic between two segments of at least
twice `block_size` units is reported at the right place, wherever it lies in
the text. Checked by simulation, which also shows where the test lacks
power.

**H-SEG-5.** A transition between two topics is reported when it is no
longer than the largest block, and is not when it is four times as long.
Checked by simulation.

**H-SEG-6.** When a text consists of two groups of identical units, each
long enough for two blocks, no arrangement of its units has a larger
statistic than the one that keeps the groups apart, and only its mirror
image has one as large. Checked by enumeration of every arrangement, for
groups of four units with blocks of two and for groups of six and of eight
units with blocks of three.

The simulations use synthetic vectors, not text.
[validation.md](validation.md) gives the results, including the cases in
which the test performs poorly.

### Known limits

* **Short texts.** A text needs at least 2 × `block_size` + 2 units to be
  tested. The field `tested` says whether it was. A text that can be tested
  is not thereby long enough for a boundary to be found: see
  [short-texts.md](short-texts.md).
* **Segments shorter than two blocks.** A boundary is never placed within
  `block_size` units of either end of the part under test, and a segment of
  fewer than twice `block_size` units has no room for the two blocks that
  make one side of a valley. Such a segment is rarely delimited: in the
  simulation, a passage of four units between two longer ones is delimited
  on both sides in at most 19% of cases with blocks of three units. A
  smaller `block_size` finds shorter segments.
* **Weakly distinct topics.** Power falls when the units of a topic are
  only weakly similar to one another or the topics are close. Since the
  blocks are bounded, it does not go on rising with the length of the
  segments. A larger `block_size` gives larger blocks.
* **Transitions.** The test does not tell a transition shorter than its
  largest block from a change at one point, and it reports part of the
  transitions that are twice as long.
* **Unrelated units.** In the simulation, an unrelated unit in the middle
  of a text of 24 units is reported as a boundary in 13% to 16% of cases,
  which is more than the nominal level, and two adjacent unrelated units in
  41% to 57%. The test does not tell a digression of two units from the
  beginning of a new topic.
* **Further boundaries.** The guarantee of H-SEG-1 concerns a text without
  any boundary. In a text that has boundaries, each further test can add a
  false one.
* **Number of reorderings.** Parts whose level is below 1 / (1 + *B*) are
  not examined. With `alpha` = 0.05, these are the parts shorter than a
  fiftieth of the text for 999 reorderings, and shorter than a tenth of the
  text for 199.
* **Dependence on the embedding.** The test inherits every limit listed in
  section 3.

### What to report

Whether boundaries were supplied or estimated; for estimated boundaries,
`block_size`, `alpha`, `n_permutations` and `seed`, the number of texts too
short to be tested, and the p-values.

---

## 7. Coherence within segments

Fields `local_coherence_within_segments` and
`global_coherence_within_segments` of the result of `coherence`.

### Definition

Local coherence within segments is the mean cosine similarity between
consecutive units that belong to the same segment. Global coherence within
segments compares each unit with the centroid of the other units of its
segment, and averages over all units in segments of at least two units. The
segments are those delimited by the boundaries of section 6. With no
boundary, the two quantities equal local and global coherence.

### Measurement hypothesis

**H-SEG-4.** When a text deals with two subjects in turn, global coherence
is lower than for a text on one subject, while global coherence within
segments is not. Checked on synthetic vectors with a supplied boundary and
with an estimated one.

### Motivating hypothesis

A speaker who moves from one subject to another on purpose and remains
coherent on each should not be scored as incoherent. The plain and the
within-segment values are reported side by side; the second is never a
replacement for the first.

### Known confounds and limits

Those of sections 3, 4 and 6, and in addition:

* **Estimated boundaries select low points.** The test places boundaries
  where similarity is lowest. Leaving these pairs out raises local coherence
  within segments mechanically. With supplied boundaries there is no such
  selection.
* **Missed boundaries.** When a change of topic is not detected, the two
  quantities equal the plain ones and the change of topic is counted as a
  loss of coherence.

### What to report

Both the plain and the within-segment values, and everything listed in
section 6.

---

## 8. Planned measures

Nothing in this section is implemented. Each measure will receive a full
entry before any code is written.

| Family | Intended content | Open question |
|---|---|---|
| Syntactic complexity | A small number of dependency-based indices | Which indices depend least on the division of spoken transcripts into sentences |
| Epistemic stance | Rates of certainty markers, hedges and evidentials | Construction and validation of the lexicon; specificity |

On specificity: Al-Mosaiwi and Johnstone (2018) found a higher rate of
absolutist words in internet forums devoted to anxiety, depression and
suicidal ideation than in control forums. They also found elevated rates in
forums devoted to borderline personality disorder and, less clearly, to
eating disorders, relative to other mental health forums. A marker of this
kind cannot be assumed to be specific to one condition or one style of
reasoning.

---

## 9. References

Al-Mosaiwi, M., & Johnstone, T. (2018). In an absolute state: Elevated use
of absolutist words is a marker specific to anxiety, depression, and
suicidal ideation. *Clinical Psychological Science*, 6(4), 529-542.
https://doi.org/10.1177/2167702617747074 (Corrigendum: 2019, 7(3), 636-637,
https://doi.org/10.1177/2167702619843297)

Bedi, G., Carrillo, F., Cecchi, G. A., Slezak, D. F., Sigman, M., Mota, N.
B., Ribeiro, S., Javitt, D. C., Copelli, M., & Corcoran, C. M. (2015).
Automated analysis of free speech predicts psychosis onset in high-risk
youths. *npj Schizophrenia*, 1, 15030.
https://doi.org/10.1038/npjschz.2015.30

Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: The
moving-average type-token ratio (MATTR). *Journal of Quantitative
Linguistics*, 17(2), 94-100. https://doi.org/10.1080/09296171003643098

Elvevåg, B., Foltz, P. W., Weinberger, D. R., & Goldberg, T. E. (2007).
Quantifying incoherence in speech: An automated methodology and novel
application to schizophrenia. *Schizophrenia Research*, 93(1-3), 304-316.
https://doi.org/10.1016/j.schres.2007.03.001

Ethayarajh, K. (2019). How contextual are contextualized word
representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings.
In *Proceedings of the 2019 Conference on Empirical Methods in Natural
Language Processing and the 9th International Joint Conference on Natural
Language Processing (EMNLP-IJCNLP)* (pp. 55-65).
https://doi.org/10.18653/v1/D19-1006

Fergadiotis, G., Wright, H. H., & West, T. M. (2013). Measuring lexical
diversity in narrative discourse of people with aphasia. *American Journal
of Speech-Language Pathology*, 22(2), S397-S408.
https://doi.org/10.1044/1058-0360(2013/12-0083)

Hearst, M. A. (1997). TextTiling: Segmenting text into multi-paragraph
subtopic passages. *Computational Linguistics*, 23(1), 33-64.
https://aclanthology.org/J97-1003/

Hoffman, P., Loginova, E., & Russell, A. (2018). Poor coherence in older
people's speech is explained by impaired semantic and executive processes.
*eLife*, 7, e38907. https://doi.org/10.7554/eLife.38907

Iter, D., Yoon, J. H., & Jurafsky, D. (2018). Automatic detection of
incoherent speech for diagnosing schizophrenia. In *Proceedings of the Fifth
Workshop on Computational Linguistics and Clinical Psychology: From Keyboard
to Clinic* (pp. 136-146). https://aclanthology.org/W18-0615/

Matteson, D. S., & James, N. A. (2014). A nonparametric approach for
multiple change point analysis of multivariate data. *Journal of the
American Statistical Association*, 109(505), 334-345.
https://doi.org/10.1080/01621459.2013.849605

Palaniyappan, L. (2021). More than a biomarker: could language be a
biosocial marker of psychosis? *npj Schizophrenia*, 7, 42.
https://doi.org/10.1038/s41537-021-00172-1

Parola, A., Lin, J. M., Simonsen, A., Bliksted, V., Zhou, Y., Wang, H.,
Inoue, L., Koelkebeck, K., & Fusaroli, R. (2023). Speech disturbances in
schizophrenia: Assessing cross-linguistic generalizability of NLP automated
measures of coherence. *Schizophrenia Research*, 259, 59-70.
https://doi.org/10.1016/j.schres.2022.07.002

Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be
zero: Calculating exact P-values when permutations are randomly drawn.
*Statistical Applications in Genetics and Molecular Biology*, 9(1), Article
39. https://doi.org/10.2202/1544-6115.1585

Unicode Consortium. (2022). *Unicode Common Locale Data Repository (CLDR)*
(Release 42). https://cldr.unicode.org/downloads/cldr-42

Unicode Consortium. (2026). *Unicode Standard Annex #29: Unicode text
segmentation* (Revision 49, Unicode 18.0.0).
https://www.unicode.org/reports/tr29/

Zenker, F., & Kyle, K. (2021). Investigating minimum text lengths for
lexical diversity indices. *Assessing Writing*, 47, 100505.
https://doi.org/10.1016/j.asw.2020.100505
