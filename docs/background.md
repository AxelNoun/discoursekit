# Background and related work

This document places the measures of the library in the literature they come
from and states what the library does and does not attempt. The measures
themselves are specified in [metrics.md](metrics.md).

## 1. Automated measures of coherence

Automated analysis of the coherence of speech rests on vector representations
of text. Elvevåg et al. (2007) applied latent semantic analysis to the speech
of patients with schizophrenia; one of their measures compares successive
windows of a response with the question that was asked (as described by Iter
et al., 2018). Bedi et al. (2015) defined first-order coherence as the
similarity of consecutive phrase vectors, averaged over the text. In a
sample of 34 young people at clinical high risk, five of whom later developed
psychosis, the feature that contributed most to classification was the
minimum coherence between two consecutive phrases. The authors noted that the
sample was small and that replication in a larger sample was needed.

Outside psychiatry, Hoffman et al. (2018) studied the speech of younger and
older adults with two measures computed on windows of 20 words: local
coherence, the similarity of a window with the previous one, and global
coherence, the similarity of a window with a composite vector of the
responses of the other participants to the same prompt. They report
correlations of 0.37 and 0.68, respectively, with human judgements of
coherence.

The measures of this library belong to this family. Local coherence follows
Bedi et al. (2015) and Hoffman et al. (2018). Global coherence uses the text
itself as the reference; reference coherence accepts an external reference,
as in Elvevåg et al. (2007) and Hoffman et al. (2018).

## 2. Known weaknesses of these measures

Several weaknesses are documented and shape the design of the library.

* **Properties of the text other than coherence.** Iter et al. (2018), in a
  study of 14 participants, found that the models of Elvevåg et al. (2007)
  and Bedi et al. (2015) gave higher scores to pairs of longer sentences,
  were disturbed by filled pauses, and gave a high score to a discourse that
  repeats the same sentence. The library therefore returns the number of
  units, reports truncated units, lets the study exclude filled pauses at
  tokenisation, and lists these confounds with each measure.
* **Scale of the similarity.** Embedding spaces are not isotropic
  (Ethayarajh, 2019), so the similarity of unrelated units is not zero and
  differs between models. The library reports local coherence together with
  the value it would take for the same units in random order.
* **Language.** Parola et al. (2023) assessed the generalisability of
  automated coherence measures across languages. The library makes no claim
  of comparability across languages.
* **Social factors.** Palaniyappan (2021) argues that language measures
  reflect social as well as biological processes, and that ignoring their
  sociolinguistic aspects limits their usefulness. Education, socioeconomic
  position and language background are listed as confounds of every
  measure.

## 3. Change of topic and loss of coherence

Clinical descriptions of disorganised speech distinguish several phenomena,
among them derailment and tangentiality (Andreasen, 1986). An automated
measure of coherence cannot by itself tell these from a deliberate change of
subject: as Hoffman et al. (2018) observe of their own measure, low local
coherence is obtained when a speaker switches abruptly between topics.

The library addresses one part of this problem. A change of topic that is
abrupt and durable gives the text a block structure, which is an ordered
property: it disappears when the units are reordered at random. Low
similarity scattered through a text does not. The test for topic boundaries
builds on the block comparison of TextTiling (Hearst, 1997) and decides by
permutation whether the deepest valley of similarity is deeper than
reordering would produce (Phipson & Smyth, 2010, for the p-value), with the
hierarchical subdivision of Matteson and James (2014). Coherence is then
reported both for the whole text and within the segments.

This does not settle the question. The test uses the order of the units and
nothing else; it does not use the cues by which a listener recognises that a
change of subject is intended, and its power is limited in short texts (see
[validation.md](validation.md)). When the elicitation procedure defines the
boundaries, for example a series of questions, they should be supplied
instead of estimated. Whether the approach separates changes of topic from
disorganised speech in natural language is to be established by the
benchmark described in the roadmap. Discriminating original from perturbed
text is the most common way of evaluating coherence models, and its limits
have been discussed (Laban et al., 2021).

## 4. Lexical diversity

The plain type-token ratio decreases with the length of the text. The
moving-average type-token ratio (Covington & McFall, 2010) removes this
dependence by averaging the ratio over windows of fixed size. In narrative
samples from people with aphasia, Fergadiotis et al. (2013) compared four
measures of lexical diversity and found the strongest evidence of freedom
from length effects for this measure and for one other (MTLD). Zenker and
Kyle (2021) examine the minimum text length at which lexical diversity
indices can be used.

## 5. Tokenisation

Every lexical measure depends on the definition of a word. The library takes
that definition from the Unicode Standard (UAX #29), which is public,
versioned and accompanied by a conformance test, so that the same word
boundaries can be obtained with other implementations of the standard. One
documented tailoring makes white space separate tokens in every case (see
[metrics.md](metrics.md), section 1).

## 6. What the library does not attempt

* **Prediction of clinical outcomes.** The use of speech measures to predict
  outcomes such as relapse is an active area of clinical research (for
  example Dalal et al., 2026). The library provides measuring instruments
  and makes no predictive claim.
* **Assessment of individuals.** See [scope-and-ethics.md](scope-and-ethics.md).
* **Division of a text into units.** Sentence and utterance segmentation is
  left to the study and to established tools.

## References

Andreasen, N. C. (1986). Scale for the Assessment of Thought, Language, and
Communication (TLC). *Schizophrenia Bulletin*, 12(3), 473-482.
https://doi.org/10.1093/schbul/12.3.473

Bedi, G., Carrillo, F., Cecchi, G. A., Slezak, D. F., Sigman, M., Mota, N.
B., Ribeiro, S., Javitt, D. C., Copelli, M., & Corcoran, C. M. (2015).
Automated analysis of free speech predicts psychosis onset in high-risk
youths. *npj Schizophrenia*, 1, 15030.
https://doi.org/10.1038/npjschz.2015.30

Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: The
moving-average type-token ratio (MATTR). *Journal of Quantitative
Linguistics*, 17(2), 94-100. https://doi.org/10.1080/09296171003643098

Dalal, T. C., Park, M. T. M., Silva, A. M., Iskhakova, S., Voppel, A.,
Brierley, N. J., MacKinley, M., Olarewaju, E., & Palaniyappan, L. (2026).
Clinical psychopathology-based early relapse prediction model using speech
and language in psychosis. *Schizophrenia Research: Cognition*, 43, 100392.
https://doi.org/10.1016/j.scog.2025.100392

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

Laban, P., Dai, L., Bandarkar, L., & Hearst, M. A. (2021). Can transformer
models measure coherence in text: Re-thinking the shuffle test. In
*Proceedings of the 59th Annual Meeting of the Association for Computational
Linguistics and the 11th International Joint Conference on Natural Language
Processing (Volume 2: Short Papers)* (pp. 1058-1064).
https://doi.org/10.18653/v1/2021.acl-short.134

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

Unicode Consortium. (2026). *Unicode Standard Annex #29: Unicode text
segmentation* (Revision 49, Unicode 18.0.0).
https://www.unicode.org/reports/tr29/

Zenker, F., & Kyle, K. (2021). Investigating minimum text lengths for
lexical diversity indices. *Assessing Writing*, 47, 100505.
https://doi.org/10.1016/j.asw.2020.100505
