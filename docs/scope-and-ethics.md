# Scope and ethics

## What the library is

A set of text measures with documented behaviour, intended for researchers
who study discourse. It takes text, or embeddings of text, and returns
numbers that describe that text.

## What it is not

* **It is not a clinical tool.** It does not detect, screen for, predict or
  diagnose any condition, and no output should be presented as doing so.
* **It has not been validated on clinical data.** No measure in the library
  has been validated against clinical assessments. Such a validation would
  require an appropriate study design, ethical approval and clinical
  expertise.
* **It does not assess individuals.** The measures vary with the length of
  the text, its topic, the task, and the education, socioeconomic position
  and language background of the speaker. A value obtained for one text does
  not support a conclusion about its author.

## Rules of this repository

1. No clinical claim is made in the code, the documentation, the examples,
   the issues or the papers derived from this repository.
2. Development, benchmarks and studies in this repository use synthetic data
   or public, non-clinical corpora only, within the terms of their licences.
3. The project does not analyse mental health forums, social media accounts
   or any other source in order to identify or flag individuals.
   Contributions that do so are declined.
4. Benchmarks based on perturbed texts (shuffled or inserted sentences) probe
   what a measure responds to. A perturbed text is not a model of anyone's
   speech, and performance on such a benchmark is not clinical validity.
5. The library produces numbers for researchers. Returning results to the
   people whose language is analysed is a clinical act that requires a
   clinical framework; it is outside the scope of the project.

## Use in a study involving people

The design and the ethical review of such a study are the responsibility of
those who conduct it. At a minimum: obtain ethical approval and informed
consent for the analysis of language; record the covariates listed as known
confounds for each measure in [metrics.md](metrics.md); and report the
parameters returned with each result.
