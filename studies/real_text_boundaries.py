"""Boundaries reported by the test for topic boundaries in real texts.

The level of :func:`discoursekit.segment_topics` holds when every order of
the units is equally likely. In simulated sequences whose consecutive units
are related, without any change of topic, the test reports boundaries far
more often than its level (``docs/validation.md``, section 4). This script
asks the same question of real texts, and three others.

1. False boundaries. How often is a boundary reported in a real text that
   keeps to one subject?
2. Order. Is the excess due to the order of the sentences? Each text is also
   tested with its sentences in a random order, in which every order is
   equally likely and the level applies.
3. Correction. Does a reordering that keeps short runs of consecutive
   sentences together reduce the excess? The circular block permutation of
   Chen (arXiv:1903.01598) is compared with the reordering of the library,
   which moves the sentences one by one.
4. Detection. What does each reordering find when the first half of a text
   is followed by the first half of another?

Corpora. Two public, non-clinical corpora, distributed with NLTK for
research: the Reuters-21578 newswire collection (ApteMod version), of which
the dispatches assigned to a single category are used, without their
headline; and the film reviews of the polarity dataset of Pang and Lee
(2004). A dispatch is taken to keep to one story and a review to one film.
This is not checked sentence by sentence: a review moves from the plot to
the verdict and a dispatch can change its angle, so that some of the
boundaries counted here as false may be changes of sub-topic. The texts are
not stored in this repository and are not reproduced in the output.

Texts. The sentences of fewer than five alphabetic words are left out, which
removes the tables of figures of many dispatches. Each text is cut to its
first 12, 24 or 48 sentences; texts with fewer sentences are not used at
that length. At most 400 texts are drawn at each length, with a fixed seed.

Embeddings. By default each sentence is embedded with the model
``sentence-transformers/all-MiniLM-L6-v2`` at a fixed revision, through
:class:`discoursekit.SentenceTransformerEmbedder`. With ``--embedder
word-vectors`` it is the mean of the spaCy ``en_core_web_md`` vectors of its
alphabetic words that are not stop words.

Test. Default block size and level, 199 reorderings, and a seed of its own
for each text, its index.

Usage, from the root of the repository::

    pip install -e ".[embeddings]" nltk
    python studies/real_text_boundaries.py

The corpora and the model are downloaded on the first run. The report is
written to ``studies/results/real_text_boundaries.md``. The run takes about
half an hour on one core.

References
----------
Chen, H. Change-point detection for multivariate and non-Euclidean data
with local dependency. arXiv:1903.01598.

Pang, B., & Lee, L. (2004). A sentimental education: Sentiment analysis
using subjectivity summarization based on minimum cuts. In Proceedings of
the 42nd Annual Meeting of the Association for Computational Linguistics
(pp. 271-278). https://doi.org/10.3115/1218955.1218990

Reuters-21578, Distribution 1.0. Available from the UCI KDD Archive and
distributed with NLTK.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import discoursekit
from discoursekit import semantic

Array = NDArray[np.float64]
Text = tuple[str, str, list[str]]  # identifier, category, sentences

LENGTHS = (12, 24, 48)
N_TEXTS = 400
N_PERMUTATIONS = 199
MIN_WORDS = 5
SELECTION_SEED = 2026
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
WORD_VECTORS = "en_core_web_md"

# Reorderings compared: the length of the runs of consecutive sentences kept
# together, None standing for the reordering of the library.
REORDERINGS: dict[str, int | None] = {
    "One by one": None,
    "Circular blocks of 2": 2,
    "Circular blocks of 3": 3,
}
CORPORA = ("Reuters", "Reviews")
OUTPUT = Path("studies/results/real_text_boundaries.md")


# Corpora


def ensure_corpora() -> None:
    """Download the NLTK resources that are missing."""
    import nltk

    for resource, path in (
        ("reuters", "corpora/reuters"),
        ("movie_reviews", "corpora/movie_reviews"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


def reuters_texts() -> list[Text]:
    """Dispatches with a single category, without their headline."""
    from nltk.corpus import reuters
    from nltk.tokenize import sent_tokenize

    texts = []
    for fileid in reuters.fileids():
        categories = reuters.categories(fileid)
        parts = reuters.raw(fileid).split("\n", 1)
        if len(categories) != 1 or len(parts) < 2:
            continue
        sentences = [" ".join(s.split()) for s in sent_tokenize(parts[1])]
        texts.append((fileid, categories[0], sentences))
    return texts


def review_texts() -> list[Text]:
    """Film reviews, one sentence per tokenised sentence of the corpus."""
    from nltk.corpus import movie_reviews

    return [
        (fileid, "review", [" ".join(words) for words in movie_reviews.sents(fileid)])
        for fileid in movie_reviews.fileids()
    ]


def usable(sentences: list[str]) -> list[str]:
    """Keep the sentences of at least MIN_WORDS alphabetic words."""
    return [s for s in sentences if sum(w.isalpha() for w in s.split()) >= MIN_WORDS]


def select(texts: list[Text], length: int, seed: int, count: int) -> list[Text]:
    """Draw texts that have at least ``length`` sentences, cut to that length."""
    long_enough = [(f, c, usable(s)) for f, c, s in texts]
    long_enough = [(f, c, s[:length]) for f, c, s in long_enough if len(s) >= length]
    order = np.random.RandomState(seed).permutation(len(long_enough))
    return [long_enough[i] for i in order[:count]]


# Embeddings


Embedder = Callable[[list[str]], Array]


def sentence_model(model: str, revision: str | None) -> tuple[Embedder, str]:
    """Return the sentence-transformers embedder and a description of it."""
    import sentence_transformers

    embed = discoursekit.SentenceTransformerEmbedder(
        model, revision=revision, device="cpu"
    )
    version = sentence_transformers.__version__
    at = f" at revision {revision}" if revision else ""
    return embed, f"{model}{at}, sentence-transformers {version}"


def word_vectors() -> tuple[Embedder, str]:
    """Return an embedder that averages spaCy word vectors, and a description."""
    import spacy

    nlp = spacy.load(WORD_VECTORS, exclude=["parser", "ner", "lemmatizer"])

    def embed(sentences: list[str]) -> Array:
        rows = []
        for doc in nlp.pipe(sentences):
            words = [t.vector for t in doc if t.is_alpha and not t.is_stop]
            words = [w for w in words if w.any()]
            rows.append(
                np.mean(words, axis=0) if words else np.zeros(nlp.vocab.vectors_length)
            )
        return np.asarray(rows, dtype=np.float64)

    description = (
        f"mean of the {WORD_VECTORS} {nlp.meta['version']} vectors of the content "
        f"words, spaCy {spacy.__version__}"
    )
    return embed, description


# Reorderings


def circular_blocks(length: int) -> Callable[[np.random.PCG64, int], NDArray[np.intp]]:
    """Circular block permutation with runs of ``length`` units (Chen).

    The units are put on a circle, which is cut at a random place into runs
    of ``length`` consecutive units (the last run may be shorter), and the
    runs are put in a random order.
    """

    def reordering(bit_generator: np.random.PCG64, n: int) -> NDArray[np.intp]:
        raw = bit_generator.random_raw(n + 1)
        rotated = np.roll(np.arange(n), -int(raw[-1] % n))
        runs = [rotated[i : i + length] for i in range(0, n, length)]
        places = np.argsort(raw[: len(runs)], kind="stable")
        return np.concatenate([runs[i] for i in places]).astype(np.intp)

    return reordering


ONE_BY_ONE = semantic._reordering


def boundaries(unit: Array, run: int | None, seed: int) -> tuple[int, ...]:
    """Boundaries found with one scheme of reordering."""
    # The private function that draws each reordering is replaced on purpose.
    reordering = ONE_BY_ONE if run is None else circular_blocks(run)
    semantic._reordering = reordering  # type: ignore[assignment]
    try:
        return discoursekit.segment_topics(
            unit, n_permutations=N_PERMUTATIONS, seed=seed
        ).boundaries
    finally:
        semantic._reordering = ONE_BY_ONE


# The study


@dataclass
class Group:
    """Texts of one corpus at one length, in one condition."""

    texts: int = 0
    reported: dict[str, int] = field(default_factory=dict)
    correct: dict[str, int] = field(default_factory=dict)
    near: list[float] = field(default_factory=list)
    far: list[float] = field(default_factory=list)


def similarities(unit: Array) -> tuple[float, float]:
    """Mean similarity of consecutive sentences, and of sentences further apart."""
    u = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    gram = u @ u.T
    near = float(np.mean(np.diagonal(gram, 1)))
    far = float(gram[np.triu_indices(len(u), 2)].mean())
    return near, far


def study(
    embed: Embedder, raw: Path | None, count: int = N_TEXTS
) -> dict[tuple[str, int, str], Group]:
    """Run every condition and return the counts."""
    ensure_corpora()
    corpora = {"Reuters": reuters_texts(), "Reviews": review_texts()}
    groups: dict[tuple[str, int, str], Group] = {}
    records = []
    embedded: dict[str, Array] = {}
    for corpus, texts in corpora.items():
        for length in LENGTHS:
            seed = SELECTION_SEED + length
            whole = select(texts, length, seed, count)
            half = length // 2
            pool = select(texts, half, seed + 1, count)
            pairs = [
                (f"{a[0]}+{b[0]}", a[2] + b[2])
                for a, b in zip(pool[0::2], pool[1::2], strict=False)
                if corpus == "Reviews" or a[1] != b[1]
            ]
            items: dict[str, list[tuple[str, list[str], tuple[int, ...]]]] = {
                "Ordered": [(f, s, ()) for f, _, s in whole],
                "Shuffled": [(f, s, ()) for f, _, s in whole],
                "Planted": [(name, s, (half,)) for name, s in pairs],
            }
            for condition, entries in items.items():
                group = groups.setdefault((corpus, length, condition), Group())
                for index, (name, sentences, truth) in enumerate(entries):
                    key = f"{corpus} {length} {name}"
                    if key not in embedded:
                        embedded[key] = embed(sentences)
                    unit = embedded[key]
                    if not unit.any(axis=1).all():
                        continue  # a sentence without a vector
                    if condition == "Shuffled":
                        unit = unit[np.random.RandomState(index).permutation(len(unit))]
                    near, far = similarities(unit)
                    group.texts += 1
                    group.near.append(near)
                    group.far.append(far)
                    found_by = {}
                    for label, run in REORDERINGS.items():
                        if condition == "Shuffled" and run is not None:
                            continue
                        found = boundaries(unit, run, index)
                        found_by[label] = list(found)
                        group.reported[label] = group.reported.get(label, 0) + bool(
                            found
                        )
                        right = len(found) == len(truth) and all(
                            abs(a - b) <= 1 for a, b in zip(found, truth, strict=True)
                        )
                        group.correct[label] = group.correct.get(label, 0) + right
                    records.append(
                        {
                            "corpus": corpus,
                            "length": length,
                            "condition": condition,
                            "text": name,
                            "near": near,
                            "far": far,
                            "truth": list(truth),
                            "found": found_by,
                        }
                    )
    if raw is not None:
        raw.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )
    return groups


# The report


def percent(count: int, total: int) -> str:
    """Format a proportion as a percentage with its 95% Wilson interval."""
    if total == 0:
        return "-"
    z = 1.959964
    p = count / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    half /= 1 + z * z / total
    return (
        f"{100 * p:.1f}% ({100 * (centre - half):.0f} to {100 * (centre + half):.0f})"
    )


def rows(
    groups: dict[tuple[str, int, str], Group], condition: str
) -> list[tuple[str, int, Group]]:
    """Return the groups of one condition that have texts, in a fixed order."""
    return [
        (corpus, length, groups[corpus, length, condition])
        for corpus in CORPORA
        for length in LENGTHS
        if (corpus, length, condition) in groups
        and groups[corpus, length, condition].texts
    ]


def report(groups: dict[tuple[str, int, str], Group], embedding: str) -> list[str]:
    """Return the report as lines of Markdown."""
    labels = list(REORDERINGS)
    heads = " | ".join(labels)
    lines = [
        "# Test for topic boundaries: results on real texts",
        "",
        "Produced by `studies/real_text_boundaries.py`. The corpora, the "
        "selection of the texts and their limits are described in the "
        "documentation of the script.",
        "",
        f"Parameters: alpha {discoursekit.DEFAULT_ALPHA}, block size "
        f"{discoursekit.DEFAULT_BLOCK_SIZE}, {N_PERMUTATIONS} reorderings, and a "
        "seed of its own for each text.",
        "",
        f"Embedding: {embedding}.",
        "",
        f"Software: discoursekit {discoursekit.__version__}, NumPy "
        f"{np.__version__}, Python {platform.python_version()}, "
        f"{platform.system()} {platform.machine()}.",
        "",
        "Data: Reuters-21578, Distribution 1.0, ApteMod version, and the "
        "polarity dataset v2.0 of Pang and Lee (2004), both as distributed "
        "with NLTK. The texts are not reproduced here.",
        "",
        "Percentages are given with their 95% Wilson interval, in points.",
        "",
        "## Texts",
        "",
        "Mean cosine similarity between consecutive sentences, and between "
        "sentences two or more apart.",
        "",
        "| Corpus | Sentences | Texts | Consecutive | Further apart |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus, length, group in rows(groups, "Ordered"):
        lines.append(
            f"| {corpus} | {length} | {group.texts} | {np.mean(group.near):.3f} "
            f"| {np.mean(group.far):.3f} |"
        )
    lines += [
        "",
        "## Texts that keep to one subject",
        "",
        "Percentage of texts in which a boundary is reported, with each scheme "
        "of reordering. The level of the test is 5%. In the last column the "
        "sentences of each text are in a random order and reordered one by "
        "one; there the level applies.",
        "",
        f"| Corpus | Sentences | {heads} | Sentences in random order |",
        "|---|---:|" + "---:|" * (len(labels) + 1),
    ]
    for corpus, length, group in rows(groups, "Ordered"):
        shuffled = groups[corpus, length, "Shuffled"]
        cells = [percent(group.reported[label], group.texts) for label in labels]
        cells.append(percent(shuffled.reported[labels[0]], shuffled.texts))
        lines.append(f"| {corpus} | {length} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Two texts joined",
        "",
        "The first half of a text followed by the first half of another: two "
        "dispatches of different categories, or reviews of two films. "
        "Percentage of joined texts in which one boundary is reported, within "
        "one sentence of the join:",
        "",
        f"| Corpus | Sentences | Texts | {heads} |",
        "|---|---:|---:|" + "---:|" * len(labels),
    ]
    planted = rows(groups, "Planted")
    for corpus, length, group in planted:
        cells = [percent(group.correct[label], group.texts) for label in labels]
        lines.append(
            f"| {corpus} | {length} | {group.texts} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "Percentage of joined texts in which at least one boundary is reported:",
        "",
        f"| Corpus | Sentences | {heads} |",
        "|---|---:|" + "---:|" * len(labels),
    ]
    for corpus, length, group in planted:
        cells = [percent(group.reported[label], group.texts) for label in labels]
        lines.append(f"| {corpus} | {length} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    """Run the study and write its report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--embedder",
        choices=("sentence-transformers", "word-vectors"),
        default="sentence-transformers",
    )
    parser.add_argument("--model", default=MODEL, help="sentence-transformers model")
    parser.add_argument("--revision", default=REVISION, help="revision of the model")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--texts",
        type=int,
        default=N_TEXTS,
        help="largest number of texts at each length (fewer for a quick trial)",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="also write one line of JSON per text (identifiers and boundaries)",
    )
    arguments = parser.parse_args()
    if arguments.embedder == "sentence-transformers":
        embed, embedding = sentence_model(arguments.model, arguments.revision or None)
    else:
        embed, embedding = word_vectors()
    groups = study(embed, arguments.raw, arguments.texts)
    arguments.output.write_text(
        "\n".join(report(groups, embedding)) + "\n", encoding="utf-8"
    )
    print(f"Report written to {arguments.output}")


if __name__ == "__main__":
    main()
