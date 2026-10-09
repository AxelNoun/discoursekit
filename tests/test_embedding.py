"""Tests of the sentence-transformers adapter.

Most of these tests need the optional dependency and are skipped without it.
They run on a very small model built on the spot with random weights, so they
need no network access: they check the adapter, not the quality of a model.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from discoursekit import SentenceTransformerEmbedder, TruncationWarning, coherence

WORDS = (
    "the a cat dog sat on mat ran in park stock market fell rose bank interest "
    "rate and of to"
).split()
MAX_TOKENS = 12


@pytest.fixture(scope="module")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Build and save a one-layer model with a 25-word vocabulary."""
    pytest.importorskip("sentence_transformers")
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import BertConfig, BertModel, BertTokenizerFast

    try:
        from sentence_transformers.sentence_transformer.modules import (
            Pooling,
            Transformer,
        )
    except ImportError:  # sentence-transformers before version 6
        from sentence_transformers.models import Pooling, Transformer

    root: Path = tmp_path_factory.mktemp("tiny-model")
    bert = root / "bert"
    bert.mkdir()
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    (bert / "vocab.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")

    torch.manual_seed(0)
    configuration = BertConfig(
        vocab_size=len(vocabulary),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    BertModel(configuration).save_pretrained(bert)
    BertTokenizerFast(vocab_file=str(bert / "vocab.txt")).save_pretrained(bert)

    transformer = Transformer(str(bert), max_seq_length=MAX_TOKENS)
    pooling = Pooling(16, pooling_mode="mean")
    saved = root / "model"
    SentenceTransformer(modules=[transformer, pooling], device="cpu").save(str(saved))
    return str(saved)


@pytest.fixture(scope="module")
def embed(model_path: str) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(model_path, device="cpu", batch_size=2)


UNITS = [
    "The cat sat on the mat.",
    "The dog ran in the park.",
    "The stock market fell.",
    "The bank rate rose.",
]

# Behaviour that does not need the library


def test_missing_library_gives_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes the import fail, as if not installed.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match=r"discoursekit\[embeddings\]"):
        SentenceTransformerEmbedder("any-model")


@pytest.mark.parametrize("model", ["", None, 3])
def test_rejects_invalid_model_name(model: object) -> None:
    with pytest.raises(TypeError, match="model must be a non-empty string"):
        SentenceTransformerEmbedder(model)  # type: ignore[arg-type]


def test_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        SentenceTransformerEmbedder("any-model", batch_size=0)
    with pytest.raises(TypeError, match="batch_size must be an int"):
        SentenceTransformerEmbedder("any-model", batch_size=2.0)  # type: ignore[arg-type]


# The wrapper around a model, with a stand-in for the model


class StandInModel:
    """The least a model must offer: no tokeniser and no stated length limit."""

    def encode(self, texts: list[str], **options: object) -> NDArray[np.float32]:
        return np.full((len(texts), 3), 0.5, dtype=np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        return 3


@pytest.fixture
def plain() -> SentenceTransformerEmbedder:
    embedder = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
    embedder.model_name = "stand-in"
    embedder.revision = None
    embedder.batch_size = 8
    embedder.max_tokens = None
    embedder._model = StandInModel()
    return embedder


def test_output_is_converted_to_double_precision(
    plain: SentenceTransformerEmbedder,
) -> None:
    vectors = plain(["one", "two"])
    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float64


def test_truncation_cannot_be_counted_without_a_tokeniser(
    plain: SentenceTransformerEmbedder,
) -> None:
    assert plain.count_truncated(["one", "two"]) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error", TruncationWarning)
        plain(["one", "two"])


def test_empty_input_with_the_older_name_of_the_dimension_method(
    plain: SentenceTransformerEmbedder,
) -> None:
    assert plain([]).shape == (0, 3)


def test_wrapper_rejects_a_single_string(plain: SentenceTransformerEmbedder) -> None:
    with pytest.raises(TypeError, match="not a single string"):
        plain("one")
    with pytest.raises(TypeError, match="units must be strings"):
        plain(["one", 2])  # type: ignore[list-item]


class StandInTokenizer:
    """One token per word, returned in the form a real tokeniser uses."""

    def __call__(
        self, texts: list[str], **options: object
    ) -> dict[str, list[list[int]]]:
        return {"input_ids": [[0] * len(text.split()) for text in texts]}


class StandInModelWithLimit(StandInModel):
    """A model that accepts four tokens and can put a prompt before its input."""

    def __init__(self) -> None:
        self.max_seq_length = 4
        self.tokenizer = StandInTokenizer()
        self.prompts = {"lead": "one two "}
        self.default_prompt_name: str | None = None


@pytest.fixture
def limited() -> SentenceTransformerEmbedder:
    embedder = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
    embedder.model_name = "stand-in"
    embedder.revision = None
    embedder.batch_size = 8
    embedder.max_tokens = 4
    embedder._model = StandInModelWithLimit()
    return embedder


def test_units_beyond_the_limit_are_counted(
    limited: SentenceTransformerEmbedder,
) -> None:
    units = ["a b c d", "a b c d e", "a"]
    assert limited.count_truncated(units) == 1
    assert limited.count_truncated([]) == 0
    with pytest.warns(TruncationWarning, match=r"1 of 3 units .* 4 tokens"):
        limited(units)


def test_the_prompt_of_a_model_counts_towards_the_limit(
    limited: SentenceTransformerEmbedder,
) -> None:
    units = ["a b", "a b c"]
    assert limited.count_truncated(units) == 0
    # With its prompt of two words, the model sees four and five tokens.
    limited._model.default_prompt_name = "lead"
    assert limited.count_truncated(units) == 1
    # A default prompt that the model does not define is no prompt.
    limited._model.default_prompt_name = "absent"
    assert limited.count_truncated(units) == 0


# Behaviour with the library


def test_records_what_must_be_reported(model_path: str) -> None:
    embedder = SentenceTransformerEmbedder(model_path, revision=None, device="cpu")
    assert embedder.model_name == model_path
    assert embedder.revision is None
    assert embedder.max_tokens == MAX_TOKENS
    assert embedder.batch_size == 32


def test_returns_one_float64_row_per_unit(embed: SentenceTransformerEmbedder) -> None:
    vectors = embed(UNITS)
    assert vectors.shape == (len(UNITS), 16)
    assert vectors.dtype == np.float64
    assert np.isfinite(vectors).all()


def test_accepts_any_sequence_of_strings(embed: SentenceTransformerEmbedder) -> None:
    assert np.array_equal(embed(tuple(UNITS)), embed(UNITS))


def test_no_units_gives_an_empty_array_of_the_right_width(
    embed: SentenceTransformerEmbedder,
) -> None:
    vectors = embed([])
    assert vectors.shape == (0, 16)
    assert vectors.dtype == np.float64


def test_same_input_gives_same_output(embed: SentenceTransformerEmbedder) -> None:
    assert np.array_equal(embed(UNITS), embed(UNITS))


def test_batch_size_does_not_change_the_result(model_path: str) -> None:
    one = SentenceTransformerEmbedder(model_path, device="cpu", batch_size=1)
    four = SentenceTransformerEmbedder(model_path, device="cpu", batch_size=4)
    assert np.allclose(one(UNITS), four(UNITS), atol=1e-6)


def test_a_unit_is_embedded_independently_of_the_others(
    embed: SentenceTransformerEmbedder,
) -> None:
    alone = embed([UNITS[0]])
    together = embed(UNITS)
    assert np.allclose(alone[0], together[0], atol=1e-6)


def test_rejects_a_single_string_and_non_strings(
    embed: SentenceTransformerEmbedder,
) -> None:
    with pytest.raises(TypeError, match="not a single string"):
        embed("The cat sat on the mat.")
    with pytest.raises(TypeError, match="units must be strings"):
        embed(["a cat", 3])  # type: ignore[list-item]


def test_short_units_raise_no_warning(embed: SentenceTransformerEmbedder) -> None:
    assert embed.count_truncated(UNITS) == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", TruncationWarning)
        embed(UNITS)


def test_long_units_are_counted_and_reported(
    embed: SentenceTransformerEmbedder,
) -> None:
    long_unit = " ".join(["the cat sat on the mat and"] * 4)
    units = [UNITS[0], long_unit, long_unit]
    assert embed.count_truncated(units) == 2
    with pytest.warns(TruncationWarning, match=r"2 of 3 units .* 12 tokens"):
        vectors = embed(units)
    assert vectors.shape == (3, 16)
    # Only the beginning is embedded: the text beyond the limit has no effect.
    with pytest.warns(TruncationWarning):
        extended = embed([long_unit + " the dog ran in the park"])
    assert np.allclose(extended[0], vectors[1], atol=1e-6)


PROMPT = "the dog ran in the park and "


@pytest.fixture(scope="module")
def prompted_model_path(
    model_path: str, tmp_path_factory: pytest.TempPathFactory
) -> str:
    """Save the same model, configured to put a prompt before every unit."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_path, device="cpu", prompts={"lead": PROMPT}, default_prompt_name="lead"
    )
    saved = tmp_path_factory.mktemp("prompted-model") / "model"
    model.save(str(saved))
    return str(saved)


def test_a_prompt_configured_in_the_model_counts_towards_the_limit(
    embed: SentenceTransformerEmbedder, prompted_model_path: str
) -> None:
    prompted = SentenceTransformerEmbedder(prompted_model_path, device="cpu")
    unit = "the cat sat on the mat"
    # Six words fit within the twelve tokens of the model; thirteen do not.
    assert embed.count_truncated([unit]) == 0
    assert prompted.count_truncated([unit]) == 1
    # The model does embed the prompt followed by the unit, cut at the limit.
    with pytest.warns(TruncationWarning, match="1 of 1 units"):
        with_prompt = prompted([unit])
    with pytest.warns(TruncationWarning):
        spelled_out = embed([PROMPT + unit])
    assert np.allclose(with_prompt, spelled_out, atol=1e-6)
    assert not np.allclose(with_prompt, embed([unit]), atol=1e-3)


def test_embeddings_feed_the_coherence_measures(
    embed: SentenceTransformerEmbedder,
) -> None:
    result = coherence(embed(UNITS), boundaries=[2])
    assert result.n_units == 4
    assert -1.0 <= result.local_coherence <= 1.0
    assert -1.0 <= result.global_coherence <= 1.0


# A published model (needs network access; not run by default)

PUBLISHED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Commit of the model repository on the Hugging Face Hub, so that every run
# uses the same weights.
PUBLISHED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

SENTENCES = [
    "The kitten slept on the sofa all afternoon.",
    "A small cat was napping on the couch.",
    "Interest rates rose again after the central bank meeting.",
]


@pytest.fixture(scope="module")
def published() -> SentenceTransformerEmbedder:
    pytest.importorskip("sentence_transformers")
    return SentenceTransformerEmbedder(
        PUBLISHED_MODEL, revision=PUBLISHED_REVISION, device="cpu"
    )


# The download can raise warnings that are not ours to act on, hence the
# filter: in this module they would otherwise be turned into errors.
@pytest.mark.network
@pytest.mark.filterwarnings("default")
def test_published_model_is_loaded_at_the_requested_revision(
    published: SentenceTransformerEmbedder,
) -> None:
    assert published.model_name == PUBLISHED_MODEL
    assert published.revision == PUBLISHED_REVISION
    assert published.max_tokens == 256
    vectors = published(SENTENCES)
    assert vectors.shape == (3, 384)
    assert vectors.dtype == np.float64
    assert np.array_equal(published(SENTENCES), vectors)


@pytest.mark.network
@pytest.mark.filterwarnings("default")
def test_published_model_orders_related_and_unrelated_sentences(
    published: SentenceTransformerEmbedder,
) -> None:
    vectors = published(SENTENCES)
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    assert unit[0] @ unit[1] > unit[0] @ unit[2]
    assert unit[0] @ unit[1] > unit[1] @ unit[2]
    result = coherence(vectors, reference=vectors[0], boundaries=[2])
    assert result.local_coherence_within_segments > result.local_coherence


@pytest.mark.network
@pytest.mark.filterwarnings("default")
def test_published_model_reports_units_beyond_its_limit(
    published: SentenceTransformerEmbedder,
) -> None:
    long_unit = " ".join(["The kitten slept on the sofa."] * 60)
    assert published.count_truncated(SENTENCES) == 0
    assert published.count_truncated([*SENTENCES, long_unit]) == 1
    with pytest.warns(TruncationWarning, match=r"1 of 4 units .* 256 tokens"):
        vectors = published([*SENTENCES, long_unit])
    assert vectors.shape == (4, 384)
