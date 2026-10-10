"""Embedding of text units.

The coherence measures take one vector per unit of text. Any method that maps
a list of strings to an array with one row per string can supply them. This
module provides an adapter for the models of the ``sentence-transformers``
library (Reimers & Gurevych, 2019), which is an optional dependency::

    pip install "discoursekit[embeddings]"

The embedding model is part of the definition of a coherence value. There is
no default model: the model, and its revision, must be chosen and reported.

Reference
---------
Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using
Siamese BERT-networks. In Proceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the 9th International Joint
Conference on Natural Language Processing (EMNLP-IJCNLP), 3982-3992.
https://doi.org/10.18653/v1/D19-1410
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["SentenceTransformerEmbedder", "TruncationWarning"]


class TruncationWarning(UserWarning):
    """A unit was longer than the model accepts and was cut short."""


class SentenceTransformerEmbedder:
    """Embed text units with a ``sentence-transformers`` model.

    Parameters
    ----------
    model
        Name of a model on the Hugging Face Hub, or path to a model saved on
        disk.
    revision
        Commit hash, tag or branch of the model on the Hub. Give a commit
        hash to make sure the same weights are used every time; without it
        the current version of the model is downloaded.
    device
        Device passed to the library, for example ``"cpu"``. By default the
        library chooses.
    batch_size
        Number of units encoded together.

    Attributes
    ----------
    model_name, revision
        As given. Report both with any result.
    max_tokens
        Largest number of model tokens a unit may have; longer units are
        truncated by the model. ``None`` if the model does not state it.

    Notes
    -----
    A unit longer than ``max_tokens`` is truncated by the model and only its
    beginning is embedded. When the model exposes its tokeniser, calling the
    embedder issues a :class:`TruncationWarning` giving the number of units
    affected. The count allows for the prompt that some models are configured
    to place before every input. It can be too low for a model that adds
    other text of its own.

    A unit that occurs several times is embedded once, and each of its
    occurrences receives the same vector. Passed to the model several times,
    the same text can receive vectors that differ in their last digits,
    depending on the units it is batched with; the test of
    :func:`discoursekit.segment_topics` would then treat its occurrences as
    different units.

    Embeddings can differ in their last digits between devices, batch sizes
    and versions of the underlying libraries. For an exactly repeatable
    analysis, compute the embeddings once and store them.

    Examples
    --------
    >>> embed = SentenceTransformerEmbedder(  # doctest: +SKIP
    ...     "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )
    >>> embed(["A first sentence.", "A second one."]).shape  # doctest: +SKIP
    (2, 384)
    """

    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if not isinstance(model, str) or not model:
            raise TypeError("model must be a non-empty string: a model name or a path")
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
            raise TypeError(
                f"batch_size must be an int, got {type(batch_size).__name__}"
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "SentenceTransformerEmbedder needs the sentence-transformers "
                'library: pip install "discoursekit[embeddings]"'
            ) from error

        self.model_name: str = model
        self.revision: str | None = revision
        self.batch_size: int = int(batch_size)
        self._model: Any = SentenceTransformer(model, device=device, revision=revision)
        limit = getattr(self._model, "max_seq_length", None)
        self.max_tokens: int | None = int(limit) if limit is not None else None

    def _dimensions(self) -> int:
        # The method was renamed in sentence-transformers 6.
        method = getattr(self._model, "get_embedding_dimension", None)
        if method is None:
            method = self._model.get_sentence_embedding_dimension
        return int(method())

    def _default_prompt(self) -> str:
        """Return the text the model places before every input, if any."""
        name = getattr(self._model, "default_prompt_name", None)
        prompts = getattr(self._model, "prompts", None)
        if name is None or not isinstance(prompts, dict):
            return ""
        prompt = prompts.get(name)
        return prompt if isinstance(prompt, str) else ""

    def count_truncated(self, units: Sequence[str]) -> int | None:
        """Return how many units exceed ``max_tokens`` and would be truncated.

        ``None`` if the model does not state a limit or does not expose a
        tokeniser from which the length of a unit can be obtained.
        """
        tokenizer = getattr(self._model, "tokenizer", None)
        if self.max_tokens is None or tokenizer is None or not callable(tokenizer):
            return None
        prompt = self._default_prompt()
        texts = [prompt + unit for unit in units]
        if not texts:
            return 0
        encoded = tokenizer(texts, truncation=False, verbose=False)
        return sum(len(ids) > self.max_tokens for ids in encoded["input_ids"])

    def __call__(self, units: Sequence[str]) -> NDArray[np.float64]:
        """Return an array of shape ``(len(units), n_dimensions)``.

        Equal units have equal rows: each distinct unit is passed to the
        model once.

        Raises
        ------
        TypeError
            If ``units`` is a single string or contains anything other than
            strings.
        """
        if isinstance(units, str):
            raise TypeError("units must be a sequence of strings, not a single string")
        texts = list(units)
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f"units must be strings, got {type(text).__name__}")
        if not texts:
            return np.empty((0, self._dimensions()), dtype=np.float64)

        truncated = self.count_truncated(texts)
        if truncated:
            warnings.warn(
                f"{truncated} of {len(texts)} units are longer than the "
                f"{self.max_tokens} tokens accepted by {self.model_name!r} and "
                "were truncated; only their beginning is embedded",
                TruncationWarning,
                stacklevel=2,
            )
        distinct = list(dict.fromkeys(texts))
        encoded = self._model.encode(
            distinct,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        vectors = np.asarray(encoded, dtype=np.float64)
        if len(distinct) < len(texts):
            row = {text: index for index, text in enumerate(distinct)}
            vectors = vectors[[row[text] for text in texts]]
        return vectors
