"""Embeddings that a model gives to a unit that occurs several times.

A ``sentence-transformers`` model encodes its input in batches and pads the
units of a batch to the length of the longest. A unit that occurs several
times in the input can fall into different batches, and the arithmetic of a
padded unit is not that of the same unit unpadded to the last digit. The
test for topic boundaries treats units as the same only when their
embeddings are equal (see ``docs/metrics.md``, section 6), so the question
matters for texts in which a sentence is repeated.

This script counts how often it happens. Each input holds 6 to 13 distinct
units of 2 to 29 words, one of which is placed three times. The input is
encoded by the model directly, in batches of four, and by
:class:`discoursekit.SentenceTransformerEmbedder`, which passes each
distinct unit to the model once. The models are small, have random weights
and are built on the spot: the script needs no network access, and its
result says nothing about the quality of a model.

Usage, from the root of the repository::

    pip install "discoursekit[embeddings]"
    python studies/repeated_unit_embeddings.py > results.md

The counts depend on the platform and on the versions of the libraries,
which the output records. The run takes less than a minute.
"""

from __future__ import annotations

import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import discoursekit
from discoursekit import SentenceTransformerEmbedder

WORDS = (
    "the a cat dog sat on mat ran in park stock market fell rose bank interest "
    "rate and of to"
).split()
N_INPUTS = 200
BATCH_SIZE = 4
MAX_TOKENS = 48

# Width and number of layers of the two models.
MODELS = ((16, 1), (64, 2))


def build_model(root: Path, width: int, layers: int) -> str:
    """Build a small model with random weights, save it and return its path."""
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

    bert = root / "bert"
    bert.mkdir(parents=True)
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    (bert / "vocab.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    torch.manual_seed(0)
    configuration = BertConfig(
        vocab_size=len(vocabulary),
        hidden_size=width,
        num_hidden_layers=layers,
        num_attention_heads=2,
        intermediate_size=2 * width,
        max_position_embeddings=64,
    )
    BertModel(configuration).save_pretrained(bert)
    BertTokenizerFast(vocab_file=str(bert / "vocab.txt")).save_pretrained(bert)
    transformer = Transformer(str(bert), max_seq_length=MAX_TOKENS)
    pooling = Pooling(width, pooling_mode="mean")
    saved = root / "model"
    SentenceTransformer(modules=[transformer, pooling], device="cpu").save(str(saved))
    return str(saved)


def draw_input(state: np.random.RandomState) -> tuple[list[str], list[int]]:
    """Draw one input and the places of the unit that occurs three times."""

    def integer(low: int, high: int) -> int:
        # The type is given because the default has 32 bits on Windows only.
        return int(state.randint(low, high, dtype=np.int64))

    units: list[str] = []
    n_distinct = integer(6, 14)
    while len(units) < n_distinct:
        words = [WORDS[integer(0, len(WORDS))] for _ in range(integer(2, 30))]
        unit = " ".join(words)
        if unit not in units:
            units.append(unit)
    repeated = units[integer(0, len(units))]
    for _ in range(2):
        units.insert(integer(0, len(units) + 1), repeated)
    return units, [place for place, unit in enumerate(units) if unit == repeated]


def largest_difference(vectors: NDArray[np.float64], places: list[int]) -> float:
    """Largest difference between the vectors of the repeated unit."""
    first = vectors[places[0]]
    return max(float(np.abs(vectors[place] - first).max()) for place in places[1:])


def count_unequal(path: str) -> tuple[int, float, int]:
    """Count the inputs in which the repeated unit has unequal vectors.

    Returns the count and the largest difference when the model encodes the
    input directly, and the count when the adapter of the library does.
    """
    embed = SentenceTransformerEmbedder(path, device="cpu", batch_size=BATCH_SIZE)
    model: Any = embed._model
    state = np.random.RandomState(0)
    direct = adapted = 0
    largest = 0.0
    for _ in range(N_INPUTS):
        units, places = draw_input(state)
        encoded = model.encode(
            units,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        difference = largest_difference(np.asarray(encoded, dtype=np.float64), places)
        direct += difference > 0.0
        largest = max(largest, difference)
        adapted += largest_difference(embed(units), places) > 0.0
    return direct, largest, adapted


def report() -> list[str]:
    """Return the report as lines of Markdown."""
    import sentence_transformers
    import torch
    import transformers

    lines = [
        "# Embeddings of a unit that occurs several times",
        "",
        "Produced by `studies/repeated_unit_embeddings.py`. The models are "
        "small and have random weights: see the documentation of the script.",
        "",
        f"Parameters: {N_INPUTS} inputs for each model, of 8 to 15 units of 2 "
        "to 29 words, one unit occurring three times; batches of "
        f"{BATCH_SIZE} units; at most {MAX_TOKENS} tokens in a unit.",
        "",
        f"Software: discoursekit {discoursekit.__version__}, "
        f"sentence-transformers {sentence_transformers.__version__}, "
        f"transformers {transformers.__version__}, PyTorch {torch.__version__}, "
        f"NumPy {np.__version__}, Python {platform.python_version()}, "
        f"{platform.system()} {platform.machine()}, CPU.",
        "",
        "Number of inputs in which the three vectors of the repeated unit are "
        "not all equal, when the model encodes the input directly and when "
        "the adapter of the library does:",
        "",
        "| Model | Encoded directly | Largest difference | Through the adapter |",
        "|---|---:|---:|---:|",
    ]
    with tempfile.TemporaryDirectory() as folder:
        for width, layers in MODELS:
            path = build_model(Path(folder) / f"{width}-{layers}", width, layers)
            direct, largest, adapted = count_unequal(path)
            plural = "layer" if layers == 1 else "layers"
            lines.append(
                f"| {width} dimensions, {layers} {plural} | {direct} of {N_INPUTS} "
                f"| {largest:.1e} | {adapted} of {N_INPUTS} |"
            )
    return lines


if __name__ == "__main__":
    print("\n".join(report()))
