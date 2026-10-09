"""discoursekit: linguistic measures for discourse research.

A research instrument for measuring properties of text. It is not a clinical
tool, and its outputs describe texts, not people.
"""

from importlib.metadata import PackageNotFoundError, version

from discoursekit.embedding import SentenceTransformerEmbedder, TruncationWarning
from discoursekit.lexical import (
    DEFAULT_WINDOW,
    LexicalDiversity,
    lexical_diversity,
    mattr,
)
from discoursekit.semantic import (
    DEFAULT_ALPHA,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_PERMUTATIONS,
    Coherence,
    TopicSegmentation,
    adjacent_similarities,
    coherence,
    segment_topics,
)
from discoursekit.tokenization import UNICODE_VERSION, tokenize

try:
    __version__ = version("discoursekit")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_PERMUTATIONS",
    "DEFAULT_WINDOW",
    "UNICODE_VERSION",
    "Coherence",
    "LexicalDiversity",
    "SentenceTransformerEmbedder",
    "TopicSegmentation",
    "TruncationWarning",
    "__version__",
    "adjacent_similarities",
    "coherence",
    "lexical_diversity",
    "mattr",
    "segment_topics",
    "tokenize",
]
