"""Lexical diversity: the moving-average type-token ratio (MATTR).

Rationale
---------
The plain type-token ratio (number of types divided by number of tokens)
falls as a text gets longer, because frequent words keep recurring while new
words become rarer. Two samples of different length cannot be compared with
it: a difference in the ratio may reflect nothing but a difference in length.

The moving-average type-token ratio (Covington & McFall, 2010) is the ratio
computed inside a window of fixed size, moved one token at a time over the
whole text, and averaged. Every ratio entering the average is computed on
the same number of tokens, which removes the systematic dependence on the
length of the text.

Design decisions
----------------
* The plain type-token ratio is not offered as a measure.
* A text with fewer tokens than the window has no value at that window size,
  and the result is NaN. The plain ratio is never substituted, since that
  would make the shortest samples appear the most diverse.

Reference
---------
Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: The
moving-average type-token ratio (MATTR). Journal of Quantitative Linguistics,
17(2), 94-100. https://doi.org/10.1080/09296171003643098
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral

from discoursekit.tokenization import _excluded_tokens, tokenize

__all__ = ["DEFAULT_WINDOW", "LexicalDiversity", "lexical_diversity", "mattr"]

#: Default window size, in tokens. This is a convention of this library, not
#: a standard: published work uses different sizes, and values obtained with
#: different windows are not comparable. Always report the window.
DEFAULT_WINDOW = 50


def _check_window(window: int) -> int:
    """Validate the window size and return it as a plain ``int``."""
    # bool is a subclass of int; mattr(tokens, True) is always a mistake.
    # Integer types of other libraries, such as NumPy, are accepted.
    if isinstance(window, bool) or not isinstance(window, Integral):
        raise TypeError(f"window must be an int, got {type(window).__name__}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return int(window)


def _mattr_and_types(tokens: Sequence[str], window: int) -> tuple[float, int]:
    """Return (MATTR, number of types in the whole sequence) in one pass."""
    n = len(tokens)
    if n < window:
        return math.nan, len(set(tokens))

    in_window = Counter(tokens[:window])
    types_in_window = len(in_window)
    sum_of_types = types_in_window
    seen = set(in_window)

    for i in range(window, n):
        leaving = tokens[i - window]
        in_window[leaving] -= 1
        if in_window[leaving] == 0:
            del in_window[leaving]
            types_in_window -= 1

        entering = tokens[i]
        if entering not in in_window:
            types_in_window += 1
        in_window[entering] += 1
        seen.add(entering)

        sum_of_types += types_in_window

    n_windows = n - window + 1
    # One integer sum, one division: the result is exactly reproducible
    # across platforms, with no accumulated floating-point error.
    return sum_of_types / (window * n_windows), len(seen)


def mattr(tokens: Sequence[str], window: int = DEFAULT_WINDOW) -> float:
    """Moving-average type-token ratio of a token sequence.

    Parameters
    ----------
    tokens
        Tokens in order of appearance, for example the output of
        :func:`discoursekit.tokenize`. Comparison is exact string equality:
        case folding or lemmatisation must be done beforehand.
    window
        Window size in tokens.

    Returns
    -------
    float
        Mean, over all ``len(tokens) - window + 1`` windows, of the proportion
        of distinct tokens in the window. Lies in ``[1 / window, 1]``.
        ``nan`` if there are fewer tokens than ``window``.

    Raises
    ------
    TypeError
        If ``tokens`` is a single string (tokenise it first) or ``window`` is
        not an integer.
    ValueError
        If ``window`` is smaller than 1.

    Examples
    --------
    >>> mattr(["a", "b", "a", "b", "c"], window=3)  # (2/3 + 2/3 + 3/3) / 3
    0.7777777777777778
    """
    if isinstance(tokens, str):
        raise TypeError(
            "tokens must be a sequence of tokens, not a single string; "
            "call discoursekit.tokenize(text) first"
        )
    return _mattr_and_types(tokens, _check_window(window))[0]


@dataclass(frozen=True)
class LexicalDiversity:
    """Lexical diversity of one text, with what is needed to interpret it.

    Attributes
    ----------
    mattr
        Moving-average type-token ratio; ``nan`` if ``n_tokens < window``.
    window
        Window size used, in tokens. Values are only comparable at equal
        window size.
    n_tokens
        Number of tokens. Report it and use it as a covariate: MATTR removes
        the systematic length effect of TTR, not every effect of length.
    n_types
        Number of distinct tokens in the whole text.
    n_windows
        Number of windows averaged (0 if the text is shorter than the
        window). With few windows the estimate rests on little data.
    casefold
        Whether the tokens were case folded.
    excluded
        Tokens that were to be left out, as normalised for the comparison
        and sorted by code point, whether or not they occur in the text.
        Empty if none was given.
    """

    mattr: float
    window: int
    n_tokens: int
    n_types: int
    n_windows: int
    casefold: bool
    excluded: tuple[str, ...]


def lexical_diversity(
    text: str,
    window: int = DEFAULT_WINDOW,
    *,
    casefold: bool = True,
    exclude: Iterable[str] = (),
) -> LexicalDiversity:
    """Tokenise ``text`` and compute its lexical diversity.

    Parameters
    ----------
    text
        Raw text or transcript.
    window
        MATTR window size in tokens.
    casefold, exclude
        Passed to :func:`discoursekit.tokenize`.

    Returns
    -------
    LexicalDiversity

    Examples
    --------
    >>> result = lexical_diversity("the cat saw the dog and the dog saw the cat",
    ...                            window=5)
    >>> result.n_tokens, result.n_types, result.n_windows
    (11, 5, 7)
    >>> result.mattr  # 28 distinct tokens over 7 windows of 5
    0.8
    """
    window = _check_window(window)
    if not isinstance(exclude, str):
        exclude = tuple(exclude)  # it is read twice, and may be an iterator
    tokens = tokenize(text, casefold=casefold, exclude=exclude)
    value, n_types = _mattr_and_types(tokens, window)
    return LexicalDiversity(
        mattr=value,
        window=window,
        n_tokens=len(tokens),
        n_types=n_types,
        n_windows=max(0, len(tokens) - window + 1),
        casefold=bool(casefold),
        excluded=tuple(sorted(_excluded_tokens(exclude, casefold))),
    )
