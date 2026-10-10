"""Word tokenisation based on the Unicode word boundary specification.

Every lexical measure depends on what counts as a word. This module fixes that
definition by reference to a published standard, so that a reported value can
be recomputed exactly, in this library or with any other implementation of
the standard.

Definition
----------
A token is a word as delimited by the default word boundary rules of Unicode
Standard Annex #29, "Unicode Text Segmentation" (UAX #29), provided that it
contains at least one letter or digit. Segments made only of white space,
punctuation or symbols are discarded. The character properties the rules rely
on are taken from the Unicode Character Database, version
:data:`UNICODE_VERSION`, shipped with this library, so the word boundaries do
not depend on the Python version.

The rules are applied with one tailoring, which makes white space separate
tokens in every case. The default rules give U+202F NARROW NO-BREAK SPACE the
class of the underscore (ExtendNumLet) and so join it to the letters and
digits on either side. In running text this character is a space: since
release 42, the Unicode Common Locale Data Repository (CLDR) places it before
AM and PM in English time formats, and it separates groups of digits in
French number formats. Under the default rules ``3:45 PM`` written in that
way would give the word ``45 PM``. The tokeniser treats U+202F as U+00A0
NO-BREAK SPACE. The same word boundaries are obtained from any implementation
of the default rules by replacing U+202F with U+00A0 in the text beforehand.

For English text the rules have the following consequences.

* An apostrophe or a full stop between two letters does not end a word
  (rules WB6 and WB7): ``don't``, ``o'clock`` and ``e.g`` are single tokens.
* A full stop, comma or apostrophe between two digits does not end a number
  (rules WB11 and WB12): ``3.14`` and ``1,000`` are single tokens.
* Letters, digits and underscores run together (rules WB5 to WB13b):
  ``covid19`` and ``snake_case`` are single tokens.
* A hyphen always ends a word: ``well-known`` gives ``well`` and ``known``.
  UAX #29 deliberately leaves the hyphen out of the default definition.
* Combining marks and format characters belong to the character that
  precedes them (rule WB4).

Normalisation
-------------
Two strings that differ only in encoding should give the same tokens. Before
segmentation the text is converted to Unicode Normalization Form C (UAX #15).
Each token is then normalised as follows.

1. The single quotation marks U+2019 and U+2018 and the fullwidth apostrophe
   U+FF07, which UAX #29 treats like the apostrophe inside a word, are
   replaced by the ASCII apostrophe U+0027.
2. Format characters (general category Cf, for example the soft hyphen) are
   removed, and so is a space that rule WB4 leaves attached to the characters
   after it. No token contains white space.
3. Unless ``casefold=False``, the token is case folded. Case folding is the
   operation the Unicode Standard defines for caseless matching; the
   canonical form is used (decomposition, case folding, recomposition).

Limits
------
* Tokens are word forms, not lemmas: ``walk`` and ``walked`` are two types.
* The default rules are language independent. In languages with elision they
  keep the clitic attached (French ``l'homme`` is one token); UAX #29 notes
  that such languages may need tailored rules. Token counts are therefore not
  comparable across languages.
* A full stop or colon typed between two letters without a space joins them
  into one token (``end.Next``). Such typing errors are best corrected before
  tokenisation.
* A number written with spaces between groups of digits (``10 000``) gives
  one token per group, whatever the kind of space.
* Nothing is removed unless ``exclude`` is given. Filled pauses, transcription
  codes and speaker labels are tokens like any other. Their treatment is a
  decision of the study and must be reported.
* Normalisation, case folding and the test for a letter or digit use the
  Unicode database of the Python interpreter, which can be older than the one
  used for the word boundaries. Tokens can therefore differ between Python
  versions, but only in text that contains characters added to Unicode after
  the version known to the older interpreter.

References
----------
Unicode Consortium. Unicode Standard Annex #29: Unicode Text Segmentation.
https://www.unicode.org/reports/tr29/

Unicode Consortium. Unicode Standard Annex #15: Unicode Normalization Forms.
https://www.unicode.org/reports/tr15/

Unicode Consortium. Unicode Common Locale Data Repository (CLDR), release 42.
https://cldr.unicode.org/downloads/cldr-42
"""

from __future__ import annotations

import unicodedata
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from functools import cache

from discoursekit import _wordbreak_data as _wb

__all__ = ["UNICODE_VERSION", "tokenize"]

#: Version of the Unicode Character Database used for word segmentation.
UNICODE_VERSION: str = _wb.UNICODE_VERSION

_NEWLINES = frozenset({_wb.CR, _wb.LF, _wb.NEWLINE})
# Characters that rule WB4 attaches to the character before them.
_IGNORED = frozenset({_wb.EXTEND, _wb.FORMAT, _wb.ZWJ})
# AHLetter in the notation of UAX #29.
_LETTERS = frozenset({_wb.ALETTER, _wb.HEBREW_LETTER})
# (MidLetter | MidNumLetQ) and (MidNum | MidNumLetQ) in the notation of UAX #29.
_MID_LETTER = frozenset({_wb.MIDLETTER, _wb.MIDNUMLET, _wb.SINGLE_QUOTE})
_MID_NUMBER = frozenset({_wb.MIDNUM, _wb.MIDNUMLET, _wb.SINGLE_QUOTE})
_WORD_CORE = frozenset({_wb.ALETTER, _wb.HEBREW_LETTER, _wb.NUMERIC, _wb.KATAKANA})

# Characters that UAX #29 classes with the apostrophe as word-internal
# punctuation (MidNumLet): the two single quotation marks and the fullwidth
# apostrophe. U+02BC MODIFIER LETTER APOSTROPHE is a letter and is left alone.
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\uff07": "'"})

# The one tailoring of the default rules. U+202F NARROW NO-BREAK SPACE is the
# only white space character that they join to letters and digits (class
# ExtendNumLet). It is replaced by U+00A0 NO-BREAK SPACE (class Other) before
# segmentation, so that white space separates tokens in every case.
_SPACES = str.maketrans({"\u202f": "\u00a0"})


@cache
def _value(character: str) -> int:
    """Return the Word_Break class of a character, with the pictographic flag."""
    return _wb.VALUES[bisect_right(_wb.STARTS, ord(character)) - 1]


def _next_class(values: Sequence[int], index: int) -> int:
    """Return the class of the first character after ``index`` kept by WB4."""
    for position in range(index + 1, len(values)):
        following = values[position] & _wb.CLASS_MASK
        if following not in _IGNORED:
            return following
    return -1


def _joined(
    before_previous: int,
    previous: int,
    current: int,
    values: Sequence[int],
    index: int,
    indicators: int,
) -> bool:
    """Tell whether rules WB5 to WB16 forbid a break before ``values[index]``.

    ``previous`` and ``before_previous`` are the classes of the two characters
    to the left of the position, once rule WB4 has been applied. ``indicators``
    is the number of consecutive regional indicators ending at ``previous``.
    """
    if previous in _LETTERS:
        if current in _LETTERS or current == _wb.NUMERIC:
            return True  # WB5, WB9
        if current == _wb.EXTENDNUMLET:
            return True  # WB13a
        if current in _MID_LETTER and _next_class(values, index) in _LETTERS:
            return True  # WB6
        if previous == _wb.HEBREW_LETTER:
            if current == _wb.SINGLE_QUOTE:
                return True  # WB7a
            if current == _wb.DOUBLE_QUOTE:
                return _next_class(values, index) == _wb.HEBREW_LETTER  # WB7b
        return False
    if previous == _wb.NUMERIC:
        if current == _wb.NUMERIC or current in _LETTERS:
            return True  # WB8, WB10
        if current == _wb.EXTENDNUMLET:
            return True  # WB13a
        return current in _MID_NUMBER and (
            _next_class(values, index) == _wb.NUMERIC
        )  # WB12
    if previous == _wb.KATAKANA:
        return current in (_wb.KATAKANA, _wb.EXTENDNUMLET)  # WB13, WB13a
    if previous == _wb.EXTENDNUMLET:
        return current in _WORD_CORE or current == _wb.EXTENDNUMLET  # WB13b, WB13a
    if previous == _wb.REGIONAL_INDICATOR:
        # WB15, WB16: regional indicators pair up from the start of their run.
        return current == _wb.REGIONAL_INDICATOR and indicators % 2 == 1
    if current in _LETTERS:
        if previous in _MID_LETTER and before_previous in _LETTERS:
            return True  # WB7
        return (
            previous == _wb.DOUBLE_QUOTE
            and before_previous == _wb.HEBREW_LETTER
            and current == _wb.HEBREW_LETTER
        )  # WB7c
    return (
        current == _wb.NUMERIC
        and previous in _MID_NUMBER
        and before_previous == _wb.NUMERIC
    )  # WB11


def _word_segments(text: str) -> list[str]:
    """Split ``text`` at the default word boundaries of UAX #29.

    Every character of ``text`` belongs to exactly one segment; joining the
    segments gives ``text`` back. The comments name the rule of the annex
    that each branch implements.
    """
    if not text:
        return []
    values = [_value(character) for character in text]
    mask = _wb.CLASS_MASK
    segments = []
    start = 0
    # Class of the character just before the position, as it stands in the text.
    left = values[0] & mask
    # Classes of the last two characters before the position after WB4, that
    # is, not counting the Extend, Format and ZWJ characters attached to them.
    previous = left
    before_previous = -1
    indicators = 1 if previous == _wb.REGIONAL_INDICATOR else 0

    for index in range(1, len(text)):
        value = values[index]
        current = value & mask
        attached = False
        if left == _wb.CR and current == _wb.LF:
            boundary = False  # WB3
        elif left in _NEWLINES or current in _NEWLINES:
            boundary = True  # WB3a, WB3b
        elif left == _wb.ZWJ and value & _wb.EXTENDED_PICTOGRAPHIC:
            boundary = False  # WB3c
        elif left == _wb.WSEGSPACE and current == _wb.WSEGSPACE:
            boundary = False  # WB3d
        elif current in _IGNORED:
            boundary = False  # WB4
            attached = True
        else:
            boundary = not _joined(
                before_previous, previous, current, values, index, indicators
            )  # WB5 to WB16, then WB999

        if boundary:
            segments.append(text[start:index])
            start = index
        left = current
        if not attached:
            before_previous = previous
            previous = current
            if current == _wb.REGIONAL_INDICATOR:
                indicators += 1
            else:
                indicators = 0

    segments.append(text[start:])
    return segments


def _normalise(token: str, casefold: bool) -> str:
    """Return the canonical form of a token (see the module documentation)."""
    if token.isascii():
        return token.lower() if casefold else token
    token = token.translate(_APOSTROPHES)
    token = "".join(
        character
        for character in token
        if unicodedata.category(character) != "Cf" and not character.isspace()
    )
    if casefold:
        token = unicodedata.normalize("NFD", token).casefold()
    return unicodedata.normalize("NFC", token)


def _words(text: str, casefold: bool) -> list[str]:
    """Return the normalised tokens of ``text``, before any exclusion."""
    text = unicodedata.normalize("NFC", text).translate(_SPACES)
    return [
        _normalise(segment, casefold)
        for segment in _word_segments(text)
        if any(character.isalnum() for character in segment)
    ]


def tokenize(
    text: str, *, casefold: bool = True, exclude: Iterable[str] = ()
) -> list[str]:
    """Split ``text`` into word tokens.

    Parameters
    ----------
    text
        Raw text or transcript.
    casefold
        If true (default), tokens are case folded, so that ``The`` and ``the``
        are the same type.
    exclude
        Tokens to leave out of the result, for example filled pauses
        (``{"uh", "um"}``). Each entry must be a single token; it is
        normalised in the same way as the text before comparison. Nothing is
        excluded by default.

    Returns
    -------
    list of str
        Tokens in order of appearance. Empty if ``text`` contains no word.

    Raises
    ------
    TypeError
        If ``text`` is not a string, or ``exclude`` is a single string.
    ValueError
        If an entry of ``exclude`` is not exactly one token.

    Examples
    --------
    >>> tokenize("Well, I don't know... It's a well-known problem!")
    ['well', 'i', "don't", 'know', "it's", 'a', 'well', 'known', 'problem']
    >>> tokenize("About 1,500 people, i.e. 3.5% of them.")
    ['about', '1,500', 'people', 'i.e', '3.5', 'of', 'them']
    >>> tokenize("So, um, I went, uh, home.", exclude={"um", "uh"})
    ['so', 'i', 'went', 'home']
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    excluded = _excluded_tokens(exclude, casefold)
    tokens = _words(text, casefold)
    if excluded:
        tokens = [token for token in tokens if token not in excluded]
    return tokens


def _excluded_tokens(exclude: Iterable[str], casefold: bool) -> set[str]:
    """Validate the tokens to leave out and normalise them like the text."""
    if isinstance(exclude, str):
        raise TypeError("exclude must be a collection of tokens, not a single string")
    excluded = set()
    for entry in exclude:
        if not isinstance(entry, str):
            raise TypeError(f"exclude entries must be str, got {type(entry).__name__}")
        entry_tokens = _words(entry, casefold)
        if len(entry_tokens) != 1:
            raise ValueError(
                f"exclude entries must be single tokens; {entry!r} "
                f"gives {len(entry_tokens)}"
            )
        excluded.add(entry_tokens[0])
    return excluded
