from __future__ import annotations

import random
import unicodedata

import pytest

from discoursekit import UNICODE_VERSION, tokenize
from discoursekit import _wordbreak_data as _wb
from discoursekit.tokenization import _value, _word_segments

# What a token is


def test_splits_on_whitespace_and_punctuation() -> None:
    assert tokenize("Hello, world! (Again.)") == ["hello", "world", "again"]


def test_empty_and_wordless_text_give_no_tokens() -> None:
    assert tokenize("") == []
    assert tokenize(" \n\t ... --- !!! \u2026 \u00ab \u00bb") == []


def test_symbols_and_emoji_are_not_tokens() -> None:
    assert tokenize("50% of $20 \u2192 10 \U0001f600") == ["50", "of", "20", "10"]


# Apostrophes (rules WB6 and WB7)


def test_apostrophe_between_letters_does_not_end_a_word() -> None:
    assert tokenize("I don't think it's o'clock") == [
        "i",
        "don't",
        "think",
        "it's",
        "o'clock",
    ]


@pytest.mark.parametrize("apostrophe", ["\u2019", "\u2018", "\uff07"])
def test_typographic_apostrophes_equal_ascii(apostrophe: str) -> None:
    assert tokenize(f"don{apostrophe}t") == tokenize("don't") == ["don't"]


def test_modifier_letter_apostrophe_is_a_letter_and_is_kept() -> None:
    # U+02BC is a letter in Unicode, not punctuation: it is not unified.
    assert tokenize("don\u02bct") == ["don\u02bct"]


def test_leading_and_trailing_apostrophes_are_not_part_of_a_word() -> None:
    assert tokenize("she said 'yes' and \u2018no\u2019") == [
        "she",
        "said",
        "yes",
        "and",
        "no",
    ]
    assert tokenize("'tis the dogs' bowl") == ["tis", "the", "dogs", "bowl"]


# Hyphens and dashes


def test_a_hyphen_ends_a_word() -> None:
    assert tokenize("a well-known fact") == ["a", "well", "known", "fact"]
    assert tokenize("wait - no -- yes \u2014 maybe") == ["wait", "no", "yes", "maybe"]


def test_hyphenated_repetitions_and_cut_off_words_in_transcripts() -> None:
    assert tokenize("I-I-I think") == ["i", "i", "i", "think"]
    assert tokenize("wh- what") == ["wh", "what"]


# Full stops and colons


def test_full_stop_between_letters_does_not_end_a_word() -> None:
    assert tokenize("in the U.S.A. today, e.g. at 9 a.m.") == [
        "in",
        "the",
        "u.s.a",
        "today",
        "e.g",
        "at",
        "9",
        "a.m",
    ]


def test_sentence_final_full_stop_and_ellipsis_are_dropped() -> None:
    assert tokenize("The end. Wait...what") == ["the", "end", "wait", "what"]


def test_missing_space_after_full_stop_or_colon_joins_two_words() -> None:
    # A documented limit of the default rules, not a feature.
    assert tokenize("The end.Next one") == ["the", "end.next", "one"]
    assert tokenize("note:this") == ["note:this"]


# Numbers (rules WB8, WB11, WB12)


def test_decimal_and_thousands_separators_stay_inside_a_number() -> None:
    assert tokenize("pi is 3.14, not 1,000.5 or 1'000") == [
        "pi",
        "is",
        "3.14",
        "not",
        "1,000.5",
        "or",
        "1'000",
    ]


def test_other_separators_split_numbers() -> None:
    assert tokenize("at 10:30 on 2024-01-05, ratio 3/4") == [
        "at",
        "10",
        "30",
        "on",
        "2024",
        "01",
        "05",
        "ratio",
        "3",
        "4",
    ]


def test_separator_between_letter_and_digit_does_not_join() -> None:
    assert tokenize("a.1 and 1.a") == ["a", "1", "and", "1", "a"]


def test_letters_digits_and_underscores_run_together() -> None:
    assert tokenize("covid19 in the 3rd snake_case B12 sample") == [
        "covid19",
        "in",
        "the",
        "3rd",
        "snake_case",
        "b12",
        "sample",
    ]


# Case


def test_case_folding_is_default_and_optional() -> None:
    assert tokenize("The THE the") == ["the", "the", "the"]
    assert tokenize("The THE the", casefold=False) == ["The", "THE", "the"]


def test_case_folding_is_full_unicode_case_folding() -> None:
    # Sharp s folds to "ss"; the ligature U+FB01 folds to "fi".
    assert tokenize("Stra\u00dfe STRASSE") == ["strasse", "strasse"]
    assert tokenize("\ufb01nd find") == ["find", "find"]


# Normalisation


def test_composed_and_decomposed_accents_are_the_same_token() -> None:
    composed = "caf\u00e9"
    decomposed = "cafe\u0301"
    assert composed != decomposed
    assert tokenize(composed) == tokenize(decomposed) == [composed]
    assert tokenize(composed.upper()) == tokenize(decomposed.upper()) == [composed]


def test_combining_mark_without_precomposed_form_stays_attached() -> None:
    # "g" with a tilde has no single code point; rule WB4 keeps it together.
    assert tokenize("ag\u0303o") == ["ag\u0303o"]


def test_format_characters_inside_a_word_are_removed() -> None:
    soft_hyphen, joiner, word_joiner = "\u00ad", "\u200d", "\u2060"
    assert tokenize(f"co{soft_hyphen}operate") == ["cooperate"]
    assert tokenize(f"co{joiner}operate") == ["cooperate"]
    assert tokenize(f"co{word_joiner}operate") == ["cooperate"]


def test_zero_width_space_separates_words() -> None:
    assert tokenize("zero\u200bwidth") == ["zero", "width"]


def test_byte_order_mark_is_ignored() -> None:
    assert tokenize("\ufeffhello") == ["hello"]


# White space (the one tailoring of the default rules)

SPACES = [
    " ",
    "\t",
    "\n",
    "\r",
    "\r\n",
    "\u0085",  # next line
    "\u00a0",  # no-break space
    "\u2009",  # thin space
    "\u2028",  # line separator
    "\u202f",  # narrow no-break space
    "\u3000",  # ideographic space
]


def test_default_rules_join_across_a_narrow_no_break_space() -> None:
    # What the tailoring replaces: UAX #29 classes U+202F with the underscore.
    assert _word_segments("3:45\u202fPM") == ["3", ":", "45\u202fPM"]
    assert _word_segments("3:45\u00a0PM") == ["3", ":", "45", "\u00a0", "PM"]


@pytest.mark.parametrize("space", SPACES)
def test_every_kind_of_space_separates_tokens(space: str) -> None:
    assert tokenize(f"3:45{space}PM") == ["3", "45", "pm"]
    assert tokenize(f"5{space}km or 10{space}000") == ["5", "km", "or", "10", "000"]
    assert tokenize(f"\u00ab{space}oui{space}\u00bb{space}!") == ["oui"]


def test_a_mark_typed_after_a_space_does_not_join_two_words() -> None:
    # The accent goes with the space (rule WB4) or stands alone after a line
    # break (rule WB3a). It never joins the letters around it.
    for space in SPACES:
        assert tokenize(f"a{space}\u0301b") == ["a", "b"]


def test_a_space_attached_to_a_letter_by_rule_wb4_is_removed() -> None:
    # The halfwidth voiced sound mark U+FF9E is a letter that combines with
    # the character before it, here a space.
    assert _word_segments("x \uff9e") == ["x", " \uff9e"]
    assert tokenize("x \uff9e") == ["x", "\uff9e"]


# Exclusion


def test_exclude_removes_tokens() -> None:
    text = "So, um, I went, uh, home. Um."
    assert tokenize(text, exclude={"um", "uh"}) == ["so", "i", "went", "home"]
    assert tokenize(text, exclude=["um"]) == ["so", "i", "went", "uh", "home"]


def test_nothing_is_excluded_by_default() -> None:
    assert tokenize("um uh") == ["um", "uh"]


def test_exclude_entries_are_normalised_like_the_text() -> None:
    assert tokenize("Um, don\u2019t", exclude={"UM", "DON'T"}) == []
    # Without case folding, exclusion is case sensitive.
    assert tokenize("Um um", casefold=False, exclude={"um"}) == ["Um"]


@pytest.mark.parametrize("entry", ["you know", "", "...", "well-known"])
def test_exclude_entries_must_be_single_tokens(entry: str) -> None:
    with pytest.raises(ValueError, match="single tokens"):
        tokenize("some text", exclude={entry})


def test_exclude_rejects_a_bare_string_and_non_strings() -> None:
    with pytest.raises(TypeError, match="not a single string"):
        tokenize("some text", exclude="um")
    with pytest.raises(TypeError, match="exclude entries must be str"):
        tokenize("some text", exclude=[1])  # type: ignore[list-item]


# General properties


def test_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="text must be a str"):
        tokenize(["already", "tokenised"])  # type: ignore[arg-type]


def test_same_input_gives_same_output() -> None:
    text = "Well, I don't know... It's a well-known problem!"
    assert tokenize(text) == tokenize(text)


def test_unicode_version_is_the_bundled_one() -> None:
    assert UNICODE_VERSION == "18.0.0"


# Characters chosen to exercise every branch of the segmentation and of the
# normalisation: at least one of each class of the word boundary rules,
# letters of several scripts, and characters that change under normalisation
# or case folding.
AWKWARD = [
    chr(code_point)
    for code_point in (
        0x61,  # a
        0x42,  # B
        0x39,  # 9
        0x20,  # space
        0x0A,  # line feed
        0x0D,  # carriage return
        0x85,  # next line
        0xA0,  # no-break space
        0x202F,  # narrow no-break space
        0x27,  # apostrophe
        0x2019,  # right single quotation mark
        0x02BC,  # modifier letter apostrophe
        0x2E,  # full stop
        0x2C,  # comma
        0x3A,  # colon
        0x2D,  # hyphen-minus
        0x5F,  # low line
        0xAD,  # soft hyphen
        0x200D,  # zero width joiner
        0x200B,  # zero width space
        0x301,  # combining acute accent
        0x345,  # combining Greek ypogegrammeni
        0xE9,  # e with acute
        0xDF,  # sharp s
        0x130,  # capital I with dot above
        0xFB01,  # ligature fi
        0x5D0,  # Hebrew alef
        0x22,  # quotation mark
        0x30A2,  # Katakana a
        0xFF9E,  # halfwidth Katakana voiced sound mark (a combining letter)
        0x1F1EB,  # regional indicator F
        0x2139,  # information source (a letter with a pictographic form)
        0x1F600,  # grinning face
    )
]


def random_text(rng: random.Random, longest: int = 12) -> str:
    return "".join(rng.choice(AWKWARD) for _ in range(rng.randint(0, longest)))


def test_the_random_strings_cover_every_class_of_the_rules() -> None:
    values = {_value(character) for character in AWKWARD}
    assert {value & _wb.CLASS_MASK for value in values} == set(range(19))
    assert any(value & _wb.EXTENDED_PICTOGRAPHIC for value in values)


@pytest.mark.parametrize("seed", range(4))
def test_invariants_on_random_strings(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(5000):
        text = random_text(rng)

        segments = _word_segments(text)
        assert "".join(segments) == text
        assert all(segments)

        for casefold in (True, False):
            for token in tokenize(text, casefold=casefold):
                assert token
                assert any(character.isalnum() for character in token)
                assert not any(character.isspace() for character in token)
                assert not any(unicodedata.category(c) == "Cf" for c in token)
                assert unicodedata.normalize("NFC", token) == token


@pytest.mark.parametrize("seed", range(4))
def test_white_space_separates_tokens_in_random_strings(seed: int) -> None:
    # Joining two strings with any white space gives the tokens of the first
    # followed by the tokens of the second.
    rng = random.Random(100 + seed)
    for _ in range(2500):
        before, after = random_text(rng, 8), random_text(rng, 8)
        space = rng.choice(SPACES)
        for casefold in (True, False):
            assert tokenize(before + space + after, casefold=casefold) == tokenize(
                before, casefold=casefold
            ) + tokenize(after, casefold=casefold)
