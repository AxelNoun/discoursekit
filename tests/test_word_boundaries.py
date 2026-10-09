"""Conformance of the word segmentation with Unicode Standard Annex #29.

``tests/data/WordBreakTest-<version>.txt`` is the conformance test published
by the Unicode Consortium with each version of the standard. Each line gives a
string as code points, with a division sign where a word boundary is expected
and a multiplication sign where it is not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from discoursekit.tokenization import UNICODE_VERSION, _word_segments

TEST_FILE = Path(__file__).parent / "data" / f"WordBreakTest-{UNICODE_VERSION}.txt"
BREAK = "\u00f7"
NO_BREAK = "\u00d7"


def read_cases() -> list[tuple[int, list[str]]]:
    """Return (line number, expected segments) for every test line."""
    cases = []
    lines = TEST_FILE.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        data = line.split("#", 1)[0].strip()
        if not data:
            continue
        fields = data.split()
        assert fields[0] == BREAK and fields[-1] == BREAK, number
        segments = []
        current = ""
        # After the leading sign, the fields alternate: code point, sign.
        for code_point, sign in zip(fields[1::2], fields[2::2], strict=True):
            current += chr(int(code_point, 16))
            assert sign in (BREAK, NO_BREAK), number
            if sign == BREAK:
                segments.append(current)
                current = ""
        assert current == "", number
        cases.append((number, segments))
    return cases


CASES = read_cases()


def test_the_file_matches_the_bundled_unicode_version() -> None:
    header = TEST_FILE.read_text(encoding="utf-8").splitlines()[0]
    assert header == f"# WordBreakTest-{UNICODE_VERSION}.txt"


def test_the_file_is_read_in_full() -> None:
    # 1,944 test lines in WordBreakTest-18.0.0.txt.
    assert len(CASES) == 1944


def test_every_conformance_case_passes() -> None:
    failures = []
    for number, expected in CASES:
        observed = _word_segments("".join(expected))
        if observed != expected:
            failures.append((number, expected, observed))
    assert not failures, f"{len(failures)} of {len(CASES)} cases fail: {failures[:5]}"


@pytest.mark.parametrize(
    "text",
    ["", "a", " ", "\n", "\u0301", "\u200d", "a\u0301", "\U0001f1eb\U0001f1f7"],
)
def test_segments_partition_the_text(text: str) -> None:
    segments = _word_segments(text)
    assert "".join(segments) == text
    assert all(segments)
