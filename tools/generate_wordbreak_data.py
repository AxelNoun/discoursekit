"""Generate ``src/discoursekit/_wordbreak_data.py`` from the Unicode Character Database.

The tokeniser segments text with the default word boundary rules of Unicode
Standard Annex #29. Those rules are written in terms of the Word_Break and
Extended_Pictographic properties of each character. This script reads the two
data files that define them and writes a compact, run-length encoded table.

Usage::

    python tools/generate_wordbreak_data.py \
        --word-break WordBreakProperty.txt \
        --emoji emoji-data.txt \
        --output src/discoursekit/_wordbreak_data.py

The input files are published by the Unicode Consortium for each version of
the standard, under ``https://www.unicode.org/Public/<version>/ucd/``:
``auxiliary/WordBreakProperty.txt`` and ``emoji/emoji-data.txt``.

Run it again when moving to a new Unicode version, then run the test suite:
``tests/test_word_boundaries.py`` checks the result against the official
conformance test file of the same version.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

# Identifiers of the Word_Break property values. The order is arbitrary but
# fixed: the tokeniser refers to these numbers through the generated module.
WORD_BREAK_CLASSES = (
    "Other",
    "CR",
    "LF",
    "Newline",
    "Extend",
    "ZWJ",
    "Regional_Indicator",
    "Format",
    "Katakana",
    "Hebrew_Letter",
    "ALetter",
    "Single_Quote",
    "Double_Quote",
    "MidNumLet",
    "MidLetter",
    "MidNum",
    "Numeric",
    "ExtendNumLet",
    "WSegSpace",
)
EXTENDED_PICTOGRAPHIC_FLAG = 0x20
N_CODE_POINTS = 0x110000

_LINE = re.compile(r"^([0-9A-F]{4,6})(?:\.\.([0-9A-F]{4,6}))?\s*;\s*(\w+)")
_VERSION = re.compile(r"-(\d+\.\d+\.\d+)\.txt")


def read_property(path: Path) -> list[tuple[int, int, str]]:
    """Return (first, last, value) for every data line of a UCD property file."""
    ranges = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _LINE.match(line)
        if match:
            first = int(match.group(1), 16)
            last = int(match.group(2) or match.group(1), 16)
            ranges.append((first, last, match.group(3)))
    return ranges


def build_table(word_break: Path, emoji: Path) -> bytearray:
    """Return one byte per code point: class identifier, plus the emoji flag."""
    table = bytearray(N_CODE_POINTS)
    identifiers = {name: index for index, name in enumerate(WORD_BREAK_CLASSES)}
    for first, last, value in read_property(word_break):
        table[first : last + 1] = bytes([identifiers[value]]) * (last - first + 1)
    for first, last, value in read_property(emoji):
        if value == "Extended_Pictographic":
            for code_point in range(first, last + 1):
                table[code_point] |= EXTENDED_PICTOGRAPHIC_FLAG
    return table


def run_length_encode(table: bytearray) -> tuple[list[int], list[int]]:
    """Return the code points where the value changes, and the values."""
    starts = [0]
    values = [table[0]]
    for code_point in range(1, len(table)):
        if table[code_point] != values[-1]:
            starts.append(code_point)
            values.append(table[code_point])
    return starts, values


def wrap(numbers: list[int], indent: str = "    ", width: int = 88) -> str:
    """Format integers as lines of a Python tuple body no wider than ``width``."""
    lines = []
    current = indent
    for number in numbers:
        item = f"{number}, "
        if len(current) + len(item.rstrip()) > width:
            lines.append(current.rstrip())
            current = indent
        current += item
    lines.append(current.rstrip())
    return "\n".join(lines)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file, to record what the table came from."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_line(path: Path) -> str:
    """Return the first header line of a UCD file, without the comment sign."""
    return path.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()


def date_line(path: Path) -> str:
    """Return the ``Date:`` header line of a UCD file, without the comment sign."""
    return path.read_text(encoding="utf-8").splitlines()[1].lstrip("# ").strip()


def main() -> None:
    """Read the two UCD files and write the generated module."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--word-break", type=Path, required=True)
    parser.add_argument("--emoji", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    header = first_line(arguments.word_break)
    version_match = _VERSION.search(header)
    if version_match is None:
        raise SystemExit(f"cannot read the Unicode version from {header!r}")
    version = version_match.group(1)

    starts, values = run_length_encode(
        build_table(arguments.word_break, arguments.emoji)
    )
    constants = "\n".join(
        f"{name.upper()} = {index}" for index, name in enumerate(WORD_BREAK_CLASSES)
    )
    module = f'''"""Word_Break and Extended_Pictographic properties, Unicode {version}.

GENERATED FILE. Do not edit by hand: run ``tools/generate_wordbreak_data.py``.

Derived from the Unicode Character Database:

* {header} ({date_line(arguments.word_break)})
  SHA-256 {sha256(arguments.word_break)}
* {first_line(arguments.emoji)} ({date_line(arguments.emoji)})
  SHA-256 {sha256(arguments.emoji)}

Copyright (c) Unicode, Inc. Distributed under the Unicode License v3; see
``LICENSES/Unicode-3.0.txt`` in the source distribution.

``STARTS[i]`` is the first code point of run ``i`` and ``VALUES[i]`` its value.
The low five bits of a value are one of the class identifiers below; the bit
``EXTENDED_PICTOGRAPHIC`` is set for characters that have that property.
"""

UNICODE_VERSION = "{version}"

{constants}

CLASS_MASK = 0x1F
EXTENDED_PICTOGRAPHIC = 0x{EXTENDED_PICTOGRAPHIC_FLAG:02X}

# fmt: off
STARTS = (
{wrap(starts)}
)

VALUES = bytes((
{wrap(values)}
))
# fmt: on
'''
    arguments.output.write_text(module, encoding="utf-8", newline="\n")
    print(f"Unicode {version}: {len(starts)} runs written to {arguments.output}")


if __name__ == "__main__":
    main()
