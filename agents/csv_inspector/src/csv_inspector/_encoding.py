"""Encoding detection and decoding of byte samples; no I/O.

Also holds the line-break pattern ``csv`` and pandas split rows on, and
:func:`split_lines`, shared by sampling and grounding.
"""

from __future__ import annotations

import codecs
import logging
import re

import chardet

logger = logging.getLogger(__name__)

# The line breaks csv and pandas split rows on. str.splitlines() also splits on
# form feeds, vertical tabs, U+001C-U+001E, U+0085, U+2028 and U+2029, which can
# occur inside fields of dirty or latin-1-decoded data.
LINE_BREAK = re.compile(r"\r\n|\r|\n")
# One line: its text and its line break, or the text after the last break.
_LINE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+")


def split_lines(text: str, *, keepends: bool = False) -> list[str]:
    """Split ``text`` into lines the way ``csv`` and pandas count rows.

    A final line break ends the last line; it does not start an empty one.

    Args:
        text: The text to split.
        keepends: Keep each line's line break (the last line may have none).

    Returns:
        The lines, in order.
    """
    lines: list[str] = _LINE.findall(text)
    return lines if keepends else [line.rstrip("\r\n") for line in lines]


def detect_encoding(raw_bytes: bytes) -> str:
    """Detect a byte sample's encoding with chardet; ``"utf-8"`` for ASCII or no verdict."""
    detection = chardet.detect(raw_bytes)
    encoding: str = detection.get("encoding") or "utf-8"
    if encoding.lower() == "ascii":
        encoding = "utf-8"
    return encoding


# The top two bits of a UTF-8 continuation byte (10xxxxxx).
_UTF8_CONTINUATION = 0x80


def is_utf8_suffix(raw_bytes: bytes) -> bool:
    """Whether ``raw_bytes`` is valid UTF-8, ignoring a character cut at its start.

    A tail window may start in the middle of a multi-byte character, so up
    to three leading continuation bytes are skipped before decoding.
    """
    start = 0
    while start < min(3, len(raw_bytes)) and raw_bytes[start] & 0xC0 == _UTF8_CONTINUATION:
        start += 1
    try:
        raw_bytes[start:].decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def canonical_codec_name(encoding: str) -> str | None:
    """Return Python's canonical codec name for ``encoding``, or ``None`` if unknown."""
    try:
        return codecs.lookup(encoding).name
    except LookupError:
        return None


def code_unit_size(encoding: str) -> int:
    """The code-unit width of ``encoding`` in bytes: 2 for UTF-16, 4 for UTF-32, else 1.

    A tail window of UTF-16 or UTF-32 must start on a code unit, or every
    character decodes as garbage.
    """
    codec = canonical_codec_name(encoding) or ""
    if codec.startswith("utf-16"):
        return 2
    if codec.startswith("utf-32"):
        return 4
    return 1


def tail_encoding(head_bytes: bytes, encoding: str) -> str:
    """The BOM-less codec for a tail sample, the byte order read from the head's BOM.

    ``utf-16``, ``utf-32`` and ``utf-8-sig`` read the byte order from a
    leading BOM, which a tail never has: decoded as is, it would silently
    take the host's byte order.
    """
    codec = canonical_codec_name(encoding)
    if codec == "utf-8-sig":
        return "utf-8"
    if codec == "utf-16":
        return "utf-16-be" if head_bytes.startswith(codecs.BOM_UTF16_BE) else "utf-16-le"
    if codec == "utf-32":
        return "utf-32-be" if head_bytes.startswith(codecs.BOM_UTF32_BE) else "utf-32-le"
    return encoding


def decode_sample(raw_bytes: bytes, encoding: str) -> str:
    """Decode a byte sample, replacing undecodable bytes; an unknown codec falls back to UTF-8."""
    try:
        return raw_bytes.decode(encoding, errors="replace")
    except LookupError:
        logger.warning("Unknown encoding '%s'; falling back to utf-8.", encoding)
        return raw_bytes.decode("utf-8", errors="replace")
