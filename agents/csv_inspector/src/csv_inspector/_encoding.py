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
    """Heuristically detect the character encoding of a byte sample.

    Args:
        raw_bytes: The raw byte sample to analyze.

    Returns:
        The detected encoding name, defaulting to ``"utf-8"`` when detection
        is inconclusive or reports plain ASCII.
    """
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
    """Return the fixed code-unit width, in bytes, of ``encoding``.

    UTF-16 and UTF-32 cannot be decoded from an arbitrary byte offset: a
    window that starts on an odd byte turns every character into garbage.
    Aligning the tail window to the code unit avoids that.

    Args:
        encoding: An encoding name, typically from :func:`detect_encoding`.

    Returns:
        ``2`` for UTF-16 variants, ``4`` for UTF-32 variants, ``1`` otherwise.
    """
    codec = canonical_codec_name(encoding) or ""
    if codec.startswith("utf-16"):
        return 2
    if codec.startswith("utf-32"):
        return 4
    return 1


def tail_encoding(head_bytes: bytes, encoding: str) -> str:
    """Pick the codec for decoding a tail sample, which never carries a BOM.

    BOM-dependent codecs (``utf-16``, ``utf-32``, ``utf-8-sig``) decide byte
    order from a leading BOM. A tail sample has none, so decoding it with
    the generic codec would silently assume the host's byte order. This
    resolves the explicit, BOM-less variant from the head's BOM instead.

    Args:
        head_bytes: The raw head sample, which may start with a BOM.
        encoding: The encoding detected for the head sample.

    Returns:
        An encoding name that decodes a BOM-less suffix of the same file.
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
    """Decode a byte sample using the given encoding, tolerating bad bytes.

    Args:
        raw_bytes: The raw byte sample to decode.
        encoding: The encoding to use, typically from :func:`detect_encoding`.

    Returns:
        The decoded text, with undecodable bytes replaced rather than raising.
    """
    try:
        return raw_bytes.decode(encoding, errors="replace")
    except LookupError:
        logger.warning("Unknown encoding '%s'; falling back to utf-8.", encoding)
        return raw_bytes.decode("utf-8", errors="replace")
