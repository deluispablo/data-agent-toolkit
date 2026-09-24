"""Grounding of the model's answer in the sampled text.

Small local models recognize headers and footers reliably but count and copy
lines poorly. The functions here recompute positions and verbatim text
deterministically from the real samples, using the model's answer as a key.
"""

from __future__ import annotations

import csv
import itertools
import logging
import re

from ._models import CSVInspectionResult
from ._sampling import _canonical_codec_name

logger = logging.getLogger(__name__)

_LABEL = r"(?:sub\s*-?\s*)?(?:grand\s+)?(?:totals?|totales|suma|sum)"

# A line starting with a totals label (optionally quoted), e.g. "TOTAL;;;12.50",
# "Subtotal,,3", '"Total general",9' or "Total registros: 250".
_TOTALS_LABEL = re.compile(rf"""^["']?\s*{_LABEL}\b""", re.IGNORECASE)

# A field that is nothing but a totals label, e.g. "TOTAL", "Grand total:" or
# "Total general".
_BARE_TOTALS_LABEL = re.compile(rf"{_LABEL}(?:\s+general)?\s*:?", re.IGNORECASE)

# The line breaks csv and pandas split rows on. str.splitlines() also splits on
# form feeds, vertical tabs, U+001C-U+001E, U+0085, U+2028 and U+2029, which can
# occur inside fields of dirty or latin-1-decoded data.
_LINE_BREAK = re.compile(r"\r\n|\r|\n")

# Encodings whose name says the file starts with a byte order mark. chardet
# reports these only when it sees one, and reading the file with any other
# codec leaves a U+FEFF glued to the first column name.
_BOM_CODECS = frozenset({"utf-8-sig", "utf-16", "utf-32"})


def _split_lines(text: str) -> list[str]:
    """Split ``text`` into lines the way ``csv``/pandas count rows."""
    lines = _LINE_BREAK.split(text)
    if lines[-1] == "":
        lines.pop()
    return lines


def _split_fields(line: str, delimiter: str, quotechar: str) -> list[str] | None:
    """Split one line into stripped fields, or ``None`` if it cannot be parsed."""
    try:
        fields = next(csv.reader([line], delimiter=delimiter, quotechar=quotechar))
    except (csv.Error, StopIteration):
        return None
    return [field.strip() for field in fields]


def _locate_header_row(
    result: CSVInspectionResult, head_sample: str
) -> tuple[int, list[str]] | None:
    """Find the head line holding the column names, and the names as written.

    Small models count preamble lines poorly and sometimes paraphrase column
    names (e.g. "Importe" as "Monto"), but reliably get the number of
    columns and at least some names right. The header row is therefore:

    1. the first line whose fields equal every inferred column name; or
    2. failing that, the first line with as many fields as inferred columns
       that is followed by a line of the same shape and shares at least one
       name with the model's answer. The shared name keeps the first data
       row of a header-less file from being mistaken for a header.

    Args:
        result: The model's validated result, whose ``columns`` and dialect
            are used to recognize the header line.
        head_sample: The decoded head sample.

    Returns:
        ``(index, names)``: the 0-based index of the header line and its
        fields as written in the file, or ``None`` if no line qualifies (e.g.
        a header-less file, or a dialect Python's ``csv`` module rejects).
    """
    expected = [column.name.strip() for column in result.columns]
    if not expected:
        return None
    rows = [
        _split_fields(line, result.delimiter, result.quotechar)
        for line in _split_lines(head_sample.lstrip("﻿"))
    ]

    for index, fields in enumerate(rows):
        if fields == expected:
            return index, fields

    width = len(expected)
    # An unnamed column (e.g. a pandas index) must not match any empty cell.
    named = set(expected) - {""}
    for index, (fields, next_fields) in enumerate(itertools.pairwise(rows)):
        if (
            fields is not None
            and next_fields is not None
            and len(fields) == len(next_fields) == width
            and set(fields) & named
        ):
            return index, fields
    return None


def _locate_footer_lines(
    footer_lines: list[str], end_of_file: str, delimiter: str, quotechar: str
) -> list[str] | None:
    """Re-read the model's footer verbatim from the real end of the file.

    The model is good at recognizing footer content but unreliable at
    copying it exactly: it tends to drop blank separator lines or skip a
    line in the middle. This anchors the footer at the earliest non-blank
    footer line the model reported that really occurs in the file's last
    lines, takes every line from there to the end of the file verbatim, and
    extends it backwards over the blank lines that separate it from the
    data.

    Args:
        footer_lines: The footer lines reported by the model.
        end_of_file: Decoded text that ends at the real end of the file (the
            tail sample, or the head sample when it covers the whole file).
        delimiter: The field delimiter, used to tell totals rows from data.
        quotechar: The quote character, used to tell totals rows from data.

    Returns:
        The grounded footer lines, or ``None`` when none of the model's
        non-blank footer lines occur in ``end_of_file`` (nothing to anchor).
    """
    reported = {line.strip() for line in footer_lines if line.strip()}
    if not reported:
        return None
    lines = _split_lines(end_of_file)
    # Last occurrence of each reported line, so text that also appears earlier
    # in the data cannot drag data rows into the footer. Line 0 is skipped: in
    # a tail sample it is usually a truncated fragment.
    last_seen = {line.strip(): i for i, line in enumerate(lines) if i > 0}
    anchors = [last_seen[text] for text in reported if text in last_seen]
    if not anchors:
        return None
    start = min(anchors)
    while start > 1 and _extends_footer(lines[start - 1], delimiter, quotechar):
        start -= 1
    return lines[start:]


def _extends_footer(line: str, delimiter: str, quotechar: str) -> bool:
    """Whether a line just above a known footer line also belongs to the footer.

    Only blank separator lines and rows labelled as totals qualify. A totals
    row often has the same number of fields as a data row, which is exactly
    what small models miss, but its label gives it away. A label prefix
    alone is not enough, since a data row can start with a name such as
    "Total Energies": the first field must be the bare label (``TOTAL``,
    ``Subtotal:``), or most of the other fields must be empty, as they are
    in a totals row that only sums a few columns.
    """
    stripped = line.strip()
    if not stripped:
        return True
    if _TOTALS_LABEL.match(stripped) is None:
        return False
    fields = _split_fields(stripped, delimiter, quotechar)
    if not fields:
        return False
    label, *others = fields
    if _BARE_TOTALS_LABEL.fullmatch(label):
        return True
    filled = sum(1 for field in others if field)
    return filled * 2 <= len(others)


def _ground_encoding(reported: str, detected: str) -> str:
    """Pick the encoding to report: the model's, unless it is unusable.

    The model sees decoded text, in which a byte order mark is invisible, so
    it tends to answer ``utf-8`` for a ``utf-8-sig`` file; and it sometimes
    answers a description rather than a codec name (``"UTF-8 with BOM"``).
    The detected encoding is kept when it comes from a BOM, which is
    certain, or when the reported name is not a codec Python knows.

    Args:
        reported: The encoding reported by the model.
        detected: The encoding detected from the raw head sample.

    Returns:
        ``reported`` or ``detected``.
    """
    reported_codec = _canonical_codec_name(reported)
    detected_codec = _canonical_codec_name(detected)
    if detected_codec in _BOM_CODECS and reported_codec != detected_codec:
        return detected
    return reported if reported_codec is not None else detected


def ground_in_samples(
    result: CSVInspectionResult,
    head_sample: str,
    tail_sample: str | None,
    *,
    covers_whole_file: bool = True,
    detected_encoding: str | None = None,
) -> CSVInspectionResult:
    """Correct what the model reported by matching it against the sampled text.

    Small local models recognize headers and footers reliably but count and
    copy lines poorly. Positions and verbatim text are therefore recomputed
    deterministically from the real samples, using the model's own answer
    as the key: the header row (and the column names as actually written)
    is located from the inferred columns, and the footer is re-read
    verbatim from the end of the file. The reported encoding is checked
    against the detected one (see :func:`_ground_encoding`). Anything that
    cannot be anchored is left exactly as the model reported it.

    Args:
        result: The model's validated result.
        head_sample: The decoded head sample.
        tail_sample: The decoded tail sample, or ``None`` when the head
            covers the whole file.
        covers_whole_file: Whether the samples reach the real end of the
            file. When ``False`` (no tail and a truncated head), any footer
            the model reported cannot be real, since the end was never
            seen, so ``footer_lines`` is cleared.
        detected_encoding: The encoding detected from the raw head sample,
            or ``None`` to leave the reported encoding unchecked.

    Returns:
        The result with ``header_row_index``, column names,
        ``footer_lines`` and ``encoding`` grounded in the samples; the same
        object when nothing changed.
    """
    updates: dict[str, object] = {}

    header = _locate_header_row(result, head_sample)
    if header is not None:
        header_row_index, names = header
        if header_row_index != result.header_row_index:
            updates["header_row_index"] = header_row_index
        if names != [column.name for column in result.columns]:
            updates["columns"] = [
                column.model_copy(update={"name": name})
                for column, name in zip(result.columns, names, strict=True)
            ]

    if tail_sample is None and not covers_whole_file:
        # The head's last lines are mid-file data, never a footer.
        if result.footer_lines:
            updates["footer_lines"] = []
    else:
        end_of_file = tail_sample if tail_sample is not None else head_sample
        footer_lines = _locate_footer_lines(
            result.footer_lines, end_of_file, result.delimiter, result.quotechar
        )
        if footer_lines is not None and footer_lines != result.footer_lines:
            updates["footer_lines"] = footer_lines

    if detected_encoding is not None:
        encoding = _ground_encoding(result.encoding, detected_encoding)
        if encoding != result.encoding:
            updates["encoding"] = encoding

    if not updates:
        return result
    logger.info("Grounded model output in the sampled text: %s", updates)
    return result.model_copy(update=updates)
