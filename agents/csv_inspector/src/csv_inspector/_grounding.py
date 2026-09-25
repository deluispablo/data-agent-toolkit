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
from collections import Counter

from ._encoding import LINE_BREAK, canonical_codec_name
from ._models import CSVInspectionResult

logger = logging.getLogger(__name__)

_LABEL = r"(?:sub\s*-?\s*)?(?:grand\s+)?(?:totals?|totales|suma|sum)"

# A line starting with a totals label (optionally quoted), e.g. "TOTAL;;;12.50",
# "Subtotal,,3", '"Total general",9' or "Total registros: 250".
_TOTALS_LABEL = re.compile(rf"""^["']?\s*{_LABEL}\b""", re.IGNORECASE)

# A field that is nothing but a totals label, e.g. "TOTAL", "Grand total:" or
# "Total general".
_BARE_TOTALS_LABEL = re.compile(rf"{_LABEL}(?:\s+general)?\s*:?", re.IGNORECASE)

# Encodings whose name says the file starts with a byte order mark. chardet
# reports these only when it sees one, and reading the file with any other
# codec leaves a U+FEFF glued to the first column name.
_BOM_CODECS = frozenset({"utf-8-sig", "utf-16", "utf-32"})

# The delimiters tried when the model's one does not split the head's lines.
_CANDIDATE_DELIMITERS = ",;\t|"
# How many lines must split into the same number of fields to pick a candidate.
_MIN_AGREEING_LINES = 2

# Field shapes compared by the header-less test (see _field_shape).
_INTEGER = re.compile(r"[+-]?\d+")
_DECIMAL = re.compile(r"[+-]?(?:\d{1,3}(?:[.,]\d{3})+|\d*)[.,]\d+")
_DATE = re.compile(
    r"\d{4}-\d{1,2}-\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"
    r"|\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})"
)


def _split_lines(text: str) -> list[str]:
    """Split ``text`` into lines the way ``csv``/pandas count rows."""
    lines = LINE_BREAK.split(text)
    if lines[-1] == "":
        lines.pop()
    return lines


def _split_fields(line: str, delimiter: str, quotechar: str) -> list[str] | None:
    """Split one line into fields as written, or ``None`` if it cannot be parsed.

    Surrounding spaces are kept, as ``csv`` and pandas keep them by default;
    callers strip the fields where they only compare values.
    """
    try:
        fields = next(csv.reader([line], delimiter=delimiter, quotechar=quotechar))
    except (csv.Error, StopIteration):
        return None
    return fields


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
        if fields is not None and [field.strip() for field in fields] == expected:
            return index, fields

    width = len(expected)
    # An unnamed column (e.g. a pandas index) must not match any empty cell.
    named = set(expected) - {""}
    for index, (fields, next_fields) in enumerate(itertools.pairwise(rows)):
        if (
            fields is not None
            and next_fields is not None
            and len(fields) == len(next_fields) == width
            and {field.strip() for field in fields} & named
        ):
            return index, fields
    return None


def _field_shape(value: str) -> str:
    """Classify one field as ``empty``, ``integer``, ``date``, ``decimal`` or ``text``.

    Surrounding spaces are ignored. A decimal may use ``.`` or ``,`` as its
    separator, with optional thousands grouping (``1.234,56``). Dates are
    ISO (``2024-01-15``, optionally with a time) or day-first European
    (``15/01/2024``, ``15.01.24``); they are tested before decimals so
    ``15.01.2024`` is not read as a number.
    """
    value = value.strip()
    if not value:
        return "empty"
    if _INTEGER.fullmatch(value):
        return "integer"
    if _DATE.fullmatch(value):
        return "date"
    if _DECIMAL.fullmatch(value):
        return "decimal"
    return "text"


def _shapes_agree(first: list[str], second: list[str]) -> bool:
    """Whether two shape signatures match field by field, an empty field matching any."""
    return all(a == b or "empty" in (a, b) for a, b in zip(first, second, strict=True))


def _first_row_is_data(result: CSVInspectionResult, head_sample: str) -> bool:
    """Whether the first line is shaped like the data below it, not like names.

    Small models asked about a header-less file still answer
    ``has_header=true`` and invent names (``date``, ``amount``) for the
    first row. That row is data when rows 0 and 1 have as many fields as
    inferred columns and either their shape signatures (see
    :func:`_field_shape`) are identical, or they agree field by field (an
    empty field matching any shape) with at least half of the fields
    non-text in both rows. A row-0 field equal to one of the model's
    column names (case-insensitive) always keeps the header. Only the
    sample is read, never the model's example values; only the first line
    is tested, so a header-less file with preamble lines is not described.
    """
    lines = _split_lines(head_sample.lstrip("\ufeff"))
    if len(lines) < _MIN_AGREEING_LINES:
        return False
    first = _split_fields(lines[0], result.delimiter, result.quotechar)
    second = _split_fields(lines[1], result.delimiter, result.quotechar)
    if not first or second is None or not len(first) == len(second) == len(result.columns):
        return False
    names = {column.name.strip().casefold() for column in result.columns} - {""}
    if any(field.strip().casefold() in names for field in first):
        return False
    first_shape = [_field_shape(field) for field in first]
    second_shape = [_field_shape(field) for field in second]
    if first_shape == second_shape:
        return True
    half = (len(first) + 1) // 2
    return (
        _shapes_agree(first_shape, second_shape)
        and sum(shape != "text" for shape in first_shape) >= half
        and sum(shape != "text" for shape in second_shape) >= half
    )


def _ground_header(result: CSVInspectionResult, head_sample: str) -> dict[str, object]:
    """Return the header fields to correct: row index, names, or "no header".

    A header-less file keeps its positional column names. A model that
    answers a header at row 0 it cannot anchor, over a first row shaped
    like the data below it, is corrected to no header row (see
    :func:`_first_row_is_data`).
    """
    if not result.has_header:
        return {}
    header = _locate_header_row(result, head_sample)
    if header is None:
        if result.header_row_index != 0 or not _first_row_is_data(result, head_sample):
            return {}
        logger.info("The first row holds data, not column names: reporting no header row.")
        return {
            "has_header": False,
            "header_row_index": None,
            "columns": [
                column.model_copy(update={"name": f"column_{number}"})
                for number, column in enumerate(result.columns, start=1)
            ],
        }
    updates: dict[str, object] = {}
    header_row_index, names = header
    if header_row_index != result.header_row_index:
        updates["header_row_index"] = header_row_index
    if names != [column.name for column in result.columns]:
        updates["columns"] = [
            column.model_copy(update={"name": name})
            for column, name in zip(result.columns, names, strict=True)
        ]
    return updates


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
    label, *others = (field.strip() for field in fields)
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
    reported_codec = canonical_codec_name(reported)
    detected_codec = canonical_codec_name(detected)
    if detected_codec in _BOM_CODECS and reported_codec != detected_codec:
        return detected
    return reported if reported_codec is not None else detected


def _agreement_score(lines: list[str], delimiter: str, quotechar: str) -> int:
    """Count the lines that ``delimiter`` splits into the modal field count (2 or more)."""
    widths = [
        len(fields)
        for line in lines
        if (fields := _split_fields(line, delimiter, quotechar)) is not None and len(fields) > 1
    ]
    return Counter(widths).most_common(1)[0][1] if widths else 0


def _ground_delimiter(result: CSVInspectionResult, head_sample: str) -> str:
    """Pick the delimiter to report: the model's, unless it clearly loses.

    Small models sometimes answer ``,`` for a tab-separated file, whose
    values may still hold a comma or two. Every delimiter is scored the
    same way: the number of head lines it splits into the same number (2
    or more) of fields. Preamble and footer lines do not block this,
    unlike ``csv.Sniffer``, which needs nearly every line to agree. The
    model's delimiter is replaced only when it scores below
    ``_MIN_AGREEING_LINES`` (a delimiter that never occurs scores 0) and
    exactly one usual delimiter scores at least that and more than it.
    Otherwise it is kept: ties, one-column files and exotic delimiters
    stay as reported.

    Args:
        result: The model's validated result.
        head_sample: The decoded head sample.

    Returns:
        The reported delimiter or the chosen candidate.
    """
    lines = _split_lines(head_sample.lstrip("﻿"))
    reported = (
        _agreement_score(lines, result.delimiter, result.quotechar)
        if result.delimiter in head_sample
        else 0
    )
    if reported >= _MIN_AGREEING_LINES:
        return result.delimiter
    scores = {
        candidate: _agreement_score(lines, candidate, result.quotechar)
        for candidate in _CANDIDATE_DELIMITERS
        if candidate not in (result.delimiter, result.quotechar, result.escapechar)
    }
    best = max(scores.values(), default=0)
    winners = [candidate for candidate, score in scores.items() if score == best]
    if best < _MIN_AGREEING_LINES or best <= reported or len(winners) != 1:
        return result.delimiter
    logger.info(
        "Replacing delimiter %r (%d agreeing lines) with %r (%d agreeing lines).",
        result.delimiter,
        reported,
        winners[0],
        best,
    )
    return winners[0]


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
    verbatim from the end of the file. A delimiter that splits too few head
    lines is replaced (see :func:`_ground_delimiter`), and the reported
    encoding is checked against the detected one (see
    :func:`_ground_encoding`). A header at row 0 that cannot be anchored
    becomes "no header" when the first row has the same field shapes
    (integer, decimal, date, empty or text) as the second: header-less
    detection is a shape test on the sample, not on the model's example
    values (see :func:`_first_row_is_data`). Any other header that cannot
    be anchored is left as the model reported it. A footer that cannot be anchored is dropped when
    the end of the file was sampled, since it is not there.

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
        The result with ``delimiter``, ``header_row_index``, column names,
        ``footer_lines`` and ``encoding`` grounded in the samples; the same
        object when nothing changed.
    """
    updates: dict[str, object] = {}

    delimiter = _ground_delimiter(result, head_sample)
    if delimiter != result.delimiter:
        updates["delimiter"] = delimiter
        # Header and footer grounding split fields with the grounded delimiter.
        result = result.model_copy(update={"delimiter": delimiter})

    updates.update(_ground_header(result, head_sample))

    if tail_sample is None and not covers_whole_file:
        # The head's last lines are mid-file data, never a footer.
        if result.footer_lines:
            updates["footer_lines"] = []
    else:
        end_of_file = tail_sample if tail_sample is not None else head_sample
        footer_lines = _locate_footer_lines(
            result.footer_lines, end_of_file, result.delimiter, result.quotechar
        )
        if footer_lines is None and any(line.strip() for line in result.footer_lines):
            # The real end of the file was seen and the reported footer is
            # not there. Keeping it would make readers drop real data rows.
            logger.warning(
                "Discarding footer lines that do not occur at the end of the file: %s",
                result.footer_lines,
            )
            footer_lines = []
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
