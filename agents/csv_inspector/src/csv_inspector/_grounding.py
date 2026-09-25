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
from ._models import CSVInspectionResult, _ModelAnswer

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
# How many times the model's score a single candidate must reach to replace it.
_DOMINANCE_RATIO = 2

# Field shapes compared by the header-less test (see _field_shape).
_INTEGER = re.compile(r"[+-]?\d+")
_DECIMAL = re.compile(r"[+-]?(?:\d{1,3}(?:[.,]\d{3})+|\d*)[.,]\d+")
_DATE = re.compile(
    r"\d{4}-\d{1,2}-\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"
    r"|\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})"
)
# The shortest reported footer text that anchors a line it only occurs in.
_MIN_SUBSTRING_ANCHOR = 8


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

    1. the first line whose fields equal every inferred column name, or
       whose non-empty fields do and which has one or more extra, empty
       fields: models tend to leave out a blank name, such as the unnamed
       index column pandas writes; or
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
        fields as written in the file, blank ones included (so there may be
        more names than inferred columns), or ``None`` if no line qualifies
        (e.g. a header-less file, or a dialect Python's ``csv`` module
        rejects).
    """
    expected = [name.strip() for name in result.columns]
    if not expected:
        return None
    rows = [
        _split_fields(line, result.delimiter, result.quotechar)
        for line in _split_lines(head_sample.lstrip("﻿"))
    ]

    for index, fields in enumerate(rows):
        if fields is None:
            continue
        stripped = [field.strip() for field in fields]
        if stripped == expected or (
            len(fields) > len(expected) and [field for field in stripped if field] == expected
        ):
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
    sample is read; only the first line is tested, so a header-less file
    with preamble lines is not described.
    """
    lines = _split_lines(head_sample.lstrip("\ufeff"))
    if len(lines) < _MIN_AGREEING_LINES:
        return False
    first = _split_fields(lines[0], result.delimiter, result.quotechar)
    second = _split_fields(lines[1], result.delimiter, result.quotechar)
    if not first or second is None or not len(first) == len(second) == len(result.columns):
        return False
    names = {name.strip().casefold() for name in result.columns} - {""}
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
            "columns": [f"column_{number}" for number in range(1, len(result.columns) + 1)],
        }
    updates: dict[str, object] = {}
    header_row_index, names = header
    if header_row_index != result.header_row_index:
        updates["header_row_index"] = header_row_index
    # The header's fields as written, blank names the model left out included.
    if names != result.columns:
        updates["columns"] = names
    return updates


def _locate_footer_lines(
    anchor: str,
    end_of_file: str,
    delimiter: str,
    quotechar: str,
    model_delimiter: str | None = None,
) -> list[str] | None:
    """Read the footer verbatim from the real end of the file, from one anchor line.

    The model is asked for the first non-blank footer line only: it is good
    at recognizing footer content but unreliable at copying several lines
    (it drops blank separators, skips a line, or takes the last data rows
    for a footer), and the file holds the footer anyway. The anchor matches
    its last occurrence in the file's last lines (see
    :func:`_anchor_matches`), also when the model wrote it with its own,
    replaced delimiter; a data-shaped anchor that occurs nowhere (a
    miscopied or made-up last data row) means the footer starts right after
    the data. The footer starts after the last data row at or after the
    anchor, since a footer follows the data (the model may point at a
    ragged data row with full rows after it); it then takes every line from
    there to the end of the file verbatim, and extends backwards over the
    blank, totals and other non-data lines that separate it from the data.
    A blank anchor (the model copied the blank separator line) starts at the
    end of the file and extends backwards the same way.

    Args:
        anchor: The first footer line reported by the model.
        end_of_file: Decoded text that ends at the real end of the file (the
            tail sample, or the head sample when it covers the whole file).
        delimiter: The field delimiter, used to tell footer lines from data.
        quotechar: The quote character, used to tell footer lines from data.
        model_delimiter: The delimiter the model answered, when grounding
            replaced it: the model copies footer lines with its own
            delimiter (``TOTAL,,12.50`` for ``TOTAL		12.50``).

    Returns:
        The grounded footer lines, or ``None`` when the anchor does not occur
        in ``end_of_file`` (and is not shaped like a data row), or only data
        rows follow it (nothing to anchor).
    """
    reported = anchor.strip()
    keys = {reported}
    if model_delimiter and model_delimiter != delimiter:
        keys.add(reported.replace(model_delimiter, delimiter))
    lines = _split_lines(end_of_file)
    width = _data_width(lines, delimiter, quotechar)

    def is_data(line: str) -> bool:
        return width is not None and _is_data_row(line, delimiter, quotechar, width)

    if reported:
        # The last occurrence, so text that also appears earlier in the data
        # cannot drag data rows into the footer. Line 0 is skipped: in a tail
        # sample it is usually a truncated fragment.
        found = next(
            (
                i
                for i in range(len(lines) - 1, 0, -1)
                if any(_anchor_matches(key, lines[i], delimiter) for key in keys)
            ),
            None,
        )
        if found is None:
            # A data row the model miscopied or made up still says where it
            # saw the footer: right after the data.
            if not any(is_data(key) for key in keys):
                return None
            found = 1
        data_rows = [index for index in range(found, len(lines)) if is_data(lines[index])]
        start = data_rows[-1] + 1 if data_rows else found
    else:
        start = len(lines)
    if start < len(lines) or not reported:
        while start > 1 and (
            _extends_footer(lines[start - 1], delimiter, quotechar)
            or (width is not None and not is_data(lines[start - 1]))
        ):
            start -= 1
    if start == len(lines):
        return None
    return lines[start:]


def _anchor_matches(reported: str, line: str, delimiter: str) -> bool:
    """Whether a stripped line the model reported designates this file line.

    The model's line is a key, matched tolerantly: the two are equal once
    surrounding whitespace and trailing empty fields (trailing delimiters)
    are stripped, or the reported text, at least
    ``_MIN_SUBSTRING_ANCHOR`` characters long, occurs within the line
    (e.g. the timestamp of a ``Generated on ...`` line).
    """
    trailing = delimiter + " \t"
    if reported.rstrip(trailing) == line.strip().rstrip(trailing):
        return True
    return len(reported) >= _MIN_SUBSTRING_ANCHOR and reported in line


def _data_width(lines: list[str], delimiter: str, quotechar: str) -> int | None:
    """Return the modal field count (2 or more) of the end of the file.

    Line 0 is skipped, as it may be a truncated fragment of a tail sample.
    ``None`` when no line has 2 or more fields (e.g. a one-column file), or
    when fewer than half of the non-blank lines have the modal count: the
    delimiter then does not split the data (a wrong one that only occurs
    inside some values), and no line can be told to be data.
    """
    rows = [_split_fields(line, delimiter, quotechar) for line in lines[1:] if line.strip()]
    widths = [len(fields) for fields in rows if fields is not None and len(fields) > 1]
    if not widths:
        return None
    width, count = Counter(widths).most_common(1)[0]
    return width if count * 2 >= len(rows) else None


def _is_data_row(line: str, delimiter: str, quotechar: str, width: int) -> bool:
    """Whether a line is a data row: never part of a footer.

    A data row splits into ``width`` fields, at least half of them
    non-empty, and is not a totals row (see :func:`_extends_footer`, whose
    label logic keeps "Total Energies SA" a data row).
    """
    fields = _split_fields(line, delimiter, quotechar)
    if fields is None or len(fields) != width:
        return False
    filled = sum(1 for field in fields if field.strip())
    return filled * 2 >= width and not _extends_footer(line, delimiter, quotechar)


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
    model's delimiter (one that never occurs scores 0) is replaced only
    when exactly one usual delimiter scores the most, at least
    ``_MIN_AGREEING_LINES``, and at least ``_DOMINANCE_RATIO`` (2) times
    the model's score. A delimiter that splits half the lines the winner
    does therefore stays. On the sample fixtures, a wrong ``,`` scored 4
    to 13 times less than the tab (#151), while no other candidate ever
    reached the right delimiter's score. Otherwise it is kept: ties,
    one-column files and exotic delimiters stay as reported.

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
    scores = {
        candidate: _agreement_score(lines, candidate, result.quotechar)
        for candidate in _CANDIDATE_DELIMITERS
        if candidate not in (result.delimiter, result.quotechar, result.escapechar)
    }
    best = max(scores.values(), default=0)
    winners = [candidate for candidate, score in scores.items() if score == best]
    if best < max(_MIN_AGREEING_LINES, _DOMINANCE_RATIO * reported) or len(winners) != 1:
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
    answer: _ModelAnswer,
    head_sample: str,
    tail_sample: str | None,
    *,
    covers_whole_file: bool = True,
    detected_encoding: str | None = None,
) -> CSVInspectionResult:
    """Build the public result from the model's answer, matched against the sampled text.

    This is the only place the pipeline constructs a
    :class:`CSVInspectionResult`. Small local models recognize headers and
    footers reliably but count and copy lines poorly. Positions and verbatim
    text are therefore recomputed deterministically from the real samples,
    using the model's own answer as the key: the header row (and the column
    names as actually written) is located from the inferred columns, blank
    names the model left out included, and the footer is read verbatim from
    the end of the file, starting at the first non-data line the model
    pointed at (its ``footer_first_line``, matched tolerantly: see
    :func:`_locate_footer_lines`). A delimiter that splits too few head
    lines, or that another usual delimiter clearly dominates, is replaced
    (see :func:`_ground_delimiter`), and the reported encoding is checked
    against the detected one (see :func:`_ground_encoding`). A header at
    row 0 that cannot be anchored becomes "no header" when the first row
    has the same field shapes (integer, decimal, date, empty or text) as the
    second: header-less detection is a shape test on the sample, not on the
    model's answer (see :func:`_first_row_is_data`). Any other header that
    cannot be anchored is left as the model reported it. A footer anchor
    that cannot be found is dropped, with a WARNING, when the end of the
    file was sampled, since the footer is not there.

    Args:
        answer: The model's validated answer.
        head_sample: The decoded head sample.
        tail_sample: The decoded tail sample, or ``None`` when the head
            covers the whole file.
        covers_whole_file: Whether the samples reach the real end of the
            file. When ``False`` (no tail and a truncated head), any footer
            the model reported cannot be real, since the end was never
            seen, so ``footer_lines`` is empty.
        detected_encoding: The encoding detected from the raw head sample,
            or ``None`` to leave the reported encoding unchecked.

    Returns:
        The result, with ``delimiter``, ``header_row_index``, column names,
        ``footer_lines`` and ``encoding`` grounded in the samples.
    """
    result = CSVInspectionResult.model_validate(answer.model_dump(exclude={"footer_first_line"}))
    updates: dict[str, object] = {}

    model_delimiter = result.delimiter
    delimiter = _ground_delimiter(result, head_sample)
    if delimiter != result.delimiter:
        updates["delimiter"] = delimiter
        # Header and footer grounding split fields with the grounded delimiter.
        result = result.model_copy(update={"delimiter": delimiter})

    updates.update(_ground_header(result, head_sample))

    anchor = answer.footer_first_line
    if anchor is not None and (tail_sample is not None or covers_whole_file):
        end_of_file = tail_sample if tail_sample is not None else head_sample
        footer_lines = _locate_footer_lines(
            anchor, end_of_file, result.delimiter, result.quotechar, model_delimiter
        )
        if footer_lines is None:
            # The real end of the file was seen and the reported footer is
            # not there. Keeping it would make readers drop real data rows.
            logger.warning(
                "Discarding a footer line that does not occur at the end of the file: %r",
                anchor,
            )
        else:
            updates["footer_lines"] = footer_lines
    # Otherwise the footer stays empty: the model reported none, or the end
    # of the file was never sampled (the head's last lines are mid-file data).

    if detected_encoding is not None:
        encoding = _ground_encoding(result.encoding, detected_encoding)
        if encoding != result.encoding:
            updates["encoding"] = encoding

    if not updates:
        return result
    logger.info("Grounded model output in the sampled text: %s", updates)
    return result.model_copy(update=updates)
