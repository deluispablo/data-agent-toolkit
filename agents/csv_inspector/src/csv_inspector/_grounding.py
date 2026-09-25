"""Grounding of the model's answer in the sampled text.

Small local models recognize headers and footers reliably but count and copy
lines poorly. The rules here recompute positions and verbatim text from the
real samples, using the answer as a key. Each sample is parsed once
(:class:`_ParsedSample`), with the grounded delimiter and quote character.
"""

from __future__ import annotations

import csv
import itertools
import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from ._encoding import canonical_codec_name, split_lines
from ._models import CSVInspectionResult, _ModelAnswer

logger = logging.getLogger(__name__)

_LABEL = r"(?:sub\s*-?\s*)?(?:grand\s+)?(?:totals?|totales|suma|sum)"
# A line starting with a totals label (optionally quoted), e.g. "TOTAL;;;12.50",
# "Subtotal,,3", '"Total general",9' or "Total registros: 250".
_TOTALS_LABEL = re.compile(rf"""^["']?\s*{_LABEL}\b""", re.IGNORECASE)
# A field that is nothing but a totals label: "TOTAL", "Grand total:", "Total general".
_BARE_TOTALS_LABEL = re.compile(rf"{_LABEL}(?:\s+general)?\s*:?", re.IGNORECASE)

# Encodings named for a byte order mark: chardet reports them only when it sees
# one, and any other codec leaves a U+FEFF glued to the first column name.
_BOM_CODECS = frozenset({"utf-8-sig", "utf-16", "utf-32"})

# The delimiters tried when the model's one does not split the head's lines.
_CANDIDATE_DELIMITERS = ",;\t|"
# How many lines must split into the same number of fields to pick a candidate.
_MIN_AGREEING_LINES = 2
# How many times the model's score a single candidate must reach to replace it.
_DOMINANCE_RATIO = 1.5
# Runs of whitespace and usual delimiters, squashed when matching a footer anchor.
_SEPARATORS = re.compile(r"[\s,;|]+")
# The shortest reported footer text that anchors a line it only occurs in.
_MIN_SUBSTRING_ANCHOR = 8

# Field shapes compared by the header-less test (see _field_shape).
_INTEGER = re.compile(r"[+-]?\d+")
_DECIMAL = re.compile(r"[+-]?(?:\d{1,3}(?:[.,]\d{3})+|\d*)[.,]\d+")
_DATE = re.compile(
    r"\d{4}-\d{1,2}-\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"
    r"|\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})"
)


def _split_fields(line: str, delimiter: str, quotechar: str) -> list[str] | None:
    """Split one line into fields as written (spaces kept), or ``None`` if unparsable."""
    try:
        return next(csv.reader([line], delimiter=delimiter, quotechar=quotechar))
    except (csv.Error, StopIteration):
        return None


def _split_rows(lines: Iterable[str], delimiter: str, quotechar: str) -> list[list[str] | None]:
    """Split every line into fields (see :func:`_split_fields`)."""
    return [_split_fields(line, delimiter, quotechar) for line in lines]


@dataclass(frozen=True)
class _ParsedSample:
    """A sample's lines, as ``csv`` and pandas count them, and each line's fields."""

    lines: list[str]
    rows: list[list[str] | None]
    delimiter: str
    quotechar: str

    @classmethod
    def parse(cls, text: str, delimiter: str, quotechar: str) -> _ParsedSample:
        """Split ``text`` into lines, then each line into fields."""
        lines = split_lines(text)
        return cls(lines, _split_rows(lines, delimiter, quotechar), delimiter, quotechar)

    def is_data_row(self, index: int, width: int) -> bool:
        """Whether line ``index`` has ``width`` fields, half of them filled, and no totals label."""
        fields = self.rows[index]
        if fields is None or len(fields) != width:
            return False
        filled = sum(1 for field in fields if field.strip())
        line = self.lines[index]
        return filled * 2 >= width and not _extends_footer(line, self.delimiter, self.quotechar)


def _modal_width(rows: Iterable[list[str] | None]) -> tuple[int, int]:
    """``(agreeing rows, field count)`` of the modal field count of 2 or more, else ``(0, 1)``."""
    widths = [len(fields) for fields in rows if fields is not None and len(fields) > 1]
    if not widths:
        return 0, 1
    width, count = Counter(widths).most_common(1)[0]
    return count, width


def head_field_count(head_sample: str) -> int:
    """The column count to size the reply for: the best usual delimiter's modal field count."""
    lines = split_lines(head_sample.lstrip("\ufeff"))
    scores = (_modal_width(_split_rows(lines, d, '"')) for d in _CANDIDATE_DELIMITERS)
    return max(scores)[1]


def _ground_delimiter(result: CSVInspectionResult, head_sample: str) -> dict[str, object]:
    """Replace the model's delimiter only when it clearly loses.

    A delimiter scores the head lines it splits into its modal field count
    (2 or more), preamble and footer lines notwithstanding. The model's (0 if
    it never occurs) yields only to exactly one usual delimiter scoring the
    most, at least ``_MIN_AGREEING_LINES`` and ``_DOMINANCE_RATIO`` times its
    score: on the fixtures a wrong ``,`` lost 1.6x to 13x to the tab (#151).
    """
    lines = split_lines(head_sample.lstrip("\ufeff"))
    quotechar = result.quotechar
    reported = 0
    if result.delimiter in head_sample:
        reported = _modal_width(_split_rows(lines, result.delimiter, quotechar))[0]
    scores = {
        candidate: _modal_width(_split_rows(lines, candidate, quotechar))[0]
        for candidate in _CANDIDATE_DELIMITERS
        if candidate not in (result.delimiter, quotechar, result.escapechar)
    }
    best = max(scores.values(), default=0)
    winners = [candidate for candidate, score in scores.items() if score == best]
    if best >= max(_MIN_AGREEING_LINES, _DOMINANCE_RATIO * reported) and len(winners) == 1:
        msg = "Replacing delimiter %r (%d agreeing lines) with %r (%d agreeing lines)."
        logger.info(msg, result.delimiter, reported, winners[0], best)
        return {"delimiter": winners[0]}
    # A delimiter that never occurs splits nothing: in a one-column file
    # report the default, which is just as inert, not the model's guess.
    comma_taken = "," in (result.delimiter, quotechar, result.escapechar)
    return {"delimiter": ","} if result.delimiter not in head_sample and not comma_taken else {}


def _has_quoted_field(text: str, delimiter: str, quote: str) -> bool:
    """Whether a line of ``text`` holds a field enclosed in ``quote``, at field edges."""
    edge = re.escape(delimiter)
    quoted = re.escape(quote)
    pattern = rf"(?:^|{edge}){quoted}[^{quoted}\r\n]+{quoted}(?={edge}|\r|$)"
    return re.search(pattern, text, re.MULTILINE) is not None


def _ground_quote_escaping(result: CSVInspectionResult, text: str) -> dict[str, object]:
    r"""Return the ``escapechar``/``doublequote`` pair the samples show, when it differs.

    ``\"`` alone means ``("\\", False)``; a doubled quote after a field
    character (``abc""``, not the empty field ``,"",``) alone means
    ``(None, True)``; neither, with a quoted field, means nothing is escaped:
    ``(None, doublequote as answered)``. Otherwise the answer stays.
    """
    quote = result.quotechar
    backslashed = f"\\{quote}" in text
    # A field character: not a delimiter, quote, backslash or line break, so
    # neither an empty field nor a backslash-escaped quote before a closing
    # quote (\"") counts as a doubled quote.
    field = f"[^{re.escape(result.delimiter + quote)}\\\\\\r\\n]"
    doubled = re.search(f"{field}{re.escape(quote * 2)}", text) is not None
    if backslashed and not doubled:
        grounded: tuple[str | None, bool] = ("\\", False)
    elif doubled and not backslashed:
        grounded = (None, True)
    elif not backslashed and _has_quoted_field(text, result.delimiter, quote):
        grounded = (None, result.doublequote)
    else:
        return {}
    if grounded == (result.escapechar, result.doublequote):
        return {}
    return {"escapechar": grounded[0], "doublequote": grounded[1]}


def _field_shape(value: str) -> str:
    """Classify one field as ``empty``, ``integer``, ``date``, ``decimal`` or ``text``.

    Spaces are ignored; decimals take ``.`` or ``,`` with optional grouping
    (``1.234,56``); dates are ISO or day-first, tested before decimals so
    ``15.01.2024`` is not a number.
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


def _rows_look_alike(first: list[str], second: list[str]) -> bool:
    """Whether two equal-width rows have the same shapes, or agreeing ones half non-text in both.

    See :func:`_field_shape`; an empty field agrees with any shape.
    """
    first_shape = [_field_shape(field) for field in first]
    second_shape = [_field_shape(field) for field in second]
    if first_shape == second_shape:
        return True
    half = (len(first) + 1) // 2
    return (
        all(a == b or "empty" in (a, b) for a, b in zip(first_shape, second_shape, strict=True))
        and sum(shape != "text" for shape in first_shape) >= half
        and sum(shape != "text" for shape in second_shape) >= half
    )


def _first_row_is_data(result: CSVInspectionResult, head: _ParsedSample) -> bool:
    """Whether line 0 is shaped like line 1, not like names: a header-less file.

    Rows 0 and 1 have one field per inferred column and look alike, and no
    row-0 field is one of the names the model invented for it.
    """
    if len(head.rows) < _MIN_AGREEING_LINES:
        return False
    first, second = head.rows[0], head.rows[1]
    if not first or second is None or not len(first) == len(second) == len(result.columns):
        return False
    names = {name.strip().casefold() for name in result.columns} - {""}
    if any(field.strip().casefold() in names for field in first):
        return False
    return _rows_look_alike(first, second)


def _locate_header_row(
    result: CSVInspectionResult, head: _ParsedSample
) -> tuple[int, list[str]] | None:
    """Find the header line and its fields as written, blank names included.

    Models miscount preamble lines and paraphrase names, but get the column
    count and some names right. The header is the first line whose (non-empty)
    fields are the answer's names; else the first line of the answer's width,
    followed by one of the same width, that shares a name with the answer.
    """
    expected = [name.strip() for name in result.columns]
    if not expected:
        return None
    for index, fields in enumerate(head.rows):
        if fields is None:
            continue
        stripped = [field.strip() for field in fields]
        if stripped == expected or (
            len(fields) > len(expected) and [field for field in stripped if field] == expected
        ):
            return index, fields
    # An unnamed column (e.g. a pandas index) must not match any empty cell.
    named = set(expected) - {""}
    for index, (fields, next_fields) in enumerate(itertools.pairwise(head.rows)):
        if (
            fields is not None
            and next_fields is not None
            and len(fields) == len(next_fields) == len(expected)
            and {field.strip() for field in fields} & named
        ):
            return index, fields
    return None


def _positional(count: int) -> list[str]:
    """Positional column names, ``column_1`` to ``column_<count>``."""
    return [f"column_{number}" for number in range(1, count + 1)]


def _one_column_header(
    result: CSVInspectionResult, head: _ParsedSample
) -> dict[str, object] | None:
    """When the delimiter splits no head line, find the one name among the lines listed.

    The line equal to the first name (else the answer's header row) is the header.
    """
    if len(result.columns) <= 1 or _modal_width(head.rows)[0]:
        return None
    at = next((i for i, line in enumerate(head.lines) if line.strip() == result.columns[0]), None)
    if at is None and result.has_header:
        at = result.header_row_index
    if at is None or at >= len(head.rows):
        return {"has_header": False, "header_row_index": None, "columns": ["column_1"]}
    return {"has_header": True, "header_row_index": at, "columns": [(head.rows[at] or [""])[0]]}


def _no_header_answer(result: CSVInspectionResult, head: _ParsedSample) -> dict[str, object] | None:
    """Ground a "no header" answer: positional names, unless its names are a line.

    Names that are exactly a head line, the last one or one followed by a line
    not shaped like it, mean the model found the header after all.
    """
    if result.has_header:
        return None
    rows = head.rows
    for index, (fields, next_fields) in enumerate(zip(rows, [*rows[1:], None], strict=True)):
        if (
            fields is not None
            and [field.strip() for field in fields] == result.columns
            and (
                next_fields is None
                or (len(next_fields) == len(fields) and not _rows_look_alike(fields, next_fields))
            )
        ):
            logger.info("The model's column names are line %d: reporting a header row.", index)
            return {"has_header": True, "header_row_index": index, "columns": fields}
    positional = _positional(len(result.columns))
    return {} if result.columns == positional else {"columns": positional}


def _header_answer(result: CSVInspectionResult, head: _ParsedSample) -> dict[str, object]:
    """Anchor a header answer; an unanchored one at row 0 over data becomes "no header"."""
    header = _locate_header_row(result, head)
    if header is None:
        if result.header_row_index != 0 or not _first_row_is_data(result, head):
            return {}
        logger.info("The first row holds data, not column names: reporting no header row.")
        columns = _positional(len(result.columns))
        return {"has_header": False, "header_row_index": None, "columns": columns}
    updates: dict[str, object] = {}
    header_row_index, names = header
    if header_row_index != result.header_row_index:
        updates["header_row_index"] = header_row_index
    # The header's fields as written, blank names the model left out included.
    if names != result.columns:
        updates["columns"] = names
    return updates


def _ground_header(result: CSVInspectionResult, head: _ParsedSample) -> dict[str, object]:
    """Return the header fields to correct, from the first header rule that applies."""
    updates = _one_column_header(result, head)
    if updates is None:
        updates = _no_header_answer(result, head)
    return _header_answer(result, head) if updates is None else updates


def _anchor_matches(reported: str, line: str, delimiter: str) -> bool:
    """Whether a stripped line the model reported designates this file line.

    Equal once trailing empty fields are stripped, or once runs of whitespace
    and usual delimiters are squashed (a row copied with spaces for tabs), or
    a reported text of ``_MIN_SUBSTRING_ANCHOR`` characters or more within it.
    """
    trailing = delimiter + " \t"
    if reported.rstrip(trailing) == line.strip().rstrip(trailing):
        return True
    if _SEPARATORS.sub(" ", reported).strip() == _SEPARATORS.sub(" ", line).strip():
        return True
    return len(reported) >= _MIN_SUBSTRING_ANCHOR and reported in line


def _extends_footer(line: str, delimiter: str, quotechar: str) -> bool:
    """Whether a line just above a footer line belongs to it: blank, or labelled totals.

    A label prefix alone is not enough ("Total Energies" can name a data
    row): the first field must be the bare label, or most others empty.
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


def _data_width(end: _ParsedSample) -> int | None:
    """The modal field count (2 or more) of the end of the file, if half its lines have it.

    Line 0 is skipped: in a tail sample it is usually a truncated fragment.
    """
    lines = zip(end.lines[1:], end.rows[1:], strict=True)
    rows = [fields for line, fields in lines if line.strip()]
    count, width = _modal_width(rows)
    return width if count and count * 2 >= len(rows) else None


def _footer_start(end: _ParsedSample, width: int | None, found: int | None) -> int:
    """After the last data row; without a width, at the anchor, up over blank and totals lines."""
    if width is not None:
        rows = range(len(end.lines) - 1, 0, -1)
        return next((i for i in rows if end.is_data_row(i, width)), 0) + 1
    start = len(end.lines) if found is None else found
    while start > 1 and _extends_footer(end.lines[start - 1], end.delimiter, end.quotechar):
        start -= 1
    return start


def _locate_footer_lines(
    anchor: str, end: _ParsedSample, model_delimiter: str | None = None
) -> list[str] | None:
    """Read the footer verbatim from the end of the file, from the model's first footer line.

    Models copy several lines poorly. The anchor must match a last line (see
    :func:`_anchor_matches`), also with the replaced ``model_delimiter``, or be
    a data row (a miscopied last one); a blank one always qualifies. ``None``
    when it does not, or when only data follows it.
    """
    reported = anchor.strip()
    keys = {reported}
    if model_delimiter and model_delimiter != end.delimiter:
        keys.add(reported.replace(model_delimiter, end.delimiter))
    width = _data_width(end)
    lines, delimiter = end.lines, end.delimiter
    matches = [
        i
        for i in range(1, len(lines))
        if any(_anchor_matches(k, lines[i], delimiter) for k in keys)
    ]
    # The last match after line 0: text also found earlier may be data.
    found = matches[-1] if matches else None
    if reported and found is None:
        rows = _split_rows(keys, end.delimiter, end.quotechar)
        keyed = _ParsedSample([*keys], rows, end.delimiter, end.quotechar)
        if width is None or not any(keyed.is_data_row(i, width) for i in range(len(keys))):
            return None
    start = _footer_start(end, width, found)
    return end.lines[start:] if start < len(end.lines) else None


def _ground_footer(answer: _ModelAnswer, end: _ParsedSample | None) -> dict[str, object]:
    """Read the footer at the model's anchor; ``end`` is ``None`` if the end was never sampled."""
    anchor = answer.footer_first_line
    if anchor is None or end is None:
        return {}
    footer_lines = _locate_footer_lines(anchor, end, answer.delimiter)
    if footer_lines is None:
        # The real end of the file was seen and the reported footer is not
        # there. Keeping it would make readers drop real data rows.
        msg = "Discarding a footer line that does not occur at the end of the file: %r"
        logger.warning(msg, anchor)
        return {}
    return {"footer_lines": footer_lines}


def _ground_encoding(reported: str, detected: str | None) -> dict[str, object]:
    """Keep the model's encoding unless it is no codec or misses a detected (invisible) BOM."""
    if detected is None:
        return {}
    reported_codec = canonical_codec_name(reported)
    detected_codec = canonical_codec_name(detected)
    if detected_codec in _BOM_CODECS and reported_codec != detected_codec:
        encoding = detected
    else:
        encoding = reported if reported_codec is not None else detected
    return {} if encoding == reported else {"encoding": encoding}


def ground_in_samples(
    answer: _ModelAnswer,
    head_sample: str,
    tail_sample: str | None,
    *,
    covers_whole_file: bool = True,
    detected_encoding: str | None = None,
) -> CSVInspectionResult:
    """Build the public result from the model's answer, matched against the sampled text.

    The only place the pipeline constructs a :class:`CSVInspectionResult`.
    The rules run in order: delimiter, quote character, quote escaping,
    header, footer (read from the end of the file, or dropped with a WARNING
    when the end was sampled and the anchor is not there), encoding.

    Args:
        answer: The model's validated answer.
        head_sample: The decoded head sample.
        tail_sample: The decoded tail sample; ``None`` when the head covers the file.
        covers_whole_file: Whether the samples reach the end of the file (else no footer).
        detected_encoding: The encoding detected from the raw head, or ``None``.

    Returns:
        The result, with its dialect, header, footer and encoding grounded.
    """
    result = CSVInspectionResult.model_validate(answer.model_dump(exclude={"footer_first_line"}))
    text = head_sample + (tail_sample or "")
    # The later rules split fields with the grounded delimiter and quote character.
    updates = _ground_delimiter(result, head_sample)
    if result.quotechar != '"' and result.quotechar not in text:
        # A quote character that never occurs quotes nothing: report the default.
        updates["quotechar"] = '"'
    result = result.model_copy(update=updates)
    head = _ParsedSample.parse(head_sample.lstrip("\ufeff"), result.delimiter, result.quotechar)
    # Without a tail, the head is the end of the file if it covers it; line 0,
    # the only one a BOM changes, is never part of a footer.
    end = head if covers_whole_file else None
    if tail_sample is not None:
        end = _ParsedSample.parse(tail_sample, result.delimiter, result.quotechar)
    updates.update(_ground_quote_escaping(result, text))
    updates.update(_ground_header(result, head))
    updates.update(_ground_footer(answer, end))
    updates.update(_ground_encoding(result.encoding, detected_encoding))
    if not updates:
        return result
    logger.info("Grounded model output in the sampled text: %s", updates)
    return result.model_copy(update=updates)
