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

logger = logging.getLogger(__name__)

# A line starting with a totals label (optionally quoted), e.g. "TOTAL;;;12.50",
# "Subtotal,,3", '"Total general",9' or "Total registros: 250".
_TOTALS_LABEL = re.compile(
    r"""^["']?\s*(?:sub\s*-?\s*)?(?:grand\s+)?(?:totals?|totales|suma|sum)\b""",
    re.IGNORECASE,
)


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
    if not expected or len(result.delimiter) != 1 or len(result.quotechar) != 1:
        return None
    rows = [
        _split_fields(line, result.delimiter, result.quotechar)
        for line in head_sample.lstrip("﻿").splitlines()
    ]

    for index, fields in enumerate(rows):
        if fields == expected:
            return index, fields

    width = len(expected)
    for index, (fields, next_fields) in enumerate(itertools.pairwise(rows)):
        if (
            fields is not None
            and next_fields is not None
            and len(fields) == len(next_fields) == width
            and set(fields) & set(expected)
        ):
            return index, fields
    return None


def _locate_footer_lines(footer_lines: list[str], end_of_file: str) -> list[str] | None:
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

    Returns:
        The grounded footer lines, or ``None`` when none of the model's
        non-blank footer lines occur in ``end_of_file`` (nothing to anchor).
    """
    reported = {line.strip() for line in footer_lines if line.strip()}
    if not reported:
        return None
    lines = end_of_file.splitlines()
    # Last occurrence of each reported line, so text that also appears earlier
    # in the data cannot drag data rows into the footer. Line 0 is skipped: in
    # a tail sample it is usually a truncated fragment.
    last_seen = {line.strip(): i for i, line in enumerate(lines) if i > 0}
    anchors = [last_seen[text] for text in reported if text in last_seen]
    if not anchors:
        return None
    start = min(anchors)
    while start > 1 and _extends_footer(lines[start - 1]):
        start -= 1
    return lines[start:]


def _extends_footer(line: str) -> bool:
    """Whether a line just above a known footer line also belongs to the footer.

    Only blank separator lines and rows labelled as totals qualify. A totals
    row often has the same number of fields as a data row, which is exactly
    what small models miss, but its label gives it away.
    """
    stripped = line.strip()
    return not stripped or _TOTALS_LABEL.match(stripped) is not None


def ground_in_samples(
    result: CSVInspectionResult,
    head_sample: str,
    tail_sample: str | None,
    *,
    covers_whole_file: bool = True,
) -> CSVInspectionResult:
    """Correct what the model reported by matching it against the sampled text.

    Small local models recognize headers and footers reliably but count and
    copy lines poorly. Positions and verbatim text are therefore recomputed
    deterministically from the real samples, using the model's own answer
    as the key: the header row (and the column names as actually written)
    is located from the inferred columns, and the footer is re-read
    verbatim from the end of the file. Anything that cannot be anchored is
    left exactly as the model reported it.

    Args:
        result: The model's validated result.
        head_sample: The decoded head sample.
        tail_sample: The decoded tail sample, or ``None`` when the head
            covers the whole file.
        covers_whole_file: Whether the samples reach the real end of the
            file. When ``False`` (no tail and a truncated head), any footer
            the model reported cannot be real, since the end was never
            seen, so ``footer_lines`` is cleared.

    Returns:
        The result with ``header_row_index``, column names and
        ``footer_lines`` grounded in the samples; the same object when
        nothing changed.
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
        footer_lines = _locate_footer_lines(result.footer_lines, end_of_file)
        if footer_lines is not None and footer_lines != result.footer_lines:
            updates["footer_lines"] = footer_lines

    if not updates:
        return result
    logger.info("Grounded model output in the sampled text: %s", updates)
    return result.model_copy(update=updates)
