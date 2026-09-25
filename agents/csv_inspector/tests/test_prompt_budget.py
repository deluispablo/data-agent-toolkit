"""Prompt size budget and golden prompts (issue #127).

The instruction template is sent with every call, so its size is a cost.
These tests fail when it grows past its budget, and show any change to its
wording as a readable diff against the golden strings below, without a
model call. A deliberate prompt change bumps ``PROMPT_VERSION`` and updates
the budget and the golden strings in the same pull request: see
``docs/evaluation.md`` "Changing the prompt".
"""

from __future__ import annotations

import re

import pytest

from csv_inspector._prompt import PROMPT_VERSION, build_prompt

# The template with empty samples: 1912 characters on PROMPT_VERSION
# 2026.09-j, plus 10 %. Lower it when the template shrinks (#133).
PROMPT_TEMPLATE_MAX_CHARS = 2104

_HEAD = "Fecha,Importe\n2024-01-15,1250.50\n"
_TAIL = "15,890.00\nTOTAL,2140.50\n"


def test_prompt_template_fits_its_budget() -> None:
    """Growing the template past the budget fails, naming the budget and the size."""
    size = len(build_prompt("", "utf-8"))

    assert size <= PROMPT_TEMPLATE_MAX_CHARS, (
        f"The prompt template is {size} characters, over PROMPT_TEMPLATE_MAX_CHARS "
        f"({PROMPT_TEMPLATE_MAX_CHARS}). Shrink it, or raise the budget deliberately "
        "and bump PROMPT_VERSION."
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"tail_sample": _TAIL}, {"covers_whole_file": False}, {}],
    ids=["tail", "truncated-head", "whole-file"],
)
def test_prompt_names_no_json_shape(kwargs: dict[str, object]) -> None:
    """The schema sent with the request is the shape; the prompt holds no copy of it (#130)."""
    prompt = build_prompt(_HEAD, "utf-8", **kwargs)  # type: ignore[arg-type]

    assert '"encoding":' not in prompt
    assert "exactly this shape" not in prompt
    assert "matching the schema you were given" in prompt


def test_prompt_version_is_date_based() -> None:
    """``YYYY.MM-x``: the month of the change and a suffix for several bumps in it."""
    assert re.fullmatch(r"\d{4}\.\d{2}-[a-z]", PROMPT_VERSION)


@pytest.mark.parametrize(
    ("kwargs", "golden"),
    [
        ({"tail_sample": _TAIL}, "_GOLDEN_WITH_TAIL"),
        ({"covers_whole_file": False}, "_GOLDEN_TRUNCATED_HEAD"),
        ({}, "_GOLDEN_WHOLE_FILE"),
    ],
    ids=["tail", "truncated-head", "whole-file"],
)
def test_prompt_matches_its_golden_string(kwargs: dict[str, object], golden: str) -> None:
    """Each of the three prompt branches is exactly its golden string."""
    prompt = build_prompt(_HEAD, "utf-8", **kwargs)  # type: ignore[arg-type]

    assert prompt.splitlines() == globals()[golden].splitlines()


# Golden prompts, one literal per line so a wording change reads as a diff.
_GOLDEN_WITH_TAIL = (
    "Byte samples of a real, possibly messy CSV file follow. Encoding guessed by chardet (may be wrong): 'utf-8'.\n"
    "\n"
    "--- HEAD SAMPLE START (first bytes of the file) ---\n"
    "Fecha,Importe\n"
    "2024-01-15,1250.50\n"
    "\n"
    "--- HEAD SAMPLE END ---\n"
    "\n"
    "--- TAIL SAMPLE START (last bytes of the file; may start mid-line or mid-word) ---\n"
    "15,890.00\n"
    "TOTAL,2140.50\n"
    "\n"
    "--- TAIL SAMPLE END ---\n"
    "\n"
    "The head stops mid-data: its last line may be cut and is never a footer. The tail is the real end of the file; its first line is likely a cut fragment: do not use it for columns.\n"
    "\n"
    "Analyze the samples and answer with a JSON object matching the schema you were given. What its fields mean:\n"
    '- "encoding" is the real encoding, e.g. utf-8, latin-1, cp1252.\n'
    '- "quotechar" is the character that wraps quoted fields: "\'" when fields look like \'Acme, S.L.\', \'"\' when they look like "Acme, S.L." or are never quoted.\n'
    '- "escapechar" and "doublequote": a quote inside a quoted field written with a backslash (\\") means "escapechar": "\\\\", "doublequote": false; written doubled ("") or never present, "escapechar": null, "doublequote": true.\n'
    '- "delimiter" is the real separator: it may also appear inside quoted fields, and rows may have uneven field counts.\n'
    '- "columns" holds each name copied character for character from the header row.\n'
    "\n"
    "HEADER:\n"
    "- Preamble lines (export banners, '#' comments, blank lines) come before the column-name row: \"header_row_index\" is their count (0-based index of that row). Never list them; footer lines are never preamble.\n"
    '- No column-name row (the first line is already data, e.g. "17,red,3.5"): "has_header": false, "header_row_index": null, columns column_1, column_2, ...\n'
    "\n"
    "FOOTER: in the last lines of the TAIL sample, every line after the last data row (a data row holds a real record, like the rows above it):\n"
    '- a totals row: a label instead of a record, other fields empty, e.g. "TOTAL,,4241.25" or "Total registros: 250"\n'
    '- an end marker, e.g. "--- Fin del informe ---"; a timestamp, e.g. "Generado el 2024-01-20 10:00:00"; a blank line before them\n'
    '"footer_first_line": the first non-blank footer line, verbatim; null only when the file ends with a data row.\n'
)

_GOLDEN_TRUNCATED_HEAD = (
    "Byte samples of a real, possibly messy CSV file follow. Encoding guessed by chardet (may be wrong): 'utf-8'.\n"
    "\n"
    "--- HEAD SAMPLE START (first bytes of the file) ---\n"
    "Fecha,Importe\n"
    "2024-01-15,1250.50\n"
    "\n"
    "--- HEAD SAMPLE END ---\n"
    "\n"
    "(The sample above is only the START of the file; its end was not sampled. Its last line may be truncated and is never a footer.)\n"
    "\n"
    "Analyze the samples and answer with a JSON object matching the schema you were given. What its fields mean:\n"
    '- "encoding" is the real encoding, e.g. utf-8, latin-1, cp1252.\n'
    '- "quotechar" is the character that wraps quoted fields: "\'" when fields look like \'Acme, S.L.\', \'"\' when they look like "Acme, S.L." or are never quoted.\n'
    '- "escapechar" and "doublequote": a quote inside a quoted field written with a backslash (\\") means "escapechar": "\\\\", "doublequote": false; written doubled ("") or never present, "escapechar": null, "doublequote": true.\n'
    '- "delimiter" is the real separator: it may also appear inside quoted fields, and rows may have uneven field counts.\n'
    '- "columns" holds each name copied character for character from the header row.\n'
    "\n"
    "HEADER:\n"
    "- Preamble lines (export banners, '#' comments, blank lines) come before the column-name row: \"header_row_index\" is their count (0-based index of that row). Never list them; footer lines are never preamble.\n"
    '- No column-name row (the first line is already data, e.g. "17,red,3.5"): "has_header": false, "header_row_index": null, columns column_1, column_2, ...\n'
    "\n"
    'FOOTER: in the end of the file (not sampled here, so "footer_first_line" must be null), every line after the last data row (a data row holds a real record, like the rows above it):\n'
    '- a totals row: a label instead of a record, other fields empty, e.g. "TOTAL,,4241.25" or "Total registros: 250"\n'
    '- an end marker, e.g. "--- Fin del informe ---"; a timestamp, e.g. "Generado el 2024-01-20 10:00:00"; a blank line before them\n'
    '"footer_first_line": the first non-blank footer line, verbatim; null only when the file ends with a data row.\n'
)

_GOLDEN_WHOLE_FILE = (
    "Byte samples of a real, possibly messy CSV file follow. Encoding guessed by chardet (may be wrong): 'utf-8'.\n"
    "\n"
    "--- HEAD SAMPLE START (first bytes of the file) ---\n"
    "Fecha,Importe\n"
    "2024-01-15,1250.50\n"
    "\n"
    "--- HEAD SAMPLE END ---\n"
    "\n"
    "(The sample above contains the ENTIRE file; there is no separate tail. Its last lines are the real end of the file.)\n"
    "\n"
    "Analyze the samples and answer with a JSON object matching the schema you were given. What its fields mean:\n"
    '- "encoding" is the real encoding, e.g. utf-8, latin-1, cp1252.\n'
    '- "quotechar" is the character that wraps quoted fields: "\'" when fields look like \'Acme, S.L.\', \'"\' when they look like "Acme, S.L." or are never quoted.\n'
    '- "escapechar" and "doublequote": a quote inside a quoted field written with a backslash (\\") means "escapechar": "\\\\", "doublequote": false; written doubled ("") or never present, "escapechar": null, "doublequote": true.\n'
    '- "delimiter" is the real separator: it may also appear inside quoted fields, and rows may have uneven field counts.\n'
    '- "columns" holds each name copied character for character from the header row.\n'
    "\n"
    "HEADER:\n"
    "- Preamble lines (export banners, '#' comments, blank lines) come before the column-name row: \"header_row_index\" is their count (0-based index of that row). Never list them; footer lines are never preamble.\n"
    '- No column-name row (the first line is already data, e.g. "17,red,3.5"): "has_header": false, "header_row_index": null, columns column_1, column_2, ...\n'
    "\n"
    "FOOTER: in the last lines of the sample above, every line after the last data row (a data row holds a real record, like the rows above it):\n"
    '- a totals row: a label instead of a record, other fields empty, e.g. "TOTAL,,4241.25" or "Total registros: 250"\n'
    '- an end marker, e.g. "--- Fin del informe ---"; a timestamp, e.g. "Generado el 2024-01-20 10:00:00"; a blank line before them\n'
    '"footer_first_line": the first non-blank footer line, verbatim; null only when the file ends with a data row.\n'
)
