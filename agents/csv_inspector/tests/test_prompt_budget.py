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

# The template with empty samples: 3540 characters on PROMPT_VERSION
# 2026.09-a, plus 10 %. Lower it when the template shrinks (#133).
PROMPT_TEMPLATE_MAX_CHARS = 3894

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
    'You are an expert data engineering agent specialized in detecting the quirks of "dirty" or non-standard CSV files.\n'
    "\n"
    "Below are byte samples from a real CSV file. The encoding heuristically detected by chardet is: 'utf-8' (it may be incorrect).\n"
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
    "The head sample stops somewhere in the middle of the data: its last line may be truncated and is never a footer. The tail sample is the real end of the file: read footer lines ONLY from its last lines. Its first visible line is very likely a truncated fragment, not a real row: do not use it to infer columns.\n"
    "\n"
    "Analyze the samples and respond ONLY with a JSON object (no extra text, no markdown, no backticks) with exactly this shape:\n"
    "\n"
    "{\n"
    '  "encoding": "<real encoding, e.g. utf-8, latin-1, cp1252>",\n'
    "  \"delimiter\": \"<field separator character, e.g. ',' or ';'>\",\n"
    '  "quotechar": "<character used to quote fields, or null if fields are never quoted>",\n'
    '  "escapechar": "<"\\\\" if quotes inside fields are written as \\", otherwise null>",\n'
    '  "doublequote": <true if quotes inside fields are written as "", false if as \\">,\n'
    '  "has_header": <true if a row holds the column names, false if the first row is already data>,\n'
    '  "header_row_index": <0-based index of the column-name row, or null if has_header is false>,\n'
    '  "footer_lines": ["<every footer line after the last data row, in file order>", "..."],\n'
    '  "columns": [\n'
    "    {\n"
    '      "name": "<name copied character for character from the header row; column_N if none>",\n'
    '      "inferred_type": "<string|integer|float|date|datetime|boolean>",\n'
    '      "nullable": <true|false>,\n'
    '      "example_values": ["<at most 3 raw values copied from this column>"]\n'
    "    }\n"
    "  ],\n"
    '  "confidence": <number between 0.0 and 1.0 indicating your confidence>,\n'
    '  "notes": "<relevant observations, or null>"\n'
    "}\n"
    "\n"
    'HEADER (start of the file): lines before the real column-name row, such as export banners, comments (e.g. starting with \'#\') or blank lines, are preamble. Do not list them anywhere; just count them: "header_row_index" is the 0-based index of the column-name row, i.e. the number of preamble lines. If the file has no column-name row at all (its first line is already a data record, e.g. "17,red,3.5"), answer "has_header": false, "header_row_index": null, and name the columns column_1, column_2, and so on.\n'
    "\n"
    "FOOTER (end of the file): check the last lines of the TAIL sample independently of the header. A data row holds a real record, with values like the rows above it (a date in the date column, a name in the name column, and so on). Any trailing line after the last data row is a footer line, for example:\n"
    '- a totals/summary row: it may have the same number of fields as a data row, but it carries a label instead of a record and leaves other fields empty, e.g. "TOTAL,,4241.25", "TOTAL;;;98765.40" or "Total registros: 250"\n'
    '- an end-of-report marker, e.g. "--- Fin del informe ---" or "*** END ***"\n'
    '- a generation timestamp or signature, e.g. "Generado el 2024-01-20 10:00:00"\n'
    "- a blank line separating the data from any of the above\n"
    'Copy every footer line verbatim into "footer_lines" (a blank line is ""), from the first footer line to the last line of the file. Use [] only when the file really ends with a data row. Footer lines are never part of the header preamble.\n'
    "\n"
    "Keep in mind:\n"
    "- The delimiter may also appear inside quoted fields; do not confuse it with the real separator.\n"
    "- Column names may contain accented characters and other special characters.\n"
    '- Check how quotes are escaped inside quoted fields: doubled ("") or backslash-escaped (\\"); see "escapechar" and "doublequote" above.\n'
    "- Rows may have an inconsistent number of fields; do not let that block your analysis.\n"
)

_GOLDEN_TRUNCATED_HEAD = (
    'You are an expert data engineering agent specialized in detecting the quirks of "dirty" or non-standard CSV files.\n'
    "\n"
    "Below are byte samples from a real CSV file. The encoding heuristically detected by chardet is: 'utf-8' (it may be incorrect).\n"
    "\n"
    "--- HEAD SAMPLE START (first bytes of the file) ---\n"
    "Fecha,Importe\n"
    "2024-01-15,1250.50\n"
    "\n"
    "--- HEAD SAMPLE END ---\n"
    "\n"
    "(The sample above is only the START of the file; its end was not sampled. Its last line may be truncated and is never a footer.)\n"
    "\n"
    "Analyze the samples and respond ONLY with a JSON object (no extra text, no markdown, no backticks) with exactly this shape:\n"
    "\n"
    "{\n"
    '  "encoding": "<real encoding, e.g. utf-8, latin-1, cp1252>",\n'
    "  \"delimiter\": \"<field separator character, e.g. ',' or ';'>\",\n"
    '  "quotechar": "<character used to quote fields, or null if fields are never quoted>",\n'
    '  "escapechar": "<"\\\\" if quotes inside fields are written as \\", otherwise null>",\n'
    '  "doublequote": <true if quotes inside fields are written as "", false if as \\">,\n'
    '  "has_header": <true if a row holds the column names, false if the first row is already data>,\n'
    '  "header_row_index": <0-based index of the column-name row, or null if has_header is false>,\n'
    '  "footer_lines": ["<every footer line after the last data row, in file order>", "..."],\n'
    '  "columns": [\n'
    "    {\n"
    '      "name": "<name copied character for character from the header row; column_N if none>",\n'
    '      "inferred_type": "<string|integer|float|date|datetime|boolean>",\n'
    '      "nullable": <true|false>,\n'
    '      "example_values": ["<at most 3 raw values copied from this column>"]\n'
    "    }\n"
    "  ],\n"
    '  "confidence": <number between 0.0 and 1.0 indicating your confidence>,\n'
    '  "notes": "<relevant observations, or null>"\n'
    "}\n"
    "\n"
    'HEADER (start of the file): lines before the real column-name row, such as export banners, comments (e.g. starting with \'#\') or blank lines, are preamble. Do not list them anywhere; just count them: "header_row_index" is the 0-based index of the column-name row, i.e. the number of preamble lines. If the file has no column-name row at all (its first line is already a data record, e.g. "17,red,3.5"), answer "has_header": false, "header_row_index": null, and name the columns column_1, column_2, and so on.\n'
    "\n"
    'FOOTER (end of the file): check the end of the file (not sampled here, so "footer_lines" must be []) independently of the header. A data row holds a real record, with values like the rows above it (a date in the date column, a name in the name column, and so on). Any trailing line after the last data row is a footer line, for example:\n'
    '- a totals/summary row: it may have the same number of fields as a data row, but it carries a label instead of a record and leaves other fields empty, e.g. "TOTAL,,4241.25", "TOTAL;;;98765.40" or "Total registros: 250"\n'
    '- an end-of-report marker, e.g. "--- Fin del informe ---" or "*** END ***"\n'
    '- a generation timestamp or signature, e.g. "Generado el 2024-01-20 10:00:00"\n'
    "- a blank line separating the data from any of the above\n"
    'Copy every footer line verbatim into "footer_lines" (a blank line is ""), from the first footer line to the last line of the file. Use [] only when the file really ends with a data row. Footer lines are never part of the header preamble.\n'
    "\n"
    "Keep in mind:\n"
    "- The delimiter may also appear inside quoted fields; do not confuse it with the real separator.\n"
    "- Column names may contain accented characters and other special characters.\n"
    '- Check how quotes are escaped inside quoted fields: doubled ("") or backslash-escaped (\\"); see "escapechar" and "doublequote" above.\n'
    "- Rows may have an inconsistent number of fields; do not let that block your analysis.\n"
)

_GOLDEN_WHOLE_FILE = (
    'You are an expert data engineering agent specialized in detecting the quirks of "dirty" or non-standard CSV files.\n'
    "\n"
    "Below are byte samples from a real CSV file. The encoding heuristically detected by chardet is: 'utf-8' (it may be incorrect).\n"
    "\n"
    "--- HEAD SAMPLE START (first bytes of the file) ---\n"
    "Fecha,Importe\n"
    "2024-01-15,1250.50\n"
    "\n"
    "--- HEAD SAMPLE END ---\n"
    "\n"
    "(The sample above contains the ENTIRE file; there is no separate tail. Its last lines are the real end of the file.)\n"
    "\n"
    "Analyze the samples and respond ONLY with a JSON object (no extra text, no markdown, no backticks) with exactly this shape:\n"
    "\n"
    "{\n"
    '  "encoding": "<real encoding, e.g. utf-8, latin-1, cp1252>",\n'
    "  \"delimiter\": \"<field separator character, e.g. ',' or ';'>\",\n"
    '  "quotechar": "<character used to quote fields, or null if fields are never quoted>",\n'
    '  "escapechar": "<"\\\\" if quotes inside fields are written as \\", otherwise null>",\n'
    '  "doublequote": <true if quotes inside fields are written as "", false if as \\">,\n'
    '  "has_header": <true if a row holds the column names, false if the first row is already data>,\n'
    '  "header_row_index": <0-based index of the column-name row, or null if has_header is false>,\n'
    '  "footer_lines": ["<every footer line after the last data row, in file order>", "..."],\n'
    '  "columns": [\n'
    "    {\n"
    '      "name": "<name copied character for character from the header row; column_N if none>",\n'
    '      "inferred_type": "<string|integer|float|date|datetime|boolean>",\n'
    '      "nullable": <true|false>,\n'
    '      "example_values": ["<at most 3 raw values copied from this column>"]\n'
    "    }\n"
    "  ],\n"
    '  "confidence": <number between 0.0 and 1.0 indicating your confidence>,\n'
    '  "notes": "<relevant observations, or null>"\n'
    "}\n"
    "\n"
    'HEADER (start of the file): lines before the real column-name row, such as export banners, comments (e.g. starting with \'#\') or blank lines, are preamble. Do not list them anywhere; just count them: "header_row_index" is the 0-based index of the column-name row, i.e. the number of preamble lines. If the file has no column-name row at all (its first line is already a data record, e.g. "17,red,3.5"), answer "has_header": false, "header_row_index": null, and name the columns column_1, column_2, and so on.\n'
    "\n"
    "FOOTER (end of the file): check the last lines of the sample above independently of the header. A data row holds a real record, with values like the rows above it (a date in the date column, a name in the name column, and so on). Any trailing line after the last data row is a footer line, for example:\n"
    '- a totals/summary row: it may have the same number of fields as a data row, but it carries a label instead of a record and leaves other fields empty, e.g. "TOTAL,,4241.25", "TOTAL;;;98765.40" or "Total registros: 250"\n'
    '- an end-of-report marker, e.g. "--- Fin del informe ---" or "*** END ***"\n'
    '- a generation timestamp or signature, e.g. "Generado el 2024-01-20 10:00:00"\n'
    "- a blank line separating the data from any of the above\n"
    'Copy every footer line verbatim into "footer_lines" (a blank line is ""), from the first footer line to the last line of the file. Use [] only when the file really ends with a data row. Footer lines are never part of the header preamble.\n'
    "\n"
    "Keep in mind:\n"
    "- The delimiter may also appear inside quoted fields; do not confuse it with the real separator.\n"
    "- Column names may contain accented characters and other special characters.\n"
    '- Check how quotes are escaped inside quoted fields: doubled ("") or backslash-escaped (\\"); see "escapechar" and "doublequote" above.\n'
    "- Rows may have an inconsistent number of fields; do not let that block your analysis.\n"
)
