"""Prompt construction and response parsing for csv_inspector.

Known limitation: the samples are delimited by plain-text marker lines
(``--- HEAD SAMPLE START ... ---``). A file that contains such a line, or
text mimicking the instructions, can confuse the model. This is accepted on
purpose: grounding recomputes the delimiter, header row, column names,
footer and encoding from the real bytes, so the model's answer is only a
key, and random markers or escaping would cost tokens on every call.
"""

from __future__ import annotations

import json
import re
from typing import get_args

from pydantic import ValidationError

from ._exceptions import ResponseParsingError, SchemaValidationError
from ._models import ColumnType, CSVInspectionResult

# Bumped by hand on any change to the prompt wording (date-based; the suffix
# tells several bumps in one month apart). Recorded in every Usage and eval
# run, so measurements of different prompts are never mixed; see
# docs/evaluation.md "Changing the prompt".
PROMPT_VERSION = "2026.09-a"

SYSTEM_PROMPT = "You always respond with valid JSON, with no explanations or markdown."

# Built from the model's vocabulary so the prompt can never drift from it.
_COLUMN_TYPE_CHOICES = "|".join(get_args(ColumnType))

_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def build_prompt(
    head_sample: str,
    detected_encoding: str,
    tail_sample: str | None = None,
    covers_whole_file: bool = True,
) -> str:
    """Build the prompt sent to the LLM to infer the CSV dialect and schema.

    Args:
        head_sample: The decoded text sample from the start of the source
            file.
        detected_encoding: The encoding heuristically detected for the head
            sample, included as a hint the model may override.
        tail_sample: The decoded text sample from the end of the source
            file, or ``None`` when the head sample already covers the whole
            file (in which case a separate tail section is omitted to save
            tokens). When present, this sample is a blind byte-suffix and
            may start mid-line or mid-character.
        covers_whole_file: Whether the samples reach the real end of the
            file. Only consulted without a tail sample: ``False`` means the
            head was truncated and the end of the file was never sampled,
            so the model is told not to report any footer.

    Returns:
        A complete prompt instructing the model to respond with a single
        JSON object matching the ``CSVInspectionResult`` schema.
    """
    if tail_sample is not None:
        tail_section = f"""
--- TAIL SAMPLE START (last bytes of the file; may start mid-line or mid-word) ---
{tail_sample}
--- TAIL SAMPLE END ---

The head sample stops somewhere in the middle of the data: its last line \
may be truncated and is never a footer. The tail sample is the real end of \
the file: read footer lines ONLY from its last lines. Its first visible \
line is very likely a truncated fragment, not a real row: do not use it to \
infer columns.
"""
        file_end = "the last lines of the TAIL sample"
    elif not covers_whole_file:
        tail_section = (
            "\n(The sample above is only the START of the file; its end was not sampled. "
            "Its last line may be truncated and is never a footer.)\n"
        )
        file_end = 'the end of the file (not sampled here, so "footer_lines" must be [])'
    else:
        tail_section = (
            "\n(The sample above contains the ENTIRE file; there is no separate tail. "
            "Its last lines are the real end of the file.)\n"
        )
        file_end = "the last lines of the sample above"

    return f"""You are an expert data engineering agent specialized in detecting \
the quirks of "dirty" or non-standard CSV files.

Below are byte samples from a real CSV file. The encoding heuristically \
detected by chardet is: {detected_encoding!r} (it may be incorrect).

--- HEAD SAMPLE START (first bytes of the file) ---
{head_sample}
--- HEAD SAMPLE END ---
{tail_section}
Analyze the samples and respond ONLY with a JSON object (no extra text, no \
markdown, no backticks) with exactly this shape:

{{
  "encoding": "<real encoding, e.g. utf-8, latin-1, cp1252>",
  "delimiter": "<field separator character, e.g. ',' or ';'>",
  "quotechar": "<character used to quote fields, or null if fields are never quoted>",
  "escapechar": "<"\\\\" if quotes inside fields are written as \\", otherwise null>",
  "doublequote": <true if quotes inside fields are written as "", false if as \\">,
  "has_header": <true if a row holds the column names, false if the first row is already data>,
  "header_row_index": <0-based index of the column-name row, or null if has_header is false>,
  "footer_lines": ["<every footer line after the last data row, in file order>", "..."],
  "columns": [
    {{
      "name": "<name copied character for character from the header row; column_N if none>",
      "inferred_type": "<{_COLUMN_TYPE_CHOICES}>",
      "nullable": <true|false>,
      "example_values": ["<at most 3 raw values copied from this column>"]
    }}
  ],
  "confidence": <number between 0.0 and 1.0 indicating your confidence>,
  "notes": "<relevant observations, or null>"
}}

HEADER (start of the file): lines before the real column-name row, such as \
export banners, comments (e.g. starting with '#') or blank lines, are \
preamble. Do not list them anywhere; just count them: "header_row_index" is \
the 0-based index of the column-name row, i.e. the number of preamble lines. \
If the file has no column-name row at all (its first line is already a data \
record, e.g. "17,red,3.5"), answer "has_header": false, \
"header_row_index": null, and name the columns column_1, column_2, and so on.

FOOTER (end of the file): check {file_end} independently of the header. \
A data row holds a real record, with values like the rows above it (a date \
in the date column, a name in the name column, and so on). Any trailing \
line after the last data row is a footer line, for example:
- a totals/summary row: it may have the same number of fields as a data \
row, but it carries a label instead of a record and leaves other fields \
empty, e.g. "TOTAL,,4241.25", "TOTAL;;;98765.40" or "Total registros: 250"
- an end-of-report marker, e.g. "--- Fin del informe ---" or "*** END ***"
- a generation timestamp or signature, e.g. "Generado el 2024-01-20 10:00:00"
- a blank line separating the data from any of the above
Copy every footer line verbatim into "footer_lines" (a blank line is ""), \
from the first footer line to the last line of the file. Use [] only when \
the file really ends with a data row. Footer lines are never part of the \
header preamble.

Keep in mind:
- The delimiter may also appear inside quoted fields; do not confuse it with the real separator.
- Column names may contain accented characters and other special characters.
- Check how quotes are escaped inside quoted fields: doubled ("") or \
backslash-escaped (\\"); see "escapechar" and "doublequote" above.
- Rows may have an inconsistent number of fields; do not let that block your analysis.
"""


def _extract_json_payload(raw_response: str) -> str:
    """Extract a bare JSON object from a raw LLM response.

    Some models wrap their JSON output in a markdown code fence, or in prose
    such as ``Here is the result: {...}``, even when explicitly instructed
    not to. A fenced object is taken as is; otherwise the text from the
    first ``{`` to the last ``}`` is taken, which drops any surrounding
    prose.

    Args:
        raw_response: The raw text returned by the model.

    Returns:
        The JSON object's text, or the stripped response when it holds no
        ``{...}`` span, so that parsing reports the real content.
    """
    match = _JSON_FENCE_PATTERN.search(raw_response)
    if match:
        return match.group(1)
    text = raw_response.strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else text


def parse_and_validate(raw_response: str, model: str) -> CSVInspectionResult:
    """Parse a raw model response into a validated inspection result.

    Args:
        raw_response: The raw text returned by the model.
        model: Name of the model that produced the response, used only for
            diagnostics.

    Returns:
        A validated :class:`CSVInspectionResult`.

    Raises:
        ResponseParsingError: If the response is not valid JSON.
        SchemaValidationError: If the JSON does not match the expected schema.
    """
    payload = _extract_json_payload(raw_response)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ResponseParsingError(f"Model '{model}' returned invalid JSON: {exc}") from exc

    try:
        return CSVInspectionResult.model_validate(data)
    except ValidationError as exc:
        raise SchemaValidationError(
            f"Model '{model}' response did not match the expected schema: {exc}"
        ) from exc
