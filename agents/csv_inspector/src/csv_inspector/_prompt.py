"""Prompt construction and response parsing for csv_inspector.

Known limitation: the samples are delimited by plain-text marker lines
(``--- HEAD SAMPLE START ... ---``). A file that contains such a line, or
text mimicking the instructions, can confuse the model. This is accepted on
purpose: grounding recomputes the delimiter, header row, column names,
footer and encoding from the real bytes, so the model's answer is only a
key, and random markers or escaping would cost tokens on every call.
"""

from __future__ import annotations

import functools
import json
import re
from typing import Any, cast

from pydantic import ValidationError

from ._exceptions import ResponseParsingError, SchemaValidationError
from ._models import _ModelAnswer

# Bumped by hand on any change to the prompt wording (date-based; the suffix
# tells several bumps in one month apart). Recorded in every Usage and eval
# run, so measurements of different prompts are never mixed; see
# docs/evaluation.md "Changing the prompt".
PROMPT_VERSION = "2026.09-h"

SYSTEM_PROMPT = "You always respond with valid JSON, with no explanations or markdown."

_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)

# Annotation keywords dropped from the response schema: they are prose for
# humans, and Ollama compiles the schema into a grammar, so every key costs.
_SCHEMA_ANNOTATIONS = frozenset({"title", "description", "default"})


def _strip_annotations(node: object) -> object:
    """Return a JSON Schema fragment without its annotation keywords."""
    if isinstance(node, list):
        return [_strip_annotations(item) for item in node]
    if not isinstance(node, dict):
        return node
    return {
        # Property names are data, not keywords: keep every one.
        key: (
            {name: _strip_annotations(sub) for name, sub in value.items()}
            if key == "properties"
            else _strip_annotations(value)
        )
        for key, value in node.items()
        if key not in _SCHEMA_ANNOTATIONS
    }


@functools.lru_cache(maxsize=1)
def response_schema() -> dict[str, Any]:
    """The JSON Schema both backends send to constrain the model's answer.

    Built from the JSON Schema of the model's answer (``_ModelAnswer``:
    ``footer_first_line``, not the result's ``footer_lines``), without ``title``,
    ``description`` and ``default`` keys, so it is small enough for Ollama to
    compile into a grammar. The answer is flat (no nested models, so no
    ``$defs`` to inline). Numeric bounds
    stay. Every property is required, nullable ones included: a grammar
    lets the model skip an optional key, and a skipped ``quotechar`` or
    ``escapechar`` silently becomes its default. The schema is the contract
    of the answer's shape; the prompt only carries what the schema cannot
    say (see ``build_prompt``). Cached: treat the returned dict as
    read-only.

    Returns:
        The JSON Schema, as a dict.
    """
    schema = _ModelAnswer.model_json_schema()
    if "$defs" in schema:  # pragma: no cover - guards a future nested field
        raise TypeError("_ModelAnswer must stay flat: Ollama gets no $defs to resolve")
    stripped = cast("dict[str, Any]", _strip_annotations(schema))
    stripped["required"] = list(stripped["properties"])
    return stripped


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
        JSON object matching :func:`response_schema`. The prompt names no
        JSON shape: the schema sent with the request is the contract, and
        the prompt carries only what the schema cannot say (what counts as
        preamble or footer, what to copy verbatim).
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
        file_end = 'the end of the file (not sampled here, so "footer_first_line" must be null)'
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
Analyze the samples and answer with a JSON object matching the schema you \
were given. What its fields mean:
- "encoding" is the real encoding, e.g. utf-8, latin-1, cp1252.
- "quotechar" is the character that wraps quoted fields: "'" when fields \
look like 'Acme, S.L.', '"' when they look like "Acme, S.L." or are never quoted.
- "escapechar" and "doublequote": a quote inside a quoted field written \
with a backslash (\\") means "escapechar": "\\\\", "doublequote": false; \
written doubled ("") or never present, "escapechar": null, "doublequote": true.
- "delimiter" is the real separator: it may also appear inside quoted \
fields, and rows may have uneven field counts.
- "columns" holds each name copied character for character from the header row.

HEADER:
- Preamble lines (export banners, '#' comments, blank lines) come before \
the column-name row: "header_row_index" is their count (0-based index of \
that row). Never list them; footer lines are never preamble.
- No column-name row (the first line is already data, e.g. "17,red,3.5"): \
"has_header": false, "header_row_index": null, columns column_1, column_2, ...

FOOTER: in {file_end}, every line after the last data row (a data row \
holds a real record, like the rows above it):
- a totals row: a label instead of a record, other fields empty, e.g. \
"TOTAL,,4241.25" or "Total registros: 250"
- an end marker, e.g. "--- Fin del informe ---"; a timestamp, e.g. \
"Generado el 2024-01-20 10:00:00"; a blank line before them
"footer_first_line": the first non-blank footer line, verbatim; null only \
when the file ends with a data row.
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


def parse_and_validate(raw_response: str, model: str) -> _ModelAnswer:
    """Parse a raw model response into a validated model answer.

    Built-in and custom invokers alike go through here; grounding then
    turns the answer into the public result.

    Args:
        raw_response: The raw text returned by the model.
        model: Name of the model that produced the response, used only for
            diagnostics.

    Returns:
        The validated answer, still to be grounded in the samples.

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
        return _ModelAnswer.model_validate(data)
    except ValidationError as exc:
        raise SchemaValidationError(
            f"Model '{model}' response did not match the expected schema: {exc}"
        ) from exc
