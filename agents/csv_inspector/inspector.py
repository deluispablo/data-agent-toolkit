"""Core inspection pipeline for the csv_inspector agent.

This module implements a stateless pipeline that:

1. Reads only the first ``n_bytes`` (head) and last ``tail_bytes`` (tail) of
   a potentially massive delimited file, via bounded, seek-based reads that
   never load the full file into memory.
2. Heuristically pre-detects the character encoding with ``chardet``.
3. Builds a concise prompt describing both samples.
4. Invokes a local LLM through Ollama (with an optional fallback model) to
   infer the file's dialect (delimiter, quoting, escaping), non-standard
   header/footer lines, and a preliminary column schema.
5. Validates the model's JSON response against
   :class:`~agents.csv_inspector.models.CSVInspectionResult`.

When the head sample already exhausts the file (i.e. the file is smaller
than ``n_bytes``), the tail sample is skipped entirely: it would only
duplicate content already visible to the model and would waste tokens,
which conflicts with this project's cost-optimization-first principle.

The LLM backend is pluggable via the ``model_invoker`` parameter, which
makes the whole flow unit-testable without a running Ollama instance.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path

import chardet
from pydantic import ValidationError

from exceptions import (
    FileSampleReadError,
    InspectionFailedError,
    ModelInvocationError,
    ResponseParsingError,
    SchemaValidationError,
)
from models import CSVInspectionResult

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_MODEL: str = "qwen2.5-coder:7b"
FALLBACK_MODEL: str = "qwen2.5-coder:7b"
DEFAULT_SAMPLE_BYTES: int = 4096
DEFAULT_TAIL_BYTES: int = 4096

_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)

ModelInvoker = Callable[[str, str], str]
"""A callable that sends ``prompt`` to ``model`` and returns the raw response text."""


def read_sample_bytes(path: str | Path, n_bytes: int = DEFAULT_SAMPLE_BYTES) -> bytes:
    """Read only the first ``n_bytes`` of a file without loading it fully into memory.

    Args:
        path: Path to the source file.
        n_bytes: Maximum number of bytes to read from the start of the file.

    Returns:
        The raw bytes read from the file. Shorter than ``n_bytes`` only when
        the file itself is smaller than ``n_bytes``.

    Raises:
        FileSampleReadError: If the file does not exist or cannot be read.
    """
    try:
        with open(path, "rb") as handle:
            return handle.read(n_bytes)
    except OSError as exc:
        raise FileSampleReadError(f"Unable to read head sample from '{path}': {exc}") from exc


def read_tail_bytes(path: str | Path, n_bytes: int = DEFAULT_TAIL_BYTES) -> bytes:
    """Read only the last ``n_bytes`` of a file without loading it fully into memory.

    Uses a bounded seek from the end of the file (``os.SEEK_END``) followed
    by a single bounded read, so the cost is independent of the file's total
    size: no data before the tail window is ever touched.

    Args:
        path: Path to the source file.
        n_bytes: Maximum number of trailing bytes to read.

    Returns:
        The raw trailing bytes. Shorter than ``n_bytes`` only when the file
        itself is smaller than ``n_bytes``; empty for a zero-byte file.
        These bytes are a blind suffix of the file and may begin mid-line
        (or mid-character, for multi-byte encodings) rather than at a clean
        row boundary.

    Raises:
        FileSampleReadError: If the file does not exist or cannot be read.
    """
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            file_size = handle.tell()
            read_size = min(n_bytes, file_size)
            handle.seek(-read_size, os.SEEK_END)
            return handle.read(read_size)
    except OSError as exc:
        raise FileSampleReadError(f"Unable to read tail sample from '{path}': {exc}") from exc


def detect_encoding(raw_bytes: bytes) -> str:
    """Heuristically detect the character encoding of a byte sample.

    Args:
        raw_bytes: The raw byte sample to analyze.

    Returns:
        The detected encoding name, defaulting to ``"utf-8"`` when detection
        is inconclusive or reports plain ASCII.
    """
    detection = chardet.detect(raw_bytes)
    encoding = detection.get("encoding") or "utf-8"
    if encoding.lower() == "ascii":
        encoding = "utf-8"
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


def build_prompt(
    head_sample: str,
    detected_encoding: str,
    tail_sample: str | None = None,
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

    Returns:
        A complete prompt instructing the model to respond with a single
        JSON object matching the ``CSVInspectionResult`` schema.
    """
    if tail_sample is not None:
        tail_section = f"""
--- TAIL SAMPLE START (last bytes of the file; may start mid-line or mid-word) ---
{tail_sample}
--- TAIL SAMPLE END ---

Use the tail sample only to detect trailing footer content (summary/total \
rows, "end of report" markers, trailing blank lines). Its first visible \
line is very likely a truncated fragment, not a real row: do not use it to \
infer columns.
"""
        footer_instruction = (
            '"footer_lines": ["<trailing footer line 1, from the tail sample>", "..."],\n'
            '  "footer_rows_to_skip": <number of trailing footer rows to ignore, usually 0>,'
        )
    else:
        tail_section = "\n(The file is fully contained in the sample above; there is no separate tail.)\n"
        footer_instruction = (
            '"footer_lines": ["<trailing footer line 1, if any, from the sample above>", "..."],\n'
            '  "footer_rows_to_skip": <number of trailing footer rows to ignore, usually 0>,'
        )

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
  "quotechar": "<character used to quote fields>",
  "escapechar": "<escape character if any, or null>",
  "doublequote": <true|false, whether embedded quotes are escaped by doubling>,
  "header_row_index": <0-based index of the row containing the real column names>,
  "metadata_lines": ["<metadata/comment line 1 preceding the header>", "..."],
  {footer_instruction}
  "columns": [
    {{
      "name": "<column name>",
      "inferred_type": "<string|integer|float|date|boolean>",
      "nullable": <true|false>,
      "example_values": ["<example value 1>", "<example value 2>"]
    }}
  ],
  "confidence": <number between 0.0 and 1.0 indicating your confidence>,
  "notes": "<relevant observations, or null>"
}}

Keep in mind:
- There may be metadata or comment lines (e.g. starting with '#') before the real header.
- The delimiter may also appear inside quoted fields; do not confuse it with the real separator.
- Column names may contain accented characters and other special characters.
- Pay attention to escaped double quotes (e.g. "" inside a quoted field).
- Rows may have an inconsistent number of fields; do not let that block your analysis.
"""


def _extract_json_payload(raw_response: str) -> str:
    """Extract a bare JSON object from a raw LLM response.

    Some models wrap their JSON output in a markdown code fence even when
    explicitly instructed not to. This strips that fence when present.

    Args:
        raw_response: The raw text returned by the model.

    Returns:
        The response text with any surrounding markdown code fence removed.
    """
    match = _JSON_FENCE_PATTERN.search(raw_response)
    return match.group(1) if match else raw_response.strip()


def invoke_ollama_model(prompt: str, model: str) -> str:
    """Send a prompt to a local Ollama model and return its raw text response.

    Args:
        prompt: The fully-built prompt to send.
        model: Name of the Ollama model to invoke (e.g. ``"qwen2.5-coder:7b"``).

    Returns:
        The raw text content of the model's response.

    Raises:
        ModelInvocationError: If the ``ollama`` package is not installed, the
            backend cannot be reached, or the model is not available locally.
    """
    try:
        import ollama  # noqa: PLC0415 - lazily imported: only this backend needs it.
    except ImportError as exc:
        raise ModelInvocationError(
            "The 'ollama' package is required to use the local Ollama backend. "
            "Install it with 'pip install ollama' or inject a custom model_invoker."
        ) from exc

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You always respond with valid JSON, with no explanations or markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={"temperature": 0.0},
        )
    except Exception as exc:  # noqa: BLE001 - the Ollama client raises varied backend errors.
        raise ModelInvocationError(f"Model '{model}' failed to respond: {exc}") from exc
    return response["message"]["content"]


def _parse_and_validate(raw_response: str, model: str) -> CSVInspectionResult:
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


def inspect_csv(
    path: str | Path,
    model: str = DEFAULT_MODEL,
    n_bytes: int = DEFAULT_SAMPLE_BYTES,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
    fallback_model: str = FALLBACK_MODEL,
    model_invoker: ModelInvoker = invoke_ollama_model,
) -> CSVInspectionResult:
    """Inspect a delimited file fragment and infer its dialect and schema.

    Reads only the first ``n_bytes`` (head) and, when the file is larger
    than that, the last ``tail_bytes`` (tail) of ``path``. Asks a local LLM
    (via ``model_invoker``) to infer the file's encoding, delimiter, quoting
    rules, non-standard header/footer lines, and a preliminary column
    schema, and returns the result validated with Pydantic.

    Args:
        path: Path to the source CSV/TSV file.
        model: Primary Ollama model name to use.
        n_bytes: Number of bytes to sample from the start of the file.
        tail_bytes: Number of bytes to sample from the end of the file. If
            the head sample already exhausts the file, no separate tail
            read is performed.
        fallback_model: Secondary model to try if ``model`` fails. Skipped
            automatically when equal to ``model``.
        model_invoker: Callable used to invoke the LLM; defaults to
            :func:`invoke_ollama_model`. Injectable for testing.

    Returns:
        A validated :class:`CSVInspectionResult`.

    Raises:
        FileSampleReadError: If the source file cannot be read.
        InspectionFailedError: If every configured model fails to produce a
            valid, schema-conformant result.
    """
    head_bytes = read_sample_bytes(path, n_bytes)
    detected_encoding = detect_encoding(head_bytes)
    head_sample = decode_sample(head_bytes, detected_encoding)

    file_fits_in_head = len(head_bytes) < n_bytes
    tail_sample: str | None = None
    if not file_fits_in_head:
        tail_raw_bytes = read_tail_bytes(path, tail_bytes)
        tail_sample = decode_sample(tail_raw_bytes, detected_encoding)

    prompt = build_prompt(head_sample, detected_encoding, tail_sample=tail_sample)

    candidate_models = [model] if model == fallback_model else [model, fallback_model]
    attempts: dict[str, Exception] = {}

    for candidate in candidate_models:
        logger.info("Inspecting '%s' with model '%s'.", path, candidate)
        try:
            raw_response = model_invoker(prompt, candidate)
            result = _parse_and_validate(raw_response, candidate)
        except (ModelInvocationError, ResponseParsingError, SchemaValidationError) as exc:
            logger.warning("Model '%s' failed: %s", candidate, exc)
            attempts[candidate] = exc
            continue
        logger.info(
            "Inspection succeeded with model '%s' (confidence=%.2f).",
            candidate,
            result.confidence,
        )
        return result

    raise InspectionFailedError(
        f"No configured model produced a valid inspection result for '{path}'. "
        f"Tried: {list(attempts)}.",
        attempts=attempts,
    )
