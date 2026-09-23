"""Unit tests for the csv_inspector agent.

The LLM backend is injected via ``inspect_csv``'s ``model_invoker``
parameter, so the full pipeline — including the fallback-model path and all
domain error branches — is exercised without requiring a running Ollama
instance or network access.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from exceptions import (
    FileSampleReadError,
    InspectionFailedError,
)
from inspector import (
    build_prompt,
    decode_sample,
    detect_encoding,
    inspect_csv,
    read_sample_bytes,
)
from inspector import _extract_json_payload  # noqa: PLC2701 - white-box unit test.
from models import CSVInspectionResult

SAMPLE_CSV_PATH = Path(__file__).resolve().parent.parent / "agents" / "csv_inspector" / "sample.csv"

VALID_RESULT_PAYLOAD: dict[str, object] = {
    "encoding": "utf-8",
    "delimiter": ";",
    "quotechar": '"',
    "escapechar": None,
    "doublequote": True,
    "header_row_index": 2,
    "metadata_lines": [
        "# Exportado desde SistemaXYZ v3.2",
        "# Fecha de generación: 2024-01-15",
    ],
    "footer_rows_to_skip": 0,
    "columns": [
        {
            "name": "Fecha",
            "inferred_type": "date",
            "nullable": False,
            "example_values": ["2024-01-15", "2024-01-16"],
        },
        {
            "name": "Importe",
            "inferred_type": "float",
            "nullable": False,
            "example_values": ["1250.50", "890.00"],
        },
    ],
    "confidence": 0.95,
    "notes": None,
}


# ---------------------------------------------------------------------
# read_sample_bytes
# ---------------------------------------------------------------------


def test_read_sample_bytes_respects_byte_limit(tmp_path: Path) -> None:
    """Only the requested number of leading bytes should be read."""
    target = tmp_path / "large.csv"
    target.write_bytes(b"a,b,c\n" * 1000)

    sample = read_sample_bytes(target, n_bytes=10)

    assert sample == b"a,b,c\na,b,"
    assert len(sample) == 10


def test_read_sample_bytes_missing_file_raises_domain_error(tmp_path: Path) -> None:
    """A missing source file should raise the domain-specific exception."""
    missing = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileSampleReadError):
        read_sample_bytes(missing)


def test_read_sample_bytes_reads_real_sample_fixture() -> None:
    """The bundled sample.csv fixture should be readable end-to-end."""
    sample = read_sample_bytes(SAMPLE_CSV_PATH, n_bytes=4096)

    assert b"Fecha;Cliente" in sample


# ---------------------------------------------------------------------
# encoding detection and decoding
# ---------------------------------------------------------------------


def test_detect_encoding_defaults_to_utf8_for_ascii_bytes() -> None:
    """Pure ASCII input should resolve to utf-8 rather than a narrow alias."""
    encoding = detect_encoding(b"col_a,col_b,col_c\n1,2,3\n")

    assert encoding.lower() == "utf-8"


def test_decode_sample_replaces_undecodable_bytes_on_unknown_encoding() -> None:
    """An unrecognized encoding name should fall back to utf-8 with replacement."""
    text = decode_sample(b"a;b;c\n", encoding="not-a-real-encoding")

    assert "a;b;c" in text


# ---------------------------------------------------------------------
# prompt construction and response parsing helpers
# ---------------------------------------------------------------------


def test_build_prompt_embeds_sample_and_encoding_hint() -> None:
    """The generated prompt must include the sample text and encoding hint."""
    prompt = build_prompt(sample_text="Fecha;Cliente\n2024-01-15;Acme", detected_encoding="utf-8")

    assert "Fecha;Cliente" in prompt
    assert "'utf-8'" in prompt


def test_extract_json_payload_strips_markdown_fence() -> None:
    """A JSON payload wrapped in a markdown code fence should be unwrapped."""
    fenced = '```json\n{"a": 1}\n```'

    assert _extract_json_payload(fenced) == '{"a": 1}'


def test_extract_json_payload_passes_through_bare_json() -> None:
    """A bare JSON payload with no fence should be returned unchanged (trimmed)."""
    bare = '  {"a": 1}  '

    assert _extract_json_payload(bare) == '{"a": 1}'


# ---------------------------------------------------------------------
# inspect_csv orchestration (with an injected fake model invoker)
# ---------------------------------------------------------------------


def test_inspect_csv_returns_validated_result_on_first_model_success() -> None:
    """A well-formed first-model response should short-circuit the fallback."""
    calls: list[str] = []

    def fake_invoker(prompt: str, model: str) -> str:
        calls.append(model)
        return json.dumps(VALID_RESULT_PAYLOAD)

    result = inspect_csv(
        SAMPLE_CSV_PATH,
        model="primary-model",
        fallback_model="fallback-model",
        model_invoker=fake_invoker,
    )

    assert isinstance(result, CSVInspectionResult)
    assert result.delimiter == ";"
    assert result.header_row_index == 2
    assert calls == ["primary-model"]


def test_inspect_csv_falls_back_to_secondary_model_on_primary_failure() -> None:
    """A primary-model failure should be retried against the fallback model."""
    calls: list[str] = []

    def fake_invoker(prompt: str, model: str) -> str:
        calls.append(model)
        if model == "primary-model":
            return "not valid json"
        return json.dumps(VALID_RESULT_PAYLOAD)

    result = inspect_csv(
        SAMPLE_CSV_PATH,
        model="primary-model",
        fallback_model="fallback-model",
        model_invoker=fake_invoker,
    )

    assert result.confidence == pytest.approx(0.95)
    assert calls == ["primary-model", "fallback-model"]


def test_inspect_csv_raises_when_every_model_fails() -> None:
    """If all configured models fail, an aggregated domain error is raised."""

    def failing_invoker(prompt: str, model: str) -> str:
        return "not valid json"

    with pytest.raises(InspectionFailedError) as exc_info:
        inspect_csv(
            SAMPLE_CSV_PATH,
            model="primary-model",
            fallback_model="fallback-model",
            model_invoker=failing_invoker,
        )

    assert set(exc_info.value.attempts) == {"primary-model", "fallback-model"}


def test_inspect_csv_does_not_duplicate_identical_primary_and_fallback() -> None:
    """When model equals fallback_model, the invoker should only run once."""
    calls: list[str] = []

    def fake_invoker(prompt: str, model: str) -> str:
        calls.append(model)
        return json.dumps(VALID_RESULT_PAYLOAD)

    inspect_csv(
        SAMPLE_CSV_PATH,
        model="only-model",
        fallback_model="only-model",
        model_invoker=fake_invoker,
    )

    assert calls == ["only-model"]


def test_inspect_csv_propagates_file_read_errors_before_invoking_model(tmp_path: Path) -> None:
    """A missing source file should fail fast, without calling the model."""
    missing = tmp_path / "missing.csv"
    invoked = False

    def fake_invoker(prompt: str, model: str) -> str:
        nonlocal invoked
        invoked = True
        return json.dumps(VALID_RESULT_PAYLOAD)

    with pytest.raises(FileSampleReadError):
        inspect_csv(missing, model_invoker=fake_invoker)

    assert invoked is False
