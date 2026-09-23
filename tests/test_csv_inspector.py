"""Unit tests for the csv_inspector agent.

The LLM backend is injected via ``inspect_csv``'s ``model_invoker``
parameter, so the full pipeline — including the fallback-model path and all
domain error branches — is exercised without requiring a running Ollama
instance or network access.

The byte-sampling layer (``read_sample_bytes`` / ``read_tail_bytes``) is
tested not only for correctness but also, via an ``open()`` spy, for *how*
it reads: these functions must never fall back to loading a whole file into
memory, since that is the entire point of sampling head/tail byte windows
against multi-gigabyte production files.
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
    read_tail_bytes,
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
    "footer_lines": [],
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


def _spy_on_open(monkeypatch: pytest.MonkeyPatch) -> list[int | None]:
    """Patch the ``open`` builtin to record every size passed to ``.read()``.

    Args:
        monkeypatch: The pytest monkeypatch fixture for the current test.

    Returns:
        A list that will be populated, in call order, with the ``size``
        argument of every ``.read()`` call made through ``open()`` while the
        patch is active. A ``None`` or negative entry would indicate an
        unbounded (whole-file) read.
    """
    read_calls: list[int | None] = []
    real_open = open

    def spy_open(*args: object, **kwargs: object):
        handle = real_open(*args, **kwargs)  # type: ignore[arg-type]
        original_read = handle.read

        def traced_read(size: int = -1) -> bytes:
            read_calls.append(size)
            return original_read(size)

        handle.read = traced_read  # type: ignore[method-assign]
        return handle

    monkeypatch.setattr("builtins.open", spy_open)
    return read_calls


# ---------------------------------------------------------------------
# read_sample_bytes (head)
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


def test_read_sample_bytes_empty_file_returns_empty_bytes(tmp_path: Path) -> None:
    """An empty file should yield an empty sample, not an error."""
    target = tmp_path / "empty.csv"
    target.write_bytes(b"")

    assert read_sample_bytes(target, n_bytes=4096) == b""


def test_read_sample_bytes_never_reads_the_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sampling a large file must issue exactly one bounded ``.read()`` call.

    This is the production-critical guarantee behind byte-budgeted
    inspection: a multi-gigabyte file must never be pulled fully into
    memory just to look at its first few kilobytes.
    """
    marker = b"HEAD_MARKER_0123456789"
    target = tmp_path / "big_head.csv"
    target.write_bytes(marker + b"x" * 5_000_000)

    read_calls = _spy_on_open(monkeypatch)

    head = read_sample_bytes(target, n_bytes=len(marker))

    assert head == marker
    assert read_calls == [len(marker)]


# ---------------------------------------------------------------------
# read_tail_bytes
# ---------------------------------------------------------------------


def test_read_tail_bytes_respects_byte_limit(tmp_path: Path) -> None:
    """Only the requested number of trailing bytes should be read."""
    target = tmp_path / "large.csv"
    target.write_bytes(b"a,b,c\n" * 1000)

    tail = read_tail_bytes(target, n_bytes=10)

    assert tail == (b"a,b,c\n" * 1000)[-10:]
    assert len(tail) == 10


def test_read_tail_bytes_smaller_than_limit_returns_whole_file(tmp_path: Path) -> None:
    """A file smaller than ``n_bytes`` should be returned in full."""
    target = tmp_path / "small.csv"
    target.write_bytes(b"a,b,c\n1,2,3\n")

    tail = read_tail_bytes(target, n_bytes=4096)

    assert tail == b"a,b,c\n1,2,3\n"


def test_read_tail_bytes_exact_file_size(tmp_path: Path) -> None:
    """A file exactly ``n_bytes`` long should be returned in full."""
    content = b"0123456789"
    target = tmp_path / "exact.csv"
    target.write_bytes(content)

    assert read_tail_bytes(target, n_bytes=len(content)) == content


def test_read_tail_bytes_empty_file_returns_empty_bytes(tmp_path: Path) -> None:
    """An empty file should yield an empty tail sample, not an error."""
    target = tmp_path / "empty.csv"
    target.write_bytes(b"")

    assert read_tail_bytes(target, n_bytes=4096) == b""


def test_read_tail_bytes_missing_file_raises_domain_error(tmp_path: Path) -> None:
    """A missing source file should raise the domain-specific exception."""
    missing = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileSampleReadError):
        read_tail_bytes(missing)


def test_read_tail_bytes_never_reads_the_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sampling the tail of a large file must never read from the start.

    A naive implementation might read the whole file and slice the last
    bytes in memory; this proves the implementation instead performs a
    single bounded read anchored at the end of the file.
    """
    marker = b"TAIL_MARKER_0123456789"
    target = tmp_path / "big_tail.csv"
    target.write_bytes(b"x" * 5_000_000 + marker)

    read_calls = _spy_on_open(monkeypatch)

    tail = read_tail_bytes(target, n_bytes=len(marker))

    assert tail == marker
    assert read_calls == [len(marker)]


def test_read_tail_bytes_on_real_sample_fixture_ends_with_last_row() -> None:
    """The tail of the bundled sample.csv should contain its final row."""
    tail = read_tail_bytes(SAMPLE_CSV_PATH, n_bytes=200)

    assert b"traslado" in tail


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


def test_build_prompt_embeds_head_sample_and_encoding_hint() -> None:
    """The generated prompt must include the head sample and encoding hint."""
    prompt = build_prompt(head_sample="Fecha;Cliente\n2024-01-15;Acme", detected_encoding="utf-8")

    assert "Fecha;Cliente" in prompt
    assert "'utf-8'" in prompt


def test_build_prompt_omits_tail_section_when_tail_sample_is_none() -> None:
    """No tail section should appear in the prompt when there is no tail sample."""
    prompt = build_prompt(head_sample="a,b,c\n1,2,3\n", detected_encoding="utf-8", tail_sample=None)

    assert "TAIL SAMPLE START" not in prompt
    assert "fully contained in the sample above" in prompt


def test_build_prompt_includes_tail_section_and_mid_line_caveat() -> None:
    """A provided tail sample must appear in its own labeled section with a caveat."""
    prompt = build_prompt(
        head_sample="a,b,c\n1,2,3\n",
        detected_encoding="utf-8",
        tail_sample=",99\nTOTAL,,999\n",
    )

    assert "TAIL SAMPLE START" in prompt
    assert ",99\nTOTAL,,999\n" in prompt
    assert "may start mid-line" in prompt


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


def test_inspect_csv_skips_tail_read_when_file_fits_in_head(tmp_path: Path) -> None:
    """A file smaller than n_bytes should not trigger a separate tail read."""
    target = tmp_path / "small.csv"
    target.write_bytes(b"a,b,c\n1,2,3\n")
    seen_prompts: list[str] = []

    def fake_invoker(prompt: str, model: str) -> str:
        seen_prompts.append(prompt)
        return json.dumps(VALID_RESULT_PAYLOAD)

    inspect_csv(target, n_bytes=4096, tail_bytes=4096, model_invoker=fake_invoker)

    assert "TAIL SAMPLE START" not in seen_prompts[0]


def test_inspect_csv_includes_tail_sample_for_files_larger_than_head(tmp_path: Path) -> None:
    """A file larger than n_bytes should trigger a separate, labeled tail read."""
    target = tmp_path / "large.csv"
    target.write_bytes(b"a,b,c\n" + b"1,2,3\n" * 2000 + b"TOTAL,,999\n")
    seen_prompts: list[str] = []

    def fake_invoker(prompt: str, model: str) -> str:
        seen_prompts.append(prompt)
        return json.dumps(VALID_RESULT_PAYLOAD)

    inspect_csv(target, n_bytes=64, tail_bytes=64, model_invoker=fake_invoker)

    assert "TAIL SAMPLE START" in seen_prompts[0]
    assert "TOTAL,,999" in seen_prompts[0]
