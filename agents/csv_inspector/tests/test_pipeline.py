"""Tests of ``inspect_csv`` end to end, with an injected fake model invoker.

The LLM backend is injected via ``inspect_csv``'s ``model_invoker``
parameter (or a fake ``ollama`` module), so the full pipeline, including
the fallback-model path and every domain error branch, runs without Ollama
or network access.
"""

from __future__ import annotations

import asyncio
import codecs
import csv
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from csv_inspector import (
    BackendConfigurationError,
    CSVInspectionResult,
    EmptySampleError,
    FileSampleReadError,
    InspectionFailedError,
    ModelInvocationError,
    Settings,
    ainspect_csv,
    inspect_csv,
)
from csv_inspector._config import DEFAULT_MODEL, FALLBACK_MODEL
from csv_inspector._invokers import ModelInvoker, _invoke_ollama
from csv_inspector._sampling import (
    MAX_SAMPLE_BYTES,
)
from fakes import install_fake_ollama
from payloads import _LEDGER, SAMPLE_CSV_PATH, VALID_RESULT_PAYLOAD, _sloppy_answer


def _tail_section(prompt: str) -> str:
    """Return the text between the tail sample markers of a built prompt."""
    start = prompt.index("--- TAIL SAMPLE START")
    end = prompt.index("--- TAIL SAMPLE END ---")
    return prompt[prompt.index("\n", start) + 1 : end]


def _capture_prompts(prompts: list[str]) -> ModelInvoker:
    """Build a fake model invoker that records prompts and returns a valid payload."""

    def fake_invoker(prompt: str, model: str) -> str:
        prompts.append(prompt)
        return json.dumps(VALID_RESULT_PAYLOAD)

    return fake_invoker


def test_a_prose_wrapped_reply_is_accepted_on_the_first_attempt() -> None:
    """A valid answer wrapped in prose no longer spends the fallback attempt."""
    models: list[str] = []

    def invoker(prompt: str, model: str) -> str:
        models.append(model)
        payload = json.dumps(VALID_RESULT_PAYLOAD)
        return f"Here is the inspection result:\n{payload}\nHope this helps!"

    result = inspect_csv(
        SAMPLE_CSV_PATH, model="primary", fallback_model="fallback", model_invoker=invoker
    )

    assert models == ["primary"]
    assert result.delimiter == ";"


def test_the_default_models_include_a_distinct_fallback() -> None:
    """With default settings, a failing primary is retried with the default fallback."""
    models: list[str] = []

    def invoker(prompt: str, model: str) -> str:
        models.append(model)
        if len(models) == 1:
            raise ModelInvocationError("primary unavailable")
        return json.dumps(VALID_RESULT_PAYLOAD)

    inspect_csv(SAMPLE_CSV_PATH, settings=Settings(), model_invoker=invoker)

    assert models == [DEFAULT_MODEL, FALLBACK_MODEL]


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


@pytest.mark.parametrize(
    ("n_bytes", "tail_bytes", "bad_name"),
    [
        (0, 64, "n_bytes"),
        (64, -1, "tail_bytes"),
        (MAX_SAMPLE_BYTES + 1, 64, "n_bytes"),
        (64, MAX_SAMPLE_BYTES + 1, "tail_bytes"),
    ],
)
def test_inspect_csv_validates_budgets_before_touching_the_file(
    tmp_path: Path, n_bytes: int, tail_bytes: int, bad_name: str
) -> None:
    """Invalid budgets fail fast, even before the (missing) file is opened."""
    with pytest.raises(ValueError, match=bad_name):
        inspect_csv(
            tmp_path / "missing.csv",
            n_bytes=n_bytes,
            tail_bytes=tail_bytes,
            model_invoker=_capture_prompts([]),
        )


def test_inspect_csv_raises_on_empty_file_without_invoking_model(tmp_path: Path) -> None:
    """An empty file must fail fast with a domain error and cost no LLM call."""
    target = tmp_path / "empty.csv"
    target.write_bytes(b"")
    prompts: list[str] = []

    with pytest.raises(EmptySampleError):
        inspect_csv(target, model_invoker=_capture_prompts(prompts))

    assert prompts == []


def test_inspect_csv_skips_tail_when_file_is_exactly_head_sized(tmp_path: Path) -> None:
    """A file of exactly ``n_bytes`` is fully covered by the head: no tail section."""
    target = tmp_path / "exact.csv"
    target.write_bytes(b"a,b,c\n" * 10)
    prompts: list[str] = []

    inspect_csv(target, n_bytes=60, tail_bytes=64, model_invoker=_capture_prompts(prompts))

    assert "TAIL SAMPLE START" not in prompts[0]


def test_inspect_csv_tail_never_overlaps_head(tmp_path: Path) -> None:
    """Only the bytes past the head window are sent as the tail sample."""
    target = tmp_path / "overlap.csv"
    target.write_bytes(b"H" * 64 + b"0123456789")
    prompts: list[str] = []

    inspect_csv(target, n_bytes=64, tail_bytes=64, model_invoker=_capture_prompts(prompts))

    assert _tail_section(prompts[0]).strip() == "0123456789"


def test_inspect_csv_zero_tail_bytes_disables_tail_sampling(tmp_path: Path) -> None:
    """``tail_bytes=0`` opts out of tail sampling entirely."""
    target = tmp_path / "large.csv"
    target.write_bytes(b"a,b,c\n" + b"1,2,3\n" * 100)
    prompts: list[str] = []

    inspect_csv(target, n_bytes=16, tail_bytes=0, model_invoker=_capture_prompts(prompts))

    assert "TAIL SAMPLE START" not in prompts[0]


@pytest.mark.parametrize(
    ("bom", "codec"),
    [(codecs.BOM_UTF16_LE, "utf-16-le"), (codecs.BOM_UTF16_BE, "utf-16-be")],
)
def test_inspect_csv_decodes_utf16_tail_with_odd_budget(
    tmp_path: Path, bom: bytes, codec: str
) -> None:
    """A UTF-16 tail must decode cleanly even for an odd budget and either byte order."""
    text = "a\tb\n" + "1\t2\n" * 200 + "TOTAL\t999\n"
    target = tmp_path / "utf16.csv"
    target.write_bytes(bom + text.encode(codec))
    prompts: list[str] = []

    inspect_csv(target, n_bytes=64, tail_bytes=33, model_invoker=_capture_prompts(prompts))

    tail = _tail_section(prompts[0])
    assert "TOTAL\t999" in tail
    assert "�" not in tail


def _install_fake_ollama(monkeypatch: pytest.MonkeyPatch, chat: Any) -> None:
    """Register a stand-in ``ollama`` module whose clients call ``chat``."""
    install_fake_ollama(monkeypatch, chat)


def test_invoke_ollama_model_returns_message_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model's message content is returned verbatim."""

    def chat(**kwargs: Any) -> Any:
        return SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))

    _install_fake_ollama(monkeypatch, chat)

    assert _invoke_ollama("prompt", "some-model").text == '{"ok": true}'


@pytest.mark.parametrize("content", [None, ""])
def test_invoke_ollama_model_rejects_empty_content(
    monkeypatch: pytest.MonkeyPatch, content: str | None
) -> None:
    """An empty or missing message is a backend failure, not an empty JSON payload."""

    def chat(**kwargs: Any) -> Any:
        return SimpleNamespace(message=SimpleNamespace(content=content))

    _install_fake_ollama(monkeypatch, chat)

    with pytest.raises(ModelInvocationError, match="empty response"):
        _invoke_ollama("prompt", "some-model")


def test_invoke_ollama_model_wraps_backend_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any client-side failure is surfaced as the domain ``ModelInvocationError``."""

    def chat(**kwargs: Any) -> Any:
        raise ConnectionError("connection refused")

    _install_fake_ollama(monkeypatch, chat)

    with pytest.raises(ModelInvocationError, match="connection refused"):
        _invoke_ollama("prompt", "some-model")


def test_invoke_ollama_model_reports_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing ``ollama`` package yields an actionable domain error."""
    monkeypatch.setitem(sys.modules, "ollama", None)

    with pytest.raises(BackendConfigurationError, match="pip install ollama"):
        _invoke_ollama("prompt", "some-model")


def test_a_conflicting_dialect_moves_on_to_the_fallback_model(tmp_path: Path) -> None:
    """A dialect csv rejects is a schema error, not a crash in grounding (issue #48)."""
    target = tmp_path / "ledger.csv"
    target.write_text(_LEDGER, encoding="utf-8")

    def invoker(prompt: str, model: str) -> str:
        quotechar = ";" if model == "primary" else '"'
        return _sloppy_answer(quotechar=quotechar)(prompt, model)

    result = inspect_csv(target, model="primary", fallback_model="fallback", model_invoker=invoker)

    assert (result.delimiter, result.quotechar) == (";", '"')
    csv.reader(
        [],
        delimiter=result.delimiter,
        quotechar=result.quotechar,
        escapechar=result.escapechar,
        doublequote=result.doublequote,
    )


def test_a_malformed_delimiter_moves_on_to_the_fallback_model(tmp_path: Path) -> None:
    """A multi-character delimiter is a schema error, so the fallback model runs (issue #11)."""
    target = tmp_path / "ledger.csv"
    target.write_text(_LEDGER, encoding="utf-8")
    calls: list[str] = []

    def invoker(prompt: str, model: str) -> str:
        calls.append(model)
        delimiter = "semicolon" if model == "primary" else ";"
        return _sloppy_answer(delimiter=delimiter)(prompt, model)

    result = inspect_csv(target, model="primary", fallback_model="fallback", model_invoker=invoker)

    assert calls == ["primary", "fallback"]
    assert result.delimiter == ";"


def test_a_failed_attempt_log_never_shows_the_configured_api_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A custom invoker's error carrying the key is logged redacted (issue #81)."""
    secret = "AIza-test-secret-value"

    def invoker(prompt: str, model: str) -> str:
        raise PermissionError(f"API key {secret} was rejected")

    settings = Settings(gemini_api_key=SecretStr(secret))
    with (
        caplog.at_level(logging.DEBUG, logger="csv_inspector"),
        pytest.raises(InspectionFailedError),
    ):
        inspect_csv(SAMPLE_CSV_PATH, settings=settings, model_invoker=invoker)

    failures = [r.getMessage() for r in caplog.records if "failed:" in r.getMessage()]
    assert len(failures) == 2
    assert all("API key *** was rejected" in message for message in failures)
    assert secret not in caplog.text


def test_a_failed_async_attempt_log_never_shows_the_configured_api_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The async entry point shares the redaction (issue #81)."""
    secret = "AIza-test-secret-value"

    async def invoker(prompt: str, model: str) -> str:
        raise PermissionError(f"API key {secret} was rejected")

    settings = Settings(gemini_api_key=SecretStr(secret))
    with (
        caplog.at_level(logging.DEBUG, logger="csv_inspector"),
        pytest.raises(InspectionFailedError),
    ):
        asyncio.run(ainspect_csv(SAMPLE_CSV_PATH, settings=settings, model_invoker=invoker))

    assert "API key *** was rejected" in caplog.text
    assert secret not in caplog.text
