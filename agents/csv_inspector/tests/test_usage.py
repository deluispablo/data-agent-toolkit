"""``result.usage``: tokens, latency, attempts, retries and prompt version (issues #121, #127)."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from csv_inspector import CSVInspectionResult, Settings, ainspect_csv, inspect_csv
from csv_inspector._prompt import PROMPT_VERSION
from csv_inspector.cli import main
from fakes import install_fake_ollama, ollama_reply

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "sample.csv"
ANSWER = json.dumps(
    {
        "encoding": "utf-8",
        "delimiter": ";",
        "header_row_index": 2,
        "columns": [{"name": "Fecha", "inferred_type": "date"}],
        "confidence": 0.9,
    }
)


def _inspect(use_async: bool, **kwargs: Any) -> CSVInspectionResult:
    """Run the sync or the async entry point with the same arguments."""
    if use_async:
        return asyncio.run(ainspect_csv(SAMPLE_CSV, **kwargs))
    return inspect_csv(SAMPLE_CSV, **kwargs)


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_ollama_counters_reach_the_usage(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, use_async: bool
) -> None:
    """prompt_eval_count, eval_count and load_duration (ns) end up in result.usage."""
    reply = ollama_reply(ANSWER, prompt_eval_count=900, eval_count=150, load_duration=2_500_000_000)
    install_fake_ollama(monkeypatch, lambda **kwargs: reply)

    with caplog.at_level(logging.INFO, logger="csv_inspector"):
        result = _inspect(use_async, settings=Settings(), model="big", fallback_model="small")

    usage = result.usage
    assert usage is not None
    assert (usage.model, usage.prompt_tokens, usage.completion_tokens) == ("big", 900, 150)
    assert (usage.attempts, usage.retries) == (1, 0)
    assert usage.load_seconds == pytest.approx(2.5)
    assert usage.latency_seconds >= 0
    assert "model=big prompt_tokens=900 completion_tokens=150" in caplog.text
    assert usage.prompt_version == PROMPT_VERSION
    assert f"prompt_version={PROMPT_VERSION}" in caplog.text


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_a_failed_primary_costs_tokens_too(
    monkeypatch: pytest.MonkeyPatch, use_async: bool
) -> None:
    """The primary's unparseable answer is summed into the fallback's usage."""

    def chat(**kwargs: Any) -> Any:
        if kwargs["model"] == "big":
            return ollama_reply("not json", prompt_eval_count=900, eval_count=40)
        return ollama_reply(ANSWER, prompt_eval_count=800, eval_count=150)

    install_fake_ollama(monkeypatch, chat)

    usage = _inspect(use_async, settings=Settings(), model="big", fallback_model="small").usage

    assert usage is not None
    assert (usage.model, usage.attempts) == ("small", 2)
    assert (usage.prompt_tokens, usage.completion_tokens) == (1700, 190)
    assert usage.load_seconds is None


def test_missing_counters_stay_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reply without counters (older Ollama) gives None, not zero."""
    install_fake_ollama(monkeypatch, lambda **kwargs: ollama_reply(ANSWER))

    usage = inspect_csv(SAMPLE_CSV, settings=Settings(), model="m", fallback_model="m").usage

    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens, usage.load_seconds) == (None,) * 3


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_a_custom_invoker_reports_no_tokens_but_counts_attempts(use_async: bool) -> None:
    """Custom invokers return text only: tokens are None, attempts are still counted."""

    def answer(model: str) -> str:
        if model == "first":
            raise RuntimeError("down")
        return ANSWER

    async def async_invoker(prompt: str, model: str) -> str:
        return answer(model)

    invoker = async_invoker if use_async else (lambda prompt, model: answer(model))
    usage = _inspect(use_async, model="first", fallback_model="second", model_invoker=invoker).usage

    assert usage is not None
    assert (usage.model, usage.attempts, usage.retries) == ("second", 2, 0)
    assert (usage.prompt_tokens, usage.completion_tokens) == (None, None)
    assert usage.prompt_version == PROMPT_VERSION


def test_usage_is_never_serialized() -> None:
    """The usage stays out of the JSON Schema sent to models and out of every dump."""
    result = inspect_csv(
        SAMPLE_CSV, model="m", fallback_model="m", model_invoker=lambda p, m: ANSWER
    )
    assert result.usage is not None

    assert "usage" not in CSVInspectionResult.model_json_schema()["properties"]
    assert "usage" not in json.dumps(CSVInspectionResult.model_json_schema())
    assert "usage" not in result.model_dump()
    assert "usage" not in json.loads(result.model_dump_json())


def test_cli_stats_prints_the_usage_to_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--stats adds the usage block on stderr; stdout stays the result JSON only."""
    reply = ollama_reply(ANSWER, prompt_eval_count=900, eval_count=150)
    install_fake_ollama(monkeypatch, lambda **kwargs: reply)

    main([str(SAMPLE_CSV), "--stats", "--model", "m", "--no-env-file", "--log-level", "ERROR"])

    captured = capsys.readouterr()
    assert "usage" not in json.loads(captured.out)
    stats = json.loads(captured.err)
    assert (stats["model"], stats["prompt_tokens"], stats["completion_tokens"]) == ("m", 900, 150)


def test_cli_without_stats_prints_nothing_to_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --stats the usage is not printed."""
    install_fake_ollama(monkeypatch, lambda **kwargs: ollama_reply(ANSWER, eval_count=1))

    main([str(SAMPLE_CSV), "--model", "m", "--no-env-file", "--log-level", "ERROR"])

    assert capsys.readouterr().err == ""
