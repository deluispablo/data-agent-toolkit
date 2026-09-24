"""Tests for the time budget (sync and async) and the asyncio API.

No test reaches the network: invokers are fakes, the ``ollama`` module is a
recording stand-in, and the Gemini client class is replaced by a recorder.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import httpx
import pytest
from fakes import install_fake_ollama, ollama_reply

from csv_inspector import (
    BackendConfigurationError,
    CSVInspectionResult,
    InspectionFailedError,
    InspectionTimeoutError,
    LLMBackend,
    ModelTimeoutError,
    Settings,
    ainspect_csv,
    inspect_csv,
)
from csv_inspector._invokers import (
    ainvoke_cloud_model,
    ainvoke_ollama_model,
    invoke_cloud_model,
    invoke_ollama_model,
)
from csv_inspector._sampling import MAX_SAMPLE_BYTES, sample_source

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "agents" / "csv_inspector" / "sample.csv"
FAKE_KEY = "AIza-fake-test-key-000"
RESULT_JSON = json.dumps(
    {
        "encoding": "utf-8",
        "delimiter": ";",
        "header_row_index": 2,
        "columns": [{"name": "Fecha", "inferred_type": "date"}],
        "confidence": 0.9,
    }
)
# Generous slack for slow CI machines; the point is "near the budget", not "exactly".
SLACK_SECONDS = 1.0

needs_cloud_extra = pytest.mark.skipif(
    importlib.util.find_spec("google") is None, reason="needs the cloud extra"
)


# ---------------------------------------------------------------------
# Sync time budget
# ---------------------------------------------------------------------


def test_a_slow_sync_invoker_is_cut_at_the_budget() -> None:
    """A custom invoker that hangs does not hold the caller past the budget."""

    def slow_invoker(prompt: str, model: str) -> str:
        time.sleep(3)
        return RESULT_JSON

    started = time.monotonic()
    with pytest.raises(InspectionTimeoutError) as exc_info:
        inspect_csv(
            SAMPLE_CSV,
            model="m",
            fallback_model="m",
            model_invoker=slow_invoker,
            timeout_seconds=0.3,
        )
    elapsed = time.monotonic() - started

    assert elapsed < 0.3 + SLACK_SECONDS
    assert isinstance(exc_info.value.attempts["m"], ModelTimeoutError)


def test_the_budget_is_shared_so_no_fallback_runs_once_it_is_spent() -> None:
    """The fallback is not attempted after the primary exhausted the budget."""
    calls: list[str] = []

    def invoker(prompt: str, model: str) -> str:
        calls.append(model)
        time.sleep(2)
        return RESULT_JSON

    with pytest.raises(InspectionTimeoutError):
        inspect_csv(
            SAMPLE_CSV,
            model="primary",
            fallback_model="fallback",
            model_invoker=invoker,
            timeout_seconds=0.3,
        )

    assert calls == ["primary"]


def test_a_fast_failure_leaves_budget_for_the_fallback() -> None:
    """A quick primary failure within budget still lets the fallback answer."""
    calls: list[str] = []

    def invoker(prompt: str, model: str) -> str:
        calls.append(model)
        return "not json" if model == "primary" else RESULT_JSON

    result = inspect_csv(
        SAMPLE_CSV,
        model="primary",
        fallback_model="fallback",
        model_invoker=invoker,
        timeout_seconds=5,
    )

    assert calls == ["primary", "fallback"]
    assert result.delimiter == ";"


def test_timeout_error_is_an_inspection_failed_error() -> None:
    """Hosts catching InspectionFailedError also catch timeouts."""
    assert issubclass(InspectionTimeoutError, InspectionFailedError)


@pytest.mark.parametrize("timeout", [0, -1, float("nan")])
def test_non_positive_timeouts_are_rejected(timeout: float) -> None:
    """Zero, negative or NaN budgets are programming errors."""
    with pytest.raises(ValueError, match="timeout_seconds"):
        inspect_csv(
            SAMPLE_CSV,
            model="m",
            fallback_model="m",
            model_invoker=lambda p, m: RESULT_JSON,
            timeout_seconds=timeout,
        )


def test_ollama_client_receives_the_remaining_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """The built-in local invoker passes the budget to its HTTP client."""
    fake = install_fake_ollama(monkeypatch, lambda **kwargs: ollama_reply(RESULT_JSON))

    inspect_csv(SAMPLE_CSV, timeout_seconds=5)

    timeout = fake.client_kwargs[0]["timeout"]
    assert 0 < timeout <= 5
    assert fake.closed_clients == 1


def test_no_budget_means_no_client_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without timeout_seconds, the client keeps its default (no limit)."""
    fake = install_fake_ollama(monkeypatch, lambda **kwargs: ollama_reply(RESULT_JSON))

    inspect_csv(SAMPLE_CSV)

    assert fake.client_kwargs[0]["timeout"] is None


@pytest.mark.parametrize("error", [httpx.ReadTimeout("read timed out"), TimeoutError("timed out")])
def test_client_timeouts_map_to_model_timeout_error(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """A transport timeout inside the Ollama client is a ModelTimeoutError."""

    def chat(**kwargs: Any) -> Any:
        raise error

    install_fake_ollama(monkeypatch, chat)

    with pytest.raises(ModelTimeoutError):
        invoke_ollama_model("prompt", "m", timeout_seconds=1)


# ---------------------------------------------------------------------
# Gemini: timeout in milliseconds, native async client
# ---------------------------------------------------------------------


class _RecordingGenai:
    """Replacement for ``google.genai.Client`` with sync and ``aio`` surfaces."""

    instances: ClassVar[list[_RecordingGenai]] = []
    error: ClassVar[Exception | None] = None

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.sync_requests: list[dict[str, Any]] = []
        self.async_requests: list[dict[str, Any]] = []
        self.models = SimpleNamespace(generate_content=self._generate)
        self.aio = _RecordingAio(self)
        type(self).instances.append(self)

    def _generate(self, **kwargs: Any) -> Any:
        self.sync_requests.append(kwargs)
        error = type(self).error
        if error is not None:
            raise error
        return SimpleNamespace(text=RESULT_JSON)

    def __enter__(self) -> _RecordingGenai:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


class _RecordingAio:
    def __init__(self, parent: _RecordingGenai) -> None:
        self._parent = parent
        self.models = SimpleNamespace(generate_content=self._generate)
        self.closed = False

    async def _generate(self, **kwargs: Any) -> Any:
        self._parent.async_requests.append(kwargs)
        return SimpleNamespace(text=RESULT_JSON)

    async def __aenter__(self) -> _RecordingAio:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.closed = True


@pytest.fixture
def recording_genai(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingGenai]:
    """Replace ``google.genai.Client`` with :class:`_RecordingGenai`."""
    genai = pytest.importorskip("google.genai")
    _RecordingGenai.instances = []
    _RecordingGenai.error = None
    monkeypatch.setattr(genai, "Client", _RecordingGenai)
    return _RecordingGenai


@needs_cloud_extra
def test_gemini_receives_the_timeout_in_milliseconds(
    recording_genai: type[_RecordingGenai],
) -> None:
    """google-genai's HttpOptions.timeout is in ms: 2.5 s must become 2500."""
    invoke_cloud_model(
        "prompt",
        "gemini-2.5-flash",
        settings=Settings(gemini_api_key=FAKE_KEY),
        timeout_seconds=2.5,
    )

    assert recording_genai.instances[0].init_kwargs["http_options"].timeout == 2500


@needs_cloud_extra
def test_gemini_without_a_timeout_sets_no_http_options(
    recording_genai: type[_RecordingGenai],
) -> None:
    """No budget: the client is built exactly as before, without http_options."""
    invoke_cloud_model("prompt", "gemini-2.5-flash", settings=Settings(gemini_api_key=FAKE_KEY))

    assert recording_genai.instances[0].init_kwargs == {"api_key": FAKE_KEY}


@needs_cloud_extra
def test_gemini_transport_timeouts_map_to_model_timeout_error(
    recording_genai: type[_RecordingGenai],
) -> None:
    """A transport timeout from the Gemini client is a ModelTimeoutError, key redacted."""
    recording_genai.error = httpx.ReadTimeout(f"timed out for {FAKE_KEY}")

    with pytest.raises(ModelTimeoutError) as exc_info:
        invoke_cloud_model(
            "prompt",
            "gemini-2.5-flash",
            settings=Settings(gemini_api_key=FAKE_KEY),
            timeout_seconds=1,
        )

    assert FAKE_KEY not in str(exc_info.value)


@needs_cloud_extra
def test_async_cloud_invoker_uses_the_native_aio_client(
    recording_genai: type[_RecordingGenai],
) -> None:
    """ainvoke_cloud_model awaits client.aio and closes it; no sync request is made."""
    text = asyncio.run(
        ainvoke_cloud_model(
            "prompt",
            "gemini-2.5-flash",
            settings=Settings(gemini_api_key=FAKE_KEY),
            timeout_seconds=3,
        )
    )

    client = recording_genai.instances[0]
    assert text == RESULT_JSON
    assert client.sync_requests == []
    assert client.async_requests[0]["config"].temperature == 0.0
    assert client.aio.closed
    assert client.init_kwargs["http_options"].timeout == 3000


# ---------------------------------------------------------------------
# ainspect_csv
# ---------------------------------------------------------------------


def test_ainspect_csv_matches_the_sync_result() -> None:
    """The async API returns exactly what the sync API returns for the same answer."""

    async def async_invoker(prompt: str, model: str) -> str:
        await asyncio.sleep(0)
        return RESULT_JSON

    sync_result = inspect_csv(
        SAMPLE_CSV, model="m", fallback_model="m", model_invoker=lambda p, m: RESULT_JSON
    )
    async_result = asyncio.run(
        ainspect_csv(SAMPLE_CSV, model="m", fallback_model="m", model_invoker=async_invoker)
    )

    assert isinstance(async_result, CSVInspectionResult)
    assert async_result == sync_result


def test_ainspect_csv_cancels_a_slow_async_invoker_at_the_budget() -> None:
    """asyncio.wait_for cancels the pending call when the budget runs out."""
    cancelled = asyncio.Event()

    async def slow_invoker(prompt: str, model: str) -> str:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return RESULT_JSON

    async def run() -> float:
        started = time.monotonic()
        with pytest.raises(InspectionTimeoutError):
            await ainspect_csv(
                SAMPLE_CSV,
                model="m",
                fallback_model="m",
                model_invoker=slow_invoker,
                timeout_seconds=0.3,
            )
        return time.monotonic() - started

    elapsed = asyncio.run(run())

    assert elapsed < 0.3 + SLACK_SECONDS
    assert cancelled.is_set()


def test_ainspect_csv_samples_off_the_event_loop_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocking sampling I/O runs in a worker thread, not on the loop's thread."""
    sampling_threads: list[threading.Thread] = []
    real_sample_source = sample_source

    def recording_sample_source(*args: Any) -> Any:
        sampling_threads.append(threading.current_thread())
        return real_sample_source(*args)

    monkeypatch.setattr("csv_inspector._inspect.sample_source", recording_sample_source)

    async def async_invoker(prompt: str, model: str) -> str:
        return RESULT_JSON

    async def run() -> threading.Thread:
        await ainspect_csv(SAMPLE_CSV, model="m", fallback_model="m", model_invoker=async_invoker)
        return threading.current_thread()

    loop_thread = asyncio.run(run())

    assert sampling_threads
    assert sampling_threads[0] is not loop_thread


def test_ainspect_csv_default_backend_uses_ollama_async_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no invoker, ainspect_csv talks to Ollama through AsyncClient."""
    fake = install_fake_ollama(monkeypatch, lambda **kwargs: ollama_reply(RESULT_JSON))

    result = asyncio.run(ainspect_csv(SAMPLE_CSV, timeout_seconds=5))

    assert result.delimiter == ";"
    assert fake.requests[0]["options"]["temperature"] == 0.0
    assert 0 < fake.client_kwargs[0]["timeout"] <= 5
    assert fake.closed_clients == 1


def test_ainvoke_ollama_model_returns_the_message_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """The async local invoker returns the model's content verbatim."""
    install_fake_ollama(monkeypatch, lambda **kwargs: ollama_reply('{"ok": true}'))

    assert asyncio.run(ainvoke_ollama_model("prompt", "m")) == '{"ok": true}'


@needs_cloud_extra
def test_ainspect_csv_fails_on_configuration_before_reading_the_source() -> None:
    """Missing cloud credentials surface before a (non-rewindable) stream is touched."""

    class ExplodingStream:
        def read(self, size: int = -1) -> bytes:
            raise AssertionError("the source must not be read")

    with pytest.raises(BackendConfigurationError):
        asyncio.run(
            ainspect_csv(
                ExplodingStream(),
                backend=LLMBackend.API,
                settings=Settings(),
            )
        )


def test_ollama_num_ctx_fits_the_largest_built_in_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Max-size numeric samples still fit the requested window, so Ollama never truncates."""
    fake = install_fake_ollama(monkeypatch, lambda **kwargs: ollama_reply(RESULT_JSON))
    target = tmp_path / "numbers.csv"
    target.write_text("a;b\n" + "1234567;7654321\n" * 5000, encoding="utf-8")

    inspect_csv(target, n_bytes=MAX_SAMPLE_BYTES, tail_bytes=MAX_SAMPLE_BYTES)

    request = fake.requests[0]
    prompt_chars = sum(len(message["content"]) for message in request["messages"])
    # ~2 characters per token for digit-heavy text, plus room for the reply.
    assert request["options"]["num_ctx"] >= prompt_chars // 2 + 1024
    assert request["options"]["num_ctx"] <= 32768


def test_ollama_num_ctx_uses_the_minimum_window_for_small_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Small prompts keep a stable minimum window instead of shrinking below it."""
    fake = install_fake_ollama(monkeypatch, lambda **kwargs: ollama_reply('{"ok": true}'))

    invoke_ollama_model("tiny", "m")

    assert fake.requests[0]["options"]["num_ctx"] == 4096
