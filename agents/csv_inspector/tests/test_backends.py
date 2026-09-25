"""Unit tests for LLM backend selection and the cloud (Gemini) invoker.

No test here reaches the network: the ``google-genai`` client class is
replaced by a recording fake, while the SDK's real request types are kept,
so the request shape is still checked against the actual SDK. The autouse
``_isolated_settings`` fixture (``conftest.py``) clears every settings
variable and runs each test from an empty directory.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from ollama import ResponseError
from pydantic import SecretStr

from csv_inspector import (
    BackendConfigurationError,
    CredentialsNotConfiguredError,
    CSVInspectionResult,
    LLMBackend,
    ModelInvocationError,
    Settings,
    ainspect_csv,
    ensure_backend_ready,
    inspect_csv,
)
from csv_inspector._config import DEFAULT_MODEL, FALLBACK_MODEL, resolve_settings
from csv_inspector._invokers import (
    ainvoke_cloud_model,
    ainvoke_ollama_model,
    invoke_cloud_model,
    invoke_ollama_model,
)
from csv_inspector._prompt import _strip_annotations, response_schema
from fakes import install_fake_ollama, ollama_reply

AGENT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV_PATH = AGENT_DIR / "sample.csv"
FAKE_KEY = "AIza-fake-test-key-000"

VALID_RESULT_JSON = json.dumps(
    {
        "encoding": "utf-8",
        "delimiter": ";",
        "header_row_index": 2,
        "columns": ["Fecha"],
        "confidence": 0.9,
    }
)

needs_cloud_extra = pytest.mark.skipif(
    importlib.util.find_spec("google") is None,
    reason="needs the [cloud] extra",
)


def _without_cloud_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a base install, without ``google-genai`` (imported lazily)."""
    monkeypatch.setitem(sys.modules, "google.genai", None)


class _RecordingClient:
    """Stand-in for ``google.genai.Client`` that records calls instead of sending them."""

    instances: ClassVar[list[_RecordingClient]] = []
    response_text: ClassVar[str | None] = VALID_RESULT_JSON
    error: ClassVar[Exception | None] = None
    # Raised by the first requests, one each, before ``error``/the response.
    errors: ClassVar[list[Exception]] = []
    response: ClassVar[Any] = None

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.requests: list[dict[str, Any]] = []
        self.closed = False
        self.models = SimpleNamespace(generate_content=self._generate_content)
        self.aio = _RecordingAsyncClient(self)
        type(self).instances.append(self)

    def _generate_content(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if type(self).errors:
            raise type(self).errors.pop(0)
        error = type(self).error
        if error is not None:
            raise error
        if type(self).response is not None:
            return type(self).response
        # The shape _GenaiResponse declares.
        return SimpleNamespace(text=type(self).response_text, prompt_feedback=None, candidates=None)

    def __enter__(self) -> _RecordingClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.closed = True


class _RecordingAsyncClient:
    """Stand-in for ``client.aio``, delegating to the sync recorder."""

    def __init__(self, client: _RecordingClient) -> None:
        self.models = SimpleNamespace(generate_content=self._generate_content)
        self._client = client

    async def _generate_content(self, **kwargs: Any) -> Any:
        return self._client._generate_content(**kwargs)

    async def __aenter__(self) -> _RecordingAsyncClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        pass


@pytest.fixture
def recording_client(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingClient]:
    """Replace ``google.genai.Client`` with :class:`_RecordingClient`."""
    genai = pytest.importorskip("google.genai")
    _RecordingClient.instances = []
    _RecordingClient.response_text = VALID_RESULT_JSON
    _RecordingClient.error = None
    _RecordingClient.errors = []
    _RecordingClient.response = None
    monkeypatch.setattr(genai, "Client", _RecordingClient)
    return _RecordingClient


# ---------------------------------------------------------------------
# Factories and backend resolution
# ---------------------------------------------------------------------


def test_backend_values_match_the_cli_choices() -> None:
    """The enum's string values are the public ``--backend`` / LLM_BACKEND values."""
    assert [backend.value for backend in LLMBackend] == ["local", "api"]


def test_local_models_default_to_the_built_in_names() -> None:
    """Without configuration, the local backend keeps today's model names."""
    settings = resolve_settings(None)

    assert settings.model_for(LLMBackend.LOCAL) == DEFAULT_MODEL
    assert settings.fallback_model_for(LLMBackend.LOCAL) == FALLBACK_MODEL


def test_local_backend_needs_no_cloud_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Base requirements only: settings load from the environment, local backend ready (#99)."""
    _without_cloud_extra(monkeypatch)
    monkeypatch.setenv("OLLAMA_FALLBACK_MODEL", FALLBACK_MODEL)

    settings = resolve_settings(None)

    assert settings.model_for(LLMBackend.LOCAL) == DEFAULT_MODEL
    assert settings.fallback_model_for(LLMBackend.LOCAL) == FALLBACK_MODEL
    assert settings.llm_backend is LLMBackend.LOCAL
    ensure_backend_ready(LLMBackend.LOCAL)


def test_api_backend_without_cloud_extra_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting the API backend on a base install fails with an actionable message."""
    _without_cloud_extra(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)

    with pytest.raises(BackendConfigurationError, match=r"csv-inspector\[cloud\]"):
        ensure_backend_ready(LLMBackend.API)


@needs_cloud_extra
def test_resolved_settings_read_the_configured_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured names are used per backend."""
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    monkeypatch.setenv("CLOUD_MODEL", "gemini-x")
    monkeypatch.setenv("CLOUD_FALLBACK_MODEL", "gemini-y")

    settings = resolve_settings(None)

    assert settings.model_for(LLMBackend.LOCAL) == "llama3.1:8b"
    assert settings.model_for(LLMBackend.API) == "gemini-x"
    assert settings.fallback_model_for(LLMBackend.API) == "gemini-y"


@needs_cloud_extra
def test_configured_backend_follows_llm_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM_BACKEND selects the default backend; unset means local."""
    assert resolve_settings(None).llm_backend is LLMBackend.LOCAL

    monkeypatch.setenv("LLM_BACKEND", "api")

    assert resolve_settings(None).llm_backend is LLMBackend.API


@needs_cloud_extra
def test_ensure_backend_ready_requires_cloud_credentials() -> None:
    """The API backend is not ready without credentials; local always is."""
    ensure_backend_ready(LLMBackend.LOCAL)

    with pytest.raises(CredentialsNotConfiguredError):
        ensure_backend_ready(LLMBackend.API)


@needs_cloud_extra
def test_ensure_backend_ready_requires_the_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """With credentials but no google-genai, the API backend is reported as not ready."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    with pytest.raises(BackendConfigurationError, match="google-genai"):
        ensure_backend_ready(LLMBackend.API)


# ---------------------------------------------------------------------
# invoke_cloud_model (real SDK types, recorded client, no network)
# ---------------------------------------------------------------------


@needs_cloud_extra
def test_cloud_invoker_without_credentials_fails_before_creating_a_client(
    recording_client: type[_RecordingClient],
) -> None:
    """No credentials: a clear domain error, and no client or request is ever made."""
    with pytest.raises(CredentialsNotConfiguredError, match="GEMINI_API_KEY"):
        invoke_cloud_model("prompt", "gemini-2.5-flash")

    assert recording_client.instances == []


@needs_cloud_extra
def test_cloud_invoker_sends_a_deterministic_json_request_with_the_schema(
    monkeypatch: pytest.MonkeyPatch, recording_client: type[_RecordingClient]
) -> None:
    """The request uses JSON mode, the shared response schema and temperature 0.0."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)

    text = invoke_cloud_model("the prompt", "gemini-2.5-flash")

    assert text == VALID_RESULT_JSON
    (client,) = recording_client.instances
    assert client.init_kwargs == {"api_key": FAKE_KEY}
    assert client.closed
    (request,) = client.requests
    assert request["model"] == "gemini-2.5-flash"
    assert request["contents"] == "the prompt"
    config = request["config"]
    assert config.temperature == 0.0
    assert config.response_mime_type == "application/json"
    # The very dict the Ollama backend sends as ``format`` (issue #130).
    assert config.response_json_schema is response_schema()
    sent = config.response_json_schema["properties"]
    assert "footer_first_line" in sent
    assert "footer_lines" not in sent
    assert config.system_instruction
    assert config.automatic_function_calling.disable


@needs_cloud_extra
def test_cloud_invoker_response_parses_into_a_validated_result(
    monkeypatch: pytest.MonkeyPatch, recording_client: type[_RecordingClient]
) -> None:
    """End to end through inspect_csv: the mocked cloud answer becomes a result."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)

    result = inspect_csv(SAMPLE_CSV_PATH, backend=LLMBackend.API)

    assert isinstance(result, CSVInspectionResult)
    assert result.delimiter == ";"
    assert recording_client.instances[0].requests[0]["model"] == "gemini-3.6-flash"


@needs_cloud_extra
def test_cloud_invoker_uses_vertex_ai_without_an_api_key(
    monkeypatch: pytest.MonkeyPatch, recording_client: type[_RecordingClient]
) -> None:
    """Project + location configure a Vertex AI client (ADC), with no key."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")

    invoke_cloud_model("prompt", "gemini-2.5-flash")

    assert recording_client.instances[0].init_kwargs == {
        "vertexai": True,
        "project": "my-project",
        "location": "europe-west1",
    }


@needs_cloud_extra
@pytest.mark.parametrize("text", [None, ""])
def test_cloud_invoker_rejects_an_empty_response(
    monkeypatch: pytest.MonkeyPatch, recording_client: type[_RecordingClient], text: str | None
) -> None:
    """An empty answer is a backend failure, not an empty JSON payload."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    recording_client.response_text = text

    with pytest.raises(ModelInvocationError, match="empty response"):
        invoke_cloud_model("prompt", "gemini-2.5-flash")


@needs_cloud_extra
@pytest.mark.parametrize(
    ("response_fields", "reason"),
    [
        ({"prompt_feedback": {"block_reason": "SAFETY"}}, "prompt blocked: SAFETY"),
        ({"candidates": [{"finish_reason": "RECITATION"}]}, "finish reason: RECITATION"),
    ],
)
def test_cloud_invoker_reports_why_a_response_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    recording_client: type[_RecordingClient],
    response_fields: dict[str, Any],
    reason: str,
) -> None:
    """A blocked prompt or a stopped candidate has no text; the error says why."""
    from google.genai import types  # noqa: PLC0415 - needs the [cloud] extra.

    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    recording_client.response = types.GenerateContentResponse.model_validate(response_fields)

    with pytest.raises(ModelInvocationError, match=rf"empty response \({reason}\)"):
        invoke_cloud_model("prompt", "gemini-2.5-flash")


@needs_cloud_extra
def test_cloud_invoker_errors_never_carry_the_api_key(
    monkeypatch: pytest.MonkeyPatch,
    recording_client: type[_RecordingClient],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A service error echoing the key is redacted, and the original is not chained."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    recording_client.error = RuntimeError(f"403 PERMISSION_DENIED for key {FAKE_KEY}")

    with (
        caplog.at_level(logging.DEBUG),
        pytest.raises(ModelInvocationError, match="PERMISSION_DENIED") as exc_info,
    ):
        invoke_cloud_model("prompt", "gemini-2.5-flash")

    assert FAKE_KEY not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__
    assert FAKE_KEY not in caplog.text


def _api_error(code: int, retry_after: str | None = None) -> Exception:
    """Build a real ``google-genai`` API error with an HTTP status (and Retry-After)."""
    import httpx  # noqa: PLC0415
    from google.genai import errors  # noqa: PLC0415

    headers = {} if retry_after is None else {"Retry-After": retry_after}
    error_class = errors.ServerError if code >= 500 else errors.ClientError
    return error_class(
        code,
        {"error": {"code": code, "message": "transient", "status": "UNAVAILABLE"}},
        httpx.Response(code, headers=headers),
    )


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the invokers' retry waits instead of sleeping."""
    waits: list[float] = []

    async def fake_async_sleep(seconds: float) -> None:
        waits.append(seconds)

    # The invokers call time.sleep / asyncio.sleep through their modules.
    monkeypatch.setattr(time, "sleep", waits.append)
    monkeypatch.setattr(asyncio, "sleep", fake_async_sleep)
    return waits


@needs_cloud_extra
@pytest.mark.parametrize("code", [429, 503])
def test_cloud_invoker_retries_a_transient_error_once(
    monkeypatch: pytest.MonkeyPatch,
    recording_client: type[_RecordingClient],
    sleeps: list[float],
    code: int,
) -> None:
    """A 429/503 is retried once on the same model, about a second later (issue #98)."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    recording_client.errors = [_api_error(code)]

    assert invoke_cloud_model("prompt", "gemini-x") == VALID_RESULT_JSON

    assert [len(client.requests) for client in recording_client.instances] == [1, 1]
    (wait,) = sleeps
    assert 0.8 <= wait <= 1.2


@needs_cloud_extra
def test_cloud_invoker_retries_only_once(
    monkeypatch: pytest.MonkeyPatch, recording_client: type[_RecordingClient], sleeps: list[float]
) -> None:
    """Two 503s in a row: ModelInvocationError after exactly two requests."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    recording_client.errors = [_api_error(503), _api_error(503)]

    with pytest.raises(ModelInvocationError, match="503"):
        invoke_cloud_model("prompt", "gemini-x")

    assert sum(len(client.requests) for client in recording_client.instances) == 2
    assert len(sleeps) == 1


@needs_cloud_extra
@pytest.mark.parametrize("code", [400, 404, 500])
def test_cloud_invoker_does_not_retry_other_errors(
    monkeypatch: pytest.MonkeyPatch,
    recording_client: type[_RecordingClient],
    sleeps: list[float],
    code: int,
) -> None:
    """Anything but 429/503 fails at once: the fallback model handles it."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    recording_client.errors = [_api_error(code)]

    with pytest.raises(ModelInvocationError):
        invoke_cloud_model("prompt", "gemini-x")

    assert len(recording_client.instances) == 1
    assert sleeps == []


@needs_cloud_extra
def test_cloud_invoker_honours_a_short_retry_after(
    monkeypatch: pytest.MonkeyPatch, recording_client: type[_RecordingClient], sleeps: list[float]
) -> None:
    """Retry-After sets the wait; a long one means a quota, so there is no retry."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    recording_client.errors = [_api_error(429, retry_after="3")]
    invoke_cloud_model("prompt", "gemini-x")
    assert sleeps == [3.0]

    recording_client.errors = [_api_error(429, retry_after="60")]
    with pytest.raises(ModelInvocationError):
        invoke_cloud_model("prompt", "gemini-x")
    assert sleeps == [3.0]


@needs_cloud_extra
def test_cloud_retry_never_outlives_the_time_budget(
    monkeypatch: pytest.MonkeyPatch, recording_client: type[_RecordingClient], sleeps: list[float]
) -> None:
    """A retry that would not fit the budget is skipped; one that fits gets the rest."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    recording_client.errors = [_api_error(503, retry_after="2")]
    with pytest.raises(ModelInvocationError):
        invoke_cloud_model("prompt", "gemini-x", timeout_seconds=2.5)
    assert sleeps == []

    recording_client.instances = []
    recording_client.errors = [_api_error(503, retry_after="2")]
    invoke_cloud_model("prompt", "gemini-x", timeout_seconds=10)
    first, retried = (
        client.init_kwargs["http_options"].timeout for client in recording_client.instances
    )
    assert first == 10_000
    assert 7_000 <= retried <= 8_000


@needs_cloud_extra
def test_async_cloud_invoker_retries_a_transient_error_once(
    monkeypatch: pytest.MonkeyPatch, recording_client: type[_RecordingClient], sleeps: list[float]
) -> None:
    """The async invoker retries a 503 once too, with asyncio.sleep."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    recording_client.errors = [_api_error(503)]

    assert asyncio.run(ainvoke_cloud_model("prompt", "gemini-x")) == VALID_RESULT_JSON

    assert len(recording_client.instances) == 2
    assert len(sleeps) == 1


@needs_cloud_extra
def test_a_transient_error_does_not_reach_the_fallback_model(
    recording_client: type[_RecordingClient], sleeps: list[float]
) -> None:
    """inspect_csv: the retried primary answers, so the fallback is never called."""
    recording_client.errors = [_api_error(503)]
    settings = Settings(gemini_api_key=SecretStr(FAKE_KEY), cloud_model="primary")

    result = inspect_csv(SAMPLE_CSV_PATH, backend=LLMBackend.API, settings=settings)

    assert result.delimiter
    models = [r["model"] for c in recording_client.instances for r in c.requests]
    assert models == ["primary", "primary"]


@needs_cloud_extra
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_cloud_usage_reports_tokens_and_the_retry(
    recording_client: type[_RecordingClient], sleeps: list[float], use_async: bool
) -> None:
    """Gemini's usage_metadata and the one 503 retry reach result.usage (#121)."""
    recording_client.errors = [_api_error(503)]
    recording_client.response = SimpleNamespace(
        text=VALID_RESULT_JSON,
        prompt_feedback=None,
        candidates=None,
        usage_metadata=SimpleNamespace(prompt_token_count=120, candidates_token_count=30),
    )
    settings = Settings(gemini_api_key=SecretStr(FAKE_KEY), cloud_model="primary")

    if use_async:
        result = asyncio.run(
            ainspect_csv(SAMPLE_CSV_PATH, backend=LLMBackend.API, settings=settings)
        )
    else:
        result = inspect_csv(SAMPLE_CSV_PATH, backend=LLMBackend.API, settings=settings)

    usage = result.usage
    assert usage is not None
    assert (usage.model, usage.prompt_tokens, usage.completion_tokens) == ("primary", 120, 30)
    assert (usage.attempts, usage.retries, usage.load_seconds) == (1, 1, None)


@needs_cloud_extra
def test_missing_application_default_credentials_is_a_credentials_error(
    monkeypatch: pytest.MonkeyPatch, recording_client: type[_RecordingClient]
) -> None:
    """Vertex AI without ADC points the user at how to create them."""
    from google.auth.exceptions import DefaultCredentialsError  # noqa: PLC0415

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")
    # google-auth ships without type annotations.
    recording_client.error = DefaultCredentialsError("File not found")  # type: ignore[no-untyped-call]

    with pytest.raises(CredentialsNotConfiguredError, match="application-default login"):
        invoke_cloud_model("prompt", "gemini-2.5-flash")


@needs_cloud_extra
def test_cloud_invoker_without_the_sdk_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credentials but no google-genai: an actionable configuration error."""
    import google  # noqa: PLC0415

    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setitem(sys.modules, "google.genai", None)
    monkeypatch.delattr(google, "genai", raising=False)

    with pytest.raises(BackendConfigurationError, match=r"csv-inspector\[cloud\]"):
        invoke_cloud_model("prompt", "gemini-2.5-flash")


# ---------------------------------------------------------------------
# inspect_csv backend wiring
# ---------------------------------------------------------------------


def test_explicit_model_invoker_takes_precedence_over_backend() -> None:
    """Regression: an injected invoker is used even when backend=API is requested."""
    calls: list[str] = []

    def fake_invoker(prompt: str, model: str) -> str:
        calls.append(model)
        return VALID_RESULT_JSON

    result = inspect_csv(
        SAMPLE_CSV_PATH,
        backend=LLMBackend.API,
        model="injected-model",
        fallback_model="injected-model",
        model_invoker=fake_invoker,
    )

    assert calls == ["injected-model"]
    assert result.delimiter == ";"


def test_default_backend_is_local_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no backend argument, inspect_csv calls Ollama with the default model."""
    models: list[str] = []

    def chat(**kwargs: Any) -> Any:
        models.append(kwargs["model"])
        return ollama_reply(VALID_RESULT_JSON)

    install_fake_ollama(monkeypatch, chat)

    inspect_csv(SAMPLE_CSV_PATH)

    assert models == [DEFAULT_MODEL]


@pytest.mark.parametrize(
    ("settings", "expected_host"),
    [(Settings(ollama_host="http://x:1"), "http://x:1"), (Settings(), None)],
)
def test_ollama_client_receives_the_configured_host(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, expected_host: str | None
) -> None:
    """Settings.ollama_host reaches the sync Ollama client; unset means SDK default (#96)."""
    fake = install_fake_ollama(monkeypatch, lambda **_: ollama_reply(VALID_RESULT_JSON))

    inspect_csv(SAMPLE_CSV_PATH, settings=settings)

    assert fake.client_kwargs[0]["host"] == expected_host


def test_async_ollama_client_receives_the_configured_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ainspect_csv forwards Settings.ollama_host to the async Ollama client (#96)."""
    fake = install_fake_ollama(monkeypatch, lambda **_: ollama_reply(VALID_RESULT_JSON))

    asyncio.run(ainspect_csv(SAMPLE_CSV_PATH, settings=Settings(ollama_host="http://x:1")))

    assert fake.client_kwargs[0]["host"] == "http://x:1"


@needs_cloud_extra
def test_configuration_errors_are_not_retried_with_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing credentials surface directly, after one attempt, not as InspectionFailedError."""
    calls: list[str] = []

    def failing_invoker(prompt: str, model: str) -> str:
        calls.append(model)
        raise CredentialsNotConfiguredError("set GEMINI_API_KEY")

    with pytest.raises(CredentialsNotConfiguredError):
        inspect_csv(
            SAMPLE_CSV_PATH,
            model="primary",
            fallback_model="fallback",
            model_invoker=failing_invoker,
        )

    assert calls == ["primary"]


# ---------------------------------------------------------------------
# The response schema both backends send (issue #130)
# ---------------------------------------------------------------------


def test_response_schema_is_small_flat_and_bounded() -> None:
    """No annotations, no $defs, no usage; names are kept and numeric bounds stay."""
    schema = response_schema()
    text = json.dumps(schema)

    assert "$defs" not in schema
    assert "$ref" not in text
    for keyword in ('"title"', '"description"', '"default"'):
        assert keyword not in text
    assert "usage" not in schema["properties"]
    assert schema["properties"]["columns"]["type"] == "array"
    # The model's answer, not the result: one footer anchor line (issue #132).
    assert "footer_first_line" in schema["properties"]
    assert "footer_lines" not in schema["properties"]
    assert "footer_rows_to_skip" not in schema["properties"]
    # A grammar lets a model skip optional keys; every one is required.
    assert schema["required"] == list(schema["properties"])
    assert schema["properties"]["confidence"] == {
        "maximum": 1.0,
        "minimum": 0.0,
        "type": "number",
    }
    assert response_schema() is schema


def test_schema_annotations_are_dropped_but_property_names_kept() -> None:
    """A property named like an annotation keyword survives; annotations do not."""
    schema = {
        "title": "T",
        "properties": {"title": {"type": "string", "description": "x"}, "n": {"default": 1}},
    }

    assert _strip_annotations(schema) == {"properties": {"title": {"type": "string"}, "n": {}}}


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_ollama_request_sends_the_response_schema(
    monkeypatch: pytest.MonkeyPatch, use_async: bool
) -> None:
    """The ``format`` of every Ollama request is the response schema, not "json"."""
    fake = install_fake_ollama(monkeypatch, lambda **_: ollama_reply(VALID_RESULT_JSON))

    if use_async:
        asyncio.run(ainvoke_ollama_model("p", "m"))
    else:
        invoke_ollama_model("p", "m")

    (request,) = fake.requests
    assert request["format"] == response_schema()
    assert request["format"]["properties"]["columns"]["type"] == "array"
    assert "footer_first_line" in request["format"]["properties"]
    assert "footer_lines" not in request["format"]["properties"]


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_ollama_schema_rejection_retries_in_json_mode(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, use_async: bool
) -> None:
    """An old server refusing a schema ``format`` gets the same call with "json", once."""

    def chat(**kwargs: Any) -> Any:
        if isinstance(kwargs["format"], dict):
            raise ResponseError('{"error": "invalid format: expected \\"json\\""}', 400)
        return ollama_reply(VALID_RESULT_JSON)

    fake = install_fake_ollama(monkeypatch, chat)

    with caplog.at_level(logging.WARNING, logger="csv_inspector"):
        if use_async:
            text = asyncio.run(ainvoke_ollama_model("p", "m"))
        else:
            text = invoke_ollama_model("p", "m")

    assert text == VALID_RESULT_JSON
    assert [request["format"] for request in fake.requests] == [response_schema(), "json"]
    assert fake.requests[1]["messages"] == fake.requests[0]["messages"]
    assert fake.requests[1]["options"] == fake.requests[0]["options"]
    (record,) = [r for r in caplog.records if "rejected the response schema" in r.message]
    assert record.levelno == logging.WARNING


@pytest.mark.parametrize(
    ("error", "status"),
    [("model 'm' not found", 404), ("invalid options", 400)],
    ids=["other-status", "400-not-about-format"],
)
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_other_ollama_errors_are_not_retried_in_json_mode(
    monkeypatch: pytest.MonkeyPatch, error: str, status: int, use_async: bool
) -> None:
    """Only a 400 about ``format`` triggers the JSON-mode retry."""

    def chat(**kwargs: Any) -> Any:
        raise ResponseError(error, status)

    fake = install_fake_ollama(monkeypatch, chat)

    def invoke() -> str:
        if use_async:
            return asyncio.run(ainvoke_ollama_model("p", "m"))
        return invoke_ollama_model("p", "m")

    with pytest.raises(ModelInvocationError, match=error):
        invoke()

    assert len(fake.requests) == 1


# ---------------------------------------------------------------------
# Import hygiene and CLI behavior (subprocesses, no network)
# ---------------------------------------------------------------------


def _run_python(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the current interpreter with the agent directory on the path."""
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )


def test_no_module_imports_google_genai_at_import_time(tmp_path: Path) -> None:
    """Importing the package, its CLI and the repo scripts must not load the cloud SDK."""
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(AGENT_DIR / 'scripts')!r}); "
        "import csv_inspector, csv_inspector.cli, eval_samples; "
        "import csv_inspector._config, csv_inspector._inspect, csv_inspector._invokers; "
        "print(sorted(name for name in sys.modules if name.startswith('google')))"
    )

    completed = _run_python("-c", code, cwd=tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


@needs_cloud_extra
@pytest.mark.parametrize(
    "command",
    [
        ("-m", "csv_inspector", str(SAMPLE_CSV_PATH)),
        (str(AGENT_DIR / "main_demo.py"),),
        (str(AGENT_DIR / "scripts" / "eval_samples.py"),),
    ],
    ids=["python -m csv_inspector", "main_demo.py", "eval_samples.py"],
)
def test_cli_api_backend_without_credentials_fails_cleanly(
    tmp_path: Path, command: tuple[str, ...]
) -> None:
    """``--backend api`` with no credentials: exit 1, a one-line error, no traceback."""
    completed = _run_python(*command, "--backend", "api", cwd=tmp_path)

    assert completed.returncode == 1
    assert "GEMINI_API_KEY" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert completed.stdout == ""


@needs_cloud_extra
def test_ensure_backend_ready_checks_explicit_settings_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With explicit settings the public check ignores the environment (issue #82)."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)

    with pytest.raises(CredentialsNotConfiguredError):
        ensure_backend_ready(LLMBackend.API, Settings())
    ensure_backend_ready(LLMBackend.API, Settings(gemini_api_key=SecretStr(FAKE_KEY)))
    ensure_backend_ready(LLMBackend.LOCAL, Settings())
