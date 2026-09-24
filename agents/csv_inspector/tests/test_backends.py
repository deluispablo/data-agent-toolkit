"""Unit tests for LLM backend selection and the cloud (Gemini) invoker.

No test here reaches the network: the ``google-genai`` client class is
replaced by a recording fake, while the SDK's real request types are kept,
so the request shape is still checked against the actual SDK. The autouse
``_isolated_settings`` fixture (``conftest.py``) clears every settings
variable and runs each test from an empty directory.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from pydantic import SecretStr

from csv_inspector import (
    BackendConfigurationError,
    CredentialsNotConfiguredError,
    CSVInspectionResult,
    LLMBackend,
    ModelInvocationError,
    Settings,
    ensure_backend_ready,
    inspect_csv,
)
from csv_inspector._config import DEFAULT_MODEL, FALLBACK_MODEL, resolve_settings
from csv_inspector._invokers import invoke_cloud_model
from fakes import install_fake_ollama, ollama_reply

AGENT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV_PATH = AGENT_DIR / "sample.csv"
FAKE_KEY = "AIza-fake-test-key-000"

VALID_RESULT_JSON = json.dumps(
    {
        "encoding": "utf-8",
        "delimiter": ";",
        "header_row_index": 2,
        "columns": [{"name": "Fecha", "inferred_type": "date"}],
        "confidence": 0.9,
    }
)

needs_cloud_extra = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("pydantic_settings", "google")),
    reason="needs the [cloud] extra",
)


def _without_cloud_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a base install, without ``pydantic-settings`` (imported lazily)."""
    monkeypatch.setitem(sys.modules, "pydantic_settings", None)


class _RecordingClient:
    """Stand-in for ``google.genai.Client`` that records calls instead of sending them."""

    instances: ClassVar[list[_RecordingClient]] = []
    response_text: ClassVar[str | None] = VALID_RESULT_JSON
    error: ClassVar[Exception | None] = None
    response: ClassVar[Any] = None

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.requests: list[dict[str, Any]] = []
        self.closed = False
        self.models = SimpleNamespace(generate_content=self._generate_content)
        type(self).instances.append(self)

    def _generate_content(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        error = type(self).error
        if error is not None:
            raise error
        if type(self).response is not None:
            return type(self).response
        return SimpleNamespace(text=type(self).response_text)

    def __enter__(self) -> _RecordingClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.closed = True


@pytest.fixture
def recording_client(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingClient]:
    """Replace ``google.genai.Client`` with :class:`_RecordingClient`."""
    genai = pytest.importorskip("google.genai")
    _RecordingClient.instances = []
    _RecordingClient.response_text = VALID_RESULT_JSON
    _RecordingClient.error = None
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
    settings = resolve_settings(None, LLMBackend.LOCAL)

    assert settings.model_for(LLMBackend.LOCAL) == DEFAULT_MODEL
    assert settings.fallback_model_for(LLMBackend.LOCAL) == FALLBACK_MODEL


def test_local_backend_needs_no_cloud_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Base requirements only: local defaults and backend resolution still work."""
    _without_cloud_extra(monkeypatch)

    settings = resolve_settings(None, LLMBackend.LOCAL)

    assert settings.model_for(LLMBackend.LOCAL) == DEFAULT_MODEL
    assert settings.fallback_model_for(LLMBackend.LOCAL) == FALLBACK_MODEL
    assert settings.llm_backend is LLMBackend.LOCAL
    ensure_backend_ready(LLMBackend.LOCAL)


def test_api_backend_without_cloud_extra_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting the API backend on a base install fails with an actionable message."""
    _without_cloud_extra(monkeypatch)

    with pytest.raises(BackendConfigurationError, match=r"csv-inspector\[cloud\]"):
        resolve_settings(None, LLMBackend.API)


@needs_cloud_extra
def test_resolved_settings_read_the_configured_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured names are used per backend."""
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    monkeypatch.setenv("CLOUD_MODEL", "gemini-x")
    monkeypatch.setenv("CLOUD_FALLBACK_MODEL", "gemini-y")

    settings = resolve_settings(None, LLMBackend.API)

    assert settings.model_for(LLMBackend.LOCAL) == "llama3.1:8b"
    assert settings.model_for(LLMBackend.API) == "gemini-x"
    assert settings.fallback_model_for(LLMBackend.API) == "gemini-y"


@needs_cloud_extra
def test_configured_backend_follows_llm_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM_BACKEND selects the default backend; unset means local."""
    assert resolve_settings(None, LLMBackend.LOCAL).llm_backend is LLMBackend.LOCAL

    monkeypatch.setenv("LLM_BACKEND", "api")

    assert resolve_settings(None, LLMBackend.LOCAL).llm_backend is LLMBackend.API


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
    """The request uses JSON mode, the result schema and temperature 0.0."""
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
    assert config.response_json_schema == CSVInspectionResult.model_json_schema()
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
