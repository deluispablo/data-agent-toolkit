"""Unit tests for csv_inspector's environment-driven settings.

The autouse ``_isolated_settings`` fixture (``conftest.py``) clears every
settings variable and runs each test from an empty directory, so each test
sets exactly the environment it describes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydantic_settings", reason="settings need the cloud extra")

from csv_inspector import (
    BackendConfigurationError,
    CredentialsNotConfiguredError,
    LLMBackend,
    Settings,
    load_settings,
)
from csv_inspector._config import (
    DEFAULT_CLOUD_FALLBACK_MODEL,
    DEFAULT_CLOUD_MODEL,
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    CloudAuthMode,
)

FAKE_KEY = "AIza-fake-test-key-000"


def test_defaults_select_the_local_backend_with_built_in_models() -> None:
    """With nothing configured, settings mirror the library's built-in defaults."""
    settings = load_settings()

    assert settings.llm_backend is LLMBackend.LOCAL
    assert settings.ollama_model == DEFAULT_MODEL
    assert settings.ollama_fallback_model == FALLBACK_MODEL
    assert settings.cloud_model == DEFAULT_CLOUD_MODEL
    assert settings.cloud_fallback_model == DEFAULT_CLOUD_FALLBACK_MODEL
    assert settings.gemini_api_key is None


@pytest.mark.parametrize("backend", list(LLMBackend))
def test_every_backend_has_a_distinct_fallback_by_default(backend: LLMBackend) -> None:
    """Out of the box, a failed primary is retried with a different model.

    Regression test for issue #13: each default fallback equalled its primary,
    so it was skipped and only one model was ever tried.
    """
    settings = Settings()

    assert settings.fallback_model_for(backend) != settings.model_for(backend)


@pytest.mark.parametrize("raw", ["api", "API", " Api "])
def test_llm_backend_is_read_case_insensitively(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """LLM_BACKEND accepts any casing and surrounding whitespace."""
    monkeypatch.setenv("LLM_BACKEND", raw)

    assert load_settings().llm_backend is LLMBackend.API


def test_invalid_llm_backend_names_the_variable_without_echoing_the_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad value fails with a domain error naming the variable, not its content."""
    monkeypatch.setenv("LLM_BACKEND", "sentinel-bad-value")

    with pytest.raises(BackendConfigurationError, match="LLM_BACKEND") as exc_info:
        load_settings()

    assert "sentinel-bad-value" not in str(exc_info.value)


def test_model_names_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every model name is configurable through its own variable."""
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    monkeypatch.setenv("OLLAMA_FALLBACK_MODEL", "qwen3:8b")
    monkeypatch.setenv("CLOUD_MODEL", "gemini-x")
    monkeypatch.setenv("CLOUD_FALLBACK_MODEL", "gemini-y")

    settings = load_settings()

    assert (settings.ollama_model, settings.ollama_fallback_model) == ("llama3.1:8b", "qwen3:8b")
    assert (settings.cloud_model, settings.cloud_fallback_model) == ("gemini-x", "gemini-y")


def test_ollama_host_is_read_from_the_sdk_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """OLLAMA_HOST, the Ollama SDK's own variable, loads into Settings (issue #96)."""
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")

    assert load_settings().ollama_host == "http://ollama:11434"


def test_blank_ollama_host_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty OLLAMA_HOST leaves the SDK default in place."""
    monkeypatch.setenv("OLLAMA_HOST", " ")

    assert load_settings().ollama_host is None


def test_a_dotenv_in_the_working_directory_is_ignored_by_default(tmp_path: Path) -> None:
    """The library never reads ``./.env`` implicitly: the host's cwd is not trusted."""
    (tmp_path / ".env").write_text("LLM_BACKEND=api\nCLOUD_MODEL=gemini-from-dotenv\n")

    settings = load_settings()

    assert settings.llm_backend is LLMBackend.LOCAL
    assert settings.cloud_model == DEFAULT_CLOUD_MODEL


def test_a_dotenv_file_is_read_only_when_requested(tmp_path: Path) -> None:
    """``env_file=`` opts in to a specific file."""
    env_file = tmp_path / "secrets.env"
    env_file.write_text("LLM_BACKEND=api\nCLOUD_MODEL=gemini-from-dotenv\n")

    settings = load_settings(env_file=env_file)

    assert settings.llm_backend is LLMBackend.API
    assert settings.cloud_model == "gemini-from-dotenv"


def test_environment_variables_win_over_the_dotenv_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Process variables take precedence over values in the requested file."""
    env_file = tmp_path / "secrets.env"
    env_file.write_text("CLOUD_MODEL=from-file\n")
    monkeypatch.setenv("CLOUD_MODEL", "from-env")

    assert load_settings(env_file=env_file).cloud_model == "from-env"


def test_loaded_settings_are_plain_settings() -> None:
    """``load_settings`` returns a plain ``Settings``, which never reads the environment."""
    assert type(load_settings()) is Settings


def test_explicit_settings_ignore_the_environment_entirely(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Constructing ``Settings`` directly never looks at variables or files."""
    monkeypatch.setenv("LLM_BACKEND", "api")
    monkeypatch.setenv("CLOUD_MODEL", "from-env")
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    (tmp_path / ".env").write_text("OLLAMA_MODEL=from-dotenv\n")

    settings = Settings()

    assert settings.llm_backend is LLMBackend.LOCAL
    assert settings.cloud_model == DEFAULT_CLOUD_MODEL
    assert settings.ollama_model == DEFAULT_MODEL
    assert settings.gemini_api_key is None


def test_explicit_settings_reject_unknown_fields() -> None:
    """A typo in an injected setting fails loudly instead of being silently ignored."""
    with pytest.raises(ValueError, match="gemini_key"):
        Settings(gemini_key="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------
# Cloud credential resolution
# ---------------------------------------------------------------------


def test_api_key_selects_the_gemini_developer_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """An API key alone is sufficient and selects the Developer API."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)

    credentials = load_settings().cloud_credentials()

    assert credentials.mode is CloudAuthMode.GEMINI_API
    assert credentials.api_key is not None
    assert credentials.api_key.get_secret_value() == FAKE_KEY


def test_project_and_location_select_vertex_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Project plus location (no key) selects Vertex AI with ADC."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")

    credentials = load_settings().cloud_credentials()

    assert credentials.mode is CloudAuthMode.VERTEX_AI
    assert (credentials.project, credentials.location) == ("my-project", "europe-west1")
    assert credentials.api_key is None


def test_api_key_takes_precedence_over_vertex(monkeypatch: pytest.MonkeyPatch) -> None:
    """With both routes configured, the API key wins (documented precedence)."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")

    assert load_settings().cloud_credentials().mode is CloudAuthMode.GEMINI_API


@pytest.mark.parametrize(
    ("set_var", "missing_var"),
    [
        ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"),
        ("GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_PROJECT"),
    ],
)
def test_half_configured_vertex_names_the_missing_variable(
    monkeypatch: pytest.MonkeyPatch, set_var: str, missing_var: str
) -> None:
    """A lone project or location is reported as exactly what is missing."""
    monkeypatch.setenv(set_var, "value")

    with pytest.raises(CredentialsNotConfiguredError, match=f"{missing_var} is not set"):
        load_settings().cloud_credentials()


def test_no_credentials_lists_both_options() -> None:
    """With nothing set, the error explains both ways to configure credentials."""
    with pytest.raises(CredentialsNotConfiguredError) as exc_info:
        load_settings().cloud_credentials()

    message = str(exc_info.value)
    for variable in ("GEMINI_API_KEY", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
        assert variable in message


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_values_count_as_unset(monkeypatch: pytest.MonkeyPatch, blank: str) -> None:
    """``GEMINI_API_KEY=`` copied from .env.example must not count as a key."""
    monkeypatch.setenv("GEMINI_API_KEY", blank)

    with pytest.raises(CredentialsNotConfiguredError):
        load_settings().cloud_credentials()


def test_the_api_key_never_appears_in_repr_or_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The key is masked wherever settings or credentials might be logged."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    settings = load_settings()
    credentials = settings.cloud_credentials()

    for text in (repr(settings), str(settings), repr(credentials), credentials.describe()):
        assert FAKE_KEY not in text
