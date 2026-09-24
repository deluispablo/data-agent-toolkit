"""``ApiSettings``: environment prefix, limits, mapping to the library, secrets."""

from __future__ import annotations

from pathlib import Path

import csv_inspector
import pytest
from pydantic import SecretStr, ValidationError

from csv_inspector_api import create_app
from csv_inspector_api.settings import ApiSettings


def test_create_app_reads_settings_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without explicit settings, the factory reads CSV_INSPECTOR_API_* variables."""
    monkeypatch.setenv("CSV_INSPECTOR_API_OLLAMA_MODEL", "tiny:1b")
    monkeypatch.setenv("CSV_INSPECTOR_API_MAX_UPLOAD_BYTES", "1024")

    app = create_app()

    assert app.state.settings.max_upload_bytes == 1024
    assert app.state.library_settings.ollama_model == "tiny:1b"
    assert app.state.model_invoker is None


def test_defaults_match_the_library() -> None:
    """Unset settings keep the library defaults and the documented limits."""
    settings = ApiSettings()

    assert settings.to_library_settings() == csv_inspector.Settings()
    assert settings.default_timeout_seconds == 60
    assert settings.max_timeout_seconds == 300
    assert settings.max_upload_bytes == 256 * 1024 * 1024
    assert settings.allow_backend_override is False


def test_to_library_settings_passes_cloud_fields_through() -> None:
    """Cloud settings, secret included, reach the library settings unchanged."""
    settings = ApiSettings(
        llm_backend=csv_inspector.LLMBackend.API,
        cloud_model="gemini-x",
        cloud_fallback_model="gemini-x-lite",
        gemini_api_key=SecretStr("secret"),
        google_cloud_project="project",
        google_cloud_location="europe-west1",
        ollama_host="http://ollama:11434",
    )

    library = settings.to_library_settings()

    assert library.llm_backend is csv_inspector.LLMBackend.API
    assert library.cloud_model == "gemini-x"
    assert library.cloud_fallback_model == "gemini-x-lite"
    assert library.gemini_api_key is not None
    assert library.gemini_api_key.get_secret_value() == "secret"
    assert library.google_cloud_project == "project"
    assert library.google_cloud_location == "europe-west1"
    assert library.ollama_host == "http://ollama:11434"


def test_settings_are_frozen() -> None:
    """Settings cannot be changed after construction."""
    settings = ApiSettings()

    with pytest.raises(ValidationError):
        settings.max_upload_bytes = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"default_timeout_seconds": 0},
        {"max_timeout_seconds": -1},
        {"max_upload_bytes": 0},
        {"default_timeout_seconds": 301, "max_timeout_seconds": 300},
    ],
)
def test_invalid_settings_are_rejected(overrides: dict[str, float]) -> None:
    """Non-positive limits and a default timeout above the cap are rejected."""
    with pytest.raises(ValidationError):
        ApiSettings.model_validate(overrides)


def test_settings_read_the_prefixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only CSV_INSPECTOR_API_* variables are read; the library's own names are not."""
    monkeypatch.setenv("CSV_INSPECTOR_API_LLM_BACKEND", "api")
    monkeypatch.setenv("CSV_INSPECTOR_API_DEFAULT_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("CSV_INSPECTOR_API_ALLOW_BACKEND_OVERRIDE", "true")
    monkeypatch.setenv("OLLAMA_MODEL", "not-for-the-api:1b")

    settings = ApiSettings()

    assert settings.llm_backend is csv_inspector.LLMBackend.API
    assert settings.default_timeout_seconds == 12.5
    assert settings.allow_backend_override is True
    assert settings.ollama_model == "qwen2.5-coder:7b"


def test_secrets_are_not_in_repr() -> None:
    """The API key is masked in repr and str, of both the API and the library settings."""
    settings = ApiSettings(gemini_api_key=SecretStr("AIza-secret"))

    for text in (repr(settings), str(settings), repr(settings.to_library_settings())):
        assert "AIza-secret" not in text


def test_nothing_turns_the_backend_override_on_by_default() -> None:
    """The cost guard is opt-in by the operator only: no shipped file enables it."""
    root = Path(__file__).resolve().parents[1]
    shipped = [root / "main_demo.py", root / ".env.example", *sorted((root / "src").rglob("*.py"))]

    for path in shipped:
        text = path.read_text(encoding="utf-8").lower()
        assert "allow_backend_override=true" not in text.replace(" ", ""), path
