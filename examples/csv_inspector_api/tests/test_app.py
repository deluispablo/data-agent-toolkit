"""The scaffold: the app boots, serves its OpenAPI schema, and maps its settings."""

from __future__ import annotations

import socket

import csv_inspector
import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr, ValidationError

from csv_inspector_api import __version__, create_app
from csv_inspector_api.settings import ApiSettings
from fakes import FakeInvoker


def test_public_api() -> None:
    """The package exports only the app factory and its version."""
    import csv_inspector_api  # noqa: PLC0415

    assert csv_inspector_api.__all__ == ["__version__", "create_app"]


@pytest.mark.anyio
async def test_openapi_is_served(client: httpx.AsyncClient) -> None:
    """The app boots and serves its OpenAPI schema with the package version."""
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    info = response.json()["info"]
    assert info["title"] == "csv-inspector API"
    assert info["version"] == __version__


@pytest.mark.anyio
async def test_docs_are_served(client: httpx.AsyncClient) -> None:
    """The interactive documentation page is served."""
    response = await client.get("/docs")

    assert response.status_code == 200


def test_create_app_stores_settings_and_invoker(
    app: FastAPI, settings: ApiSettings, invoker: FakeInvoker
) -> None:
    """The factory stores both settings and the model invoker on app.state."""
    assert app.state.settings is settings
    assert app.state.library_settings == settings.to_library_settings()
    assert app.state.model_invoker is invoker


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


def test_to_library_settings_passes_cloud_fields_through() -> None:
    """Cloud settings, secret included, reach the library settings unchanged."""
    settings = ApiSettings(
        llm_backend=csv_inspector.LLMBackend.API,
        cloud_model="gemini-x",
        cloud_fallback_model="gemini-x-lite",
        gemini_api_key=SecretStr("secret"),
        google_cloud_project="project",
        google_cloud_location="europe-west1",
    )

    library = settings.to_library_settings()

    assert library.llm_backend is csv_inspector.LLMBackend.API
    assert library.cloud_model == "gemini-x"
    assert library.cloud_fallback_model == "gemini-x-lite"
    assert library.gemini_api_key is not None
    assert library.gemini_api_key.get_secret_value() == "secret"
    assert library.google_cloud_project == "project"
    assert library.google_cloud_location == "europe-west1"


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


def test_network_is_blocked() -> None:
    """The hermetic guard refuses outgoing connections."""
    with pytest.raises(RuntimeError, match="network access"):
        socket.create_connection(("127.0.0.1", 11434))
    with socket.socket() as sock, pytest.raises(RuntimeError, match="network access"):
        sock.connect(("127.0.0.1", 11434))


def test_socketpair_still_works() -> None:
    """The guard leaves socket.socketpair() usable for the event loop."""
    left, right = socket.socketpair()
    with left, right:
        left.sendall(b"x")
        assert right.recv(1) == b"x"
