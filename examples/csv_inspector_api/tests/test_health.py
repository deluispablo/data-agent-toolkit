"""``GET /health``: cheap, secret-free, never calls the model."""

from __future__ import annotations

import csv_inspector
import httpx
import pytest
from pydantic import SecretStr

from csv_inspector_api import __version__, create_app
from csv_inspector_api.errors import PROBLEM_MEDIA_TYPE
from csv_inspector_api.settings import ApiSettings
from fakes import FakeInvoker


@pytest.mark.anyio
async def test_reports_the_local_configuration(
    client: httpx.AsyncClient, invoker: FakeInvoker
) -> None:
    """The default deployment reports the local backend and models, without a model call."""
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "csv_inspector_version": csv_inspector.__version__,
        "api_version": __version__,
        "backend": "local",
        "model": "qwen2.5-coder:7b",
        "fallback_model": "qwen2.5-coder:7b",
    }
    assert invoker.calls == []


@pytest.mark.anyio
async def test_reports_cloud_models_without_secrets() -> None:
    """With the cloud backend, the models are reported; key, project and location are not."""
    settings = ApiSettings(
        llm_backend=csv_inspector.LLMBackend.API,
        cloud_model="gemini-x",
        cloud_fallback_model="gemini-x-lite",
        gemini_api_key=SecretStr("AIza-secret"),
        google_cloud_project="my-project",
        google_cloud_location="europe-west1",
    )
    transport = httpx.ASGITransport(app=create_app(settings, model_invoker=FakeInvoker()))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")

    body = response.json()
    assert (body["backend"], body["model"], body["fallback_model"]) == (
        "api",
        "gemini-x",
        "gemini-x-lite",
    )
    for private in ("AIza-secret", "my-project", "europe-west1"):
        assert private not in response.text


@pytest.mark.anyio
async def test_openapi_declares_the_health_response(client: httpx.AsyncClient) -> None:
    """The route is tagged ``meta`` and documented with HealthResponse."""
    operation = (await client.get("/openapi.json")).json()["paths"]["/health"]["get"]

    assert operation["tags"] == ["meta"]
    schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema == {"$ref": "#/components/schemas/HealthResponse"}


async def _get_health(app_settings: ApiSettings, invoker: FakeInvoker | None) -> httpx.Response:
    """GET ``/health?probe=true`` on an app built with these settings and invoker."""
    transport = httpx.ASGITransport(app=create_app(app_settings, model_invoker=invoker))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.get("/health", params={"probe": "true"})


@pytest.mark.anyio
async def test_probe_passes_on_the_local_backend() -> None:
    """The local backend needs no configuration, so the probe passes without a model call."""
    response = await _get_health(ApiSettings(), None)

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.anyio
async def test_probe_reports_a_misconfigured_cloud_backend() -> None:
    """The cloud backend without credentials is a 503 problem; nothing is contacted."""
    settings = ApiSettings(llm_backend=csv_inspector.LLMBackend.API)

    response = await _get_health(settings, None)

    assert response.status_code == 503
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert response.json()["error"] == "CredentialsNotConfiguredError"


@pytest.mark.anyio
async def test_probe_passes_on_a_configured_cloud_backend() -> None:
    """With a key the cloud backend is ready: the probe checks configuration, not the network."""
    settings = ApiSettings(llm_backend=csv_inspector.LLMBackend.API, gemini_api_key=SecretStr("k"))

    response = await _get_health(settings, None)

    assert response.status_code == 200


@pytest.mark.anyio
async def test_probe_is_skipped_with_a_custom_invoker() -> None:
    """A custom invoker replaces the backend, so its configuration is not checked."""
    settings = ApiSettings(llm_backend=csv_inspector.LLMBackend.API)

    response = await _get_health(settings, FakeInvoker())

    assert response.status_code == 200


@pytest.mark.anyio
async def test_default_health_does_not_probe() -> None:
    """Without probe=true a misconfigured cloud backend still reports ok (liveness only)."""
    app = create_app(ApiSettings(llm_backend=csv_inspector.LLMBackend.API))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")

    assert response.status_code == 200
