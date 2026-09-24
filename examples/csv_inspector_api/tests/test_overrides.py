"""Per-request ``backend``, ``model`` and ``fallback_model`` on both inspection routes."""

from __future__ import annotations

import logging
from typing import Any

import csv_inspector
import httpx
import pytest

from csv_inspector_api import create_app
from csv_inspector_api.errors import PROBLEM_MEDIA_TYPE
from csv_inspector_api.settings import ApiSettings
from fakes import SAMPLE_CSV, FakeInvoker

SAMPLE = SAMPLE_CSV.read_bytes()
CLOUD_DEFAULTS = csv_inspector.Settings()


async def _post(client: httpx.AsyncClient, route: str, params: dict[str, str]) -> httpx.Response:
    """POST the sample to ``route``, as a multipart upload or as the raw body."""
    if route == "/inspect":
        return await client.post(route, params=params, files={"file": ("s.csv", SAMPLE)})
    return await client.post(route, params=params, content=SAMPLE)


def _client(invoker: FakeInvoker, **settings: Any) -> httpx.AsyncClient:
    """A client for an app built with ``settings``."""
    app = create_app(ApiSettings(**settings), model_invoker=invoker)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


ROUTES = pytest.mark.parametrize("route", ["/inspect", "/inspect/raw"])


@pytest.mark.anyio
@ROUTES
async def test_model_override_reaches_the_model(
    client: httpx.AsyncClient,
    invoker: FakeInvoker,
    caplog: pytest.LogCaptureFixture,
    route: str,
) -> None:
    """The requested model is the one called, and the one logged."""
    with caplog.at_level(logging.INFO, logger="csv_inspector_api.inspect"):
        response = await _post(client, route, {"model": "tiny:1b"})

    assert response.status_code == 200
    assert [model for _, model in invoker.calls] == ["tiny:1b"]
    assert "local/tiny:1b" in caplog.records[-1].getMessage()


@pytest.mark.anyio
@ROUTES
async def test_fallback_model_override_is_tried_second(
    client: httpx.AsyncClient, invoker: FakeInvoker, route: str
) -> None:
    """When the primary fails, the requested fallback model is tried."""
    invoker.error = ConnectionError("down")

    response = await _post(client, route, {"model": "a:1b", "fallback_model": "b:1b"})

    assert response.status_code == 502
    assert [model for _, model in invoker.calls] == ["a:1b", "b:1b"]


@pytest.mark.anyio
@ROUTES
async def test_cloud_backend_is_refused_by_default(invoker: FakeInvoker, route: str) -> None:
    """``backend=api`` is a 403 problem unless the deployment opts in; no model is called."""
    async with _client(invoker) as client:
        response = await _post(client, route, {"backend": "api"})

    assert response.status_code == 403
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    body = response.json()
    assert body["title"] == "Backend override disabled"
    assert body["error"] == "BackendOverrideDisabledError"
    assert "CSV_INSPECTOR_API_ALLOW_BACKEND_OVERRIDE" in body["detail"]
    assert invoker.calls == []


@pytest.mark.anyio
@ROUTES
async def test_cloud_backend_when_allowed(invoker: FakeInvoker, route: str) -> None:
    """With the opt-in, ``backend=api`` uses the configured cloud models."""
    async with _client(invoker, allow_backend_override=True) as client:
        response = await _post(client, route, {"backend": "api"})

    assert response.status_code == 200
    assert [model for _, model in invoker.calls] == [CLOUD_DEFAULTS.cloud_model]


@pytest.mark.anyio
async def test_local_backend_is_always_allowed(invoker: FakeInvoker) -> None:
    """Switching a cloud deployment to the free local backend needs no opt-in."""
    async with _client(invoker, llm_backend=csv_inspector.LLMBackend.API) as client:
        response = await _post(client, "/inspect", {"backend": "local"})

    assert response.status_code == 200
    assert [model for _, model in invoker.calls] == [CLOUD_DEFAULTS.ollama_model]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "params",
    [
        {"backend": "paid"},
        {"model": ""},
        {"model": "bad name"},
        {"fallback_model": "line\nbreak"},
        {"model": "m" * 201},
    ],
)
async def test_invalid_overrides_are_422(
    client: httpx.AsyncClient, invoker: FakeInvoker, params: dict[str, str]
) -> None:
    """Unknown backends and model names that are not plain tokens are FastAPI 422s."""
    response = await _post(client, "/inspect/raw", params)

    assert response.status_code == 422
    assert invoker.calls == []


@pytest.mark.anyio
async def test_openapi_documents_the_overrides(client: httpx.AsyncClient) -> None:
    """Both routes document the three parameters and the 403."""
    paths = (await client.get("/openapi.json")).json()["paths"]

    for route in ("/inspect", "/inspect/raw"):
        operation = paths[route]["post"]
        names = {parameter["name"] for parameter in operation["parameters"]}
        assert {"backend", "model", "fallback_model"} <= names
        assert operation["responses"]["403"]["description"] == "Backend override disabled"


@pytest.mark.anyio
async def test_explicit_api_backend_on_a_cloud_deployment_is_allowed(invoker: FakeInvoker) -> None:
    """``backend=api`` on a deployment already on the cloud backend changes nothing: allowed."""
    async with _client(invoker, llm_backend=csv_inspector.LLMBackend.API) as client:
        response = await _post(client, "/inspect/raw", {"backend": "api"})

    assert response.status_code == 200
    assert [model for _, model in invoker.calls] == [CLOUD_DEFAULTS.cloud_model]


@pytest.mark.anyio
@ROUTES
@pytest.mark.parametrize(
    ("configured", "params"),
    [
        (csv_inspector.LLMBackend.API, {"model": "gemini-pro-x"}),
        (csv_inspector.LLMBackend.API, {"fallback_model": "gemini-pro-x"}),
        (csv_inspector.LLMBackend.LOCAL, {"backend": "api", "model": "gemini-pro-x"}),
    ],
)
async def test_cloud_model_choice_is_refused_by_default(
    invoker: FakeInvoker,
    route: str,
    configured: csv_inspector.LLMBackend,
    params: dict[str, str],
) -> None:
    """Picking the models of a cloud call (possibly pricier ones) needs the opt-in too."""
    async with _client(invoker, llm_backend=configured) as client:
        response = await _post(client, route, params)

    assert response.status_code == 403
    assert response.json()["error"] == "BackendOverrideDisabledError"
    assert invoker.calls == []


@pytest.mark.anyio
async def test_cloud_model_choice_when_allowed(invoker: FakeInvoker) -> None:
    """With the opt-in, a caller may pick the cloud models."""
    async with _client(invoker, allow_backend_override=True) as client:
        response = await _post(client, "/inspect", {"backend": "api", "model": "gemini-pro-x"})

    assert response.status_code == 200
    assert [model for _, model in invoker.calls] == ["gemini-pro-x"]
