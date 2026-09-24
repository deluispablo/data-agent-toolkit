"""The app factory: it boots, serves its OpenAPI schema, and keeps the network away."""

from __future__ import annotations

import socket

import httpx
import pytest
from fastapi import FastAPI

from csv_inspector_api import __version__
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
