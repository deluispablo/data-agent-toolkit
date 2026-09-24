"""Shared fixtures: hermetic settings, no sockets, an app with a fake model, an ASGI client.

The suite doubles as a reference for testing a host that embeds the agent:
``create_app(settings, model_invoker=...)`` takes a fake async model
(``fakes.py``) that answers, fails, stalls or answers garbage, and an
``httpx.AsyncClient`` over ``httpx.ASGITransport`` calls the app in
process while a guard here fails any network connection. Notable checks:
``test_inspect_raw.py`` streams 20 MiB under ``tracemalloc`` and cancels a
request mid-body (the reader, not its timeout, releases the worker
thread); ``test_inspect_gcs.py`` reads through a fake Cloud Storage reader
that records every ``read`` and ``seek`` (only the two windows are read);
``test_gcs_errors.py`` covers the error table; ``test_errors.py`` fails when
a new library exception has no status; ``test_embedding_rules.py`` checks
with ``ast`` that only ``main_demo.py`` prints or configures logging and
that only the library's public API is imported.
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator
from contextvars import ContextVar
from typing import Any, NoReturn

import httpx
import pytest
from fastapi import FastAPI

from csv_inspector_api import create_app
from csv_inspector_api.settings import ApiSettings
from fakes import FakeInvoker

ENV_PREFIX = "CSV_INSPECTOR_API_"


@pytest.fixture
def anyio_backend() -> str:
    """Run the ``anyio``-marked tests on asyncio only."""
    return "asyncio"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every ``CSV_INSPECTOR_API_*`` variable, so no test sees the developer's."""
    for name in list(os.environ):
        if name.upper().startswith(ENV_PREFIX):
            monkeypatch.delenv(name)


_socketpair_in_progress: ContextVar[bool] = ContextVar("_socketpair_in_progress", default=False)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any attempt to open a network connection fail loudly.

    The ASGI client calls the app in-process, so a connection attempt means a
    test reached a real backend (Ollama, a cloud API) instead of the fake
    invoker. The one exception is ``socket.socketpair()``: the event loop
    needs it for its self-pipe, and on Windows it is built by connecting two
    loopback sockets.
    """
    original_connect = socket.socket.connect
    original_socketpair = socket.socketpair

    def _refuse(*args: object, **kwargs: object) -> NoReturn:
        msg = "network access in a hermetic test"
        raise RuntimeError(msg)

    def _connect(self: socket.socket, address: Any) -> None:
        if not _socketpair_in_progress.get():
            _refuse()
        original_connect(self, address)

    def _socketpair(*args: Any, **kwargs: Any) -> tuple[socket.socket, socket.socket]:
        token = _socketpair_in_progress.set(True)
        try:
            return original_socketpair(*args, **kwargs)
        finally:
            _socketpair_in_progress.reset(token)

    monkeypatch.setattr(socket.socket, "connect", _connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
    monkeypatch.setattr(socket, "socketpair", _socketpair)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)


@pytest.fixture
def invoker() -> FakeInvoker:
    """The fake model: answers :data:`fakes.SAMPLE_ANSWER` unless a test scripts it."""
    return FakeInvoker()


@pytest.fixture
def settings() -> ApiSettings:
    """API settings built in code, independent of the environment, with short time budgets."""
    return ApiSettings(default_timeout_seconds=2, max_timeout_seconds=5)


@pytest.fixture
def app(settings: ApiSettings, invoker: FakeInvoker) -> FastAPI:
    """The application wired to the fake model invoker."""
    return create_app(settings, model_invoker=invoker)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """An HTTP client that calls the app in-process through ASGI."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
