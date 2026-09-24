"""The error contract: every library error maps to a status and a problem body."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import csv_inspector
import httpx
import pytest
from csv_inspector import CSVInspectorError
from fastapi import FastAPI

from csv_inspector_api import create_app
from csv_inspector_api.errors import PROBLEM_MEDIA_TYPE, status_for
from csv_inspector_api.settings import ApiSettings
from fakes import FakeInvoker

EXPECTED_STATUS = {
    "EmptySampleError": 422,
    "FileSampleReadError": 422,
    "InspectionTimeoutError": 504,
    "CredentialsNotConfiguredError": 503,
    "BackendConfigurationError": 503,
    "InspectionFailedError": 502,
    "ModelInvocationError": 502,
    "ModelTimeoutError": 502,
    "ResponseParsingError": 502,
    "SchemaValidationError": 502,
    "CSVInspectorError": 500,
}
"""Expected status of every exception the library exports; a new one must be added here."""


def _exported_errors() -> list[type[CSVInspectorError]]:
    """Every ``CSVInspectorError`` subclass in ``csv_inspector.__all__``."""
    exported = (getattr(csv_inspector, name) for name in csv_inspector.__all__)
    return [obj for obj in exported if isinstance(obj, type) and issubclass(obj, CSVInspectorError)]


def _make(error_type: type[CSVInspectorError], message: str) -> CSVInspectorError:
    """Instantiate a library error, whatever its constructor needs."""
    if issubclass(error_type, csv_inspector.InspectionFailedError):
        return error_type(message, attempts={})
    return error_type(message)


@pytest.mark.parametrize("error_type", _exported_errors(), ids=lambda t: t.__name__)
def test_every_exported_error_is_mapped(error_type: type[CSVInspectorError]) -> None:
    """Each exported library error has the documented status."""
    assert error_type.__name__ in EXPECTED_STATUS, f"map {error_type.__name__} to a status"
    assert status_for(_make(error_type, "boom")) == EXPECTED_STATUS[error_type.__name__]


def test_expected_table_has_no_stale_entries() -> None:
    """The table lists only exceptions the library still exports."""
    assert set(EXPECTED_STATUS) == {t.__name__ for t in _exported_errors()}


def test_unknown_subclass_falls_back_to_500() -> None:
    """A library error outside the table is a 500."""

    class NewError(CSVInspectorError):
        pass

    assert status_for(NewError("new")) == 500


_to_raise: list[Exception] = []


async def _raise() -> None:
    """Route body: raise whatever the test queued."""
    raise _to_raise.pop()


@pytest.fixture
async def raising_client() -> AsyncIterator[httpx.AsyncClient]:
    """A client for an app with one route that raises the queued exception."""
    app: FastAPI = create_app(ApiSettings(), model_invoker=FakeInvoker())
    app.add_api_route("/boom", _raise)
    # Let unhandled errors become responses, as they would under a server.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.mark.anyio
@pytest.mark.parametrize("error_type", _exported_errors(), ids=lambda t: t.__name__)
async def test_errors_become_problem_responses(
    raising_client: httpx.AsyncClient, error_type: type[CSVInspectorError]
) -> None:
    """The handler answers with an RFC 9457 problem body naming the exception."""
    _to_raise.append(_make(error_type, "something broke"))

    response = await raising_client.get("/boom")

    status = EXPECTED_STATUS[error_type.__name__]
    assert response.status_code == status
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    body = response.json()
    assert body["type"] == "about:blank"
    assert body["status"] == status
    assert body["detail"] == "something broke"
    assert body["error"] == error_type.__name__
    assert body["title"]


@pytest.mark.anyio
async def test_backend_misconfiguration_says_it_is_the_server(
    raising_client: httpx.AsyncClient,
) -> None:
    """A 503 tells the client the deployment, not its request, is at fault."""
    _to_raise.append(csv_inspector.CredentialsNotConfiguredError("no key"))

    response = await raising_client.get("/boom")

    assert response.status_code == 503
    assert "misconfigured on the server" in response.json()["title"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "level", "with_traceback"),
    [
        (csv_inspector.EmptySampleError("empty"), logging.WARNING, False),
        (csv_inspector.InspectionTimeoutError("slow", attempts={}), logging.WARNING, False),
        (csv_inspector.InspectionFailedError("bad", attempts={}), logging.WARNING, False),
        (csv_inspector.BackendConfigurationError("no sdk"), logging.ERROR, False),
        (CSVInspectorError("odd"), logging.ERROR, True),
    ],
)
async def test_errors_are_logged_once_at_the_right_level(
    raising_client: httpx.AsyncClient,
    caplog: pytest.LogCaptureFixture,
    error: CSVInspectorError,
    level: int,
    with_traceback: bool,
) -> None:
    """Client and model failures warn, server faults are errors, only 500 has a traceback."""
    _to_raise.append(error)

    with caplog.at_level(logging.DEBUG, logger="csv_inspector_api"):
        await raising_client.get("/boom")

    records = [r for r in caplog.records if r.name == "csv_inspector_api.errors"]
    assert len(records) == 1
    assert records[0].levelno == level
    assert (records[0].exc_info is not None) is with_traceback


@pytest.mark.anyio
@pytest.mark.parametrize("error", [ValueError("bad budget"), TypeError("text stream")])
async def test_host_bugs_are_not_mapped(
    raising_client: httpx.AsyncClient, error: Exception
) -> None:
    """ValueError and TypeError are host bugs: plain 500, no problem body."""
    _to_raise.append(error)

    response = await raising_client.get("/boom")

    assert response.status_code == 500
    assert response.headers["content-type"] != PROBLEM_MEDIA_TYPE
