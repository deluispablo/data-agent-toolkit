"""Cloud Storage errors of ``POST /inspect/gcs``: every row of the table, through a fake client.

These tests need ``google-api-core`` and ``google-auth`` (the ``[gcs]``
extra) for the exception classes, and are skipped without them.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

import csv_inspector_api.routes.inspect as inspect_route
from csv_inspector_api import create_app
from csv_inspector_api.errors import PROBLEM_MEDIA_TYPE
from csv_inspector_api.settings import ApiSettings
from fakes import FakeClient, FakeInvoker

api = pytest.importorskip("google.api_core.exceptions")
auth = pytest.importorskip("google.auth.exceptions")

URI = "gs://my-bucket/exports/sales.csv"


def _rate_limited(retry_after: str | None) -> Exception:
    """A 429 as the SDK raises it, with the HTTP response it came from."""
    headers = {"Retry-After": retry_after} if retry_after else {}
    response = SimpleNamespace(headers=headers)
    error: Exception = api.TooManyRequests("rate limited", response=response)
    return error


TABLE: list[tuple[Exception, int]] = [
    (api.NotFound("No such object: my-bucket/exports/sales.csv"), 404),
    (api.NotFound("The specified bucket does not exist."), 404),
    (api.Forbidden("sa@p.iam.gserviceaccount.com does not have storage.objects.get"), 403),
    (api.Unauthorized("Invalid Credentials"), 503),
    (auth.DefaultCredentialsError("Your default credentials were not found."), 503),
    (auth.RefreshError("metadata server unreachable"), 503),
    (_rate_limited("30"), 429),
    (_rate_limited(None), 429),
    (api.TooManyRequests("rate limited"), 429),
    (api.InternalServerError("backend error"), 502),
    (api.ServiceUnavailable("try again"), 502),
    (api.BadRequest("Bucket is a requester pays bucket but no user project provided."), 502),
    (api.RetryError("Deadline of 120.0s exceeded", cause=api.ServiceUnavailable("x")), 502),
]
"""The issue's mapping table, one or more exceptions per row."""


@pytest.fixture
def gcs() -> FakeClient:
    """A fake client; each test scripts the error its reader raises."""
    return FakeClient()


@pytest.fixture
def app(settings: ApiSettings, invoker: FakeInvoker, gcs: FakeClient) -> FastAPI:
    """The application wired to the fake model and the fake Cloud Storage client."""
    return create_app(settings, model_invoker=invoker, gcs_client=gcs)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "status"), TABLE, ids=lambda v: type(v).__name__ if isinstance(v, Exception) else ""
)
async def test_every_gcs_error_is_mapped(  # noqa: PLR0913, PLR0917 - fixtures and parameters
    client: httpx.AsyncClient,
    gcs: FakeClient,
    invoker: FakeInvoker,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    status: int,
) -> None:
    """Each exception of the table becomes its status, as a problem body without SDK text."""
    gcs.error = error

    with caplog.at_level(logging.DEBUG, logger="csv_inspector_api"):
        response = await client.post("/inspect/gcs", json={"uri": URI})

    assert response.status_code == status
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    body = response.json()
    assert body["status"] == status
    assert body["error"] == type(error).__name__
    assert str(error) not in body["detail"]
    assert "my-bucket" not in response.text
    assert gcs.blobs[0].readers[0].closed
    assert invoker.calls == []
    (record,) = [r for r in caplog.records if r.name == "csv_inspector_api.errors"]
    assert record.levelno == (logging.ERROR if status == 503 else logging.WARNING)
    assert str(error) in record.getMessage()


@pytest.mark.anyio
async def test_missing_bucket_and_missing_object_look_the_same(
    client: httpx.AsyncClient, gcs: FakeClient
) -> None:
    """A client cannot tell a missing bucket from a missing object: no bucket enumeration."""
    gcs.error = api.NotFound("No such object: my-bucket/exports/sales.csv")
    missing_object = await client.post("/inspect/gcs", json={"uri": URI})
    gcs.error = api.NotFound("The specified bucket does not exist.")
    missing_bucket = await client.post("/inspect/gcs", json={"uri": URI})

    assert missing_object.status_code == missing_bucket.status_code == 404
    assert missing_object.json() == missing_bucket.json()


@pytest.mark.anyio
@pytest.mark.parametrize(("retry_after", "expected"), [("30", "30"), (None, None)])
async def test_retry_after_is_passed_through(
    client: httpx.AsyncClient, gcs: FakeClient, retry_after: str | None, expected: str | None
) -> None:
    """A 429 carries Cloud Storage's ``Retry-After`` when it sent one."""
    gcs.error = _rate_limited(retry_after)

    response = await client.post("/inspect/gcs", json={"uri": URI})

    assert response.status_code == 429
    assert response.headers.get("Retry-After") == expected


@pytest.fixture
async def unconfigured_client(
    settings: ApiSettings, invoker: FakeInvoker
) -> AsyncIterator[httpx.AsyncClient]:
    """A client for an app without a Cloud Storage client."""
    app = create_app(settings, model_invoker=invoker)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.mark.anyio
async def test_no_credentials_at_request_time_is_a_503(
    unconfigured_client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building the client without Application Default Credentials is a 503, not a 500."""

    def _no_adc(project: str | None) -> FakeClient:
        raise auth.DefaultCredentialsError("Your default credentials were not found.")

    monkeypatch.setattr(inspect_route, "create_client", _no_adc)

    response = await unconfigured_client.post("/inspect/gcs", json={"uri": URI})

    assert response.status_code == 503
    assert response.json()["error"] == "DefaultCredentialsError"


def test_handlers_are_skipped_without_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``google-api-core`` the app still builds, with no Google handler."""
    monkeypatch.setitem(sys.modules, "google.api_core", None)
    monkeypatch.setitem(sys.modules, "google.api_core.exceptions", None)

    app = create_app(ApiSettings(), model_invoker=FakeInvoker())

    handled = [key for key in app.exception_handlers if isinstance(key, type)]
    modules = {error_type.__module__ for error_type in handled}
    assert not {module for module in modules if module.startswith("google")}


@pytest.mark.anyio
async def test_openapi_documents_the_gcs_statuses(client: httpx.AsyncClient) -> None:
    """The route declares 404 and 429 besides the library's statuses."""
    schema = (await client.get("/openapi.json")).json()

    responses = schema["paths"]["/inspect/gcs"]["post"]["responses"]
    assert {"403", "404", "422", "429", "502", "503", "504"} <= set(responses)
    assert "Access to the object denied".lower() in responses["403"]["description"].lower()
