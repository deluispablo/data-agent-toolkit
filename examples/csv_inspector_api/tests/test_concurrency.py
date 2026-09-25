"""The inspection slots: at most N inspections at once, a bounded wait, then 503."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx
import pytest
from csv_inspector import ModelInvocationError
from fastapi import FastAPI
from pydantic import ValidationError

from csv_inspector_api import create_app
from csv_inspector_api.errors import PROBLEM_MEDIA_TYPE
from csv_inspector_api.settings import ApiSettings
from fakes import SAMPLE_CSV, FakeClient, GatedInvoker

SAMPLE = SAMPLE_CSV.read_bytes()
URI = "gs://my-bucket/sample.csv"
MAX_CONCURRENT = 2

Send = Callable[[httpx.AsyncClient], Awaitable[httpx.Response]]


@pytest.fixture
def gated() -> GatedInvoker:
    """A fake model that answers only once the test releases it."""
    return GatedInvoker()


@pytest.fixture
def settings() -> ApiSettings:
    """Two slots, and a queue timeout long enough for the tests that release in time."""
    return ApiSettings(
        default_timeout_seconds=5,
        max_timeout_seconds=5,
        max_concurrent_inspections=MAX_CONCURRENT,
        queue_timeout_seconds=5,
    )


@pytest.fixture
def app(settings: ApiSettings, gated: GatedInvoker) -> FastAPI:
    """The application wired to the gated model and a fake Cloud Storage object."""
    gcs = FakeClient({("my-bucket", "sample.csv"): SAMPLE})
    return create_app(settings, model_invoker=gated, gcs_client=gcs)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """An in-process client, with the app's lifespan running."""
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        yield client


def _upload(client: httpx.AsyncClient) -> Awaitable[httpx.Response]:
    return client.post("/inspect", files={"file": ("sample.csv", SAMPLE, "text/csv")})


def _raw(client: httpx.AsyncClient) -> Awaitable[httpx.Response]:
    return client.post("/inspect/raw", content=SAMPLE)


def _gcs(client: httpx.AsyncClient) -> Awaitable[httpx.Response]:
    return client.post("/inspect/gcs", json={"uri": URI})


async def _settle() -> None:
    """Give queued requests time to run as far as they can."""
    await asyncio.sleep(0.1)


def test_create_app_creates_the_slots(app: FastAPI) -> None:
    """One place builds the semaphore; it binds to the serving loop on first use."""
    slots = app.state.inspection_slots
    assert isinstance(slots, asyncio.Semaphore)
    assert not slots.locked()


@pytest.mark.anyio
async def test_at_most_n_inspections_run_at_once(
    client: httpx.AsyncClient, gated: GatedInvoker
) -> None:
    """Five requests over the three routes: two run, the others wait, then all succeed."""
    requests = [_upload(client), _raw(client), _gcs(client), _upload(client), _raw(client)]
    tasks = [asyncio.ensure_future(request) for request in requests]

    await gated.wait_for_in_flight(MAX_CONCURRENT)
    await _settle()
    assert gated.in_flight == MAX_CONCURRENT
    assert gated.max_in_flight == MAX_CONCURRENT  # the third and later never reached the model
    assert not any(task.done() for task in tasks)

    gated.release.set()
    responses = await asyncio.gather(*tasks)

    assert [response.status_code for response in responses] == [200] * 5
    assert all(response.json()["header_row_index"] == 2 for response in responses)
    assert gated.max_in_flight == MAX_CONCURRENT
    assert len(gated.calls) == 5


@pytest.mark.anyio
async def test_a_raw_body_waiting_for_a_slot_is_still_read_whole(
    client: httpx.AsyncClient, gated: GatedInvoker
) -> None:
    """The streamed body is only consumed once the request holds a slot."""
    busy = [asyncio.ensure_future(_upload(client)) for _ in range(MAX_CONCURRENT)]
    await gated.wait_for_in_flight(MAX_CONCURRENT)
    waiting = asyncio.ensure_future(_raw(client))
    await _settle()
    assert not waiting.done()

    gated.release.set()
    response = await waiting
    await asyncio.gather(*busy)

    assert response.status_code == 200
    assert response.json()["columns"][0] == "Fecha"
    assert "Fecha;Cliente" in gated.calls[-1][0]  # the prompt holds the sample's header


@pytest.mark.anyio
@pytest.mark.parametrize("send", [_upload, _raw, _gcs], ids=["inspect", "raw", "gcs"])
async def test_queue_timeout_is_a_503_problem_with_retry_after(
    app: FastAPI,
    gated: GatedInvoker,
    send: Send,
) -> None:
    """A request that finds no slot within the queue timeout gets 503 and Retry-After."""
    app.state.settings = app.state.settings.model_copy(update={"queue_timeout_seconds": 0.2})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        busy = [asyncio.ensure_future(_upload(client)) for _ in range(MAX_CONCURRENT)]
        await gated.wait_for_in_flight(MAX_CONCURRENT)

        response = await send(client)

        gated.release.set()
        assert [r.status_code for r in await asyncio.gather(*busy)] == [200] * MAX_CONCURRENT

    assert response.status_code == 503
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert response.headers["retry-after"] == "1"
    body = response.json()
    assert body["status"] == 503
    assert body["error"] == "ServerBusyError"
    assert "busy" in body["title"].lower()
    assert "2 inspection slots" in body["detail"]
    assert len(gated.calls) == MAX_CONCURRENT


@pytest.mark.anyio
async def test_retry_after_rounds_the_queue_timeout_up(app: FastAPI, gated: GatedInvoker) -> None:
    """``Retry-After`` is whole seconds: the queue timeout rounded up."""
    app.state.settings = app.state.settings.model_copy(update={"queue_timeout_seconds": 1.2})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        busy = [asyncio.ensure_future(_upload(client)) for _ in range(MAX_CONCURRENT)]
        await gated.wait_for_in_flight(MAX_CONCURRENT)
        response = await _upload(client)
        gated.release.set()
        await asyncio.gather(*busy)

    assert response.status_code == 503
    assert response.headers["retry-after"] == "2"


@pytest.mark.anyio
async def test_health_is_never_gated(client: httpx.AsyncClient, gated: GatedInvoker) -> None:
    """With every slot taken and a request queued, ``GET /health`` still answers."""
    busy = [asyncio.ensure_future(_upload(client)) for _ in range(MAX_CONCURRENT + 1)]
    await gated.wait_for_in_flight(MAX_CONCURRENT)

    response = await client.get("/health", params={"probe": "true"})

    assert response.status_code == 200
    gated.release.set()
    await asyncio.gather(*busy)


@pytest.mark.anyio
async def test_slots_are_released_when_inspections_fail(app: FastAPI, gated: GatedInvoker) -> None:
    """A model failure (502) frees its slot: more failures than slots never turn into 503."""
    gated.error = ModelInvocationError("model down")
    gated.release.set()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        responses = [await _upload(client) for _ in range(MAX_CONCURRENT + 2)]

    assert {response.status_code for response in responses} == {502}
    slots = app.state.inspection_slots
    assert isinstance(slots, asyncio.Semaphore)
    assert not slots.locked()


@pytest.mark.anyio
async def test_slot_is_released_when_the_request_is_cancelled(
    app: FastAPI, client: httpx.AsyncClient, gated: GatedInvoker
) -> None:
    """A client that goes away (the request task is cancelled) gives its slot back."""
    tasks = [asyncio.ensure_future(_upload(client)) for _ in range(MAX_CONCURRENT)]
    await gated.wait_for_in_flight(MAX_CONCURRENT)
    slots = app.state.inspection_slots
    assert slots.locked()

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await gated.wait_for_in_flight(0)

    assert not slots.locked()
    gated.release.set()
    assert (await _upload(client)).status_code == 200


@pytest.mark.anyio
async def test_slots_are_created_on_first_use_without_a_lifespan(
    app: FastAPI, gated: GatedInvoker
) -> None:
    """A host that skips lifespan events still gets the cap."""
    gated.release.set()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        assert (await _upload(client)).status_code == 200

    assert isinstance(app.state.inspection_slots, asyncio.Semaphore)


def test_openapi_documents_the_busy_503_on_inspection_routes_only(app: FastAPI) -> None:
    """The three inspection routes say a 503 may mean busy; ``/health`` does not."""
    paths = app.openapi()["paths"]
    for path in ("/inspect", "/inspect/raw", "/inspect/gcs"):
        assert "busy" in paths[path]["post"]["responses"]["503"]["description"]
    assert "busy" not in paths["/health"]["get"]["responses"]["503"]["description"]


def test_concurrency_defaults() -> None:
    """Four inspections at once, and a ten-second wait for a slot."""
    settings = ApiSettings()

    assert settings.max_concurrent_inspections == 4
    assert settings.queue_timeout_seconds == 10


def test_concurrency_settings_are_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both settings follow the ``CSV_INSPECTOR_API_`` prefix."""
    monkeypatch.setenv("CSV_INSPECTOR_API_MAX_CONCURRENT_INSPECTIONS", "1")
    monkeypatch.setenv("CSV_INSPECTOR_API_QUEUE_TIMEOUT_SECONDS", "2.5")

    settings = ApiSettings()

    assert settings.max_concurrent_inspections == 1
    assert settings.queue_timeout_seconds == 2.5


@pytest.mark.parametrize("field", ["max_concurrent_inspections", "queue_timeout_seconds"], ids=str)
@pytest.mark.parametrize("value", [0, -1])
def test_concurrency_settings_must_be_positive(field: str, value: int) -> None:
    """Zero slots would refuse everything, and a zero wait would never queue."""
    with pytest.raises(ValidationError, match=field):
        ApiSettings.model_validate({field: value})
