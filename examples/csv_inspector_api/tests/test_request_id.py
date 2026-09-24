"""``X-Request-ID``: generated or echoed, bound to every log record, one access line."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

import anyio
import httpx
import pytest
from starlette.types import Message, Receive, Scope, Send

from csv_inspector_api.request_id import (
    NO_REQUEST_ID,
    RequestIdFilter,
    RequestIdMiddleware,
    request_id_var,
)
from fakes import SAMPLE_CSV, FakeInvoker

SAMPLE = SAMPLE_CSV.read_bytes()


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """``caplog``, with the request-id filter on its capturing handler.

    As in ``main_demo.py``: the filter sits on the handler, so records of
    every logger it receives are annotated, the library's included.
    """
    request_filter = RequestIdFilter()
    caplog.handler.addFilter(request_filter)
    with caplog.at_level(logging.INFO):
        yield caplog
    caplog.handler.removeFilter(request_filter)


@pytest.mark.anyio
async def test_client_id_is_echoed_and_logged(
    client: httpx.AsyncClient, logs: pytest.LogCaptureFixture
) -> None:
    """The client's id comes back and is on the API's and the library's records."""
    response = await client.post(
        "/inspect", headers={"X-Request-ID": "req-42"}, files={"file": ("s.csv", SAMPLE)}
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-42"
    by_logger = {record.name: record for record in logs.records}
    library = [r for r in logs.records if r.name.split(".")[0] == "csv_inspector"]
    assert library, "the library logs its model attempts at INFO"
    assert {getattr(r, "request_id", None) for r in library} == {"req-42"}
    assert getattr(by_logger["csv_inspector_api.inspect"], "request_id", None) == "req-42"
    access = by_logger["csv_inspector_api.access"]
    assert getattr(access, "request_id", None) == "req-42"
    assert re.fullmatch(r"POST /inspect 200 \d+\.\d ms", access.getMessage())


@pytest.mark.anyio
async def test_id_is_generated_when_missing(
    client: httpx.AsyncClient, logs: pytest.LogCaptureFixture
) -> None:
    """Without a header, a fresh uuid4 is used, and each request gets its own."""
    first = await client.get("/health")
    second = await client.get("/health")

    ids = [first.headers["X-Request-ID"], second.headers["X-Request-ID"]]
    assert all(re.fullmatch(r"[0-9a-f]{32}", request_id) for request_id in ids)
    assert ids[0] != ids[1]
    access = [r for r in logs.records if r.name == "csv_inspector_api.access"]
    assert [getattr(r, "request_id", None) for r in access] == ids


@pytest.mark.anyio
@pytest.mark.parametrize("claimed", ["two words", "x" * 129, "café"])
async def test_unsafe_client_id_is_replaced(client: httpx.AsyncClient, claimed: str) -> None:
    """An id that could break a log line is replaced by a generated one."""
    response = await client.get(
        "/health", headers=[(b"X-Request-ID", claimed.encode("latin-1", "replace"))]
    )

    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["X-Request-ID"])


@pytest.mark.anyio
async def test_id_is_echoed_on_problem_responses(client: httpx.AsyncClient) -> None:
    """Error responses carry the id too, so a client can quote it in a report."""
    response = await client.post("/inspect/raw", headers={"X-Request-ID": "req-empty"}, content=b"")

    assert response.status_code == 422
    assert response.headers["X-Request-ID"] == "req-empty"


@pytest.mark.anyio
async def test_cancelled_request_is_logged_as_499(
    client: httpx.AsyncClient, invoker: FakeInvoker, logs: pytest.LogCaptureFixture
) -> None:
    """A request cancelled before its response still gets its access line."""
    invoker.delay = 30

    with anyio.move_on_after(0.2):
        await client.post(
            "/inspect", headers={"X-Request-ID": "gone"}, files={"file": ("s.csv", SAMPLE)}
        )

    (access,) = [r for r in logs.records if r.name == "csv_inspector_api.access"]
    assert access.getMessage().startswith("POST /inspect 499 ")
    assert getattr(access, "request_id", None) == "gone"


def test_filter_outside_a_request() -> None:
    """Records logged outside a request get the placeholder id, never dropped."""
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "message", None, None)

    assert RequestIdFilter().filter(record)
    assert getattr(record, "request_id", None) == NO_REQUEST_ID
    assert request_id_var.get() == NO_REQUEST_ID


@pytest.mark.anyio
async def test_non_http_scopes_pass_through() -> None:
    """Lifespan (and websocket) scopes reach the app untouched, without an id."""
    seen: list[str] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(request_id_var.get())

    async def receive() -> Message:  # pragma: no cover - not called
        return {"type": "lifespan.startup"}

    async def send(message: Message) -> None:  # pragma: no cover - not called
        pass

    await RequestIdMiddleware(app)({"type": "lifespan"}, receive, send)

    assert seen == [NO_REQUEST_ID]
