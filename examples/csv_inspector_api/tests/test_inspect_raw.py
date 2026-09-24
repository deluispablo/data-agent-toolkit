"""``POST /inspect/raw``: a streamed body, bounded memory, cancellation."""

from __future__ import annotations

import asyncio
import io
import threading
import time
import tracemalloc
from collections.abc import AsyncIterator
from typing import Any

import anyio
import csv_inspector
import httpx
import pytest

import csv_inspector_api.routes.inspect as inspect_route
from csv_inspector_api import create_app
from csv_inspector_api.errors import PROBLEM_MEDIA_TYPE
from csv_inspector_api.settings import ApiSettings
from csv_inspector_api.streaming import AsyncIteratorReader, ReaderClosedError
from fakes import SAMPLE_CSV, FakeInvoker

SAMPLE = SAMPLE_CSV.read_bytes()
OCTET_STREAM = {"Content-Type": "application/octet-stream"}
CHUNK_BYTES = 64 * 1024
ROW = "2024-01-20;Cliente;Descripción;100.00;Nada\n".encode()


async def _chunks(data: bytes, size: int = 1000) -> AsyncIterator[bytes]:
    """``data`` as an async iterator: httpx then sends it chunked, without Content-Length."""
    for start in range(0, len(data), size):
        yield data[start : start + size]


async def _generated_body(total: int) -> AsyncIterator[bytes]:
    """The sample, then fresh 64 KiB chunks of data rows up to ``total`` bytes.

    Each chunk is a new object, as a server would receive it: nothing of the
    body exists in memory before it is sent. When the second chunk is asked
    for, the library has read and decoded its head: the ``tracemalloc`` peak
    is reset there, so what is measured is the stream flowing through the
    reader, not the one-off encoding detection of the 4 KiB head.
    """
    yield SAMPLE
    sent = len(SAMPLE)
    rows_per_chunk = CHUNK_BYTES // len(ROW)
    chunks_sent = 0
    while sent < total:
        if chunks_sent == 1 and tracemalloc.is_tracing():
            tracemalloc.reset_peak()
        sent += len(ROW) * rows_per_chunk
        chunks_sent += 1
        yield ROW * rows_per_chunk  # no local: the generator never holds a sent chunk


async def _stalled_body(stalled: asyncio.Event) -> AsyncIterator[bytes]:
    """The sample, then a sender that never sends again (a stalled or vanished client)."""
    yield SAMPLE * 20  # past the head window, so the library goes on to the tail
    stalled.set()
    await asyncio.Event().wait()
    yield b""  # pragma: no cover - never reached


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record the keyword arguments the route passes to ``ainspect_csv``."""
    calls: list[dict[str, Any]] = []
    real = csv_inspector.ainspect_csv

    async def _spy(source: Any, /, **kwargs: Any) -> csv_inspector.CSVInspectionResult:
        calls.append({"source": source, **kwargs})
        return await real(source, **kwargs)

    monkeypatch.setattr(inspect_route, "ainspect_csv", _spy)
    return calls


@pytest.mark.anyio
async def test_inspects_the_body(
    client: httpx.AsyncClient, invoker: FakeInvoker, spy: list[dict[str, Any]]
) -> None:
    """The sample sent as the body comes back inspected, from a non-seekable stream."""
    response = await client.post("/inspect/raw", content=SAMPLE, headers=OCTET_STREAM)

    assert response.status_code == 200
    body = response.json()
    assert body["delimiter"] == ";"
    assert body["header_row_index"] == 2
    assert [model for _, model in invoker.calls] == ["qwen2.5-coder:7b"]
    (call,) = spy
    assert isinstance(call["source"], AsyncIteratorReader)
    assert not call["source"].seekable()
    assert call["source"].closed


@pytest.mark.anyio
async def test_chunked_body_without_content_length(client: httpx.AsyncClient) -> None:
    """A body streamed in small chunks, with no Content-Length, is inspected."""
    request = client.build_request(
        "POST", "/inspect/raw", content=_chunks(SAMPLE), headers={"Content-Type": "text/csv"}
    )
    assert "content-length" not in request.headers

    response = await client.send(request)

    assert response.status_code == 200
    assert response.json()["header_row_index"] == 2


@pytest.mark.anyio
async def test_shares_the_query_parameters(
    client: httpx.AsyncClient, settings: ApiSettings, spy: list[dict[str, Any]]
) -> None:
    """Defaults and explicit values reach the library as for ``/inspect``."""
    await client.post("/inspect/raw", content=SAMPLE)
    await client.post(
        "/inspect/raw",
        params={"n_bytes": 512, "tail_bytes": 0, "timeout_seconds": 5},
        content=SAMPLE,
    )

    default, explicit = spy
    assert (default["n_bytes"], default["tail_bytes"]) == (4096, 4096)
    assert default["timeout_seconds"] == settings.default_timeout_seconds
    assert default["settings"] == settings.to_library_settings()
    assert (explicit["n_bytes"], explicit["tail_bytes"], explicit["timeout_seconds"]) == (
        512,
        0,
        5,
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "params", [{"n_bytes": 511}, {"tail_bytes": 16385}, {"timeout_seconds": 6}]
)
async def test_query_out_of_bounds_is_rejected(
    client: httpx.AsyncClient, invoker: FakeInvoker, params: dict[str, int]
) -> None:
    """The bounds are the same as ``/inspect``'s: FastAPI 422, no model call."""
    response = await client.post("/inspect/raw", params=params, content=SAMPLE)

    assert response.status_code == 422
    assert invoker.calls == []


@pytest.mark.anyio
async def test_empty_body_is_422(client: httpx.AsyncClient, invoker: FakeInvoker) -> None:
    """An empty body is an EmptySampleError problem."""
    response = await client.post("/inspect/raw", content=b"", headers=OCTET_STREAM)

    assert response.status_code == 422
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert response.json()["error"] == "EmptySampleError"
    assert invoker.calls == []


def _limited_client(max_upload_bytes: int, invoker: FakeInvoker) -> httpx.AsyncClient:
    """A client for an app with a small upload limit."""
    app = create_app(ApiSettings(max_upload_bytes=max_upload_bytes), model_invoker=invoker)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


@pytest.mark.anyio
async def test_declared_oversize_body_is_413(invoker: FakeInvoker) -> None:
    """A Content-Length over the limit is refused before the body is read."""
    async with _limited_client(len(SAMPLE) - 1, invoker) as client:
        response = await client.post("/inspect/raw", content=SAMPLE)

    assert response.status_code == 413
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert response.json()["error"] == "UploadTooLargeError"
    assert str(len(SAMPLE)) in response.json()["detail"]
    assert invoker.calls == []


@pytest.mark.anyio
async def test_streamed_oversize_body_is_413(invoker: FakeInvoker) -> None:
    """Without Content-Length, the bytes counted while streaming trigger the 413."""
    limit = 10_000
    async with _limited_client(limit, invoker) as client:
        response = await client.post("/inspect/raw", content=_chunks(SAMPLE * 100))

    assert response.status_code == 413
    assert response.json()["error"] == "UploadTooLargeError"
    assert f"{limit}-byte limit" in response.json()["detail"]
    assert invoker.calls == []


@pytest.mark.anyio
async def test_body_at_the_limit_is_accepted(invoker: FakeInvoker) -> None:
    """A streamed body of exactly max_upload_bytes is inspected."""
    async with _limited_client(len(SAMPLE), invoker) as client:
        response = await client.post("/inspect/raw", content=_chunks(SAMPLE))

    assert response.status_code == 200


@pytest.mark.anyio
async def test_broken_body_is_422(client: httpx.AsyncClient, invoker: FakeInvoker) -> None:
    """A body that fails mid-stream is a FileSampleReadError problem."""

    async def _broken() -> AsyncIterator[bytes]:
        yield SAMPLE * 20
        raise ConnectionResetError("connection lost")

    response = await client.post("/inspect/raw", content=_broken())

    assert response.status_code == 422
    assert response.json()["error"] == "FileSampleReadError"
    assert invoker.calls == []


@pytest.mark.anyio
async def test_memory_stays_bounded_on_a_20_mib_body(invoker: FakeInvoker) -> None:
    """Streaming 20 MiB through the reader holds only the windows and about one chunk.

    ``tracemalloc`` traces the reader and the library's sampling of it, from
    the end of the head (see :func:`_generated_body`) to the model call, i.e.
    the whole forward scan of the body and the decoding of the tail.
    """
    n_bytes = tail_bytes = 4096
    total = 20 * 1024 * 1024
    loop = asyncio.get_running_loop()
    peak_at_model_call: list[int] = []

    async def _measuring_invoker(prompt: str, model: str) -> str:
        peak_at_model_call.append(tracemalloc.get_traced_memory()[1])
        return await invoker(prompt, model)

    async def _inspect(body: AsyncIterator[bytes]) -> AsyncIteratorReader:
        reader = AsyncIteratorReader(body, loop, max_bytes=total * 2, read_timeout_seconds=5)
        await csv_inspector.ainspect_csv(
            reader, n_bytes=n_bytes, tail_bytes=tail_bytes, model_invoker=_measuring_invoker
        )
        return reader

    await _inspect(_chunks(SAMPLE))  # warm up: imports and caches are not the reader's
    tracemalloc.start()
    try:
        reader = await _inspect(_generated_body(total))
    finally:
        tracemalloc.stop()

    assert reader.bytes_read >= total
    assert peak_at_model_call[-1] < 2 * (n_bytes + tail_bytes + CHUNK_BYTES)


class _TrackedReader(io.RawIOBase):
    """Passes reads through to the route's reader, recording when one fails."""

    def __init__(self, inner: AsyncIteratorReader) -> None:
        super().__init__()
        self.inner = inner
        self.failed = threading.Event()
        self.error: OSError | None = None

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1, /) -> bytes:
        try:
            return self.inner.read(size)
        except OSError as exc:
            self.error = exc
            self.failed.set()
            raise


@pytest.mark.anyio
async def test_client_disconnect_releases_the_worker_thread(
    client: httpx.AsyncClient,
    invoker: FakeInvoker,
    settings: ApiSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client gone mid-body cancels the request, and the blocked read returns at once.

    The library's worker thread waits in ``read()`` for a chunk that never
    comes. Cancelling the request closes the reader, which must fail that
    read well before the read timeout (the request's time budget) would.
    """
    tracked: list[_TrackedReader] = []
    real = csv_inspector.ainspect_csv

    async def _tracking(source: Any, /, **kwargs: Any) -> csv_inspector.CSVInspectionResult:
        tracked.append(_TrackedReader(source))
        return await real(tracked[-1], **kwargs)

    monkeypatch.setattr(inspect_route, "ainspect_csv", _tracking)
    stalled = asyncio.Event()
    request = client.build_request("POST", "/inspect/raw", content=_stalled_body(stalled))

    with anyio.move_on_after(0.5) as scope:
        await client.send(request)
    assert scope.cancelled_caught
    assert stalled.is_set()

    (reader,) = tracked
    started = time.monotonic()
    assert await asyncio.to_thread(reader.failed.wait, settings.default_timeout_seconds)
    assert time.monotonic() - started < settings.default_timeout_seconds / 2
    assert isinstance(reader.error, ReaderClosedError)
    assert reader.inner.closed
    assert invoker.calls == []


class _Chunks:
    """An async iterator of scripted chunks or errors, with a hook between chunks."""

    def __init__(self, *items: bytes | BaseException) -> None:
        self.items = list(items)

    def __aiter__(self) -> _Chunks:
        return self

    async def __anext__(self) -> bytes:
        if not self.items:
            raise StopAsyncIteration
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


async def _read_in_thread(reader: io.RawIOBase, size: int = 100) -> bytes | None:
    """Read from ``reader`` in a worker thread, as the library does."""
    return await asyncio.to_thread(reader.read, size)


@pytest.mark.anyio
async def test_reader_short_reads_skip_empty_chunks_and_end_cleanly() -> None:
    """Reads never cross a chunk, empty chunks are skipped, EOF is sticky."""
    loop = asyncio.get_running_loop()
    reader = AsyncIteratorReader(
        _Chunks(b"abc", b"", b"defgh"), loop, max_bytes=100, read_timeout_seconds=1
    )

    assert await _read_in_thread(reader, 2) == b"ab"
    assert await _read_in_thread(reader) == b"c"
    assert await _read_in_thread(reader) == b"defgh"
    assert await _read_in_thread(reader) == b""
    assert await _read_in_thread(reader) == b""
    assert await _read_in_thread(reader, 0) == b""
    assert reader.bytes_read == 8
    assert reader.readable()


@pytest.mark.anyio
async def test_reader_readinto_and_readall() -> None:
    """The ``io`` helpers built on ``read`` work too."""
    loop = asyncio.get_running_loop()
    reader = AsyncIteratorReader(
        _Chunks(b"abc", b"def"), loop, max_bytes=100, read_timeout_seconds=1
    )
    buffer = bytearray(2)

    assert await asyncio.to_thread(reader.readinto, buffer) == 2
    assert buffer == b"ab"
    assert await asyncio.to_thread(reader.read, -1) == b"cdef"


@pytest.mark.anyio
async def test_reader_times_out_on_a_stalled_sender() -> None:
    """No chunk within the read timeout is a TimeoutError, an OSError."""

    async def _stalled() -> AsyncIterator[bytes]:
        await asyncio.Event().wait()
        yield b""  # pragma: no cover - never reached

    loop = asyncio.get_running_loop()
    reader = AsyncIteratorReader(_stalled(), loop, max_bytes=100, read_timeout_seconds=0.05)

    with pytest.raises(TimeoutError, match="no data received"):
        await _read_in_thread(reader)


@pytest.mark.anyio
async def test_reader_wraps_iterator_errors_in_oserror() -> None:
    """Any failure of the iterator reaches the library as an OSError."""
    loop = asyncio.get_running_loop()
    errors = AsyncIteratorReader(
        _Chunks(RuntimeError("disconnected"), BrokenPipeError("pipe")),
        loop,
        max_bytes=100,
        read_timeout_seconds=1,
    )

    with pytest.raises(OSError, match="RuntimeError: disconnected"):
        await _read_in_thread(errors)
    with pytest.raises(BrokenPipeError):
        await _read_in_thread(errors)


@pytest.mark.anyio
async def test_reader_closed_or_loop_gone_fails_the_read() -> None:
    """A closed reader, or a closed loop, fails the read instead of blocking."""
    loop = asyncio.get_running_loop()
    closed = AsyncIteratorReader(_Chunks(b"abc"), loop, max_bytes=100, read_timeout_seconds=1)
    closed.close()
    closed.close()

    with pytest.raises(ReaderClosedError, match="closed"):
        await _read_in_thread(closed)

    dead_loop = asyncio.new_event_loop()
    dead_loop.close()
    orphan = AsyncIteratorReader(_Chunks(b"abc"), dead_loop, max_bytes=100, read_timeout_seconds=1)
    with pytest.raises(ReaderClosedError):
        await _read_in_thread(orphan)


@pytest.mark.anyio
async def test_openapi_documents_the_raw_body(client: httpx.AsyncClient) -> None:
    """The schema shows the binary body, the shared parameters and the problem responses."""
    operation = (await client.get("/openapi.json")).json()["paths"]["/inspect/raw"]["post"]

    assert set(operation["requestBody"]["content"]) == {"application/octet-stream", "text/csv"}
    assert {p["name"] for p in operation["parameters"]} >= {
        "n_bytes",
        "tail_bytes",
        "timeout_seconds",
    }
    assert {"200", "413", "422", "502", "503", "504"} <= set(operation["responses"])
