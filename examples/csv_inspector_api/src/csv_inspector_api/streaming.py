"""A blocking, non-seekable file object over an async byte iterator.

``ainspect_csv`` samples its source in a worker thread through a sync
``read()``, while Starlette exposes a request body as the async iterator
``request.stream()``. :class:`AsyncIteratorReader` bridges the two: each
``read()`` in the worker thread schedules the next chunk on the event loop
with :func:`asyncio.run_coroutine_threadsafe` and waits for it. The body is
consumed once and never buffered beyond the current chunk, so the library's
non-seekable stream path keeps memory bounded by ``n_bytes + tail_bytes``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import threading
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import WriteableBuffer


MAX_READ_BYTES = 8 * 1024
"""Largest piece one ``read()`` returns.

The library keeps its previous read alive while it asks for the next one,
and appends each read to its tail window before trimming it: pieces this
small keep that working set near ``n_bytes + tail_bytes`` even when the
server delivers 64 KiB chunks.
"""

_EMPTY = memoryview(b"")


class BodyTooLargeError(OSError):
    """The body grew past the reader's byte limit while it was being read."""


class ReaderClosedError(OSError):
    """The reader was closed, e.g. because the request was cancelled."""


class AsyncIteratorReader(io.RawIOBase):
    """Non-seekable, read-only binary stream over an async iterator of ``bytes``.

    Read it from a thread other than the event loop's (``asyncio.to_thread``,
    which ``ainspect_csv`` uses): reading on the loop thread would deadlock.

    Every failure surfaces as an :class:`OSError` from ``read()``, which the
    library reports as ``FileSampleReadError``. A worker thread blocked in
    ``read()`` is released by :meth:`close` (call it from the loop when the
    request ends, including on cancellation) or after ``read_timeout_seconds``.

    Attributes:
        bytes_read: Bytes received from the iterator so far.
        limit_exceeded: Whether the body grew past ``max_bytes``.
    """

    def __init__(
        self,
        chunks: AsyncIterator[bytes],
        loop: asyncio.AbstractEventLoop,
        *,
        max_bytes: int,
        read_timeout_seconds: float,
    ) -> None:
        """Wrap ``chunks``.

        Args:
            chunks: The async byte iterator, e.g. ``request.stream()``.
                Empty chunks are skipped; its end is the end of the stream.
            loop: The event loop that owns ``chunks``.
            max_bytes: Largest body accepted; one more byte fails the read
                with :class:`BodyTooLargeError`.
            read_timeout_seconds: Longest wait for one chunk; a stalled
                sender fails the read with :class:`TimeoutError`.
        """
        super().__init__()
        self._chunks = chunks
        self._loop = loop
        self._max_bytes = max_bytes
        self._read_timeout_seconds = read_timeout_seconds
        self._pending: concurrent.futures.Future[bytes] | None = None
        self._lock = threading.Lock()
        self._buffer = _EMPTY
        self._eof = False
        self.bytes_read = 0
        self.limit_exceeded = False

    def readable(self) -> bool:
        """Whether the stream can be read: always true."""
        return True

    def read(self, size: int = -1, /) -> bytes:
        """Read up to ``size`` bytes, blocking for the next chunk when none is buffered.

        A read returns at most :data:`MAX_READ_BYTES` and never crosses a
        chunk; the spent chunk is released before the next one is awaited.

        Args:
            size: Most bytes to return; negative reads to the end of the stream.

        Returns:
            The bytes read; empty only at the end of the stream.

        Raises:
            ReaderClosedError: If the reader is closed, before or during the wait.
            BodyTooLargeError: If the body exceeds ``max_bytes``.
            TimeoutError: If no chunk arrives within ``read_timeout_seconds``.
            OSError: If the iterator fails (e.g. the client disconnected).
        """
        if size < 0:
            return self.readall()
        if size and not self._buffer and not self._eof:
            self._buffer = _EMPTY  # drop the spent chunk before the next one arrives
            self._buffer = memoryview(self._next_chunk())
        size = min(size, MAX_READ_BYTES)
        data = bytes(self._buffer[:size])
        self._buffer = self._buffer[size:]
        return data

    def readinto(self, buffer: WriteableBuffer, /) -> int:
        """Read up to ``len(buffer)`` bytes into ``buffer``; see :meth:`read`.

        Args:
            buffer: A writable bytes-like object.

        Returns:
            The number of bytes copied; ``0`` only at the end of the stream.
        """
        view = memoryview(buffer).cast("B")
        data = self.read(len(view))
        view[: len(data)] = data
        return len(data)

    def close(self) -> None:
        """Close the reader and release a worker thread waiting in ``read()``.

        Safe to call from any thread, more than once.
        """
        with self._lock:
            if self._pending is not None:
                self._pending.cancel()
            super().close()

    def _next_chunk(self) -> bytes:
        """Fetch the next chunk from the event loop; ``b""`` at the end of the stream."""
        with self._lock:
            if self.closed:
                raise ReaderClosedError("the reader is closed")
            fetch = self._anext()
            try:
                pending = asyncio.run_coroutine_threadsafe(fetch, self._loop)
            except RuntimeError as exc:  # the loop is closed
                fetch.close()
                raise ReaderClosedError(str(exc)) from exc
            self._pending = pending
        try:
            chunk = pending.result(self._read_timeout_seconds)
        except concurrent.futures.CancelledError as exc:
            raise ReaderClosedError("the reader was closed while waiting for data") from exc
        except concurrent.futures.TimeoutError as exc:
            pending.cancel()
            msg = f"no data received for {self._read_timeout_seconds} s"
            raise TimeoutError(msg) from exc
        except OSError:
            raise
        except Exception as exc:  # any iterator failure is a read failure
            msg = f"the body could not be read: {type(exc).__name__}: {exc}"
            raise OSError(msg) from exc
        finally:
            with self._lock:
                self._pending = None
        if not chunk:
            self._eof = True
            return b""
        self.bytes_read += len(chunk)
        if self.bytes_read > self._max_bytes:
            self.limit_exceeded = True
            msg = f"body exceeds the {self._max_bytes}-byte limit"
            raise BodyTooLargeError(msg)
        return chunk

    async def _anext(self) -> bytes:
        """Next non-empty chunk, or ``b""`` at the end of the iterator; runs on the loop."""
        while True:
            try:
                chunk = await self._chunks.__anext__()
            except StopAsyncIteration:
                return b""
            if chunk:
                return chunk
