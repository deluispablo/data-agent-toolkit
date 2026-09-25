"""Fakes: every inspection in the tests runs without a model or Cloud Storage.

A deliberately small copy of the idea in the agent's own test fakes, which
are not shipped with the library (and whose module name would collide).
The library accepts any async ``(prompt, model) -> text`` callable as
``model_invoker``, so these stand in for Ollama or Gemini. The Cloud Storage
fakes implement the protocols of ``csv_inspector_api.sources.gcs`` and never
touch ``google.*``.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

SAMPLE_CSV = Path(__file__).resolve().parents[3] / "agents" / "csv_inspector" / "sample.csv"
"""The agent's demo file: two preamble lines, a ``;``-delimited header, five rows."""

SAMPLE_ANSWER = json.dumps(
    {
        "encoding": "utf-8",
        "delimiter": ";",
        "quotechar": '"',
        "escapechar": None,
        "doublequote": True,
        "header_row_index": 2,
        "footer_lines": [],
        "columns": ["Fecha", "Cliente", "Descripción", "Importe", "Observaciones"],
        "confidence": 0.9,
    }
)
"""A valid model answer for :data:`SAMPLE_CSV`."""


class FakeInvoker:
    """Async model invoker with a scripted behaviour; records every call.

    Attributes:
        answer: Text returned to the library (valid JSON by default).
        error: Exception raised instead of answering, if set.
        delay: Seconds to sleep before answering, to exceed a time budget.
        calls: ``(prompt, model)`` of every call, in order.
    """

    def __init__(
        self,
        answer: str = SAMPLE_ANSWER,
        *,
        error: Exception | None = None,
        delay: float = 0,
    ) -> None:
        """Script the invoker.

        Args:
            answer: Text to return; not JSON drives the ``ResponseParsingError`` path.
            error: Exception to raise on every call instead of answering.
            delay: Seconds to sleep before answering or raising.
        """
        self.answer = answer
        self.error = error
        self.delay = delay
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, prompt: str, model: str) -> str:
        """Answer like a model would: ``(prompt, model) -> raw text``."""
        self.calls.append((prompt, model))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.answer


class RecordingReader(io.BytesIO):
    """A seekable reader over fixed bytes that records how it is used.

    Stands in for ``google.cloud.storage.fileio.BlobReader``.

    Attributes:
        reads: ``(position, bytes returned)`` of every ``read`` call.
        seeks: ``(offset, whence)`` of every ``seek`` call.
        seekable_answer: What ``seekable()`` reports.
    """

    def __init__(self, data: bytes, *, seekable: bool = True) -> None:
        """Serve ``data``; ``seekable=False`` imitates a forward-only stream."""
        super().__init__(data)
        self.reads: list[tuple[int, int]] = []
        self.seeks: list[tuple[int, int]] = []
        self.seekable_answer = seekable

    def read(self, size: int | None = -1, /) -> bytes:
        """Read like ``BytesIO``, recording the position and the byte count."""
        position = self.tell()
        chunk = super().read(size)
        self.reads.append((position, len(chunk)))
        return chunk

    def seek(self, offset: int, whence: int = 0, /) -> int:
        """Seek like ``BytesIO``, recording the call."""
        self.seeks.append((offset, whence))
        return super().seek(offset, whence)

    def seekable(self) -> bool:
        """Report :attr:`seekable_answer`."""
        return self.seekable_answer

    @property
    def bytes_read(self) -> int:
        """Total bytes returned by ``read``."""
        return sum(count for _, count in self.reads)


class FakeBlob:
    """A Cloud Storage blob over fixture bytes: no request, ever.

    Attributes:
        name: Object name.
        generation: Generation requested, or the object's one when none was.
        size: Object size, as the reader learns it from the metadata.
        readers: Every reader returned by :meth:`open`.
        open_calls: ``(mode, chunk_size)`` of every :meth:`open` call.
        error: Raised by the reader's first ``seek``, as a metadata request would.
    """

    def __init__(
        self,
        name: str,
        data: bytes,
        *,
        generation: int | None,
        error: Exception | None = None,
        seekable: bool = True,
    ) -> None:
        """Describe an object; ``error`` makes reading it fail."""
        self.name = name
        self._data = data
        self.generation = generation if generation is not None else 1700000000000001
        self.size: int | None = len(data)
        self.error = error
        self._seekable = seekable
        self.readers: list[RecordingReader] = []
        self.open_calls: list[tuple[str, int | None]] = []

    def open(self, mode: str, chunk_size: int | None = None) -> RecordingReader:
        """Open the object like ``Blob.open("rb")``: no request yet."""
        self.open_calls.append((mode, chunk_size))
        reader = (
            _FailingReader(self._data, self.error)
            if self.error is not None
            else RecordingReader(self._data, seekable=self._seekable)
        )
        self.readers.append(reader)
        return reader


class _FailingReader(RecordingReader):
    """A reader whose first ``seek`` raises, like a failed metadata request."""

    def __init__(self, data: bytes, error: Exception) -> None:
        super().__init__(data)
        self._error = error

    def seek(self, offset: int, whence: int = 0, /) -> int:
        """Raise the scripted error."""
        raise self._error


class FakeBucket:
    """A Cloud Storage bucket holding fixture objects."""

    def __init__(self, client: FakeClient, name: str) -> None:
        """A handle on bucket ``name`` of ``client``."""
        self._client = client
        self.name = name

    def blob(self, blob_name: str, *, generation: int | None = None) -> FakeBlob:
        """Build a blob handle like ``Bucket.blob``; records it on the client."""
        blob = FakeBlob(
            blob_name,
            self._client.objects.get((self.name, blob_name), b""),
            generation=generation,
            error=self._client.error,
            seekable=self._client.seekable,
        )
        self._client.blobs.append(blob)
        return blob


class FakeClient:
    """A Cloud Storage client over in-memory objects.

    Attributes:
        objects: Object bytes by ``(bucket, name)``; a missing one reads as empty.
        error: Exception every reader raises on its first ``seek``, if set.
        seekable: What every reader's ``seekable()`` reports.
        blobs: Every blob handle built, in order.
    """

    def __init__(
        self,
        objects: dict[tuple[str, str], bytes] | None = None,
        *,
        error: Exception | None = None,
        seekable: bool = True,
    ) -> None:
        """Serve ``objects``; ``error`` makes every read fail with it."""
        self.objects = objects or {}
        self.error = error
        self.seekable = seekable
        self.blobs: list[FakeBlob] = []

    def bucket(self, bucket_name: str) -> FakeBucket:
        """Build a bucket handle like ``Client.bucket``: no request."""
        return FakeBucket(self, bucket_name)
