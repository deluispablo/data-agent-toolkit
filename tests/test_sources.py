"""Tests for inspecting in-memory and streamed sources, not just paths.

Every source type must produce exactly the same prompt, and therefore the
same result, as the equivalent file path, while reading only bounded windows.
"""

from __future__ import annotations

import io
import json
import tracemalloc
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from csv_inspector import (
    CSVInspectionResult,
    EmptySampleError,
    FileSampleReadError,
    inspect_csv,
)
from csv_inspector._sampling import STREAM_CHUNK_BYTES, describe_source, sample_source

AGENT_DIR = Path(__file__).resolve().parent.parent / "agents" / "csv_inspector"
SAMPLES_DIR = AGENT_DIR / "samples"

# Small (head only), production-sized (head + tail) and UTF-16 (aligned tail).
PARITY_FIXTURES = [
    AGENT_DIR / "sample.csv",
    SAMPLES_DIR / "header_and_footer_combined.csv",
    SAMPLES_DIR / "footer_end_marker.csv",
    SAMPLES_DIR / "encoding_utf16le_bom.csv",
]

RESULT_JSON = json.dumps(
    {
        "encoding": "utf-8",
        "delimiter": ";",
        "header_row_index": 0,
        "columns": [{"name": "Fecha", "inferred_type": "date"}],
        "confidence": 0.9,
    }
)


class NonSeekableStream(io.RawIOBase):
    """A read-only, non-seekable byte stream (like an HTTP request body).

    Serves ``data`` in chunks of at most ``max_chunk`` bytes and records the
    size of every read request.
    """

    def __init__(self, data: bytes, *, max_chunk: int = 1 << 30) -> None:
        self._data = data
        self._position = 0
        self._max_chunk = max_chunk
        self.read_sizes: list[int] = []

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size < 0:
            size = len(self._data) - self._position
        size = min(size, self._max_chunk)
        chunk = self._data[self._position : self._position + size]
        self._position += len(chunk)
        return chunk


class GeneratedStream(io.RawIOBase):
    """A non-seekable stream that generates ``total`` bytes lazily, never holding them."""

    def __init__(self, total: int, footer: bytes) -> None:
        self._remaining = total
        self._footer = footer
        self._row = b"2024-01-01;Acme;10.00\n"

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            footer, self._footer = self._footer, b""
            return footer[:size] if size >= 0 else footer
        size = min(size if size >= 0 else STREAM_CHUNK_BYTES, self._remaining)
        repeats = size // len(self._row) + 1
        chunk = (self._row * repeats)[:size]
        self._remaining -= len(chunk)
        return chunk


def _capture() -> tuple[list[str], Callable[[str, str], str]]:
    prompts: list[str] = []

    def invoker(prompt: str, model: str) -> str:
        prompts.append(prompt)
        return RESULT_JSON

    return prompts, invoker


def _inspect(source: object) -> tuple[str, CSVInspectionResult]:
    prompts, invoker = _capture()
    result = inspect_csv(source, model="m", fallback_model="m", model_invoker=invoker)  # type: ignore[arg-type]
    return prompts[0], result


def _variants(data: bytes) -> Iterator[tuple[str, object]]:
    yield "bytes", data
    yield "bytearray", bytearray(data)
    yield "memoryview", memoryview(data)
    yield "BytesIO", io.BytesIO(data)
    yield "non-seekable stream", NonSeekableStream(data)
    yield "short-read stream", NonSeekableStream(data, max_chunk=7)


# ---------------------------------------------------------------------
# Parity with the path case
# ---------------------------------------------------------------------


@pytest.mark.parametrize("fixture", PARITY_FIXTURES, ids=lambda path: path.name)
def test_every_source_type_matches_the_path_case(fixture: Path) -> None:
    """Bytes, buffers and streams yield the same prompt and result as the path."""
    path_prompt, path_result = _inspect(fixture)

    for label, source in _variants(fixture.read_bytes()):
        prompt, result = _inspect(source)
        assert prompt == path_prompt, label
        assert result == path_result, label


@pytest.mark.parametrize("fixture", PARITY_FIXTURES, ids=lambda path: path.name)
def test_a_partially_consumed_stream_is_sampled_from_its_current_position(
    fixture: Path,
) -> None:
    """A stream positioned after a prefix is inspected as if the prefix were not there."""
    path_prompt, _ = _inspect(fixture)
    stream = io.BytesIO(b"PREFIX THAT IS NOT PART OF THE CSV\n" + fixture.read_bytes())
    stream.seek(len(b"PREFIX THAT IS NOT PART OF THE CSV\n"))

    prompt, _ = _inspect(stream)

    assert prompt == path_prompt


def test_the_seekable_stream_position_is_restored() -> None:
    """The host's stream is left where it was, even after head and tail reads."""
    data = (SAMPLES_DIR / "footer_end_marker.csv").read_bytes()
    stream = io.BytesIO(b"xx" + data)
    stream.seek(2)

    _inspect(stream)

    assert stream.tell() == 2


def test_an_open_binary_file_is_accepted(tmp_path: Path) -> None:
    """A real file object opened with 'rb' works and keeps its position."""
    target = tmp_path / "data.csv"
    target.write_bytes((AGENT_DIR / "sample.csv").read_bytes())
    path_prompt, _ = _inspect(target)

    with target.open("rb") as handle:
        prompt, _ = _inspect(handle)
        assert handle.tell() == 0

    assert prompt == path_prompt


class DuckTypedSeekableStream:
    """A stream with read/seek/tell but no ``seekable()``, whose ``seek`` returns ``None``."""

    def __init__(self, data: bytes) -> None:
        self._buffer = io.BytesIO(data)

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> None:
        self._buffer.seek(offset, whence)

    def tell(self) -> int:
        return self._buffer.tell()


def test_a_seek_that_returns_none_is_supported() -> None:
    """The end of a duck-typed stream is found with ``tell()``, not ``seek()``'s return value.

    Regression test for issue #18.
    """
    fixture = SAMPLES_DIR / "header_and_footer_combined.csv"
    path_prompt, _ = _inspect(fixture)
    stream = DuckTypedSeekableStream(fixture.read_bytes())

    prompt, _ = _inspect(stream)

    assert prompt == path_prompt
    assert stream.tell() == 0


# ---------------------------------------------------------------------
# Bounded reads and memory
# ---------------------------------------------------------------------


def test_non_seekable_streams_are_read_in_bounded_chunks() -> None:
    """No single read of a stream exceeds the chunk size, however large the stream."""
    stream = NonSeekableStream(b"a;b\n" + b"1;2\n" * 200_000)

    sample_source(stream, 4096, 4096)

    assert stream.read_sizes
    assert max(stream.read_sizes) <= STREAM_CHUNK_BYTES
    assert -1 not in stream.read_sizes


def test_non_seekable_stream_memory_stays_bounded() -> None:
    """Sampling a 50 MiB generated stream never holds more than a few chunks in memory."""
    footer = b"\nTOTAL;;999\n--- Fin del informe ---\n"
    stream = GeneratedStream(50 * 1024 * 1024, footer)

    tracemalloc.start()
    try:
        samples = sample_source(stream, 4096, 4096)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert samples.tail_text is not None
    assert samples.tail_text.endswith("TOTAL;;999\n--- Fin del informe ---\n")
    assert peak < 1 * 1024 * 1024


def test_tail_bytes_zero_does_not_consume_a_non_seekable_stream_past_the_head() -> None:
    """With tail sampling disabled, only the head is read from a forward-only stream."""
    stream = NonSeekableStream(b"a;b\n" + b"1;2\n" * 10_000)

    samples = sample_source(stream, 64, 0)

    assert samples.tail_text is None
    assert sum(size for size in stream.read_sizes if size > 0) <= 64


SCAN_LIMIT = 4096


def test_a_non_seekable_stream_is_read_no_further_than_the_scan_limit(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Past the scan limit the tail is skipped and the end counts as unsampled.

    Regression test for issue #17: a forward-only stream used to be read to
    its end, however long it was, just to sample its tail.
    """
    monkeypatch.setattr("csv_inspector._sampling.MAX_FORWARD_SCAN_BYTES", SCAN_LIMIT)
    stream = NonSeekableStream(b"a;b\n" + b"1;2\n" * 100_000)

    samples = sample_source(stream, 64, 64)

    assert samples.tail_text is None
    assert samples.covers_whole_file is False
    assert sum(size for size in stream.read_sizes if size > 0) <= 64 + SCAN_LIMIT + 1
    assert "skipping the tail sample" in caplog.text


def test_a_non_seekable_stream_ending_at_the_scan_limit_keeps_its_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream that ends exactly at the limit is still sampled to its real end."""
    monkeypatch.setattr("csv_inspector._sampling.MAX_FORWARD_SCAN_BYTES", SCAN_LIMIT)
    head = b"a;b\n" + b"1;2\n" * 15
    rest = b"1;2\n" * (SCAN_LIMIT // 4 - 1) + b"END\n"
    assert (len(head), len(rest)) == (64, SCAN_LIMIT)

    samples = sample_source(NonSeekableStream(head + rest, max_chunk=1000), 64, 64)

    assert samples.covers_whole_file is True
    assert samples.tail_text is not None
    assert samples.tail_text.endswith("1;2\nEND\n")


def test_an_over_long_stream_is_inspected_without_a_footer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model is told the end was not sampled, and no footer is reported."""
    monkeypatch.setattr("csv_inspector._sampling.MAX_FORWARD_SCAN_BYTES", SCAN_LIMIT)
    reported = json.loads(RESULT_JSON) | {"footer_lines": ["1;2"]}
    prompts: list[str] = []

    def invoker(prompt: str, model: str) -> str:
        prompts.append(prompt)
        return json.dumps(reported)

    stream = NonSeekableStream(b"Fecha;Cliente\n" + b"1;2\n" * 100_000)
    result = inspect_csv(stream, model="m", fallback_model="m", model_invoker=invoker)

    assert "its end was not sampled" in prompts[0]
    assert result.footer_lines == []


@pytest.mark.parametrize(
    ("data", "n_bytes", "tail_bytes", "expected"),
    [
        pytest.param(b"a;b\n1;2\n", 64, 0, True, id="head-holds-everything"),
        pytest.param(b"a;b\n" + b"1;2\n" * 100, 64, 64, True, id="tail-reaches-the-end"),
        pytest.param(b"a;b\n" + b"1;2\n" * 100, 64, 0, False, id="truncated-head-no-tail"),
    ],
)
def test_samples_report_whether_they_reach_the_end_of_the_source(
    data: bytes, n_bytes: int, tail_bytes: int, expected: bool
) -> None:
    """``covers_whole_file`` is false only when the end of the source was never sampled."""
    samples = sample_source(data, n_bytes, tail_bytes)

    assert samples.covers_whole_file is expected


# ---------------------------------------------------------------------
# Rejected and failing sources
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [io.StringIO("a;b\n1;2\n"), 42, None, ["a;b"]],
    ids=["text stream", "int", "None", "list"],
)
def test_unsupported_sources_are_rejected_with_type_error(source: object) -> None:
    """Text streams and non-source objects fail fast with a clear TypeError."""
    with pytest.raises(TypeError):
        _inspect(source)


def test_a_text_mode_file_is_rejected(tmp_path: Path) -> None:
    """Files opened in text mode are rejected; the message says to use 'rb'."""
    target = tmp_path / "data.csv"
    target.write_text("a;b\n1;2\n", encoding="utf-8")

    with target.open(encoding="utf-8") as handle, pytest.raises(TypeError, match="'rb'"):
        _inspect(handle)


@pytest.mark.parametrize(
    "source",
    [b"", bytearray(), io.BytesIO(b""), NonSeekableStream(b"")],
    ids=["bytes", "bytearray", "BytesIO", "non-seekable"],
)
def test_empty_sources_raise_before_any_model_call(source: object) -> None:
    """An empty buffer or stream is an EmptySampleError, and costs no LLM call."""
    prompts, invoker = _capture()

    with pytest.raises(EmptySampleError):
        inspect_csv(source, model="m", fallback_model="m", model_invoker=invoker)  # type: ignore[arg-type]

    assert prompts == []


def test_stream_read_errors_become_domain_errors() -> None:
    """An I/O failure while reading a stream is reported as FileSampleReadError."""

    class BrokenStream(io.RawIOBase):
        def readable(self) -> bool:
            return True

        def read(self, size: int = -1) -> bytes:
            raise OSError("connection reset by peer")

    with pytest.raises(FileSampleReadError, match="connection reset"):
        _inspect(BrokenStream())


class NonBlockingStream(io.RawIOBase):
    """A non-blocking stream that serves ``data``, then has nothing available yet."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes | None:
        if not self._data:
            return None
        chunk, self._data = self._data[:size], self._data[size:]
        return chunk


@pytest.mark.parametrize(
    "available",
    [b"", b"Fecha;Cliente\n", b"Fecha;Cliente\n" * 1000],
    ids=["nothing", "part of the head", "past the head"],
)
def test_a_stream_with_no_data_available_yet_is_not_taken_as_ended(available: bytes) -> None:
    """``read()`` returning ``None`` is a read error, never the end of the stream.

    Regression test for issue #19: it used to be treated as the end, so the
    sample was silently truncated, or an EmptySampleError was raised.
    """
    with pytest.raises(FileSampleReadError, match="non-blocking"):
        _inspect(NonBlockingStream(available))


# ---------------------------------------------------------------------
# Descriptions (used in logs and errors) never leak content
# ---------------------------------------------------------------------


def test_source_descriptions_never_include_content(tmp_path: Path) -> None:
    """Buffers and anonymous streams are described by size or type, never content."""
    secret = b"iban;ES7621000418401234567891\n"

    assert describe_source(secret) == f"<{len(secret)} bytes in memory>"
    assert describe_source(io.BytesIO(secret)) == "<BytesIO stream>"
    assert describe_source(tmp_path / "x.csv") == str(tmp_path / "x.csv")
