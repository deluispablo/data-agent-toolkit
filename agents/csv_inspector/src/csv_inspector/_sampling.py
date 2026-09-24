"""Bounded byte sampling and text decoding for csv_inspector.

Reads only a small head window and, when the source is larger, a small tail
window of a delimited source, never loading a file or stream into memory in
full. Sources can be a filesystem path, an in-memory buffer, or a binary
stream (seekable or not); see :data:`CSVSource`.
"""

from __future__ import annotations

import codecs
import errno
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, TypeAlias, cast

import chardet

from ._exceptions import EmptySampleError, FileSampleReadError

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_BYTES: int = 4096
DEFAULT_TAIL_BYTES: int = 4096
MAX_SAMPLE_BYTES: int = 16384
"""Upper bound for each sample window, so the prompt fits a local model's context."""


def _validate_byte_budget(
    name: str, value: int, *, minimum: int, maximum: int | None = None
) -> None:
    """Reject byte budgets that would break the bounded-read guarantee.

    A negative size passed to ``file.read()`` means "read everything", so an
    unchecked budget would silently load a multi-gigabyte file into memory.

    Args:
        name: Parameter name, used in the error message.
        value: The requested byte budget.
        minimum: The smallest accepted value.
        maximum: The largest accepted value, or ``None`` for no upper bound.

    Raises:
        ValueError: If ``value`` is outside ``[minimum, maximum]``.
    """
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}.")


def read_sample_bytes(path: str | os.PathLike[str], n_bytes: int = DEFAULT_SAMPLE_BYTES) -> bytes:
    """Read only the first ``n_bytes`` of a file without loading it fully into memory.

    Args:
        path: Path to the source file.
        n_bytes: Maximum number of bytes to read from the start of the file.
            Must be at least 1.

    Returns:
        The raw bytes read from the file. Shorter than ``n_bytes`` only when
        the file itself is smaller than ``n_bytes``.

    Raises:
        ValueError: If ``n_bytes`` is less than 1.
        FileSampleReadError: If the file does not exist or cannot be read.
    """
    _validate_byte_budget("n_bytes", n_bytes, minimum=1)
    try:
        with Path(path).open("rb") as handle:
            return handle.read(n_bytes)
    except OSError as exc:
        raise FileSampleReadError(f"Unable to read head sample from '{path}': {exc}") from exc


def read_tail_bytes(path: str | os.PathLike[str], n_bytes: int = DEFAULT_TAIL_BYTES) -> bytes:
    """Read only the last ``n_bytes`` of a file without loading it fully into memory.

    Uses a bounded seek from the end of the file (``os.SEEK_END``) followed
    by a single bounded read, so the cost is independent of the file's total
    size: no data before the tail window is ever touched.

    Args:
        path: Path to the source file.
        n_bytes: Maximum number of trailing bytes to read. ``0`` returns an
            empty sample without touching the file contents.

    Returns:
        The raw trailing bytes. Shorter than ``n_bytes`` only when the file
        itself is smaller than ``n_bytes``; empty for a zero-byte file.
        These bytes are a blind suffix of the file and may begin mid-line
        (or mid-character, for multi-byte encodings) rather than at a clean
        row boundary.

    Raises:
        ValueError: If ``n_bytes`` is negative.
        FileSampleReadError: If the file does not exist or cannot be read.
    """
    _validate_byte_budget("n_bytes", n_bytes, minimum=0)
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            file_size = handle.tell()
            read_size = min(n_bytes, file_size)
            handle.seek(-read_size, os.SEEK_END)
            return handle.read(read_size)
    except OSError as exc:
        raise FileSampleReadError(f"Unable to read tail sample from '{path}': {exc}") from exc


def _file_size(path: str | os.PathLike[str]) -> int:
    """Return the size of ``path`` in bytes.

    Raises:
        FileSampleReadError: If the file cannot be stat'ed.
    """
    try:
        return Path(path).stat().st_size
    except OSError as exc:
        raise FileSampleReadError(f"Unable to stat '{path}': {exc}") from exc


def detect_encoding(raw_bytes: bytes) -> str:
    """Heuristically detect the character encoding of a byte sample.

    Args:
        raw_bytes: The raw byte sample to analyze.

    Returns:
        The detected encoding name, defaulting to ``"utf-8"`` when detection
        is inconclusive or reports plain ASCII.
    """
    detection = chardet.detect(raw_bytes)
    encoding: str = detection.get("encoding") or "utf-8"
    if encoding.lower() == "ascii":
        encoding = "utf-8"
    return encoding


def _canonical_codec_name(encoding: str) -> str | None:
    """Return Python's canonical codec name for ``encoding``, or ``None`` if unknown."""
    try:
        return codecs.lookup(encoding).name
    except LookupError:
        return None


def _code_unit_size(encoding: str) -> int:
    """Return the fixed code-unit width, in bytes, of ``encoding``.

    UTF-16 and UTF-32 cannot be decoded from an arbitrary byte offset: a
    window that starts on an odd byte turns every character into garbage.
    Aligning the tail window to the code unit avoids that.

    Args:
        encoding: An encoding name, typically from :func:`detect_encoding`.

    Returns:
        ``2`` for UTF-16 variants, ``4`` for UTF-32 variants, ``1`` otherwise.
    """
    codec = _canonical_codec_name(encoding) or ""
    if codec.startswith("utf-16"):
        return 2
    if codec.startswith("utf-32"):
        return 4
    return 1


def _tail_encoding(head_bytes: bytes, encoding: str) -> str:
    """Pick the codec for decoding a tail sample, which never carries a BOM.

    BOM-dependent codecs (``utf-16``, ``utf-32``, ``utf-8-sig``) decide byte
    order from a leading BOM. A tail sample has none, so decoding it with
    the generic codec would silently assume the host's byte order. This
    resolves the explicit, BOM-less variant from the head's BOM instead.

    Args:
        head_bytes: The raw head sample, which may start with a BOM.
        encoding: The encoding detected for the head sample.

    Returns:
        An encoding name that decodes a BOM-less suffix of the same file.
    """
    codec = _canonical_codec_name(encoding)
    if codec == "utf-8-sig":
        return "utf-8"
    if codec == "utf-16":
        return "utf-16-be" if head_bytes.startswith(codecs.BOM_UTF16_BE) else "utf-16-le"
    if codec == "utf-32":
        return "utf-32-be" if head_bytes.startswith(codecs.BOM_UTF32_BE) else "utf-32-le"
    return encoding


def decode_sample(raw_bytes: bytes, encoding: str) -> str:
    """Decode a byte sample using the given encoding, tolerating bad bytes.

    Args:
        raw_bytes: The raw byte sample to decode.
        encoding: The encoding to use, typically from :func:`detect_encoding`.

    Returns:
        The decoded text, with undecodable bytes replaced rather than raising.
    """
    try:
        return raw_bytes.decode(encoding, errors="replace")
    except LookupError:
        logger.warning("Unknown encoding '%s'; falling back to utf-8.", encoding)
        return raw_bytes.decode("utf-8", errors="replace")


class SupportsBinaryRead(Protocol):
    """Any object with a ``read(size)`` returning bytes, e.g. a file opened with ``"rb"``.

    Structural, so ``io.BytesIO``, ``io.RawIOBase`` subclasses (request bodies),
    ``tempfile.SpooledTemporaryFile`` and similar objects all qualify.
    """

    def read(self, size: int = ..., /) -> bytes | None:
        """Read up to ``size`` bytes (``None``: no data available yet).

        Sampling needs a blocking stream: a ``None`` read fails with
        :class:`FileSampleReadError` rather than being taken as the end.
        """
        ...


CSVSource: TypeAlias = str | os.PathLike[str] | bytes | bytearray | memoryview | SupportsBinaryRead
"""Anything :func:`csv_inspector.inspect_csv` can sample.

* ``str`` / ``os.PathLike``: a path to a file on disk (a ``str`` is always a
  path, never CSV content; pass CSV content as ``bytes``).
* ``bytes`` / ``bytearray`` / ``memoryview``: content already in memory.
* A binary file-like object (:class:`SupportsBinaryRead`), seekable or not.
"""

# Chunk size for reading streams; also the largest single read ever issued.
STREAM_CHUNK_BYTES = 64 * 1024

MAX_FORWARD_SCAN_BYTES: int = 64 * 1024 * 1024
"""Most bytes read past the head of a non-seekable stream while looking for its end.

A non-seekable stream can only reach its tail by reading everything before
it, which for an unbounded body or pipe would take unbounded time. Past this
limit the tail is skipped and the end of the stream is treated as unsampled.
"""

_TEXT_STREAM_MESSAGE = (
    "csv_inspector needs a binary stream: open the file in binary mode ('rb') or pass "
    "the content as bytes."
)


@dataclass(frozen=True)
class Samples:
    """The decoded head and tail samples of a source.

    Attributes:
        head_text: The decoded head sample.
        tail_text: The decoded tail sample, or ``None`` when the head
            already covers the whole source (or tail sampling is disabled).
        encoding: The encoding detected for the head sample.
        description: A log-safe description of the source.
        covers_whole_file: Whether the samples reach the real end of the
            source: the head covers it all, or a tail was sampled. ``False``
            when the head was truncated and tail sampling is disabled, or a
            non-seekable stream ran past :data:`MAX_FORWARD_SCAN_BYTES`, in
            which case the end of the source was never seen.
    """

    head_text: str
    tail_text: str | None
    encoding: str
    description: str
    covers_whole_file: bool = True


class _Reader(Protocol):
    """Bounded access to the start and end of a source."""

    def head(self, n_bytes: int) -> bytes:
        """Return up to ``n_bytes`` from the start of the source."""
        ...

    def tail(self, head_length: int, max_bytes: int, unit: int) -> bytes | None:
        """Return the end of the source that lies past the head window.

        At most ``max_bytes`` bytes are returned, trimmed to a multiple of
        ``unit`` (the encoding's code-unit size) so the sample starts on a
        character boundary for UTF-16/UTF-32. ``None`` means the end of the
        source could not be reached within the reader's limits.
        """
        ...


def _tail_window(uncovered: int, max_bytes: int, unit: int) -> int:
    """Size of the tail window: at most ``max_bytes``, never overlapping the head."""
    window = min(max_bytes, uncovered)
    return max(window - window % unit, 0)


class _PathReader:
    """Reads a file on disk with one bounded read per window."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = path

    def head(self, n_bytes: int) -> bytes:
        return read_sample_bytes(self._path, n_bytes)

    def tail(self, head_length: int, max_bytes: int, unit: int) -> bytes:
        window = _tail_window(_file_size(self._path) - head_length, max_bytes, unit)
        return read_tail_bytes(self._path, window) if window else b""


class _BufferReader:
    """Slices an in-memory buffer; only the sampled windows are copied."""

    def __init__(self, data: bytes | bytearray | memoryview) -> None:
        self._view = memoryview(data).cast("B")

    def head(self, n_bytes: int) -> bytes:
        return bytes(self._view[:n_bytes])

    def tail(self, head_length: int, max_bytes: int, unit: int) -> bytes:
        size = len(self._view)
        window = _tail_window(size - head_length, max_bytes, unit)
        return bytes(self._view[size - window :]) if window else b""


def _read_chunk(stream: SupportsBinaryRead, size: int) -> bytes:
    """Read at most ``size`` bytes from ``stream``, rejecting text streams.

    Returns:
        The bytes read; empty only at the end of the stream.

    Raises:
        TypeError: If the stream yields ``str`` (a text-mode stream).
        BlockingIOError: If the stream yields ``None``: it is non-blocking
            and has no data available yet, which is not the end of the
            stream, so sampling cannot go on without a wrong result.
    """
    chunk: bytes | str | None = stream.read(size)
    if isinstance(chunk, str):
        raise TypeError(_TEXT_STREAM_MESSAGE)
    if chunk is None:
        raise BlockingIOError(
            errno.EAGAIN,
            "the stream is non-blocking and has no data available; pass a blocking stream",
        )
    return chunk


def _read_up_to(stream: SupportsBinaryRead, n_bytes: int) -> bytes:
    """Read up to ``n_bytes``, looping over short reads, stopping at end of stream."""
    chunks: list[bytes] = []
    remaining = n_bytes
    while remaining > 0:
        chunk = _read_chunk(stream, min(remaining, STREAM_CHUNK_BYTES))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class _SeekableStreamReader:
    """Reads a seekable stream from its current position, like a file."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self.start = stream.tell()
        # Not seek()'s return value: duck-typed streams may return None.
        stream.seek(0, os.SEEK_END)
        self._end = stream.tell()

    def head(self, n_bytes: int) -> bytes:
        self._stream.seek(self.start)
        return _read_up_to(self._stream, n_bytes)

    def tail(self, head_length: int, max_bytes: int, unit: int) -> bytes:
        window = _tail_window(self._end - self.start - head_length, max_bytes, unit)
        if not window:
            return b""
        self._stream.seek(self._end - window)
        return _read_up_to(self._stream, window)

    def restore(self) -> None:
        """Put the stream back where the host left it."""
        self._stream.seek(self.start)


class _ForwardStreamReader:
    """Reads a non-seekable stream in one pass with bounded memory.

    The head is read first; the rest of the stream is then consumed in
    chunks while only the last ``max_bytes`` bytes are retained, so memory
    stays bounded by ``n_bytes + max_bytes`` however long the stream is.
    At most :data:`MAX_FORWARD_SCAN_BYTES` are read past the head, so the
    time spent is bounded too: a longer stream gets no tail.
    """

    def __init__(self, stream: SupportsBinaryRead) -> None:
        self._stream = stream

    def head(self, n_bytes: int) -> bytes:
        return _read_up_to(self._stream, n_bytes)

    def tail(self, head_length: int, max_bytes: int, unit: int) -> bytes | None:
        if max_bytes <= 0:
            return b""
        retained = bytearray()
        uncovered = 0
        # One byte past the limit tells a stream that ends exactly there from a longer one.
        while chunk := _read_chunk(
            self._stream, min(STREAM_CHUNK_BYTES, MAX_FORWARD_SCAN_BYTES + 1 - uncovered)
        ):
            uncovered += len(chunk)
            if uncovered > MAX_FORWARD_SCAN_BYTES:
                logger.warning(
                    "Stream continues past %d bytes after the head; skipping the tail sample.",
                    MAX_FORWARD_SCAN_BYTES,
                )
                return None
            retained += chunk
            del retained[:-max_bytes]
        window = _tail_window(uncovered, max_bytes, unit)
        return bytes(retained[len(retained) - window :]) if window else b""


def _is_seekable(stream: object) -> bool:
    """Whether ``stream`` supports ``tell``/``seek`` (``seekable()`` when available)."""
    seekable = getattr(stream, "seekable", None)
    if callable(seekable):
        try:
            return bool(seekable())
        except (OSError, ValueError):
            return False
    return callable(getattr(stream, "seek", None)) and callable(getattr(stream, "tell", None))


def describe_source(source: CSVSource) -> str:
    """Return a short, log-safe description of ``source`` (never its content)."""
    if isinstance(source, bytes | bytearray | memoryview):
        return f"<{memoryview(source).nbytes} bytes in memory>"
    if isinstance(source, str | os.PathLike):
        return os.fspath(source)
    name = getattr(source, "name", None)
    return name if isinstance(name, str) else f"<{type(source).__name__} stream>"


def _reader_for(source: object) -> _Reader:
    """Pick the reader for ``source``'s type.

    Raises:
        TypeError: If ``source`` is not a supported type, or is a text stream.
    """
    if isinstance(source, bytes | bytearray | memoryview):
        return _BufferReader(source)
    if isinstance(source, str | os.PathLike):
        return _PathReader(source)
    if isinstance(source, io.TextIOBase):
        raise TypeError(_TEXT_STREAM_MESSAGE)
    if callable(getattr(source, "read", None)):
        if _is_seekable(source):
            return _SeekableStreamReader(cast("BinaryIO", source))
        return _ForwardStreamReader(cast("SupportsBinaryRead", source))
    raise TypeError(
        f"Unsupported source type {type(source).__name__!r}: pass a path, bytes, "
        "or a binary file-like object."
    )


def sample_source(source: CSVSource, n_bytes: int, tail_bytes: int) -> Samples:
    """Read and decode the bounded head and tail samples of ``source``.

    Paths are read with one bounded read per window. Buffers are sliced.
    Seekable streams are sampled from their **current position** to their
    end, and that position is restored afterwards. Non-seekable streams are
    consumed once, with memory bounded by ``n_bytes + tail_bytes``; when one
    runs on for more than :data:`MAX_FORWARD_SCAN_BYTES` past the head,
    reading stops there and the tail is skipped (``covers_whole_file`` is
    then ``False``). The tail never overlaps the head and is skipped when
    the head already covers the whole source. No single read exceeds
    ``max(n_bytes, tail_bytes)`` for files, or :data:`STREAM_CHUNK_BYTES`
    for streams.

    Args:
        source: The source to sample; see :data:`CSVSource`.
        n_bytes: Head window size, in bytes. Must be between 1 and
            :data:`MAX_SAMPLE_BYTES`.
        tail_bytes: Maximum tail window size, in bytes; ``0`` disables it.
            At most :data:`MAX_SAMPLE_BYTES`.

    Returns:
        The decoded :class:`Samples`.

    Raises:
        ValueError: If a byte budget is out of range.
        TypeError: If ``source`` is not a supported type, or is a text stream.
        FileSampleReadError: If the source cannot be read.
        EmptySampleError: If the source is empty.
    """
    _validate_byte_budget("n_bytes", n_bytes, minimum=1, maximum=MAX_SAMPLE_BYTES)
    _validate_byte_budget("tail_bytes", tail_bytes, minimum=0, maximum=MAX_SAMPLE_BYTES)
    description = describe_source(source)

    try:
        reader = _reader_for(source)
    except OSError as exc:
        raise FileSampleReadError(f"Unable to read sample from '{description}': {exc}") from exc

    try:
        head_raw = reader.head(n_bytes)
        if not head_raw:
            raise EmptySampleError(f"'{description}' is empty; there is nothing to inspect.")
        encoding = detect_encoding(head_raw)
        head_text = decode_sample(head_raw, encoding)

        tail_text: str | None = None
        covers_whole_file = True
        if len(head_raw) == n_bytes:
            tail_raw = reader.tail(len(head_raw), tail_bytes, _code_unit_size(encoding))
            if tail_raw:
                tail_text = decode_sample(tail_raw, _tail_encoding(head_raw, encoding))
            elif tail_raw is None or tail_bytes == 0:
                # The end was out of reach, or nothing tells whether the source
                # ends here: assume it does not.
                covers_whole_file = False
    except OSError as exc:
        raise FileSampleReadError(f"Unable to read sample from '{description}': {exc}") from exc
    finally:
        if isinstance(reader, _SeekableStreamReader):
            reader.restore()

    return Samples(
        head_text=head_text,
        tail_text=tail_text,
        encoding=encoding,
        description=description,
        covers_whole_file=covers_whole_file,
    )
