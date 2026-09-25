"""Tests of the bounded byte reads and of encoding detection.

``read_sample_bytes`` and ``read_tail_bytes`` are tested not only for
correctness but also, via a ``Path.open()`` spy, for *how* they read: they
must never load a whole file into memory, since that is the entire point of
sampling head/tail byte windows against multi-gigabyte production files.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any

import pytest

from csv_inspector import (
    FileSampleReadError,
)
from csv_inspector._encoding import decode_sample, detect_encoding
from csv_inspector._sampling import (
    read_sample_bytes,
    read_tail_bytes,
)
from payloads import SAMPLE_CSV_PATH


def _spy_on_open(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Patch ``Path.open`` to record every size passed to ``.read()``.

    The sampling functions open files via ``Path.open``. It is patched
    directly because how it reaches ``io.open`` varies across Python
    versions (3.10 binds it at import time), so patching ``io.open`` would
    not intercept the call everywhere.

    Args:
        monkeypatch: The pytest monkeypatch fixture for the current test.

    Returns:
        A list that will be populated, in call order, with the ``size``
        argument of every ``.read()`` call made through ``Path.open()``
        while the patch is active. A negative entry would indicate an
        unbounded (whole-file) read.
    """
    read_calls: list[int] = []
    real_open = Path.open

    def spy_open(self: Path, *args: Any, **kwargs: Any) -> IO[Any]:
        handle: IO[Any] = real_open(self, *args, **kwargs)
        original_read = handle.read

        def traced_read(size: int = -1) -> Any:
            read_calls.append(size)
            return original_read(size)

        handle.read = traced_read  # type: ignore[method-assign]
        return handle

    monkeypatch.setattr(Path, "open", spy_open)
    return read_calls


def test_read_sample_bytes_respects_byte_limit(tmp_path: Path) -> None:
    """Only the requested number of leading bytes should be read."""
    target = tmp_path / "large.csv"
    target.write_bytes(b"a,b,c\n" * 1000)

    sample = read_sample_bytes(target, n_bytes=10)

    assert sample == b"a,b,c\na,b,"
    assert len(sample) == 10


def test_read_sample_bytes_missing_file_raises_domain_error(tmp_path: Path) -> None:
    """A missing source file should raise the domain-specific exception."""
    missing = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileSampleReadError):
        read_sample_bytes(missing)


def test_read_sample_bytes_reads_real_sample_fixture() -> None:
    """The bundled sample.csv fixture should be readable end-to-end."""
    sample = read_sample_bytes(SAMPLE_CSV_PATH, n_bytes=4096)

    assert b"Fecha;Cliente" in sample


def test_read_sample_bytes_empty_file_returns_empty_bytes(tmp_path: Path) -> None:
    """An empty file should yield an empty sample, not an error."""
    target = tmp_path / "empty.csv"
    target.write_bytes(b"")

    assert read_sample_bytes(target, n_bytes=4096) == b""


def test_read_sample_bytes_never_reads_the_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sampling a large file must issue exactly one bounded ``.read()`` call.

    This is the production-critical guarantee behind byte-budgeted
    inspection: a multi-gigabyte file must never be pulled fully into
    memory just to look at its first few kilobytes.
    """
    marker = b"HEAD_MARKER_0123456789"
    target = tmp_path / "big_head.csv"
    target.write_bytes(marker + b"x" * 5_000_000)

    read_calls = _spy_on_open(monkeypatch)

    head = read_sample_bytes(target, n_bytes=len(marker))

    assert head == marker
    assert read_calls == [len(marker)]


def test_read_tail_bytes_respects_byte_limit(tmp_path: Path) -> None:
    """Only the requested number of trailing bytes should be read."""
    target = tmp_path / "large.csv"
    target.write_bytes(b"a,b,c\n" * 1000)

    tail = read_tail_bytes(target, n_bytes=10)

    assert tail == (b"a,b,c\n" * 1000)[-10:]
    assert len(tail) == 10


def test_read_tail_bytes_smaller_than_limit_returns_whole_file(tmp_path: Path) -> None:
    """A file smaller than ``n_bytes`` should be returned in full."""
    target = tmp_path / "small.csv"
    target.write_bytes(b"a,b,c\n1,2,3\n")

    tail = read_tail_bytes(target, n_bytes=4096)

    assert tail == b"a,b,c\n1,2,3\n"


def test_read_tail_bytes_exact_file_size(tmp_path: Path) -> None:
    """A file exactly ``n_bytes`` long should be returned in full."""
    content = b"0123456789"
    target = tmp_path / "exact.csv"
    target.write_bytes(content)

    assert read_tail_bytes(target, n_bytes=len(content)) == content


def test_read_tail_bytes_empty_file_returns_empty_bytes(tmp_path: Path) -> None:
    """An empty file should yield an empty tail sample, not an error."""
    target = tmp_path / "empty.csv"
    target.write_bytes(b"")

    assert read_tail_bytes(target, n_bytes=4096) == b""


def test_read_tail_bytes_missing_file_raises_domain_error(tmp_path: Path) -> None:
    """A missing source file should raise the domain-specific exception."""
    missing = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileSampleReadError):
        read_tail_bytes(missing)


def test_read_tail_bytes_never_reads_the_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sampling the tail of a large file must never read from the start.

    A naive implementation might read the whole file and slice the last
    bytes in memory; this proves the implementation instead performs a
    single bounded read anchored at the end of the file.
    """
    marker = b"TAIL_MARKER_0123456789"
    target = tmp_path / "big_tail.csv"
    target.write_bytes(b"x" * 5_000_000 + marker)

    read_calls = _spy_on_open(monkeypatch)

    tail = read_tail_bytes(target, n_bytes=len(marker))

    assert tail == marker
    assert read_calls == [len(marker)]


def test_read_tail_bytes_on_real_sample_fixture_ends_with_last_row() -> None:
    """The tail of the bundled sample.csv should contain its final row."""
    tail = read_tail_bytes(SAMPLE_CSV_PATH, n_bytes=200)

    assert b"traslado" in tail


def test_detect_encoding_defaults_to_utf8_for_ascii_bytes() -> None:
    """Pure ASCII input should resolve to utf-8 rather than a narrow alias."""
    encoding = detect_encoding(b"col_a,col_b,col_c\n1,2,3\n")

    assert encoding.lower() == "utf-8"


def test_decode_sample_replaces_undecodable_bytes_on_unknown_encoding() -> None:
    """An unrecognized encoding name should fall back to utf-8 with replacement."""
    text = decode_sample(b"a;b;c\n", encoding="not-a-real-encoding")

    assert "a;b;c" in text


@pytest.mark.parametrize("n_bytes", [0, -1])
def test_read_sample_bytes_rejects_non_positive_budget(tmp_path: Path, n_bytes: int) -> None:
    """A non-positive head budget must be rejected: ``read(-1)`` reads the whole file."""
    target = tmp_path / "data.csv"
    target.write_bytes(b"a,b,c\n")

    with pytest.raises(ValueError, match="n_bytes"):
        read_sample_bytes(target, n_bytes=n_bytes)


def test_read_tail_bytes_rejects_negative_budget(tmp_path: Path) -> None:
    """A negative tail budget must be rejected rather than silently misread."""
    target = tmp_path / "data.csv"
    target.write_bytes(b"a,b,c\n")

    with pytest.raises(ValueError, match="n_bytes"):
        read_tail_bytes(target, n_bytes=-1)


def test_read_tail_bytes_zero_budget_returns_empty_bytes(tmp_path: Path) -> None:
    """A zero tail budget is valid and yields an empty sample."""
    target = tmp_path / "data.csv"
    target.write_bytes(b"a,b,c\n")

    assert read_tail_bytes(target, n_bytes=0) == b""
