"""Deterministic, LLM-free tests over the samples/ fixture catalog.

These tests validate only the byte-sampling and encoding-detection layer
against every fixture in ``agents/csv_inspector/samples/`` — never the
LLM's inferred dialect/schema, which is covered separately by the manual
``eval_samples.py`` harness (non-deterministic, run against a live Ollama
instance and not part of this suite).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from inspector import decode_sample, detect_encoding, read_sample_bytes, read_tail_bytes

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "agents" / "csv_inspector" / "samples"
MANIFEST_PATH = SAMPLES_DIR / "manifest.json"

# Non-fixture files living alongside the generated CSVs.
_NON_FIXTURE_FILENAMES = {"generate_samples.py", "manifest.json"}


def _load_manifest() -> dict[str, dict[str, Any]]:
    """Load the sample catalog's ground-truth manifest."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


MANIFEST: dict[str, dict[str, Any]] = _load_manifest()


# ---------------------------------------------------------------------
# Manifest <-> filesystem consistency
# ---------------------------------------------------------------------


def test_manifest_file_exists() -> None:
    """The generated manifest must exist alongside the fixtures."""
    assert MANIFEST_PATH.exists()


def test_every_manifest_entry_has_a_file_on_disk() -> None:
    """Every fixture referenced in the manifest must actually be present."""
    missing = [name for name in MANIFEST if not (SAMPLES_DIR / name).exists()]
    assert missing == []


def test_every_fixture_file_is_listed_in_manifest() -> None:
    """No fixture on disk should be missing from the manifest (or vice versa)."""
    on_disk = {
        entry.name
        for entry in SAMPLES_DIR.iterdir()
        if entry.is_file() and entry.name not in _NON_FIXTURE_FILENAMES
    }
    unlisted = on_disk - set(MANIFEST)
    assert unlisted == set(), f"Fixtures on disk but missing from manifest.json: {unlisted}"


@pytest.mark.parametrize("filename", sorted(MANIFEST))
def test_manifest_entry_has_category_and_description(filename: str) -> None:
    """Every catalog entry must be self-documenting (category + description)."""
    entry = MANIFEST[filename]
    assert entry["category"]
    assert entry["description"]
    assert isinstance(entry["known_limitation"], bool)


# ---------------------------------------------------------------------
# Byte-sampling layer against the whole catalog
# ---------------------------------------------------------------------


@pytest.mark.parametrize("filename", sorted(MANIFEST))
def test_head_and_tail_reads_do_not_raise_for_any_fixture(filename: str) -> None:
    """Every fixture, however malformed, must be sampleable without raising."""
    path = SAMPLES_DIR / filename

    head = read_sample_bytes(path, n_bytes=4096)
    tail = read_tail_bytes(path, n_bytes=4096)

    assert isinstance(head, bytes)
    assert isinstance(tail, bytes)


def test_empty_file_reads_as_empty_bytes_from_both_ends() -> None:
    """The zero-byte fixture must yield an empty sample from head and tail."""
    path = SAMPLES_DIR / "empty_file.csv"

    assert read_sample_bytes(path, n_bytes=4096) == b""
    assert read_tail_bytes(path, n_bytes=4096) == b""


# ---------------------------------------------------------------------
# Encoding-specific deterministic checks
# ---------------------------------------------------------------------


def test_utf8_bom_fixture_starts_with_the_bom_marker() -> None:
    """The UTF-8-BOM fixture must carry the literal BOM byte sequence."""
    head = read_sample_bytes(SAMPLES_DIR / "encoding_utf8_bom.csv", n_bytes=16)

    assert head.startswith(b"\xef\xbb\xbf")


def test_utf16le_bom_fixture_starts_with_the_bom_marker() -> None:
    """The UTF-16LE fixture must carry its BOM and show the classic null-byte pattern."""
    head = read_sample_bytes(SAMPLES_DIR / "encoding_utf16le_bom.csv", n_bytes=16)

    assert head.startswith(b"\xff\xfe")
    # ASCII-range characters encoded as UTF-16LE are followed by a 0x00 byte.
    assert head[3] == 0x00


def test_latin1_fixture_is_not_valid_utf8() -> None:
    """The Latin-1 fixture's accented characters must not be valid UTF-8 bytes."""
    raw = read_sample_bytes(SAMPLES_DIR / "encoding_latin1.csv", n_bytes=4096)

    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")


def test_latin1_fixture_decodes_cleanly_as_latin1() -> None:
    """The Latin-1 fixture must round-trip cleanly once decoded with the right codec."""
    raw = read_sample_bytes(SAMPLES_DIR / "encoding_latin1.csv", n_bytes=4096)

    text = raw.decode("latin-1")

    assert "García" in text


# ---------------------------------------------------------------------
# Footer visibility in the tail sample
# ---------------------------------------------------------------------


@pytest.mark.parametrize("filename", ["footer_summary_totals.csv", "footer_end_marker.csv"])
def test_footer_content_is_visible_in_the_tail_sample(filename: str) -> None:
    """A fixture with a documented footer must expose it within a small tail window."""
    entry = MANIFEST[filename]
    footer_lines: list[str] = entry["expected"].get("footer_lines") or []
    non_empty_footer_lines = [line for line in footer_lines if line]
    assert non_empty_footer_lines, f"{filename} manifest entry should list non-empty footer_lines."

    tail_raw = read_tail_bytes(SAMPLES_DIR / filename, n_bytes=200)
    tail_text = decode_sample(tail_raw, detect_encoding(tail_raw))

    for line in non_empty_footer_lines:
        assert line in tail_text
