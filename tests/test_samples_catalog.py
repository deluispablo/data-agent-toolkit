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

from exceptions import InspectionFailedError, ModelInvocationError
from generate_samples import CASES, SampleCase, build_manifest
from inspector import (
    DEFAULT_SAMPLE_BYTES,
    DEFAULT_TAIL_BYTES,
    decode_sample,
    detect_encoding,
    inspect_csv,
    read_sample_bytes,
    read_tail_bytes,
)

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "agents" / "csv_inspector" / "samples"
MANIFEST_PATH = SAMPLES_DIR / "manifest.json"

# Non-fixture files living alongside the generated CSVs.
_NON_FIXTURE_FILENAMES = {"generate_samples.py", "manifest.json"}


def _load_manifest() -> dict[str, dict[str, Any]]:
    """Load the sample catalog's ground-truth manifest."""
    manifest: dict[str, dict[str, Any]] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return manifest


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
# Generator <-> committed catalog drift
# ---------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.filename)
def test_fixture_on_disk_matches_generator_byte_for_byte(case: SampleCase) -> None:
    """Committed fixtures must equal what generate_samples.py produces.

    Catches both a forgotten regeneration and silent rewriting by git
    (e.g. ``core.autocrlf`` normalizing the mixed CRLF/LF fixture).
    """
    assert (SAMPLES_DIR / case.filename).read_bytes() == case.raw_bytes


def test_manifest_matches_generator() -> None:
    """The committed manifest.json must equal the generator's current output."""
    assert build_manifest(CASES) == MANIFEST


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
# Footer fixtures exercise the real head/tail split (issue #2)
# ---------------------------------------------------------------------

_FOOTER_FIXTURES = sorted(
    name for name, entry in MANIFEST.items() if entry["expected"].get("footer_lines")
)


def test_catalog_has_footer_fixtures_including_header_and_footer_combined() -> None:
    """Guard against the parametrized footer tests below silently running on nothing."""
    assert "footer_summary_totals.csv" in _FOOTER_FIXTURES
    assert "footer_end_marker.csv" in _FOOTER_FIXTURES
    assert "header_and_footer_combined.csv" in _FOOTER_FIXTURES


@pytest.mark.parametrize("filename", _FOOTER_FIXTURES)
def test_footer_fixture_is_larger_than_the_default_sampling_budget(filename: str) -> None:
    """Footer fixtures must be production-sized: head and tail never meet in the middle."""
    size = (SAMPLES_DIR / filename).stat().st_size

    assert size > DEFAULT_SAMPLE_BYTES + DEFAULT_TAIL_BYTES


@pytest.mark.parametrize("filename", _FOOTER_FIXTURES)
def test_footer_is_only_visible_through_the_default_tail_window(filename: str) -> None:
    """Every non-blank footer line sits in the default tail window and never in the head."""
    path = SAMPLES_DIR / filename
    head_raw = read_sample_bytes(path, n_bytes=DEFAULT_SAMPLE_BYTES)
    encoding = detect_encoding(head_raw)
    head_text = decode_sample(head_raw, encoding)
    tail_text = decode_sample(read_tail_bytes(path, n_bytes=DEFAULT_TAIL_BYTES), encoding)

    footer_lines = [line for line in MANIFEST[filename]["expected"]["footer_lines"] if line]
    assert footer_lines, f"{filename} should list at least one non-blank footer line."
    for line in footer_lines:
        assert line in tail_text
        assert line not in head_text


@pytest.mark.parametrize("filename", _FOOTER_FIXTURES)
def test_inspect_csv_sends_the_footer_to_the_model_in_the_tail_section(filename: str) -> None:
    """End to end with default budgets: the prompt's tail section carries the footer."""
    prompts: list[str] = []

    def fake_invoker(prompt: str, model: str) -> str:
        prompts.append(prompt)
        raise ModelInvocationError("prompt captured")

    with pytest.raises(InspectionFailedError):
        inspect_csv(SAMPLES_DIR / filename, model_invoker=fake_invoker)

    tail_section = prompts[0].split("--- TAIL SAMPLE START", 1)[1]
    for line in MANIFEST[filename]["expected"]["footer_lines"]:
        if line:
            assert line in tail_section


def test_combined_fixture_keeps_the_header_preamble_in_the_head() -> None:
    """The header+footer fixture: banner and header in the head, footer only in the tail."""
    path = SAMPLES_DIR / "header_and_footer_combined.csv"
    head_text = decode_sample(read_sample_bytes(path, n_bytes=DEFAULT_SAMPLE_BYTES), "utf-8")
    header_row_index = MANIFEST["header_and_footer_combined.csv"]["expected"]["header_row_index"]

    lines = head_text.splitlines()
    assert all(line.startswith("#") for line in lines[:header_row_index])
    assert lines[header_row_index] == "Fecha;Cliente;Concepto;Importe"
