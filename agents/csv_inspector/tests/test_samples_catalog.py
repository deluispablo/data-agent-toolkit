"""Deterministic, LLM-free tests over the samples/ fixture catalog.

These tests validate only the byte-sampling and encoding-detection layer
against every fixture in ``agents/csv_inspector/samples/`` — never the
LLM's inferred dialect/schema, which is covered separately by the manual
``eval_samples.py`` harness (non-deterministic, run against a live Ollama
instance and not part of this suite).
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import pytest

from csv_inspector import InspectionFailedError, ModelInvocationError, inspect_csv
from csv_inspector._encoding import decode_sample, detect_encoding
from csv_inspector._sampling import (
    DEFAULT_SAMPLE_BYTES,
    DEFAULT_TAIL_BYTES,
    MAX_HEAD_LINES,
    MAX_TAIL_LINES,
    _PathReader,
    sample_source,
)
from generate_samples import CASES, SampleCase, build_manifest, derive_columns
from matrix import MATRIX, FixtureSpec, render

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
MANIFEST_PATH = SAMPLES_DIR / "manifest.json"

# Non-fixture files living alongside the generated CSVs.
_NON_FIXTURE_FILENAMES = {"generate_samples.py", "matrix.py", "manifest.json"}


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

    head = _PathReader(path).head(4096)
    tail = _PathReader(path).tail(0, 4096, 1)

    assert isinstance(head, bytes)
    assert isinstance(tail, bytes)


def test_empty_file_reads_as_empty_bytes_from_both_ends() -> None:
    """The zero-byte fixture must yield an empty sample from head and tail."""
    path = SAMPLES_DIR / "empty_file.csv"

    assert _PathReader(path).head(4096) == b""
    assert _PathReader(path).tail(0, 4096, 1) == b""


# ---------------------------------------------------------------------
# Encoding-specific deterministic checks
# ---------------------------------------------------------------------


def test_utf8_bom_fixture_starts_with_the_bom_marker() -> None:
    """The UTF-8-BOM fixture must carry the literal BOM byte sequence."""
    head = _PathReader(SAMPLES_DIR / "encoding_utf8_bom.csv").head(16)

    assert head.startswith(b"\xef\xbb\xbf")


def test_utf16le_bom_fixture_starts_with_the_bom_marker() -> None:
    """The UTF-16LE fixture must carry its BOM and show the classic null-byte pattern."""
    head = _PathReader(SAMPLES_DIR / "encoding_utf16le_bom.csv").head(16)

    assert head.startswith(b"\xff\xfe")
    # ASCII-range characters encoded as UTF-16LE are followed by a 0x00 byte.
    assert head[3] == 0x00


def test_latin1_fixture_is_not_valid_utf8() -> None:
    """The Latin-1 fixture's accented characters must not be valid UTF-8 bytes."""
    raw = _PathReader(SAMPLES_DIR / "encoding_latin1.csv").head(4096)

    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")


def test_latin1_fixture_decodes_cleanly_as_latin1() -> None:
    """The Latin-1 fixture must round-trip cleanly once decoded with the right codec."""
    raw = _PathReader(SAMPLES_DIR / "encoding_latin1.csv").head(4096)

    text = raw.decode("latin-1")

    assert "García" in text


# ---------------------------------------------------------------------
# Footer fixtures exercise the real head/tail split (issue #2)
# ---------------------------------------------------------------------

# A known-limitation footer (longer than the tail window) is checked on its own below.
_FOOTER_FIXTURES = sorted(
    name
    for name, entry in MANIFEST.items()
    if entry["expected"].get("footer_lines") and not entry["known_limitation"]
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
    head_raw = _PathReader(path).head(DEFAULT_SAMPLE_BYTES)
    encoding = detect_encoding(head_raw)
    head_text = decode_sample(head_raw, encoding)
    tail_text = decode_sample(_PathReader(path).tail(0, DEFAULT_TAIL_BYTES, 1), encoding)

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
    head_text = decode_sample(_PathReader(path).head(DEFAULT_SAMPLE_BYTES), "utf-8")
    header_row_index = MANIFEST["header_and_footer_combined.csv"]["expected"]["header_row_index"]

    lines = head_text.splitlines()
    assert all(line.startswith("#") for line in lines[:header_row_index])
    assert lines[header_row_index] == "Fecha;Cliente;Concepto;Importe"


# ---------------------------------------------------------------------
# Expected column names (issue #123)
# ---------------------------------------------------------------------

_HEADER_FIXTURES = sorted(
    name
    for name, entry in MANIFEST.items()
    if entry["expected"].get("has_header", True)
    and entry["expected"].get("header_row_index") is not None
)


def test_catalog_has_header_fixtures() -> None:
    """Guard against the parametrized column tests below silently running on nothing."""
    assert "delimiter_comma.csv" in _HEADER_FIXTURES
    assert "whitespace_padded_fields.csv" in _HEADER_FIXTURES
    assert "header_none_data_only.csv" not in _HEADER_FIXTURES


@pytest.mark.parametrize("filename", sorted(MANIFEST))
def test_every_manifest_entry_lists_expected_columns(filename: str) -> None:
    """Every fixture has a ``columns`` list, non-empty whenever the file has a header."""
    columns = MANIFEST[filename]["expected"]["columns"]

    assert isinstance(columns, list)
    if filename in _HEADER_FIXTURES:
        assert columns
        assert all(isinstance(name, str) for name in columns)


@pytest.mark.parametrize("filename", _HEADER_FIXTURES)
def test_expected_columns_equal_the_header_line_parsed_with_the_manifest_dialect(
    filename: str,
) -> None:
    """``columns`` is the file's header line, split by ``csv`` with the manifest dialect."""
    expected = MANIFEST[filename]["expected"]
    codec = expected["encoding"].split(" or ")[0].strip()
    text = (SAMPLES_DIR / filename).read_bytes().decode(codec).lstrip("﻿")
    header_line = text.splitlines()[expected["header_row_index"]]

    fields = next(
        csv.reader(
            [header_line],
            delimiter=expected["delimiter"],
            quotechar=expected.get("quotechar") or '"',
            escapechar=expected.get("escapechar"),
        )
    )

    assert expected["columns"] == fields


def test_header_less_fixture_expects_positional_column_names() -> None:
    """A file without a header row expects ``column_1..N``, one per field of its first row."""
    expected = MANIFEST["header_none_data_only.csv"]["expected"]

    assert expected["columns"] == ["column_1", "column_2", "column_3"]


def test_whitespace_padded_header_keeps_the_names_as_written() -> None:
    """Expected names are verbatim, surrounding spaces included, as grounding reports them."""
    expected = MANIFEST["whitespace_padded_fields.csv"]["expected"]

    assert expected["columns"] == ["Fecha ", " Cliente ", " Importe"]


@pytest.mark.parametrize(
    "case", [case for case in CASES if case.columns is not None], ids=lambda case: case.filename
)
def test_column_overrides_agree_with_the_derivation_when_it_can_parse(case: SampleCase) -> None:
    """An explicit ``SampleCase.columns`` never contradicts what the bytes say."""
    derived = derive_columns(case)

    if derived is not None:
        assert case.columns == derived


# ---------------------------------------------------------------------
# Hard-case and known-limitation fixtures (issue #125)
# ---------------------------------------------------------------------

_HARD_CASE_FILENAMES = {
    "header_none_after_preamble.csv",
    "quoting_newline_in_head_window.csv",
    "quoting_newline_in_tail_window.csv",
    "footer_longer_than_tail_window.csv",
    "footer_like_data_row_numeric_label.csv",
    "footer_after_total_energies_row.csv",
    "delimiter_semicolon_decimal_comma.csv",
    "delimiter_tab_commas_quoted_header.tsv",
    "delimiter_pipe_inside_quotes.csv",
    "encoding_cp1252_tail_only.csv",
    "single_column.csv",
    "single_data_row.csv",
    "exactly_head_window_size.csv",
    "header_duplicate_and_blank_names.csv",
    "data_contains_sample_marker.csv",
    "header_years.csv",
}
_HARD_CASES = [case for case in CASES if case.filename in _HARD_CASE_FILENAMES]


def test_catalog_has_every_hard_case() -> None:
    """Every hand-written hard case is in the catalog."""
    assert {case.filename for case in _HARD_CASES} == _HARD_CASE_FILENAMES


@pytest.mark.parametrize("case", _HARD_CASES, ids=lambda case: case.filename)
def test_hard_case_notes_name_the_rule_it_guards(case: SampleCase) -> None:
    """Each hard case has a one-line note naming the rule it guards."""
    assert case.notes
    assert "\n" not in case.notes
    assert "_grounding." in case.notes or "_sampling." in case.notes or "#105" in case.notes


def test_header_less_fixture_after_preamble_expects_positional_names() -> None:
    """Preamble lines do not set the column count of a header-less file."""
    expected = MANIFEST["header_none_after_preamble.csv"]["expected"]

    assert expected["has_header"] is False
    assert expected["columns"] == ["column_1", "column_2", "column_3"]


@pytest.mark.parametrize(
    ("filename", "in_head"),
    [("quoting_newline_in_head_window.csv", True), ("quoting_newline_in_tail_window.csv", False)],
)
def test_multiline_record_sits_in_the_intended_window(filename: str, in_head: bool) -> None:
    """The quoted line break lies only in the head window, or only in the tail window."""
    path = SAMPLES_DIR / filename
    marker = b'"Pedido urgente\nentregado'
    head = _PathReader(path).head(DEFAULT_SAMPLE_BYTES)
    tail = _PathReader(path).tail(0, DEFAULT_TAIL_BYTES, 1)

    assert path.stat().st_size > DEFAULT_SAMPLE_BYTES + DEFAULT_TAIL_BYTES
    assert (marker in head, marker in tail) == (in_head, not in_head)


def test_long_footer_does_not_fit_the_default_tail_window() -> None:
    """The known limitation holds: the footer's last line is in the tail, its first is not."""
    filename = "footer_longer_than_tail_window.csv"
    footer = MANIFEST[filename]["expected"]["footer_lines"]
    samples = sample_source(SAMPLES_DIR / filename, DEFAULT_SAMPLE_BYTES, DEFAULT_TAIL_BYTES)

    assert MANIFEST[filename]["known_limitation"] is True
    assert samples.tail_text is not None
    assert len("\n".join(footer).encode("utf-8")) > DEFAULT_TAIL_BYTES
    assert footer[-1] in samples.tail_text
    assert footer[0] not in samples.tail_text


def test_cp1252_tail_is_detected_although_the_head_is_ascii() -> None:
    """The head is pure ASCII; the accented tail still moves the encoding off UTF-8."""
    path = SAMPLES_DIR / "encoding_cp1252_tail_only.csv"
    head = _PathReader(path).head(DEFAULT_SAMPLE_BYTES)

    samples = sample_source(path, DEFAULT_SAMPLE_BYTES, DEFAULT_TAIL_BYTES)

    assert head.isascii()
    assert samples.encoding.lower() in {"windows-1252", "cp1252", "iso-8859-1", "latin-1"}
    assert samples.tail_text is not None
    assert "Muñoz Hermanos" in samples.tail_text


def test_file_of_exactly_the_head_window_is_sampled_whole() -> None:
    """A file exactly one head window long is fully covered, and cut only by the line bounds."""
    path = SAMPLES_DIR / "exactly_head_window_size.csv"
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines(keepends=True)

    samples = sample_source(path, DEFAULT_SAMPLE_BYTES, DEFAULT_TAIL_BYTES)

    assert len(raw) == DEFAULT_SAMPLE_BYTES
    assert raw.endswith(b"\n")
    assert samples.covers_whole_file is True
    assert samples.head_text == "".join(lines[:MAX_HEAD_LINES])
    assert samples.tail_text == "".join(lines[-MAX_TAIL_LINES:])
    assert samples.lines_omitted == len(lines) - MAX_HEAD_LINES - MAX_TAIL_LINES


def test_duplicate_and_blank_column_names_are_kept_verbatim() -> None:
    """A blank index name and duplicated names are expected as written."""
    expected = MANIFEST["header_duplicate_and_blank_names.csv"]["expected"]

    assert expected["columns"] == ["", "id", "id", "value"]


# ---------------------------------------------------------------------
# Parametric fixtures rendered from samples/matrix.py (issue #124)
# ---------------------------------------------------------------------

_GENERATED_CASES = [case for case in CASES if case.generated]


def test_catalog_has_at_least_sixty_fixtures() -> None:
    """The matrix grows the catalog to 60 fixtures or more."""
    assert len(MANIFEST) >= 60


def test_generated_cases_are_the_rendered_matrix_in_order() -> None:
    """Every spec renders exactly one fixture, named after its unique slug."""
    assert [case.filename for case in _GENERATED_CASES] == [spec.filename for spec in MATRIX]
    assert len({spec.slug for spec in MATRIX}) == len(MATRIX)
    assert [render(spec) for spec in MATRIX] == _GENERATED_CASES


@pytest.mark.parametrize("filename", sorted(MANIFEST))
def test_manifest_flags_exactly_the_matrix_fixtures_as_generated(filename: str) -> None:
    """``generated`` is true for the ``gen_`` files and false for hand-written ones."""
    assert MANIFEST[filename]["generated"] is filename.startswith("gen_")


def test_matrix_covers_the_planned_dimensions() -> None:
    """Footer kinds x widths, encodings x line endings, preambles, sizes and combos."""
    footer_widths = {(spec.footer_kind, spec.n_columns) for spec in MATRIX}
    encodings = {(spec.encoding, spec.bom, spec.newline) for spec in MATRIX}
    sizes = [len(case.raw_bytes) for case in _GENERATED_CASES]

    for kind in ("none", "totals", "marker", "timestamp", "blank_totals"):
        assert {(kind, 3), (kind, 40)} <= footer_widths
    for encoding, bom in (
        ("utf-8", False),
        ("utf-8", True),
        ("cp1252", False),
        ("utf-16-le", True),
    ):
        assert {(encoding, bom, "\n"), (encoding, bom, "\r\n")} <= encodings
    assert {0, 1, 5} <= {spec.preamble_lines for spec in MATRIX}
    assert sum(size > 64 * 1024 for size in sizes) >= 2
    assert max(sizes) < 1024 * 1024
    assert sum(spec.category == "combo" for spec in MATRIX) >= 3


@pytest.mark.parametrize("case", _GENERATED_CASES, ids=lambda case: case.filename)
def test_generated_fixture_header_and_footer_match_the_manifest(case: SampleCase) -> None:
    """``csv`` with the manifest dialect reads the header; the last lines are the footer."""
    expected = MANIFEST[case.filename]["expected"]
    codec = expected["encoding"].split(" or ")[0].strip()
    text = (SAMPLES_DIR / case.filename).read_bytes().decode(codec).lstrip("﻿")
    rows = list(
        csv.reader(
            io.StringIO(text, newline=""),
            delimiter=expected["delimiter"],
            quotechar=expected["quotechar"],
        )
    )

    if expected.get("has_header", True):
        assert rows[expected["header_row_index"]] == expected["columns"]
    else:
        assert expected["header_row_index"] is None
        assert len(rows[0]) == len(expected["columns"])
    lines = text.splitlines()
    footer_start = len(lines) - expected["footer_rows_to_skip"]
    assert lines[footer_start:] == expected["footer_lines"]


def test_long_line_fixture_has_no_complete_line_in_the_default_head() -> None:
    """Issue #125 case 13: the default head window ends inside the header line."""
    filename = "gen_very_wide_long_lines.csv"

    assert b"\n" not in _PathReader(SAMPLES_DIR / filename).head(DEFAULT_SAMPLE_BYTES)
    assert MANIFEST[filename]["known_limitation"] is True
    assert len(MANIFEST[filename]["expected"]["columns"]) == 200


@pytest.mark.parametrize(
    "overrides",
    [
        {"encoding": "utf-16-le"},
        {"encoding": "cp1252", "bom": True},
        {"newline": "\r"},
        {"preamble_lines": 6},
        {"has_header": False, "preamble_lines": 1},
        {"ragged": True, "footer_kind": "totals"},
    ],
)
def test_inconsistent_fixture_specs_are_rejected(overrides: dict[str, Any]) -> None:
    """The renderer only accepts specs whose ground truth it can state exactly."""
    with pytest.raises(ValueError, match="bad"):
        FixtureSpec(slug="bad", category="structural", **overrides)
