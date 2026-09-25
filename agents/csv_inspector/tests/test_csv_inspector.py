"""Unit tests for the csv_inspector agent.

The LLM backend is injected via ``inspect_csv``'s ``model_invoker``
parameter, so the full pipeline — including the fallback-model path and all
domain error branches — is exercised without requiring a running Ollama
instance or network access.

The byte-sampling layer (``read_sample_bytes`` / ``read_tail_bytes``) is
tested not only for correctness but also, via a ``Path.open()`` spy, for *how*
it reads: these functions must never fall back to loading a whole file into
memory, since that is the entire point of sampling head/tail byte windows
against multi-gigabyte production files.
"""

from __future__ import annotations

import asyncio
import codecs
import csv
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any, get_args

import pytest
from pydantic import SecretStr, ValidationError

from csv_inspector import (
    BackendConfigurationError,
    ColumnSchema,
    ColumnType,
    CSVInspectionResult,
    EmptySampleError,
    FileSampleReadError,
    InspectionFailedError,
    ModelInvocationError,
    ResponseParsingError,
    Settings,
    ainspect_csv,
    inspect_csv,
)
from csv_inspector._config import DEFAULT_MODEL, FALLBACK_MODEL
from csv_inspector._encoding import decode_sample, detect_encoding
from csv_inspector._grounding import (
    _extends_footer,
    _field_shape,
    _first_row_is_data,
    ground_in_samples,
)
from csv_inspector._invokers import ModelInvoker, invoke_ollama_model
from csv_inspector._prompt import _extract_json_payload, build_prompt, parse_and_validate
from csv_inspector._sampling import (
    MAX_SAMPLE_BYTES,
    read_sample_bytes,
    read_tail_bytes,
)
from fakes import install_fake_ollama
from generate_samples import CASES, SampleCase, derive_columns
from matrix import MATRIX, render

SAMPLE_CSV_PATH = Path(__file__).resolve().parent.parent / "sample.csv"

VALID_RESULT_PAYLOAD: dict[str, object] = {
    "encoding": "utf-8",
    "delimiter": ";",
    "quotechar": '"',
    "escapechar": None,
    "doublequote": True,
    "header_row_index": 2,
    "footer_lines": [],
    "columns": [
        {
            "name": "Fecha",
            "inferred_type": "date",
            "nullable": False,
            "example_values": ["2024-01-15", "2024-01-16"],
        },
        {
            "name": "Importe",
            "inferred_type": "float",
            "nullable": False,
            "example_values": ["1250.50", "890.00"],
        },
    ],
    "confidence": 0.95,
    "notes": None,
}


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


def _tail_section(prompt: str) -> str:
    """Return the text between the tail sample markers of a built prompt."""
    start = prompt.index("--- TAIL SAMPLE START")
    end = prompt.index("--- TAIL SAMPLE END ---")
    return prompt[prompt.index("\n", start) + 1 : end]


def _capture_prompts(prompts: list[str]) -> ModelInvoker:
    """Build a fake model invoker that records prompts and returns a valid payload."""

    def fake_invoker(prompt: str, model: str) -> str:
        prompts.append(prompt)
        return json.dumps(VALID_RESULT_PAYLOAD)

    return fake_invoker


# ---------------------------------------------------------------------
# read_sample_bytes (head)
# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------
# read_tail_bytes
# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------
# encoding detection and decoding
# ---------------------------------------------------------------------


def test_detect_encoding_defaults_to_utf8_for_ascii_bytes() -> None:
    """Pure ASCII input should resolve to utf-8 rather than a narrow alias."""
    encoding = detect_encoding(b"col_a,col_b,col_c\n1,2,3\n")

    assert encoding.lower() == "utf-8"


def test_decode_sample_replaces_undecodable_bytes_on_unknown_encoding() -> None:
    """An unrecognized encoding name should fall back to utf-8 with replacement."""
    text = decode_sample(b"a;b;c\n", encoding="not-a-real-encoding")

    assert "a;b;c" in text


# ---------------------------------------------------------------------
# prompt construction and response parsing helpers
# ---------------------------------------------------------------------


def test_build_prompt_embeds_head_sample_and_encoding_hint() -> None:
    """The generated prompt must include the head sample and encoding hint."""
    prompt = build_prompt(head_sample="Fecha;Cliente\n2024-01-15;Acme", detected_encoding="utf-8")

    assert "Fecha;Cliente" in prompt
    assert "'utf-8'" in prompt


def test_build_prompt_omits_tail_section_when_tail_sample_is_none() -> None:
    """No tail section should appear in the prompt when there is no tail sample."""
    prompt = build_prompt(head_sample="a,b,c\n1,2,3\n", detected_encoding="utf-8", tail_sample=None)

    assert "TAIL SAMPLE START" not in prompt
    assert "contains the ENTIRE file" in prompt


def test_build_prompt_forbids_a_footer_when_the_end_was_not_sampled() -> None:
    """A truncated head with no tail must not be presented as the whole file (issue #10)."""
    prompt = build_prompt(
        head_sample="a,b,c\n1,2,3\n4,5",
        detected_encoding="utf-8",
        tail_sample=None,
        covers_whole_file=False,
    )

    assert "TAIL SAMPLE START" not in prompt
    assert "contains the ENTIRE file" not in prompt
    assert "its end was not sampled" in prompt
    assert '"footer_lines" must be []' in prompt


def test_build_prompt_includes_tail_section_and_mid_line_caveat() -> None:
    """A provided tail sample must appear in its own labeled section with a caveat."""
    prompt = build_prompt(
        head_sample="a,b,c\n1,2,3\n",
        detected_encoding="utf-8",
        tail_sample=",99\nTOTAL,,999\n",
    )

    assert "TAIL SAMPLE START" in prompt
    assert ",99\nTOTAL,,999\n" in prompt
    assert "may start mid-line" in prompt


@pytest.mark.parametrize("tail_sample", [None, "2,3\nTOTAL,,5\n"])
def test_build_prompt_gives_concrete_footer_rules(tail_sample: str | None) -> None:
    """Footers get explicit rules and examples in both prompt variants.

    Regression test for issue #2: with only a metadata example in the prompt,
    the model filed end-of-report markers and timestamps under metadata.
    """
    prompt = build_prompt(
        head_sample="a,b\n1,2\n", detected_encoding="utf-8", tail_sample=tail_sample
    )

    assert "FOOTER (end of the file)" in prompt
    for example in ("TOTAL,,4241.25", "--- Fin del informe ---", "Generado el 2024-01-20"):
        assert example in prompt
    assert "Footer lines are never part of the header preamble" in prompt


def test_build_prompt_reads_footer_only_from_the_tail_when_present() -> None:
    """With a tail sample, the head's (mid-file) last line must not be read as a footer."""
    prompt = build_prompt(head_sample="a,b\n1,2\n", detected_encoding="utf-8", tail_sample="9,9\n")

    assert "check the last lines of the TAIL sample" in prompt
    assert "read footer lines ONLY from its last lines" in prompt


def test_build_prompt_no_longer_requests_derived_or_removed_fields() -> None:
    """The model is asked only for footer_lines: the count is derived, metadata is gone."""
    prompt = build_prompt(head_sample="a,b\n1,2\n", detected_encoding="utf-8")

    assert '"footer_lines"' in prompt
    assert "footer_rows_to_skip" not in prompt
    assert "metadata_lines" not in prompt


# ---------------------------------------------------------------------
# output contract
# ---------------------------------------------------------------------


def test_footer_rows_to_skip_is_derived_from_footer_lines() -> None:
    """The footer count always equals the number of footer lines, blank ones included."""
    payload = {**VALID_RESULT_PAYLOAD, "footer_lines": ["", "--- Fin ---", "Generado el X"]}

    result = CSVInspectionResult.model_validate(payload)

    assert result.footer_rows_to_skip == 3
    assert result.model_dump()["footer_rows_to_skip"] == 3


def test_contradictory_or_removed_model_fields_are_ignored() -> None:
    """A stale count or a legacy metadata_lines key from the model cannot leak into the result."""
    payload = {
        **VALID_RESULT_PAYLOAD,
        "footer_lines": ["TOTAL,,999"],
        "footer_rows_to_skip": 0,
        "metadata_lines": ["# banner"],
    }

    result = CSVInspectionResult.model_validate(payload)

    assert result.footer_rows_to_skip == 1
    assert "metadata_lines" not in result.model_dump()


# ---------------------------------------------------------------------
# Header-less files (issue #94)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected_index"),
    [
        pytest.param({"has_header": False, "header_row_index": None}, None, id="explicit"),
        pytest.param({"header_row_index": None}, None, id="null-infers-no-header"),
        pytest.param({"header_row_index": -1}, None, id="minus-one-means-null"),
        pytest.param({"header_row_index": 0}, 0, id="index-infers-header"),
        pytest.param({"has_header": True, "header_row_index": 3}, 3, id="explicit-header"),
    ],
)
def test_header_less_answers_validate(
    answer: dict[str, object], expected_index: int | None
) -> None:
    """Null (or -1) means "no header row"; has_header follows unless given."""
    result = CSVInspectionResult.model_validate({**VALID_RESULT_PAYLOAD, **answer})

    assert result.header_row_index == expected_index
    assert result.has_header is (expected_index is not None)


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        pytest.param(
            {"has_header": True, "header_row_index": None}, "required when has_header", id="null"
        ),
        pytest.param({"has_header": False, "header_row_index": 0}, "must be null when", id="index"),
        pytest.param({"header_row_index": -2}, "greater than or equal to 0", id="negative"),
    ],
)
def test_contradictory_header_answers_fail(answer: dict[str, object], message: str) -> None:
    """has_header and header_row_index must agree, with a clear message."""
    with pytest.raises(ValidationError, match=message):
        CSVInspectionResult.model_validate({**VALID_RESULT_PAYLOAD, **answer})


def test_a_missing_header_row_index_still_fails() -> None:
    """Omitting both fields is a malformed answer, not a header-less file."""
    payload = {k: v for k, v in VALID_RESULT_PAYLOAD.items() if k != "header_row_index"}

    with pytest.raises(ValidationError, match="required when has_header"):
        CSVInspectionResult.model_validate(payload)


def test_a_header_less_fixture_keeps_its_positional_columns() -> None:
    """End to end: grounding never renames positional columns or invents a header."""
    fixture = SAMPLE_CSV_PATH.parent / "samples" / "header_none_data_only.csv"
    answer = {
        **VALID_RESULT_PAYLOAD,
        "delimiter": ",",
        "has_header": False,
        "header_row_index": None,
        "columns": [
            {"name": f"column_{n}", "inferred_type": kind}
            for n, kind in enumerate(["date", "string", "float"], start=1)
        ],
    }

    result = inspect_csv(fixture, model_invoker=lambda prompt, model: json.dumps(answer))

    assert result.has_header is False
    assert result.header_row_index is None
    assert [column.name for column in result.columns] == ["column_1", "column_2", "column_3"]
    assert '"has_header"' in build_prompt("a,b\n", "utf-8")


def test_grounding_detects_a_first_row_shaped_like_data() -> None:
    """A model that invents names for a data row is corrected to no header (#94, #131)."""
    fixture = SAMPLE_CSV_PATH.parent / "samples" / "header_none_data_only.csv"
    answer = {
        **VALID_RESULT_PAYLOAD,
        "delimiter": ",",
        "header_row_index": 0,
        "columns": [
            {"name": "date", "inferred_type": "date"},
            {"name": "name", "inferred_type": "string"},
            {"name": "amount", "inferred_type": "float"},
        ],
    }

    result = inspect_csv(fixture, model_invoker=lambda prompt, model: json.dumps(answer))

    assert (result.has_header, result.header_row_index) == (False, None)
    assert [column.name for column in result.columns] == ["column_1", "column_2", "column_3"]


def test_grounding_keeps_a_header_whose_names_are_not_examples(tmp_path: Path) -> None:
    """A real header row with paraphrased names is never taken for data."""
    target = tmp_path / "plain.csv"
    target.write_text("Fecha,Cliente,Importe\n2024-01-15,Acme,10\n", encoding="utf-8")
    answer = {
        **VALID_RESULT_PAYLOAD,
        "delimiter": ",",
        "header_row_index": 0,
        "columns": [
            {"name": "date", "inferred_type": "date", "example_values": ["2024-01-15"]},
            {"name": "client", "inferred_type": "string", "example_values": ["Acme"]},
            {"name": "amount", "inferred_type": "float", "example_values": ["10"]},
        ],
    }

    result = inspect_csv(target, model_invoker=lambda prompt, model: json.dumps(answer))

    assert (result.has_header, result.header_row_index) == (True, 0)


@pytest.mark.parametrize(
    ("value", "shape"),
    [
        ("", "empty"),
        ("  ", "empty"),
        ("42", "integer"),
        (" -7 ", "integer"),
        ("2023", "integer"),
        ("1250.50", "decimal"),
        ("1250,50", "decimal"),
        ("1.234,56", "decimal"),
        ("-0.5", "decimal"),
        ("2024-01-15", "date"),
        ("2024-01-15 08:30:00", "date"),
        ("15/01/2024", "date"),
        ("15.01.2024", "date"),
        ("1-2-24", "date"),
        ("Acme S.L.", "text"),
        ("A-0042", "text"),
        ("Importe", "text"),
    ],
)
def test_field_shape_classifies_a_field(value: str, shape: str) -> None:
    """Each field has exactly one shape: integer, decimal, date, empty or text."""
    assert _field_shape(value) == shape


def _ground_case(case: SampleCase, names: list[str]) -> CSVInspectionResult:
    """Ground a model answer claiming a header at row 0 with ``names`` in a fixture."""
    encoding = case.expected["encoding"]
    answer = {
        **VALID_RESULT_PAYLOAD,
        "encoding": encoding,
        "delimiter": case.expected["delimiter"],
        "header_row_index": 0,
        "columns": [{"name": name, "inferred_type": "string"} for name in names],
    }
    head = case.raw_bytes.decode(encoding)
    return ground_in_samples(CSVInspectionResult.model_validate(answer), head, None)


_HEADERLESS_CASES = [
    *(case for case in CASES if case.filename == "header_none_data_only.csv"),
    *(render(spec) for spec in MATRIX if spec.slug in {"headerless_narrow", "headerless_marker"}),
]


@pytest.mark.parametrize("case", _HEADERLESS_CASES, ids=lambda case: case.filename)
def test_grounding_reports_no_header_for_header_less_fixtures(case: SampleCase) -> None:
    """Invented names over a header-less fixture become positional columns (#131)."""
    width = len(derive_columns(case) or [])
    assert len(_HEADERLESS_CASES) == 3
    assert width >= 3

    result = _ground_case(case, [f"invented_{n}" for n in range(width)])

    assert (result.has_header, result.header_row_index) == (False, None)
    assert [column.name for column in result.columns] == derive_columns(case)


def test_grounding_keeps_an_all_text_header_it_cannot_anchor() -> None:
    """A header of names above typed data is kept, even with paraphrased names (#131)."""
    spec = next(
        spec
        for spec in MATRIX
        if spec.has_header and spec.preamble_lines == 0 and spec.encoding == "utf-8"
    )
    case = render(spec)
    width = len(derive_columns(case) or [])

    result = _ground_case(case, [f"paraphrased_{n}" for n in range(width)])

    assert (result.has_header, result.header_row_index) == (True, 0)


@pytest.mark.parametrize("names", [["2023", "2024", "2025"], ["y1", "y2", "y3"]])
def test_grounding_keeps_a_header_of_years(names: list[str]) -> None:
    """Integer years above decimal data stay a header, anchored or not (#131)."""
    case = next(case for case in CASES if case.filename == "header_years.csv")

    result = _ground_case(case, names)

    assert (result.has_header, result.header_row_index) == (True, 0)
    assert [column.name for column in result.columns] == names


def test_grounding_keeps_a_header_named_by_the_model_in_another_case() -> None:
    """A row-0 field equal to a model's name, case aside, keeps the header (#131)."""
    answer = {
        **VALID_RESULT_PAYLOAD,
        "delimiter": ",",
        "header_row_index": 0,
        "columns": [
            {"name": "fecha", "inferred_type": "string"},
            {"name": "concepto", "inferred_type": "string"},
        ],
    }
    head = "Fecha,Cliente\nAcme,Beta\nGamma,Delta\n"

    result = ground_in_samples(CSVInspectionResult.model_validate(answer), head, None)

    assert (result.has_header, result.header_row_index) == (True, 0)


@pytest.mark.parametrize(
    ("head", "is_data"),
    [
        pytest.param("2024-01-15,Acme,10.5\n", False, id="one-line"),
        pytest.param("2024-01-15,Acme,10.5\n2024-01-16,Beta\n", False, id="ragged-second-row"),
        pytest.param("Fecha,Cliente,Importe\n2024-01-15,Acme,10.5\n", False, id="text-over-data"),
        pytest.param("2024-01-15,,10.5\n2024-01-16,Beta,3.5\n", True, id="empty-cell-matches"),
        pytest.param("2024-01-15,Acme,Norte\n2024-01-16,Beta,\n", False, id="mostly-text"),
    ],
)
def test_first_row_is_data_needs_a_second_row_of_the_same_shape(head: str, is_data: bool) -> None:
    """Only two rows of equal width and agreeing shapes flag row 0 as data."""
    answer = {
        **VALID_RESULT_PAYLOAD,
        "delimiter": ",",
        "header_row_index": 0,
        "columns": [
            {"name": name, "inferred_type": "string"} for name in ("date", "client", "amount")
        ],
    }
    result = CSVInspectionResult.model_validate(answer)

    assert _first_row_is_data(result, head) is is_data


def test_extract_json_payload_strips_markdown_fence() -> None:
    """A JSON payload wrapped in a markdown code fence should be unwrapped."""
    fenced = '```json\n{"a": 1}\n```'

    assert _extract_json_payload(fenced) == '{"a": 1}'


def test_extract_json_payload_passes_through_bare_json() -> None:
    """A bare JSON payload with no fence should be returned unchanged (trimmed)."""
    bare = '  {"a": 1}  '

    assert _extract_json_payload(bare) == '{"a": 1}'


@pytest.mark.parametrize(
    "raw",
    [
        'Here is the result: {"a": {"b": 1}}',
        '{"a": {"b": 1}}\nLet me know if you need anything else.',
        'Sure!\n{"a": {"b": 1}}\nThe delimiter is ";".',
    ],
    ids=["leading-prose", "trailing-prose", "both"],
)
def test_extract_json_payload_drops_prose_around_an_unfenced_object(raw: str) -> None:
    """Without a fence, the object is taken from the first ``{`` to the last ``}``.

    Regression test for issue #23: such replies used to fail as invalid JSON.
    """
    assert _extract_json_payload(raw) == '{"a": {"b": 1}}'


def test_a_reply_without_any_object_is_still_invalid_json() -> None:
    """Text with no ``{...}`` span is parsed as is, and fails as invalid JSON."""
    with pytest.raises(ResponseParsingError, match="invalid JSON"):
        parse_and_validate("I could not determine the dialect.", "m")


def test_a_prose_wrapped_reply_is_accepted_on_the_first_attempt() -> None:
    """A valid answer wrapped in prose no longer spends the fallback attempt."""
    models: list[str] = []

    def invoker(prompt: str, model: str) -> str:
        models.append(model)
        payload = json.dumps(VALID_RESULT_PAYLOAD)
        return f"Here is the inspection result:\n{payload}\nHope this helps!"

    result = inspect_csv(
        SAMPLE_CSV_PATH, model="primary", fallback_model="fallback", model_invoker=invoker
    )

    assert models == ["primary"]
    assert result.delimiter == ";"


def test_the_default_models_include_a_distinct_fallback() -> None:
    """With default settings, a failing primary is retried with the default fallback."""
    models: list[str] = []

    def invoker(prompt: str, model: str) -> str:
        models.append(model)
        if len(models) == 1:
            raise ModelInvocationError("primary unavailable")
        return json.dumps(VALID_RESULT_PAYLOAD)

    inspect_csv(SAMPLE_CSV_PATH, settings=Settings(), model_invoker=invoker)

    assert models == [DEFAULT_MODEL, FALLBACK_MODEL]


# ---------------------------------------------------------------------
# inspect_csv orchestration (with an injected fake model invoker)
# ---------------------------------------------------------------------


def test_inspect_csv_returns_validated_result_on_first_model_success() -> None:
    """A well-formed first-model response should short-circuit the fallback."""
    calls: list[str] = []

    def fake_invoker(prompt: str, model: str) -> str:
        calls.append(model)
        return json.dumps(VALID_RESULT_PAYLOAD)

    result = inspect_csv(
        SAMPLE_CSV_PATH,
        model="primary-model",
        fallback_model="fallback-model",
        model_invoker=fake_invoker,
    )

    assert isinstance(result, CSVInspectionResult)
    assert result.delimiter == ";"
    assert result.header_row_index == 2
    assert calls == ["primary-model"]


def test_inspect_csv_falls_back_to_secondary_model_on_primary_failure() -> None:
    """A primary-model failure should be retried against the fallback model."""
    calls: list[str] = []

    def fake_invoker(prompt: str, model: str) -> str:
        calls.append(model)
        if model == "primary-model":
            return "not valid json"
        return json.dumps(VALID_RESULT_PAYLOAD)

    result = inspect_csv(
        SAMPLE_CSV_PATH,
        model="primary-model",
        fallback_model="fallback-model",
        model_invoker=fake_invoker,
    )

    assert result.confidence == pytest.approx(0.95)
    assert calls == ["primary-model", "fallback-model"]


def test_inspect_csv_raises_when_every_model_fails() -> None:
    """If all configured models fail, an aggregated domain error is raised."""

    def failing_invoker(prompt: str, model: str) -> str:
        return "not valid json"

    with pytest.raises(InspectionFailedError) as exc_info:
        inspect_csv(
            SAMPLE_CSV_PATH,
            model="primary-model",
            fallback_model="fallback-model",
            model_invoker=failing_invoker,
        )

    assert set(exc_info.value.attempts) == {"primary-model", "fallback-model"}


def test_inspect_csv_does_not_duplicate_identical_primary_and_fallback() -> None:
    """When model equals fallback_model, the invoker should only run once."""
    calls: list[str] = []

    def fake_invoker(prompt: str, model: str) -> str:
        calls.append(model)
        return json.dumps(VALID_RESULT_PAYLOAD)

    inspect_csv(
        SAMPLE_CSV_PATH,
        model="only-model",
        fallback_model="only-model",
        model_invoker=fake_invoker,
    )

    assert calls == ["only-model"]


def test_inspect_csv_propagates_file_read_errors_before_invoking_model(tmp_path: Path) -> None:
    """A missing source file should fail fast, without calling the model."""
    missing = tmp_path / "missing.csv"
    invoked = False

    def fake_invoker(prompt: str, model: str) -> str:
        nonlocal invoked
        invoked = True
        return json.dumps(VALID_RESULT_PAYLOAD)

    with pytest.raises(FileSampleReadError):
        inspect_csv(missing, model_invoker=fake_invoker)

    assert invoked is False


def test_inspect_csv_skips_tail_read_when_file_fits_in_head(tmp_path: Path) -> None:
    """A file smaller than n_bytes should not trigger a separate tail read."""
    target = tmp_path / "small.csv"
    target.write_bytes(b"a,b,c\n1,2,3\n")
    seen_prompts: list[str] = []

    def fake_invoker(prompt: str, model: str) -> str:
        seen_prompts.append(prompt)
        return json.dumps(VALID_RESULT_PAYLOAD)

    inspect_csv(target, n_bytes=4096, tail_bytes=4096, model_invoker=fake_invoker)

    assert "TAIL SAMPLE START" not in seen_prompts[0]


def test_inspect_csv_includes_tail_sample_for_files_larger_than_head(tmp_path: Path) -> None:
    """A file larger than n_bytes should trigger a separate, labeled tail read."""
    target = tmp_path / "large.csv"
    target.write_bytes(b"a,b,c\n" + b"1,2,3\n" * 2000 + b"TOTAL,,999\n")
    seen_prompts: list[str] = []

    def fake_invoker(prompt: str, model: str) -> str:
        seen_prompts.append(prompt)
        return json.dumps(VALID_RESULT_PAYLOAD)

    inspect_csv(target, n_bytes=64, tail_bytes=64, model_invoker=fake_invoker)

    assert "TAIL SAMPLE START" in seen_prompts[0]
    assert "TOTAL,,999" in seen_prompts[0]


# ---------------------------------------------------------------------
# byte-budget validation
# ---------------------------------------------------------------------


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


@pytest.mark.parametrize(
    ("n_bytes", "tail_bytes", "bad_name"),
    [
        (0, 64, "n_bytes"),
        (64, -1, "tail_bytes"),
        (MAX_SAMPLE_BYTES + 1, 64, "n_bytes"),
        (64, MAX_SAMPLE_BYTES + 1, "tail_bytes"),
    ],
)
def test_inspect_csv_validates_budgets_before_touching_the_file(
    tmp_path: Path, n_bytes: int, tail_bytes: int, bad_name: str
) -> None:
    """Invalid budgets fail fast, even before the (missing) file is opened."""
    with pytest.raises(ValueError, match=bad_name):
        inspect_csv(
            tmp_path / "missing.csv",
            n_bytes=n_bytes,
            tail_bytes=tail_bytes,
            model_invoker=_capture_prompts([]),
        )


# ---------------------------------------------------------------------
# empty input
# ---------------------------------------------------------------------


def test_inspect_csv_raises_on_empty_file_without_invoking_model(tmp_path: Path) -> None:
    """An empty file must fail fast with a domain error and cost no LLM call."""
    target = tmp_path / "empty.csv"
    target.write_bytes(b"")
    prompts: list[str] = []

    with pytest.raises(EmptySampleError):
        inspect_csv(target, model_invoker=_capture_prompts(prompts))

    assert prompts == []


# ---------------------------------------------------------------------
# tail window: overlap, exact fit, alignment
# ---------------------------------------------------------------------


def test_inspect_csv_skips_tail_when_file_is_exactly_head_sized(tmp_path: Path) -> None:
    """A file of exactly ``n_bytes`` is fully covered by the head: no tail section."""
    target = tmp_path / "exact.csv"
    target.write_bytes(b"a,b,c\n" * 10)
    prompts: list[str] = []

    inspect_csv(target, n_bytes=60, tail_bytes=64, model_invoker=_capture_prompts(prompts))

    assert "TAIL SAMPLE START" not in prompts[0]


def test_inspect_csv_tail_never_overlaps_head(tmp_path: Path) -> None:
    """Only the bytes past the head window are sent as the tail sample."""
    target = tmp_path / "overlap.csv"
    target.write_bytes(b"H" * 64 + b"0123456789")
    prompts: list[str] = []

    inspect_csv(target, n_bytes=64, tail_bytes=64, model_invoker=_capture_prompts(prompts))

    assert _tail_section(prompts[0]).strip() == "0123456789"


def test_inspect_csv_zero_tail_bytes_disables_tail_sampling(tmp_path: Path) -> None:
    """``tail_bytes=0`` opts out of tail sampling entirely."""
    target = tmp_path / "large.csv"
    target.write_bytes(b"a,b,c\n" + b"1,2,3\n" * 100)
    prompts: list[str] = []

    inspect_csv(target, n_bytes=16, tail_bytes=0, model_invoker=_capture_prompts(prompts))

    assert "TAIL SAMPLE START" not in prompts[0]


@pytest.mark.parametrize(
    ("bom", "codec"),
    [(codecs.BOM_UTF16_LE, "utf-16-le"), (codecs.BOM_UTF16_BE, "utf-16-be")],
)
def test_inspect_csv_decodes_utf16_tail_with_odd_budget(
    tmp_path: Path, bom: bytes, codec: str
) -> None:
    """A UTF-16 tail must decode cleanly even for an odd budget and either byte order."""
    text = "a\tb\n" + "1\t2\n" * 200 + "TOTAL\t999\n"
    target = tmp_path / "utf16.csv"
    target.write_bytes(bom + text.encode(codec))
    prompts: list[str] = []

    inspect_csv(target, n_bytes=64, tail_bytes=33, model_invoker=_capture_prompts(prompts))

    tail = _tail_section(prompts[0])
    assert "TOTAL\t999" in tail
    assert "�" not in tail


# ---------------------------------------------------------------------
# invoke_ollama_model (with a fake ``ollama`` module)
# ---------------------------------------------------------------------


def _install_fake_ollama(monkeypatch: pytest.MonkeyPatch, chat: Any) -> None:
    """Register a stand-in ``ollama`` module whose clients call ``chat``."""
    install_fake_ollama(monkeypatch, chat)


def test_invoke_ollama_model_returns_message_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model's message content is returned verbatim."""

    def chat(**kwargs: Any) -> Any:
        return SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))

    _install_fake_ollama(monkeypatch, chat)

    assert invoke_ollama_model("prompt", "some-model") == '{"ok": true}'


@pytest.mark.parametrize("content", [None, ""])
def test_invoke_ollama_model_rejects_empty_content(
    monkeypatch: pytest.MonkeyPatch, content: str | None
) -> None:
    """An empty or missing message is a backend failure, not an empty JSON payload."""

    def chat(**kwargs: Any) -> Any:
        return SimpleNamespace(message=SimpleNamespace(content=content))

    _install_fake_ollama(monkeypatch, chat)

    with pytest.raises(ModelInvocationError, match="empty response"):
        invoke_ollama_model("prompt", "some-model")


def test_invoke_ollama_model_wraps_backend_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any client-side failure is surfaced as the domain ``ModelInvocationError``."""

    def chat(**kwargs: Any) -> Any:
        raise ConnectionError("connection refused")

    _install_fake_ollama(monkeypatch, chat)

    with pytest.raises(ModelInvocationError, match="connection refused"):
        invoke_ollama_model("prompt", "some-model")


def test_invoke_ollama_model_reports_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing ``ollama`` package yields an actionable domain error."""
    monkeypatch.setitem(sys.modules, "ollama", None)

    with pytest.raises(BackendConfigurationError, match="pip install ollama"):
        invoke_ollama_model("prompt", "some-model")


# ---------------------------------------------------------------------
# grounding the model's answer in the sampled text (issue #2)
# ---------------------------------------------------------------------

_LEDGER = (
    "# Exportado desde SistemaXYZ v3.2\n"
    "# Periodo: 2024-01-01 a 2024-01-03\n"
    "Fecha;Cliente;Importe\n"
    "2024-01-01;Acme;10.00\n"
    "2024-01-02;Beta;20.00\n"
    "2024-01-03;Gamma;30.00\n"
    "\n"
    "TOTAL;;60.00\n"
    "--- Fin del informe ---\n"
)


def _sloppy_answer(**overrides: object) -> ModelInvoker:
    """Fake model that recognizes the structure but miscounts, paraphrases and skips lines.

    By default it spots the totals row but drops the blank line before it
    and the end-of-report marker after it.
    """
    payload = {
        **VALID_RESULT_PAYLOAD,
        "header_row_index": 0,
        "columns": [
            {"name": "Fecha", "inferred_type": "date"},
            {"name": "Proveedor", "inferred_type": "string"},
            {"name": "Monto", "inferred_type": "float"},
        ],
        "footer_lines": ["TOTAL;;60.00"],
        **overrides,
    }

    def fake_invoker(prompt: str, model: str) -> str:
        return json.dumps(payload)

    return fake_invoker


def test_grounding_recovers_header_row_and_literal_column_names(tmp_path: Path) -> None:
    """A miscounted preamble and paraphrased names are corrected from the head."""
    target = tmp_path / "ledger.csv"
    target.write_text(_LEDGER, encoding="utf-8")

    result = inspect_csv(target, model_invoker=_sloppy_answer())

    assert result.header_row_index == 2
    assert [column.name for column in result.columns] == ["Fecha", "Cliente", "Importe"]
    assert [column.inferred_type for column in result.columns] == ["date", "string", "float"]


def test_grounded_column_names_keep_padding_like_csv_reader(tmp_path: Path) -> None:
    """Names match what csv/pandas read: surrounding spaces are kept, not stripped."""
    target = tmp_path / "padded.csv"
    target.write_text("Fecha; Cliente ;Importe\n2024-01-01;Acme;10.00\n", encoding="utf-8")
    columns = [
        {"name": name, "inferred_type": "string"} for name in ("Fecha", "Cliente", "Importe")
    ]

    result = inspect_csv(target, model_invoker=_sloppy_answer(columns=columns, footer_lines=[]))

    assert [column.name for column in result.columns] == ["Fecha", " Cliente ", "Importe"]


def test_grounding_recovers_skipped_footer_lines_verbatim(tmp_path: Path) -> None:
    """Lines after the reported footer and blank separators before it are recovered."""
    target = tmp_path / "ledger.csv"
    target.write_text(_LEDGER, encoding="utf-8")

    result = inspect_csv(target, model_invoker=_sloppy_answer())

    assert result.footer_lines == ["", "TOTAL;;60.00", "--- Fin del informe ---"]
    assert result.footer_rows_to_skip == 3


def test_grounding_reads_the_footer_from_the_tail_of_a_large_file() -> None:
    """On the real head/tail path, the footer is anchored in the tail sample."""
    fixture = SAMPLE_CSV_PATH.parent / "samples" / "header_and_footer_combined.csv"
    totals_row = fixture.read_text(encoding="utf-8").splitlines()[-2]

    # Mirrors what qwen2.5-coder:7b actually returned for this fixture.
    columns = [
        {"name": name, "inferred_type": "string"}
        for name in ("Fecha", "Proveedor", "Descripción", "Monto")
    ]

    result = inspect_csv(
        fixture, model_invoker=_sloppy_answer(columns=columns, footer_lines=[totals_row])
    )

    assert result.header_row_index == 2
    assert [column.name for column in result.columns] == [
        "Fecha",
        "Cliente",
        "Concepto",
        "Importe",
    ]
    assert result.footer_lines == ["", totals_row, "--- Fin del informe ---"]


def test_grounding_leaves_a_header_less_file_alone(tmp_path: Path) -> None:
    """Invented names that share nothing with the data never promote a data row to header.

    The first row has the shape of the second, so the header the model
    claimed at row 0 is dropped and the columns become positional (#131).
    """
    target = tmp_path / "no_header.csv"
    target.write_text("2024-01-01;Acme;10.00\n2024-01-02;Beta;20.00\n", encoding="utf-8")
    columns = [
        {"name": "col_1", "inferred_type": "date"},
        {"name": "col_2", "inferred_type": "string"},
        {"name": "col_3", "inferred_type": "float"},
    ]

    result = inspect_csv(
        target, model_invoker=_sloppy_answer(columns=columns, header_row_index=0, footer_lines=[])
    )

    assert (result.has_header, result.header_row_index) == (False, None)
    assert [column.name for column in result.columns] == ["column_1", "column_2", "column_3"]
    assert result.footer_lines == []


def test_grounding_anchors_the_footer_on_its_last_occurrence(tmp_path: Path) -> None:
    """Footer text that also appears in the data must not drag data rows into the footer."""
    target = tmp_path / "repeated.csv"
    target.write_text(
        "Fecha;Cliente;Importe\n2024-01-01;Acme;10.00\nRevisado\n2024-01-02;Beta;20.00\nRevisado\n",
        encoding="utf-8",
    )

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(footer_lines=["Revisado"], header_row_index=0),
    )

    assert result.footer_lines == ["Revisado"]


def test_grounding_drops_a_footer_that_is_not_at_the_end_of_the_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A reported footer absent from the sampled end would drop real data rows (issue #52)."""
    target = tmp_path / "plain.csv"
    target.write_text("Fecha;Cliente;Importe\n2024-01-01;Acme;10.00\n", encoding="utf-8")

    result = inspect_csv(target, model_invoker=_sloppy_answer(footer_lines=["*** END ***"]))

    assert result.footer_lines == []
    assert result.footer_rows_to_skip == 0
    assert "do not occur at the end of the file" in caplog.text


def test_grounding_drops_an_unanchored_footer_on_the_head_and_tail_path(tmp_path: Path) -> None:
    """The same applies when the end of the file comes from the tail sample (issue #52)."""
    target = tmp_path / "long.csv"
    rows = "".join(f"2024-01-{day % 28 + 1:02d};Cliente {day};{day}.00\n" for day in range(400))
    target.write_text("Fecha;Proveedor;Monto\n" + rows, encoding="utf-8")

    result = inspect_csv(
        target, n_bytes=512, tail_bytes=512, model_invoker=_sloppy_answer(footer_lines=["TOTAL"])
    )

    assert result.footer_lines == []


# ---------------------------------------------------------------------
# Delimiter grounding
# ---------------------------------------------------------------------


def test_grounding_replaces_a_delimiter_that_never_occurs(tmp_path: Path) -> None:
    """A ``,`` answer for a tab-separated file is sniffed from the head (issue #53)."""
    target = tmp_path / "ledger.tsv"
    target.write_text(_LEDGER.replace(";", "\t"), encoding="utf-8")

    result = inspect_csv(target, model_invoker=_sloppy_answer(delimiter=",", header_row_index=0))

    assert result.delimiter == "\t"
    # Header grounding ran with the grounded delimiter.
    assert [column.name for column in result.columns] == ["Fecha", "Cliente", "Importe"]
    assert result.header_row_index == 2


def test_grounding_keeps_a_delimiter_that_occurs(tmp_path: Path) -> None:
    """A delimiter found in the head is never second-guessed (issue #53)."""
    target = tmp_path / "ledger.csv"
    target.write_text(_LEDGER.replace(";", "\t", 1), encoding="utf-8")

    result = inspect_csv(target, model_invoker=_sloppy_answer())

    assert result.delimiter == ";"


def test_grounding_keeps_the_delimiter_of_a_one_column_file(tmp_path: Path) -> None:
    """With nothing to sniff, the model's delimiter is kept (issue #53)."""
    target = tmp_path / "ids.csv"
    target.write_text("id\n1\n2\n3\n", encoding="utf-8")

    result = inspect_csv(target, model_invoker=_sloppy_answer(delimiter=",", footer_lines=[]))

    assert result.delimiter == ","


def test_grounding_replaces_a_delimiter_that_occurs_only_inside_a_value(
    tmp_path: Path,
) -> None:
    """One comma inside a TSV value does not make ``,`` the delimiter (issue #97)."""
    target = tmp_path / "ledger.tsv"
    target.write_text(_LEDGER.replace(";", "\t").replace("Acme", "Smith, John"), encoding="utf-8")

    result = inspect_csv(target, model_invoker=_sloppy_answer(delimiter=",", header_row_index=0))

    assert result.delimiter == "\t"
    assert [column.name for column in result.columns] == ["Fecha", "Cliente", "Importe"]


def test_grounding_replaces_a_delimiter_dominated_by_another(tmp_path: Path) -> None:
    """A ``,`` that splits a few TSV rows evenly still loses to the tab (#151)."""
    rows = "".join(
        f"2024-01-{day:02d}\tFernández, Asociados\t{day},50\n"
        if day % 4 == 0
        else f"2024-01-{day:02d}\tAcme\t{day}.00\n"
        for day in range(1, 21)
    )
    target = tmp_path / "ledger.tsv"
    target.write_text("Fecha\tCliente\tImporte\n" + rows, encoding="utf-8")

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(delimiter=",", header_row_index=0, footer_lines=[]),
    )

    assert result.delimiter == "\t"
    assert [column.name for column in result.columns] == ["Fecha", "Cliente", "Importe"]


def test_grounding_keeps_a_delimiter_that_is_not_clearly_dominated(tmp_path: Path) -> None:
    """A ``,`` splitting more than half as many rows as the tab stays (#151)."""
    rows = "".join(
        f"2024-01-{day:02d}\tFernández, Asociados\t{day},50\n"
        if day % 2 == 0 or day % 3 == 0
        else f"2024-01-{day:02d}\tAcme\t{day}.00\n"
        for day in range(1, 21)
    )
    target = tmp_path / "ledger.tsv"
    target.write_text("Fecha\tCliente\tImporte\n" + rows, encoding="utf-8")

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(delimiter=",", header_row_index=0, footer_lines=[]),
    )

    assert result.delimiter == ","


def test_grounding_keeps_a_reported_delimiter_that_splits_the_rows(tmp_path: Path) -> None:
    """``;`` that really separates the fields stays, even with commas in values (#97)."""
    target = tmp_path / "ledger.csv"
    target.write_text(_LEDGER.replace("10.00", "10,00").replace("20.00", "20,00"), encoding="utf-8")

    result = inspect_csv(target, model_invoker=_sloppy_answer())

    assert result.delimiter == ";"


def test_grounding_keeps_the_reported_delimiter_when_candidates_tie(tmp_path: Path) -> None:
    """With no single clear winner, the model's answer is kept (#97)."""
    target = tmp_path / "ambiguous.csv"
    target.write_text("a,b;c|d\n1,2;3\n4,5;6\n7,8;9\n", encoding="utf-8")

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(
            delimiter="|", footer_lines=[], columns=[{"name": "a", "inferred_type": "string"}]
        ),
    )

    assert result.delimiter == "|"


def test_numeric_example_values_are_accepted_as_text() -> None:
    """JSON numbers or nulls in example_values must not fail the whole inspection."""
    payload = {
        **VALID_RESULT_PAYLOAD,
        "columns": [
            {"name": "Importe", "inferred_type": "float", "example_values": [1447.44, 3, None]}
        ],
    }

    result = CSVInspectionResult.model_validate(payload)

    assert result.columns[0].example_values == ["1447.44", "3", ""]


@pytest.mark.parametrize(
    ("answered", "expected"),
    [
        ("integer", "integer"),
        ("DateTime", "datetime"),
        (" Boolean ", "boolean"),
        ("int", "integer"),
        ("BIGINT", "integer"),
        ("int64", "integer"),
        ("number", "float"),
        ("decimal", "float"),
        ("double", "float"),
        ("numeric", "float"),
        ("text", "string"),
        ("str", "string"),
        ("varchar", "string"),
        ("bool", "boolean"),
        ("timestamp", "datetime"),
        ("currency", "string"),
        ("", "string"),
    ],
)
def test_inferred_type_is_normalized_to_the_vocabulary(answered: str, expected: str) -> None:
    """Aliases map onto the closed vocabulary; unknown words fall back to string."""
    column = ColumnSchema(name="c", inferred_type=answered)

    assert column.inferred_type == expected


def test_non_string_inferred_type_fails_validation() -> None:
    """Only strings are normalized; any other JSON value is a malformed answer."""
    with pytest.raises(ValidationError):
        ColumnSchema(name="c", inferred_type=3)


def test_json_schema_lists_the_type_vocabulary() -> None:
    """The schema sent to Gemini constrains inferred_type to the vocabulary."""
    schema = CSVInspectionResult.model_json_schema()

    assert schema["$defs"]["ColumnSchema"]["properties"]["inferred_type"]["enum"] == list(
        get_args(ColumnType)
    )


def test_prompt_lists_the_type_vocabulary() -> None:
    """The prompt asks for exactly the types the model accepts."""
    assert "string|integer|float|date|datetime|boolean" in build_prompt("a\n1\n", "utf-8")


def test_grounding_recovers_an_unreported_totals_row_above_the_footer(tmp_path: Path) -> None:
    """If the model only spots the closing marker, the totals row above it is still found."""
    target = tmp_path / "ledger.csv"
    target.write_text(_LEDGER, encoding="utf-8")

    result = inspect_csv(
        target, model_invoker=_sloppy_answer(footer_lines=["--- Fin del informe ---"])
    )

    assert result.footer_lines == ["", "TOTAL;;60.00", "--- Fin del informe ---"]


def test_grounding_never_extends_the_footer_past_a_data_row(tmp_path: Path) -> None:
    """Backward extension stops at the first line that is neither blank nor a totals row."""
    target = tmp_path / "ledger.csv"
    target.write_text(
        "Fecha;Cliente;Importe\n"
        "2024-01-01;Total Care S.L.;10.00\n"
        "2024-01-02;Beta;20.00\n"
        "--- Fin del informe ---\n",
        encoding="utf-8",
    )

    result = inspect_csv(
        target, model_invoker=_sloppy_answer(footer_lines=["--- Fin del informe ---"])
    )

    assert result.footer_lines == ["--- Fin del informe ---"]


def test_grounding_drops_a_footer_when_the_end_of_the_file_was_not_sampled(
    tmp_path: Path,
) -> None:
    """With tail sampling disabled, the truncated head's last rows are never a footer.

    Regression test for issue #10: the prompt claimed the head was the entire
    file, the model filed the last sampled rows as footer, and grounding
    copied them into ``footer_lines``, so consumers skipped real data.
    """
    target = tmp_path / "ledger.csv"
    target.write_text("Fecha;Cliente;Importe\n" + "2024-01-01;Acme;10.00\n" * 200, encoding="utf-8")
    prompts: list[str] = []
    answer = _sloppy_answer(footer_lines=["2024-01-01;Acme;10.00"])

    def recording_invoker(prompt: str, model: str) -> str:
        prompts.append(prompt)
        return answer(prompt, model)

    result = inspect_csv(target, n_bytes=256, tail_bytes=0, model_invoker=recording_invoker)

    assert "contains the ENTIRE file" not in prompts[0]
    assert "its end was not sampled" in prompts[0]
    assert result.footer_lines == []
    assert result.footer_rows_to_skip == 0


@pytest.mark.parametrize(
    ("line", "delimiter", "expected"),
    [
        ("", ";", True),
        ("TOTAL;;;12.50", ";", True),
        ("Subtotal,,3", ",", True),
        ('"Total general",9', ",", True),
        ("Total registros: 250", ",", True),
        ("SUMA;;;1", ";", True),
        ("TOTAL;120;340;460", ";", True),
        ("Total ventas;;;460", ";", True),
        ("2024-01-01;Acme;10.00", ";", False),
        ("Totalmente nuevo,1,2", ",", False),
        ("Summary report", ",", False),
        ("--- Fin del informe ---", ",", False),
        ("Total Energies,2024-01-01,10.00", ",", False),
        ("Sum Holdings;Madrid;2024-01-01;10.00", ";", False),
    ],
)
def test_extends_footer_accepts_only_blank_and_totals_rows(
    line: str, delimiter: str, expected: bool
) -> None:
    """Only blank separators and totals rows, not data rows named "Total...", extend a footer."""
    assert _extends_footer(line, delimiter, '"') is expected


def test_grounding_keeps_a_data_row_named_like_a_totals_label_out_of_the_footer(
    tmp_path: Path,
) -> None:
    """A data row whose first field starts with "Total" is not pulled into the footer.

    Regression test for issue #20.
    """
    target = tmp_path / "companies.csv"
    target.write_text(
        "Empresa,Fecha,Importe\n"
        "Acme,2024-01-01,10.00\n"
        "Total Energies,2024-01-02,20.00\n"
        "--- Fin del informe ---\n",
        encoding="utf-8",
    )
    columns = [
        {"name": name, "inferred_type": "string"} for name in ("Empresa", "Fecha", "Importe")
    ]

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(
            delimiter=",", columns=columns, footer_lines=["--- Fin del informe ---"]
        ),
    )

    assert result.footer_lines == ["--- Fin del informe ---"]


def test_grounding_counts_lines_like_csv_does(tmp_path: Path) -> None:
    """Characters str.splitlines() breaks on, but csv does not, never shift line indexes.

    Regression test for issue #16: stray form feeds and U+2028 in fields made
    the preamble look longer, and the blank "line" after a data row's form
    feed was taken as a footer separator, so a data row would be skipped.
    """
    target = tmp_path / "dirty.csv"
    target.write_text(
        "# Export\x0cv2\n"
        "Fecha;Cliente;Importe\n"
        "2024-01-01;Acme\u2028S.L.;10.00\n"
        "2024-01-02;Beta;20.00\x0c\n"
        "--- Fin del informe ---\n",
        encoding="utf-8",
    )

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(footer_lines=["--- Fin del informe ---"]),
    )

    assert result.header_row_index == 1
    assert result.footer_lines == ["--- Fin del informe ---"]


def test_header_fallback_ignores_empty_names(tmp_path: Path) -> None:
    """An unnamed column never makes a data row with an empty cell look like the header.

    Regression test for issue #21: with columns ``["", "a", "b"]`` (a pandas
    index), the paraphrase fallback matched the first same-width data row
    that had an empty cell.
    """
    target = tmp_path / "index.csv"
    target.write_text("# Export\n,1,\n,2,3\n,a,b\n0,4,5\n1,6,7\n", encoding="utf-8")
    columns = [{"name": name, "inferred_type": "string"} for name in ("", "x", "b")]

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(
            delimiter=",", columns=columns, header_row_index=0, footer_lines=[]
        ),
    )

    assert result.header_row_index == 3
    assert [column.name for column in result.columns] == ["", "a", "b"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [("\t", "\t"), ("\\t", "\t"), ("tab", "\t"), ("TAB", "\t"), (";", ";")],
)
def test_dialect_characters_accept_common_spellings_of_tab(value: str, expected: str) -> None:
    """A tab written as an escape sequence or a word becomes a real tab (issue #11)."""
    result = CSVInspectionResult.model_validate({**VALID_RESULT_PAYLOAD, "delimiter": value})

    assert result.delimiter == expected


@pytest.mark.parametrize("value", ["", "null", "None", None])
def test_an_empty_escapechar_means_none(value: str | None) -> None:
    """``""``, ``"null"`` and ``"none"`` mean there is no escape character (issue #11)."""
    result = CSVInspectionResult.model_validate({**VALID_RESULT_PAYLOAD, "escapechar": value})

    assert result.escapechar is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("delimiter", ""),
        ("delimiter", "null"),
        ("delimiter", ";;"),
        ("delimiter", "comma"),
        ("quotechar", "''"),
        ("escapechar", "\\\\"),
    ],
)
def test_dialect_characters_must_be_one_character(field: str, value: str) -> None:
    """Anything else that is not one character fails validation (issue #11)."""
    with pytest.raises(ValidationError, match="exactly one character"):
        CSVInspectionResult.model_validate({**VALID_RESULT_PAYLOAD, field: value})


@pytest.mark.parametrize("value", ["", "null", "None", "NONE", " none ", None])
def test_no_quoting_spellings_map_quotechar_to_default(value: str | None) -> None:
    """Unquoted files: empty/null quotechar answers keep the inert default (issue #93)."""
    result = CSVInspectionResult.model_validate({**VALID_RESULT_PAYLOAD, "quotechar": value})
    assert result.quotechar == '"'


@pytest.mark.parametrize(
    "dialect",
    [
        {"delimiter": ",", "quotechar": ","},
        {"delimiter": ",", "escapechar": ","},
        {"delimiter": "\n"},
        {"quotechar": "\r"},
    ],
)
def test_a_dialect_csv_cannot_read_fails_validation(dialect: dict[str, str]) -> None:
    """Characters that are valid alone but conflict fail validation (issue #48)."""
    with pytest.raises(ValidationError, match=r"line break|must differ"):
        CSVInspectionResult.model_validate({**VALID_RESULT_PAYLOAD, **dialect})


def test_an_escapechar_equal_to_the_quotechar_means_doubled_quotes() -> None:
    """``escapechar='"'`` describes RFC 4180 doubled quotes (issue #48)."""
    payload = {**VALID_RESULT_PAYLOAD, "escapechar": '"', "doublequote": False}

    result = CSVInspectionResult.model_validate(payload)

    assert result.escapechar is None
    assert result.doublequote is True


def test_a_conflicting_dialect_moves_on_to_the_fallback_model(tmp_path: Path) -> None:
    """A dialect csv rejects is a schema error, not a crash in grounding (issue #48)."""
    target = tmp_path / "ledger.csv"
    target.write_text(_LEDGER, encoding="utf-8")

    def invoker(prompt: str, model: str) -> str:
        quotechar = ";" if model == "primary" else '"'
        return _sloppy_answer(quotechar=quotechar)(prompt, model)

    result = inspect_csv(target, model="primary", fallback_model="fallback", model_invoker=invoker)

    assert (result.delimiter, result.quotechar) == (";", '"')
    csv.reader(
        [],
        delimiter=result.delimiter,
        quotechar=result.quotechar,
        escapechar=result.escapechar,
        doublequote=result.doublequote,
    )


def test_a_malformed_delimiter_moves_on_to_the_fallback_model(tmp_path: Path) -> None:
    """A multi-character delimiter is a schema error, so the fallback model runs (issue #11)."""
    target = tmp_path / "ledger.csv"
    target.write_text(_LEDGER, encoding="utf-8")
    calls: list[str] = []

    def invoker(prompt: str, model: str) -> str:
        calls.append(model)
        delimiter = "semicolon" if model == "primary" else ";"
        return _sloppy_answer(delimiter=delimiter)(prompt, model)

    result = inspect_csv(target, model="primary", fallback_model="fallback", model_invoker=invoker)

    assert calls == ["primary", "fallback"]
    assert result.delimiter == ";"


@pytest.mark.parametrize(
    ("content", "reported", "expected"),
    [
        (codecs.BOM_UTF8 + _LEDGER.encode("utf-8"), "utf-8", "UTF-8-SIG"),
        (codecs.BOM_UTF16_LE + _LEDGER.encode("utf-16-le"), "utf-16-le", "UTF-16"),
        (_LEDGER.encode("utf-8"), "UTF-8 with BOM", "utf-8"),
        (codecs.BOM_UTF8 + _LEDGER.encode("utf-8"), "utf_8_sig", "utf_8_sig"),
        (_LEDGER.encode("cp1252"), "cp1252", "cp1252"),
    ],
    ids=["utf-8 BOM", "utf-16 BOM", "not a codec", "same codec, other spelling", "no BOM"],
)
def test_grounding_keeps_a_bom_encoding_and_rejects_unknown_codecs(
    tmp_path: Path, content: bytes, reported: str, expected: str
) -> None:
    """A detected BOM, or a valid codec name, always wins over the model's answer (issue #22)."""
    target = tmp_path / "ledger.csv"
    target.write_bytes(content)

    result = inspect_csv(target, model_invoker=_sloppy_answer(encoding=reported))

    assert result.encoding == expected
    assert result.columns[0].name == "Fecha"


def test_a_failed_attempt_log_never_shows_the_configured_api_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A custom invoker's error carrying the key is logged redacted (issue #81)."""
    secret = "AIza-test-secret-value"

    def invoker(prompt: str, model: str) -> str:
        raise PermissionError(f"API key {secret} was rejected")

    settings = Settings(gemini_api_key=SecretStr(secret))
    with (
        caplog.at_level(logging.DEBUG, logger="csv_inspector"),
        pytest.raises(InspectionFailedError),
    ):
        inspect_csv(SAMPLE_CSV_PATH, settings=settings, model_invoker=invoker)

    failures = [r.getMessage() for r in caplog.records if "failed:" in r.getMessage()]
    assert len(failures) == 2
    assert all("API key *** was rejected" in message for message in failures)
    assert secret not in caplog.text


def test_a_failed_async_attempt_log_never_shows_the_configured_api_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The async entry point shares the redaction (issue #81)."""
    secret = "AIza-test-secret-value"

    async def invoker(prompt: str, model: str) -> str:
        raise PermissionError(f"API key {secret} was rejected")

    settings = Settings(gemini_api_key=SecretStr(secret))
    with (
        caplog.at_level(logging.DEBUG, logger="csv_inspector"),
        pytest.raises(InspectionFailedError),
    ):
        asyncio.run(ainspect_csv(SAMPLE_CSV_PATH, settings=settings, model_invoker=invoker))

    assert "API key *** was rejected" in caplog.text
    assert secret not in caplog.text
