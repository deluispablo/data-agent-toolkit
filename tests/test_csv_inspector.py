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

import codecs
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any

import pytest

from exceptions import (
    EmptySampleError,
    FileSampleReadError,
    InspectionFailedError,
    ModelInvocationError,
)
from inspector import (
    ModelInvoker,
    _extends_footer,
    _extract_json_payload,
    build_prompt,
    decode_sample,
    detect_encoding,
    inspect_csv,
    invoke_ollama_model,
    read_sample_bytes,
    read_tail_bytes,
)
from models import CSVInspectionResult

SAMPLE_CSV_PATH = Path(__file__).resolve().parent.parent / "agents" / "csv_inspector" / "sample.csv"

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


def test_extract_json_payload_strips_markdown_fence() -> None:
    """A JSON payload wrapped in a markdown code fence should be unwrapped."""
    fenced = '```json\n{"a": 1}\n```'

    assert _extract_json_payload(fenced) == '{"a": 1}'


def test_extract_json_payload_passes_through_bare_json() -> None:
    """A bare JSON payload with no fence should be returned unchanged (trimmed)."""
    bare = '  {"a": 1}  '

    assert _extract_json_payload(bare) == '{"a": 1}'


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
    [(0, 64, "n_bytes"), (64, -1, "tail_bytes")],
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
    """Register a stand-in ``ollama`` module exposing the given ``chat`` callable."""
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=chat))


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

    with pytest.raises(ModelInvocationError, match="pip install ollama"):
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
    """Invented names that share nothing with the data never promote a data row to header."""
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

    assert [column.name for column in result.columns] == ["col_1", "col_2", "col_3"]
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


def test_grounding_keeps_an_unanchored_footer_as_reported(tmp_path: Path) -> None:
    """If none of the reported footer lines occur in the file, the answer is left untouched."""
    target = tmp_path / "plain.csv"
    target.write_text("Fecha;Cliente;Importe\n2024-01-01;Acme;10.00\n", encoding="utf-8")

    result = inspect_csv(target, model_invoker=_sloppy_answer(footer_lines=["*** END ***"]))

    assert result.footer_lines == ["*** END ***"]


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


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("", True),
        ("TOTAL;;;12.50", True),
        ("Subtotal,,3", True),
        ('"Total general",9', True),
        ("Total registros: 250", True),
        ("SUMA;;;1", True),
        ("2024-01-01;Acme;10.00", False),
        ("Totalmente nuevo,1,2", False),
        ("Summary report", False),
        ("--- Fin del informe ---", False),
    ],
)
def test_extends_footer_accepts_only_blank_and_totals_rows(line: str, expected: bool) -> None:
    """Only blank separators and totals-labelled rows extend a footer upwards."""
    assert _extends_footer(line) is expected
