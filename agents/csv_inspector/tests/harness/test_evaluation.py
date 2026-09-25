"""Tests of ``eval_harness.evaluation``: one scored inspection and its run-file line."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any

import pytest
from harness_support import _answer, _evaluate_with, _fake_inspect, _result_with_columns, _usage

from csv_inspector import (
    InspectionFailedError,
    InspectionTimeoutError,
    LLMBackend,
    ModelInvocationError,
    ResponseParsingError,
    Settings,
    load_settings,
)
from csv_inspector import _inspect as inspect_module
from csv_inspector._invokers import builtin_invoker
from csv_inspector._prompt import PROMPT_VERSION
from eval_harness import evaluation as evaluation_module
from eval_harness.evaluation import FileEvaluation, evaluate_file
from eval_harness.guards import MANIFEST_PATH
from eval_harness.runs import (
    summarize,
)
from fakes import install_fake_ollama, ollama_reply


def test_file_evaluation_score_is_none_when_nothing_is_comparable() -> None:
    """A fixture with no ground truth has no score rather than a fake 0%."""
    evaluation = FileEvaluation(filename="f.csv", category="c", known_limitation=False)

    assert evaluation.score is None


def test_file_evaluation_score_is_fraction_of_matched_fields() -> None:
    """The score is matched / (matched + mismatched)."""
    evaluation = FileEvaluation(
        filename="f.csv",
        category="c",
        known_limitation=False,
        matched_fields=["encoding", "delimiter", "quotechar"],
        mismatched_fields=[("header_row_index", 0, 1)],
    )

    assert evaluation.score == pytest.approx(0.75)


def test_file_evaluation_score_is_none_on_pipeline_error() -> None:
    """An errored inspection is excluded from scoring."""
    evaluation = FileEvaluation(
        filename="f.csv",
        category="c",
        known_limitation=False,
        matched_fields=["encoding"],
        error="InspectionFailedError: boom",
    )

    assert evaluation.score is None


def test_evaluate_file_reports_a_timed_out_fixture_as_errored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The time budget reaches inspect_csv, and running out is a pipeline error (issue #54)."""
    received: dict[str, Any] = {}

    def fake_inspect_csv(source: object, /, **kwargs: Any) -> None:
        received.update(kwargs)
        raise InspectionTimeoutError("ran out of its 5s budget", attempts={})

    monkeypatch.setattr(evaluation_module, "inspect_csv", fake_inspect_csv)

    evaluation = evaluate_file(
        "delimiter_comma.csv",
        {"category": "delimiter", "known_limitation": False, "expected": {}},
        backend=LLMBackend.LOCAL,
        settings=Settings(),
        model="primary",
        fallback_model="fallback",
        n_bytes=4096,
        tail_bytes=4096,
        timeout_seconds=5.0,
    )

    assert received["timeout_seconds"] == 5.0
    assert evaluation.error is not None
    assert evaluation.error.startswith("InspectionTimeoutError")
    assert evaluation.score is None


def test_line_record_has_the_documented_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """One JSON line carries verdicts, usage (prompt version included) and cost."""
    _fake_inspect(monkeypatch, _answer(_usage(attempts=2, retries=1)))
    evaluation = evaluate_file(
        "f.csv",
        {"category": "c", "known_limitation": False, "expected": {"delimiter": ";"}},
        backend=LLMBackend.LOCAL,
        settings=Settings(),
        model="primary",
        fallback_model="fallback",
        n_bytes=4096,
        tail_bytes=4096,
        repeat=2,
    )

    record = json.loads(json.dumps(evaluation.to_record()))

    assert set(record) == {
        "fixture",
        "category",
        "repeat",
        "known_limitation",
        "model_used",
        "matched",
        "mismatched",
        "skipped",
        "score",
        "columns_recall",
        "columns_count_match",
        "usage",
        "latency_seconds",
        "calls",
        "error",
        "attempt_errors",
        "retries",
    }
    assert record["repeat"] == 2
    assert record["model_used"] == "primary"
    assert record["mismatched"] == [{"field": "delimiter", "expected": ";", "actual": ","}]
    assert record["usage"]["prompt_version"] == PROMPT_VERSION
    assert record["usage"]["prompt_tokens"] == 100
    assert record["calls"] == 3  # attempts + retries
    assert record["latency_seconds"] >= 0


def test_keep_raw_records_every_answer_and_keeps_the_token_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recording seam wraps the built-in invoker: raw text kept, usage intact, then undone."""
    answer = _result_with_columns("a").model_dump_json()
    replies = iter(
        [
            ollama_reply("not json", prompt_eval_count=50, eval_count=5),
            ollama_reply(answer, prompt_eval_count=70, eval_count=9),
        ]
    )
    install_fake_ollama(monkeypatch, lambda **kwargs: next(replies))
    assert inspect_module.builtin_invoker is builtin_invoker  # type: ignore[attr-defined]

    evaluation = evaluate_file(
        "delimiter_comma.csv",
        {"category": "delimiter", "known_limitation": False, "expected": {}},
        backend=LLMBackend.LOCAL,
        settings=Settings(),
        model="primary",
        fallback_model="fallback",
        n_bytes=4096,
        tail_bytes=4096,
        keep_raw=True,
    )

    assert inspect_module.builtin_invoker is builtin_invoker  # type: ignore[attr-defined]
    assert evaluation.error is None
    assert evaluation.to_record()["raw_response"] == [
        {"model": "primary", "text": "not json"},
        {"model": "fallback", "text": answer},
    ]
    assert evaluation.usage is not None
    assert (evaluation.usage.prompt_tokens, evaluation.usage.completion_tokens) == (120, 14)
    assert evaluation.calls == 2


def test_without_keep_raw_there_is_no_raw_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """``raw_response`` only appears when asked for."""
    evaluation = _evaluate_with(monkeypatch, _answer(), {})

    assert evaluation.raw_responses is None
    assert "raw_response" not in evaluation.to_record()


@pytest.mark.parametrize(
    ("backend", "fallback", "expected_calls"),
    [
        (LLMBackend.LOCAL, "fallback", 2),
        (LLMBackend.LOCAL, "primary", 1),
        (LLMBackend.API, "fallback", 4),
    ],
)
def test_a_failed_inspection_is_charged_its_worst_case(
    monkeypatch: pytest.MonkeyPatch, backend: LLMBackend, fallback: str, expected_calls: int
) -> None:
    """No usage on failure: one call per model, doubled for the cloud's one retry."""
    attempts: dict[str, Exception] = {
        "primary": ResponseParsingError("bad json"),
        "fallback": ModelInvocationError("Cloud model 'fallback' failed: 503 UNAVAILABLE"),
    }
    _fake_inspect(monkeypatch, InspectionFailedError("all failed", attempts=attempts))

    evaluation = evaluate_file(
        "f.csv",
        {"category": "c", "known_limitation": False, "expected": {}},
        backend=backend,
        settings=Settings(),
        model="primary",
        fallback_model=fallback,
        n_bytes=4096,
        tail_bytes=4096,
    )

    assert evaluation.calls == expected_calls
    assert evaluation.usage is None
    assert evaluation.attempt_errors == {
        "primary": "ResponseParsingError: bad json",
        "fallback": "ModelInvocationError: Cloud model 'fallback' failed: 503 UNAVAILABLE",
    }


def test_the_expected_error_scores_as_a_pass_not_as_an_error() -> None:
    """``empty_file.csv`` raises EmptySampleError before any model call: a pass."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    evaluation = evaluate_file(
        "empty_file.csv",
        manifest["empty_file.csv"],
        backend=LLMBackend.LOCAL,
        settings=Settings(),
        model="primary",
        fallback_model="fallback",
        n_bytes=4096,
        tail_bytes=4096,
    )

    assert evaluation.error is None
    assert evaluation.matched_fields == ["expected_error"]
    assert (evaluation.score, evaluation.calls) == (1.0, 0)
    summary = summarize([evaluation], {})
    assert summary["errors"] == {}
    assert summary["errored_lines"] == 0
    assert summary["fixture_verdicts"] == {"empty_file.csv": "pass"}


def test_a_missing_expected_error_is_a_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A result where an exception was expected fails on that pseudo-field."""
    _fake_inspect(monkeypatch, _answer())

    evaluation = evaluate_file(
        "f.csv",
        {"category": "c", "known_limitation": False, "expected": {}, "expected_error": "X"},
        backend=LLMBackend.LOCAL,
        settings=Settings(),
        model="primary",
        fallback_model="fallback",
        n_bytes=4096,
        tail_bytes=4096,
    )

    assert evaluation.mismatched_fields == [("expected_error", "X", None)]


def test_the_retries_of_a_failed_cloud_fixture_reach_the_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini answering 503 twice: the inspection fails, and its one retry is counted."""
    genai = pytest.importorskip("google.genai")
    import httpx  # noqa: PLC0415
    from google.genai import errors  # noqa: PLC0415

    def unavailable(**kwargs: Any) -> Any:
        body = {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}
        raise errors.ServerError(503, body, httpx.Response(503))

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            self.models = SimpleNamespace(generate_content=unavailable)

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *exc_info: object) -> None:
            pass

    monkeypatch.setattr(genai, "Client", Client)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")

    evaluation = evaluate_file(
        "delimiter_comma.csv",
        {"category": "delimiter", "known_limitation": False, "expected": {}},
        backend=LLMBackend.API,
        settings=load_settings(),
        model="gemini-x",
        fallback_model="gemini-x",
        n_bytes=4096,
        tail_bytes=4096,
    )

    assert evaluation.error is not None
    assert "503" in evaluation.attempt_errors["gemini-x"]
    assert evaluation.retries == 1
    assert summarize([evaluation], {})["retries"] == 1
