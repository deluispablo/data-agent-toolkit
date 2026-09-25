"""Unit tests for the LLM-free logic of the manual evaluation harness (no model is called)."""

from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import eval_samples
from csv_inspector import (
    CSVInspectionResult,
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
from csv_inspector._models import Usage
from csv_inspector._prompt import PROMPT_VERSION
from csv_inspector.cli import DEFAULT_CLI_TIMEOUT_SECONDS
from eval_samples import (
    HARNESS_VERSION,
    MANIFEST_PATH,
    SUBSETS,
    CallBudget,
    FileEvaluation,
    RateLimiter,
    _column_diagnostics,
    _format_file_line,
    _matches_encoding,
    _parse_args,
    _percentile,
    evaluate_file,
    line_record,
    main,
    output_path,
    print_report,
    repeat_stats,
    run_info,
    select_fixtures,
    summarize,
    summarize_file,
)
from fakes import install_fake_ollama, ollama_reply


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        ("utf-8", "UTF-8"),
        ("utf-8", "utf_8"),
        ("utf-8-sig", "UTF-8-SIG"),
        ("latin-1 or cp1252 (not utf-8)", "ISO-8859-1"),
        ("latin-1 or cp1252 (not utf-8)", "windows-1252"),
        ("utf-16 or utf-16-le", "UTF-16LE"),
    ],
)
def test_matches_encoding_accepts_aliases_of_an_expected_codec(expected: str, actual: str) -> None:
    """Different spellings of the same codec count as a match."""
    assert _matches_encoding(expected, actual)


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        ("latin-1 or cp1252 (not utf-8)", "utf-8"),
        ("utf-8", "utf-8-sig"),
        ("utf-8-sig", "utf-8"),
        ("utf-16 or utf-16-le", "utf-16-be"),
    ],
)
def test_matches_encoding_rejects_different_codecs(expected: str, actual: str) -> None:
    """Parenthesized remarks are ignored and near-miss codecs do not match."""
    assert not _matches_encoding(expected, actual)


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

    monkeypatch.setattr(eval_samples, "inspect_csv", fake_inspect_csv)

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


def _result_with_columns(*names: str) -> CSVInspectionResult:
    """Build a minimal inspection result reporting ``names`` as its columns.

    The names are set as written, surrounding spaces included, like grounding
    does after validation (the validator strips a model's names).
    """
    result = CSVInspectionResult(
        encoding="utf-8",
        delimiter=",",
        header_row_index=0,
        columns=list(names),
        confidence=0.9,
    )
    return result.model_copy(update={"columns": list(names)})


def _evaluate_with(
    monkeypatch: pytest.MonkeyPatch, result: CSVInspectionResult, expected: dict[str, Any]
) -> FileEvaluation:
    """Run ``evaluate_file`` with ``inspect_csv`` faked to return ``result``."""

    def fake_inspect_csv(source: object, /, **kwargs: Any) -> CSVInspectionResult:
        return result

    monkeypatch.setattr(eval_samples, "inspect_csv", fake_inspect_csv)
    return evaluate_file(
        "f.csv",
        {"category": "c", "known_limitation": False, "expected": expected},
        backend=LLMBackend.LOCAL,
        settings=Settings(),
        model="primary",
        fallback_model="fallback",
        n_bytes=4096,
        tail_bytes=4096,
    )


def test_exact_column_names_match_and_score_full_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Names as written in the file count as a match, with full recall and count (issue #123)."""
    evaluation = _evaluate_with(
        monkeypatch,
        _result_with_columns("Fecha ", " Importe"),
        {"columns": ["Fecha ", " Importe"]},
    )

    assert evaluation.matched_fields == ["columns"]
    assert evaluation.columns_recall == 1.0
    assert evaluation.columns_count_match is True


def test_paraphrased_column_name_is_a_mismatch_with_partial_recall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Monto`` for ``Importe`` fails the exact match; recall and count show why (issue #123)."""
    evaluation = _evaluate_with(
        monkeypatch,
        _result_with_columns("Fecha", "Cliente", "Monto"),
        {"columns": ["Fecha", "Cliente", "Importe"]},
    )

    assert evaluation.mismatched_fields == [
        ("columns", ["Fecha", "Cliente", "Importe"], ["Fecha", "Cliente", "Monto"])
    ]
    assert evaluation.columns_recall == pytest.approx(2 / 3)
    assert evaluation.columns_count_match is True
    assert "[columns recall 67%, count match]" in _format_file_line(evaluation)


def test_column_recall_ignores_padding_but_the_exact_match_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stripped names count for recall; a dropped column shows as a count mismatch."""
    evaluation = _evaluate_with(
        monkeypatch,
        _result_with_columns("Fecha", "Cliente"),
        {"columns": ["Fecha ", " Cliente ", " Importe"]},
    )

    assert [name for name, _, _ in evaluation.mismatched_fields] == ["columns"]
    assert evaluation.columns_recall == pytest.approx(2 / 3)
    assert evaluation.columns_count_match is False
    assert "count mismatch" in _format_file_line(evaluation)


def test_column_diagnostics_are_none_without_expected_columns() -> None:
    """No ground truth means no diagnostics, and an empty expectation has no recall."""
    assert _column_diagnostics(None, ["a"]) == (None, None)
    assert _column_diagnostics([], []) == (None, True)


def test_column_recall_counts_duplicated_names_once_per_occurrence() -> None:
    """A name expected twice is only fully recalled when reported twice."""
    assert _column_diagnostics(["Fecha", "Fecha"], ["Fecha", "Otra"]) == (0.5, True)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], DEFAULT_CLI_TIMEOUT_SECONDS), (["--timeout", "30"], 30.0), (["--timeout", "0"], None)],
)
def test_timeout_option_matches_the_cli(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expected: float | None
) -> None:
    """``--timeout`` defaults to the CLI's budget, and 0 disables it (issue #54)."""
    monkeypatch.setattr(sys, "argv", ["eval_samples.py", *argv])

    assert _parse_args().timeout == expected


# ---------------------------------------------------------------------
# Runs: JSONL lines, repeats, summary, quota guards (issue #122)
# ---------------------------------------------------------------------

FIXTURES = ["delimiter_comma.csv", "delimiter_pipe.csv", "delimiter_semicolon.csv"]


def _usage(**overrides: Any) -> Usage:
    """A ``Usage`` with small, recognizable counters."""
    values: dict[str, Any] = {
        "model": "primary",
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "latency_seconds": 1.5,
        "attempts": 1,
        "prompt_version": PROMPT_VERSION,
    }
    values.update(overrides)
    return Usage(**values)


def _answer(usage: Usage | None = None) -> CSVInspectionResult:
    """A comma-delimited result, with ``usage`` attached."""
    return _result_with_columns("a", "b").model_copy(update={"usage": usage or _usage()})


def _evaluation(
    filename: str = "f.csv",
    *,
    matched: list[str] | None = None,
    mismatched: list[tuple[str, Any, Any]] | None = None,
    **overrides: Any,
) -> FileEvaluation:
    """A synthetic evaluation, as a fake run would produce it."""
    values: dict[str, Any] = {"filename": filename, "category": "c", "known_limitation": False}
    values.update(overrides)
    return FileEvaluation(
        matched_fields=matched or [], mismatched_fields=mismatched or [], **values
    )


def _fake_inspect(monkeypatch: pytest.MonkeyPatch, answer: Any) -> list[dict[str, Any]]:
    """Replace ``inspect_csv`` by ``answer`` (a result, an exception or a callable)."""
    calls: list[dict[str, Any]] = []

    def fake_inspect_csv(source: object, /, **kwargs: Any) -> CSVInspectionResult:
        calls.append({"source": source, **kwargs})
        if isinstance(answer, Exception):
            raise answer
        return answer(**kwargs) if callable(answer) else answer  # type: ignore[no-any-return]

    monkeypatch.setattr(eval_samples, "inspect_csv", fake_inspect_csv)
    return calls


def _read_run(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


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

    record = json.loads(json.dumps(line_record(evaluation)))

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
    assert line_record(evaluation)["raw_response"] == [
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
    assert "raw_response" not in line_record(evaluation)


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


def test_percentile_interpolates_and_handles_edges() -> None:
    """Linear interpolation between ranks; one value is every percentile; none is None."""
    assert _percentile([], 0.5) is None
    assert _percentile([3.0], 0.95) == 3.0
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.5) == pytest.approx(2.5)
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.95) == pytest.approx(4.8)


def test_majority_vote_and_agreement_per_field() -> None:
    """Majority needs more than half the repeats; agreement is the modal answer's share."""
    evaluations = [
        # a.csv: delimiter right 2 of 3 times; encoding always right.
        _evaluation("a.csv", matched=["encoding", "delimiter"], repeat=1),
        _evaluation("a.csv", matched=["encoding"], mismatched=[("delimiter", ",", ";")], repeat=2),
        _evaluation("a.csv", matched=["encoding", "delimiter"], repeat=3),
        # b.csv: delimiter wrong twice with different answers, and one error.
        _evaluation("b.csv", mismatched=[("delimiter", ",", ";")], repeat=1),
        _evaluation("b.csv", mismatched=[("delimiter", ",", "|")], repeat=2),
        _evaluation("b.csv", error="InspectionFailedError: boom", repeat=3),
        # c.csv: stable; a known limitation, so out of the scores.
        _evaluation("c.csv", mismatched=[("delimiter", ",", ";")], known_limitation=True),
        # d.csv: every repeat failed.
        _evaluation("d.csv", error="InspectionTimeoutError: slow", repeat=1),
        _evaluation("d.csv", error="InspectionTimeoutError: slow", repeat=2),
    ]

    stats = repeat_stats(evaluations)

    assert stats.field_majority == {"encoding": 1.0, "delimiter": 0.5}
    assert stats.field_agreement["encoding"] == 1.0
    assert stats.field_agreement["delimiter"] == pytest.approx((2 / 3 + 1 / 3) / 2)
    assert stats.majority_score == pytest.approx((1.0 + 0.0) / 2)
    assert stats.disagreeing == ["a.csv", "b.csv"]
    assert stats.verdicts == {
        "a.csv": "pass",
        "b.csv": "fail: delimiter",
        "c.csv": "fail: delimiter",
        "d.csv": "error",
    }


def test_a_fixture_without_ground_truth_is_unscored() -> None:
    """Nothing compared and no error: the verdict says so instead of pass."""
    assert repeat_stats([_evaluation("x.csv")]).verdicts == {"x.csv": "unscored"}


def test_summary_aggregates_scores_cost_and_errors() -> None:
    """Scores exclude errors and known limitations; costs count every line."""
    evaluations = [
        _evaluation(
            "a.csv",
            matched=["delimiter", "columns"],
            category="delimiter",
            usage=_usage(attempts=2, retries=1),
            retries=1,
            latency_seconds=1.0,
            calls=3,
            columns_recall=1.0,
            columns_count_match=True,
        ),
        _evaluation(
            "b.csv",
            matched=["delimiter"],
            mismatched=[("columns", ["x"], ["y"])],
            category="quoting",
            usage=_usage(prompt_tokens=None, completion_tokens=40),
            latency_seconds=3.0,
            calls=1,
            columns_recall=0.0,
            columns_count_match=True,
        ),
        _evaluation(
            "c.csv",
            error="InspectionFailedError: all failed",
            attempt_errors={
                "primary": "ModelInvocationError: 429 RESOURCE_EXHAUSTED",
                "fallback": "ModelInvocationError: 503 UNAVAILABLE",
            },
            latency_seconds=5.0,
            calls=2,
        ),
        _evaluation("d.csv", mismatched=[("delimiter", ",", ";")], known_limitation=True),
    ]
    info = run_info(
        backend=LLMBackend.LOCAL,
        model="primary",
        fallback_model="fallback",
        n_bytes=4096,
        tail_bytes=4096,
        timeout_seconds=None,
        repeat=1,
        started_at="2026-09-24T00:00:00+00:00",
        stopped_early=None,
        fixtures_planned=4,
    )

    summary = json.loads(json.dumps(summarize(evaluations, info)))

    assert summary["harness_version"] == HARNESS_VERSION
    assert summary["prompt_version"] == PROMPT_VERSION
    assert (summary["fixtures_run"], summary["lines"], summary["scored_lines"]) == (4, 4, 2)
    assert (summary["errored_lines"], summary["known_limitation_lines"]) == (1, 1)
    assert summary["aggregate_score"] == pytest.approx(0.75)
    assert summary["category_scores"] == {"delimiter": 1.0, "quoting": 0.5}
    assert summary["field_scores"] == {"delimiter": 1.0, "columns": 0.5}
    assert summary["tokens"] == {
        "prompt_total": 100,
        "completion_total": 60,
        "prompt_mean": 100,
        "completion_mean": 30,
    }
    assert summary["latency_seconds"]["p50"] == pytest.approx(3.0)
    assert (summary["calls"], summary["fallback_used"], summary["retries"]) == (6, 1, 1)
    assert summary["errors"] == {"InspectionFailedError": 1}
    assert summary["attempt_errors"] == {"ModelInvocationError": 2}
    assert (summary["http_429"], summary["http_503"]) == (1, 1)
    assert summary["columns_recall_mean"] == pytest.approx(0.5)
    assert summary["columns_count_match_rate"] == 1.0
    assert summary["fixture_verdicts"]["b.csv"] == "fail: columns"


def test_report_shows_repeats_agreement_and_an_early_stop(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With repeats the report numbers each line and lists the disagreeing fixtures."""
    evaluations = [
        _evaluation("a.csv", matched=["delimiter"], repeat=1, usage=_usage(), calls=1),
        _evaluation("a.csv", mismatched=[("delimiter", ",", ";")], repeat=2, calls=1),
    ]
    info = run_info(
        backend=LLMBackend.LOCAL,
        model="primary",
        fallback_model="fallback",
        n_bytes=4096,
        tail_bytes=4096,
        timeout_seconds=None,
        repeat=2,
        started_at="now",
        stopped_early="--max-calls 2 reached",
        fixtures_planned=1,
    )

    print_report(
        evaluations,
        backend=LLMBackend.LOCAL,
        model="primary",
        fallback_model="fallback",
        summary=summarize(evaluations, info),
    )

    out = capsys.readouterr().out
    assert "[c] a.csv #2:" in out
    assert "Model calls: 2" in out
    assert "Agreement between repeats: delimiter 50%" in out
    assert "Answers differ between repeats: a.csv" in out
    assert "Stopped early: --max-calls 2 reached" in out


def test_report_without_scoreable_files(capsys: pytest.CaptureFixture[str]) -> None:
    """Only errors and known limitations: no aggregate, and both sections listed."""
    print_report(
        [
            _evaluation("a.csv", error="EmptySampleError: empty"),
            _evaluation("b.csv", known_limitation=True),
        ],
        backend=LLMBackend.LOCAL,
        model="primary",
        fallback_model="fallback",
    )

    out = capsys.readouterr().out
    assert "Pipeline errors" in out
    assert "Known limitations" in out
    assert "No scoreable files" in out


def test_call_budget_admits_a_fixture_only_when_its_worst_case_fits() -> None:
    """The budget never lets the run exceed ``max_calls``; ``None`` is unlimited."""
    budget = CallBudget(3)
    budget.charge(2)

    assert budget.allows(1)
    assert not budget.allows(2)
    assert CallBudget(None).allows(10**6)


def test_rate_limiter_waits_for_the_window_to_free_up() -> None:
    """With the window full it sleeps until the oldest call is a minute old."""
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(2, clock=lambda: now[0], sleep=sleep)
    limiter.wait(2)
    limiter.record(2)
    now[0] = 10.0
    limiter.wait(5)  # capped at rpm: waits for the whole window

    assert sleeps == [pytest.approx(50.0)]
    RateLimiter(None, sleep=sleep).wait(100)
    assert len(sleeps) == 1


def test_select_fixtures_filters_and_caps() -> None:
    """``--fixture``, ``--category`` and ``--max-fixtures`` narrow the sorted catalog."""
    manifest = {
        "b.csv": {"category": "x"},
        "a.csv": {"category": "x"},
        "c.csv": {"category": "y"},
    }

    assert [f for f, _ in select_fixtures(manifest, category="x", names=None, max_fixtures=1)] == [
        "a.csv"
    ]
    assert [
        f for f, _ in select_fixtures(manifest, category=None, names=["c.csv"], max_fixtures=None)
    ] == ["c.csv"]
    with pytest.raises(ValueError, match=r"Unknown --fixture: z.csv"):
        select_fixtures(manifest, category=None, names=["z.csv"], max_fixtures=None)
    with pytest.raises(ValueError, match="No fixture"):
        select_fixtures(manifest, category="none", names=None, max_fixtures=None)


def test_output_path_forms(tmp_path: Path) -> None:
    """A template, a file, or a directory; never overwrite, never one file for two models."""
    stamp = "20260924T000000Z"

    assert output_path(None, "m", several_models=False, stamp=stamp) is None
    assert output_path(
        "runs/{model}.jsonl", "qwen2.5-coder:7b", several_models=True, stamp=stamp
    ) == (Path("runs/qwen2.5-coder-7b.jsonl"))
    assert output_path("x.jsonl", "m", several_models=False, stamp=stamp) == Path("x.jsonl")
    assert output_path("runs", "a/b", several_models=True, stamp=stamp) == Path(
        f"runs/{stamp}-a-b.jsonl"
    )
    with pytest.raises(ValueError, match="several --model"):
        output_path("x.jsonl", "m", several_models=True, stamp=stamp)
    (tmp_path / "taken.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        output_path(str(tmp_path / "taken.jsonl"), "m", several_models=False, stamp=stamp)


def _main_args(*extra: str) -> list[str]:
    fixtures = [arg for name in FIXTURES for arg in ("--fixture", name)]
    return ["--no-env-file", "--log-level", "ERROR", *fixtures, *extra]


def test_repeat_and_out_write_one_line_per_fixture_and_repeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--repeat 3 --out`` writes 3 x fixtures lines and a final summary line."""
    calls = _fake_inspect(monkeypatch, _answer())
    out = tmp_path / "runs" / "x.jsonl"

    main(_main_args("--repeat", "3", "--out", str(out), "--model", "primary"))

    lines = _read_run(out)
    assert len(calls) == 9
    assert len(lines) == 1 + 3 * len(FIXTURES) + 1
    assert lines[0]["run"]["fixtures_planned"] == len(FIXTURES)
    assert [line["repeat"] for line in lines[1:4]] == [1, 2, 3]
    summary = lines[-1]["summary"]
    assert (summary["repeat"], summary["fixtures_run"], summary["model"]) == (3, 3, "primary")
    assert summary["stopped_early"] is None
    assert summary["incomplete"] is False
    assert "Wrote 9 line(s)" in capsys.readouterr().out


def test_several_models_run_in_sequence_into_separate_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--model`` is repeatable, and ``{model}`` names each file."""
    calls = _fake_inspect(monkeypatch, lambda **kwargs: _answer(_usage(model=kwargs["model"])))

    main(
        _main_args(
            "--model",
            "a",
            "--model",
            "b",
            "--max-fixtures",
            "1",
            "--out",
            str(tmp_path / "{model}.jsonl"),
        )
    )

    assert [call["model"] for call in calls] == ["a", "b"]
    assert _read_run(tmp_path / "a.jsonl")[1]["model_used"] == "a"
    assert _read_run(tmp_path / "b.jsonl")[-1]["summary"]["model"] == "b"


def test_max_calls_stops_the_run_before_the_budget_is_exceeded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each fixture may need 2 calls (primary + fallback): with 3 allowed, two fixtures run."""
    calls = _fake_inspect(monkeypatch, _answer())
    out = tmp_path / "x.jsonl"

    main(
        _main_args(
            "--model",
            "primary",
            "--fallback-model",
            "fallback",
            "--max-calls",
            "3",
            "--out",
            str(out),
        )
    )

    lines = _read_run(out)
    assert len(calls) == 2
    assert len(lines) == 4
    assert lines[-1]["summary"]["calls"] == 2
    assert "--max-calls 3 reached" in lines[-1]["summary"]["stopped_early"]


def test_dry_run_calls_no_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--dry-run`` prints prompt sizes and the planned calls; an empty file is reported."""
    calls = _fake_inspect(monkeypatch, AssertionError("no model call in a dry run"))

    main(
        _main_args("--dry-run", "--backend", "api", "--fixture", "empty_file.csv", "--repeat", "2")
    )

    out = capsys.readouterr().out
    assert calls == []
    assert "delimiter_comma.csv:" in out
    assert "tokens" in out
    assert "empty_file.csv: cannot sample" in out
    # 4 fixtures x 2 repeats x (primary + fallback) x 2 for the cloud retry.
    assert "up to 32 model call(s) planned" in out


def test_cloud_run_without_max_calls_warns_with_the_planned_calls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No ``--max-calls`` on the api backend: a warning names the worst case first."""
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")
    _fake_inspect(monkeypatch, _answer())

    main(_main_args("--backend", "api", "--max-fixtures", "1"))
    warned = capsys.readouterr().err

    main(_main_args("--backend", "api", "--max-fixtures", "1", "--max-calls", "4"))
    not_warned = capsys.readouterr().err

    assert "without --max-calls: this run may make up to 4 model request(s)" in warned
    assert "--max-calls" not in not_warned


def test_invalid_filters_exit_before_any_call(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown fixture is a usage error, reported once."""
    calls = _fake_inspect(monkeypatch, _answer())

    with pytest.raises(SystemExit) as exit_info:
        main(["--no-env-file", "--fixture", "nope.csv"])

    assert exit_info.value.code == 2
    assert calls == []


def test_unusable_backend_exits_before_any_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The api backend without credentials fails once, up front."""
    calls = _fake_inspect(monkeypatch, _answer())

    with pytest.raises(SystemExit) as exit_info:
        main(["--no-env-file", "--backend", "api"])

    assert exit_info.value.code == 1
    assert calls == []


def test_run_script_as_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module's ``__main__`` guard reaches ``main``."""
    monkeypatch.setattr(sys, "argv", ["eval_samples.py", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(eval_samples.__file__, run_name="__main__")
    assert exit_info.value.code == 0


# ---------------------------------------------------------------------
# Streamed runs, --summarize, subsets, expected errors, retries (issue #149)
# ---------------------------------------------------------------------


def test_an_interrupted_run_keeps_its_finished_lines_and_can_be_summarized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Lines are flushed as they finish; --summarize appends an incomplete summary."""
    finished = 0

    def answer_then_interrupt(**kwargs: Any) -> CSVInspectionResult:
        nonlocal finished
        if finished == 2:
            raise KeyboardInterrupt
        finished += 1
        return _answer()

    _fake_inspect(monkeypatch, answer_then_interrupt)
    out = tmp_path / "cut.jsonl"

    with pytest.raises(SystemExit) as exit_info:
        main(_main_args("--model", "primary", "--out", str(out)))

    assert exit_info.value.code == 130
    lines = _read_run(out)
    assert [set(line) & {"run", "fixture"} for line in lines] == [{"run"}, {"fixture"}, {"fixture"}]

    main(["--summarize", str(out)])

    summary = _read_run(out)[-1]["summary"]
    assert summary["incomplete"] is True
    assert (summary["fixtures_run"], summary["fixtures_planned"]) == (2, len(FIXTURES))
    assert summary["stopped_early"] == "interrupted"
    assert summary["model"] == "primary"
    assert "Appended an incomplete summary" in capsys.readouterr().out


def test_summarize_refuses_a_finished_run_and_a_file_without_run_line(tmp_path: Path) -> None:
    """Only an interrupted harness-version-2 run can be summarized."""
    finished = tmp_path / "done.jsonl"
    finished.write_text('{"run": {}}\n{"summary": {}}\n', encoding="utf-8")
    legacy = tmp_path / "v1.jsonl"
    legacy.write_text('{"fixture": "a.csv"}\n', encoding="utf-8")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already has a summary"):
        summarize_file(finished)
    with pytest.raises(ValueError, match="version 2"):
        summarize_file(legacy)
    with pytest.raises(ValueError, match="no run line"):
        summarize_file(empty)
    with pytest.raises(SystemExit) as exit_info:
        main(["--summarize", str(legacy)])
    assert exit_info.value.code == 2


def test_subsets_name_manifest_fixtures() -> None:
    """The quick subset covers every category; cloud is the documented 15 regular fixtures."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for names in SUBSETS.values():
        assert set(names) <= manifest.keys()
        assert len(set(names)) == len(names)
    assert {manifest[name]["category"] for name in SUBSETS["quick"]} == {
        entry["category"] for entry in manifest.values()
    }
    assert len(SUBSETS["cloud"]) == 15
    assert not any(manifest[name]["known_limitation"] for name in SUBSETS["cloud"])


def test_subset_selects_its_fixtures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``--subset cloud`` runs exactly the cloud list."""
    calls = _fake_inspect(monkeypatch, _answer())

    main(["--no-env-file", "--log-level", "ERROR", "--subset", "cloud"])

    assert sorted(Path(str(call["source"])).name for call in calls) == sorted(SUBSETS["cloud"])


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


_REPLAY_FIXTURES = ["delimiter_comma.csv", "delimiter_pipe.csv"]


def _raw_answer(delimiter: str) -> str:
    """A model answer for the two replay fixtures, which share their header."""
    return json.dumps(
        {
            "encoding": "utf-8",
            "delimiter": delimiter,
            "quotechar": '"',
            "escapechar": None,
            "doublequote": True,
            "has_header": True,
            "header_row_index": 0,
            "footer_first_line": None,
            "columns": ["Fecha", "Cliente", "Descripcion", "Importe", "Observaciones"],
            "confidence": 0.9,
        }
    )


def _write_replayable_run(path: Path, *, keep_raw: bool = True) -> Path:
    """A two-fixture, two-repeat run file as ``--keep-raw`` writes it.

    ``delimiter_comma.csv`` answered at once; on ``delimiter_pipe.csv`` the
    primary's text was not JSON and the fallback answered.
    """
    info = run_info(
        backend=LLMBackend.LOCAL,
        model="primary",
        fallback_model="fallback",
        n_bytes=4096,
        tail_bytes=4096,
        timeout_seconds=300.0,
        repeat=2,
        started_at="2026-09-26T00:00:00+00:00",
        stopped_early=None,
        fixtures_planned=2,
    )
    info["prompt_version"] = "recorded"
    raw = {
        "delimiter_comma.csv": [{"model": "primary", "text": _raw_answer(",")}],
        "delimiter_pipe.csv": [
            {"model": "primary", "text": "not json"},
            {"model": "fallback", "text": _raw_answer("|")},
        ],
    }
    lines: list[dict[str, Any]] = [{"run": info}]
    for fixture in _REPLAY_FIXTURES:
        for repeat in (1, 2):
            line: dict[str, Any] = {"fixture": fixture, "repeat": repeat}
            if keep_raw:
                line["raw_response"] = raw[fixture]
            lines.append(line)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def _replay_args(run: Path, out: Path, *extra: str) -> list[str]:
    fixtures = [arg for name in _REPLAY_FIXTURES for arg in ("--fixture", name)]
    return ["--replay", str(run), "--out", str(out), "--log-level", "ERROR", *fixtures, *extra]


def test_replay_answers_from_the_run_and_calls_no_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each (fixture, repeat) is answered from its recorded text, the fallback's included."""

    def no_model(*args: object) -> object:
        raise AssertionError("a replay must not build a model invoker")

    monkeypatch.setattr(inspect_module, "builtin_invoker", no_model)
    out = tmp_path / "replay.jsonl"

    main(_replay_args(_write_replayable_run(tmp_path / "run.jsonl"), out))

    lines = _read_run(out)
    summary = lines[-1]["summary"]
    assert summary["replay_of"] == str(tmp_path / "run.jsonl")
    assert (summary["prompt_version"], summary["model"], summary["repeat"]) == (
        "recorded",
        "primary",
        2,
    )
    assert summary["aggregate_score"] == 1.0
    assert summary["fallback_used"] == 2
    assert summary["tokens"]["prompt_mean"] is None
    pipe = lines[3]
    assert pipe["fixture"] == "delimiter_pipe.csv"
    assert pipe["model_used"] == "fallback"
    assert [attempt["model"] for attempt in pipe["raw_response"]] == ["primary", "fallback"]


def test_replay_measures_a_grounding_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same answers, different grounding: the verdicts move, as they would live."""
    original = inspect_module.ground_in_samples  # type: ignore[attr-defined]

    def comma_only(answer: Any, *args: Any, **kwargs: Any) -> CSVInspectionResult:
        result: CSVInspectionResult = original(answer, *args, **kwargs)
        return result.model_copy(update={"delimiter": ","})

    monkeypatch.setattr(inspect_module, "ground_in_samples", comma_only)
    out = tmp_path / "replay.jsonl"

    main(_replay_args(_write_replayable_run(tmp_path / "run.jsonl"), out))

    verdicts = _read_run(out)[-1]["summary"]["fixture_verdicts"]
    assert verdicts["delimiter_comma.csv"] == "pass"
    assert verdicts["delimiter_pipe.csv"].startswith("fail")


def test_a_replayed_attempt_without_text_fails_like_the_live_one() -> None:
    """A model with no recorded answer left is a failed attempt, not a crash."""
    evaluation = evaluate_file(
        "delimiter_comma.csv",
        {"category": "delimiter", "known_limitation": False, "expected": {"delimiter": ","}},
        backend=LLMBackend.LOCAL,
        settings=Settings(),
        model="primary",
        fallback_model="fallback",
        n_bytes=4096,
        tail_bytes=4096,
        replayed=[{"model": "fallback", "text": _raw_answer(",")}],
    )

    assert evaluation.error is None
    assert evaluation.usage is not None
    assert (evaluation.usage.model, evaluation.usage.attempts) == ("fallback", 2)
    assert evaluation.raw_responses == [{"model": "fallback", "text": _raw_answer(",")}]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--backend", "api"), "drop --backend"),
        (("--rpm", "5", "--max-calls", "3"), "drop --rpm, --max-calls"),
        (("--dry-run",), "drop --dry-run"),
        (("--model", "a", "--model", "b"), "at most one --model"),
        (("--bytes", "8192"), "--bytes 8192 differs from the run's 4096"),
        (("--tail-bytes", "0"), "--tail-bytes 0 differs"),
        (("--category", "quoting"), "No fixture matches"),
    ],
)
def test_replay_refuses_what_it_cannot_honour(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, extra: tuple[str, ...], message: str
) -> None:
    """Quota flags, several models, other windows: refused before anything runs."""
    run = _write_replayable_run(tmp_path / "run.jsonl")

    with pytest.raises(SystemExit) as exit_info:
        main(_replay_args(run, tmp_path / "replay.jsonl", *extra))

    assert exit_info.value.code == 2
    assert message in caplog.text
    assert not (tmp_path / "replay.jsonl").exists()


def test_replay_refuses_a_run_without_raw_answers_or_run_line(tmp_path: Path) -> None:
    """Only a ``--keep-raw`` harness-version-2 run can be replayed."""
    without_raw = _write_replayable_run(tmp_path / "plain.jsonl", keep_raw=False)
    legacy = tmp_path / "v1.jsonl"
    legacy.write_text('{"fixture": "a.csv"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="--keep-raw"):
        eval_samples.load_replay(without_raw, n_bytes=None, tail_bytes=None)
    with pytest.raises(ValueError, match="no run line"):
        eval_samples.load_replay(legacy, n_bytes=None, tail_bytes=None)


def test_replay_skips_and_counts_fixtures_the_run_does_not_hold(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A selected fixture missing from the run is skipped, counted and logged."""
    out = tmp_path / "replay.jsonl"
    args = _replay_args(_write_replayable_run(tmp_path / "run.jsonl"), out)

    main([*args, "--fixture", "delimiter_semicolon.csv", "--log-level", "WARNING"])

    summary = _read_run(out)[-1]["summary"]
    assert summary["replay_skipped"] == 1
    assert summary["fixtures_run"] == 2
    assert "Skipping 1 selected fixture(s)" in caplog.text
