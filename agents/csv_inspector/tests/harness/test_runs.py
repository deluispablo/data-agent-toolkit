"""Tests of ``eval_harness.runs``: run files, summaries and interrupted runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from harness_support import (
    FIXTURES,
    _answer,
    _evaluation,
    _fake_inspect,
    _main_args,
    _read_run,
    _usage,
)

from csv_inspector import (
    CSVInspectionResult,
    LLMBackend,
)
from csv_inspector._prompt import PROMPT_VERSION
from eval_harness.cli import main
from eval_harness.evaluation import FileEvaluation
from eval_harness.runs import (
    HARNESS_VERSION,
    _percentile,
    output_path,
    run_info,
    summarize,
    summarize_file,
)


def test_percentile_interpolates_and_handles_edges() -> None:
    """Linear interpolation between ranks; one value is every percentile; none is None."""
    assert _percentile([], 0.5) is None
    assert _percentile([3.0], 0.95) == 3.0
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.5) == pytest.approx(2.5)
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.95) == pytest.approx(4.8)


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


def test_a_0_5_0_run_file_round_trips_and_resummarizes_unchanged(tmp_path: Path) -> None:
    """The run-file format is stable: a 0.5.0 excerpt reads back line for line, same summary."""
    lines = (Path(__file__).parent / "data" / "run-050-excerpt.jsonl").read_text("utf-8")
    run, *records, final = [json.loads(line) for line in lines.splitlines()]
    evaluations = [FileEvaluation.from_record(record) for record in records]

    assert [evaluation.to_record() for evaluation in evaluations] == records
    info = {**run["run"], "stopped_early": final["summary"]["stopped_early"]}
    assert summarize(evaluations, info) == final["summary"]
    interrupted = tmp_path / "interrupted.jsonl"
    interrupted.write_text("".join(lines.splitlines(keepends=True)[:-1]), encoding="utf-8")
    _, recovered = summarize_file(interrupted)
    assert recovered == {**final["summary"], "stopped_early": "interrupted", "incomplete": True}
