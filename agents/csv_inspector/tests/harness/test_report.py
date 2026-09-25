"""Tests of ``eval_harness.report``: the per-run report and the multi-run table."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from csv_inspector import LLMBackend
from eval_harness.evaluation import FileEvaluation
from eval_harness.report import (
    RunFileError,
    compare_main,
    load_summary,
    print_report,
    render,
    run_label,
    verdict_changes,
)
from eval_harness.runs import (
    RunWriter,
    run_info,
    summarize,
    summarize_file,
)
from harness_support import _evaluation, _usage

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


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
        backend="local",
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
        backend="local",
        model="primary",
        fallback_model="fallback",
    )

    out = capsys.readouterr().out
    assert "Pipeline errors" in out
    assert "Known limitations" in out
    assert "No scoreable files" in out


def _write_run(
    path: Path,
    model: str,
    verdicts: dict[str, bool],
    *,
    prompt_version: str = "v1",
    fixtures_planned: int | None = None,
    finish: bool = True,
) -> Path:
    """Write a run through the harness's own writer: one line per fixture, then the summary.

    With ``finish=False`` the summary line is left out, as in an interrupted run.
    """
    evaluations = [
        FileEvaluation(
            filename=fixture,
            category="delimiter" if fixture.startswith("d") else "quoting",
            known_limitation=False,
            matched_fields=["delimiter"] if passed else [],
            mismatched_fields=[] if passed else [("delimiter", ",", ";")],
            latency_seconds=2.0,
        )
        for fixture, passed in verdicts.items()
    ]
    info = run_info(
        backend=LLMBackend.LOCAL,
        model=model,
        fallback_model="fallback",
        n_bytes=4096,
        tail_bytes=4096,
        timeout_seconds=300.0,
        repeat=1,
        started_at="2026-09-24T00:00:00+00:00",
        stopped_early=None,
        fixtures_planned=fixtures_planned or len(verdicts),
    )
    info["prompt_version"] = prompt_version
    writer = RunWriter(path, info)
    for evaluation in evaluations:
        writer.add(evaluation)
    if finish:
        writer.finish(summarize(evaluations, info))
    else:
        writer.close()
    return path


@pytest.fixture
def two_runs(tmp_path: Path) -> tuple[Path, Path]:
    """A baseline and a candidate that fixes one fixture and breaks another."""
    baseline = _write_run(
        tmp_path / "baseline.jsonl",
        "qwen2.5-coder:7b",
        {"d1.csv": True, "d2.csv": False, "q1.csv": True},
    )
    candidate = _write_run(
        tmp_path / "candidate.jsonl",
        "qwen3:8b",
        {"d1.csv": True, "d2.csv": True, "q1.csv": False},
        prompt_version="v2",
    )
    return baseline, candidate


def test_table_puts_each_run_in_a_column(two_runs: tuple[Path, Path]) -> None:
    """Model, prompt version, cost, accuracy per category and per field, side by side."""
    summaries = [load_summary(path) for path in two_runs]

    report = render(["baseline", "candidate"], summaries)

    assert "| metric | baseline | candidate |" in report
    assert "| model | qwen2.5-coder:7b | qwen3:8b |" in report
    assert "| prompt version | v1 | v2 |" in report
    assert "| **accuracy** | **66.7%** | **66.7%** |" in report
    assert "| category `delimiter` | 50.0% | 100.0% |" in report
    assert "| category `quoting` | 100.0% | 0.0% |" in report
    assert "| field `delimiter` | 66.7% | 66.7% |" in report
    assert "| latency p50 | 2.00s | 2.00s |" in report
    assert "| prompt tokens (mean) | n/a | n/a |" in report


def test_verdict_changes_list_fixed_and_broken_fixtures(two_runs: tuple[Path, Path]) -> None:
    """Every fixture whose verdict moved is listed against the first run."""
    report = render(["baseline", "candidate"], [load_summary(path) for path in two_runs])

    assert "Verdict changes, baseline -> candidate:" in report
    assert "- `d2.csv`: fail: delimiter -> pass" in report
    assert "- `q1.csv`: pass -> fail: delimiter" in report
    assert "d1.csv" not in report.split("Verdict changes")[1]


def test_verdict_changes_mark_fixtures_run_only_once() -> None:
    """A fixture missing from one side shows as not run; identical runs change nothing."""
    first: dict[str, Any] = {"fixture_verdicts": {"a.csv": "pass"}}
    later: dict[str, Any] = {"fixture_verdicts": {"b.csv": "error"}}

    assert verdict_changes(first, later) == [
        "`a.csv`: pass -> not run",
        "`b.csv`: not run -> error",
    ]
    assert verdict_changes(first, first) == []


def test_main_prints_the_report(
    two_runs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """File stems label the columns; an unchanged run reports no verdict change."""
    baseline, _ = two_runs

    compare_main([str(baseline), str(baseline)])

    out = capsys.readouterr().out
    assert "| metric | baseline | baseline |" in out
    assert "- none" in out


def test_main_needs_two_runs(two_runs: tuple[Path, Path]) -> None:
    """One file is not a comparison."""
    with pytest.raises(SystemExit) as exit_info:
        compare_main([str(two_runs[0])])

    assert exit_info.value.code == 2


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "Cannot read"),
        ('{"fixture": "a.csv"}\nnot json\n', "line 2 is not JSON"),
        ('{"fixture": "a.csv"}\n\n', "has no summary line"),
    ],
)
def test_unusable_run_files_are_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str | None, message: str
) -> None:
    """A missing, malformed or interrupted run fails with a one-line message."""
    path = tmp_path / "run.jsonl"
    if content is not None:
        path.write_text(content, encoding="utf-8")

    with pytest.raises(RunFileError, match=message):
        load_summary(path)
    with pytest.raises(SystemExit) as exit_info:
        compare_main([str(path), str(path)])

    assert exit_info.value.code == 1
    assert message in capsys.readouterr().err


def test_summary_is_read_from_the_last_summary_line(tmp_path: Path) -> None:
    """Per-fixture lines are skipped; the summary payload is returned as written."""
    path = tmp_path / "run.jsonl"
    lines = [{"fixture": "a.csv"}, {"summary": {"model": "m"}}]
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")

    assert load_summary(path) == {"model": "m"}


def test_an_interrupted_run_is_refused_then_labelled_once_summarized(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], two_runs: tuple[Path, Path]
) -> None:
    """No summary: refused with the recovery hint; after --summarize: an incomplete column."""
    interrupted = _write_run(
        tmp_path / "cut.jsonl",
        "m",
        {"d1.csv": True, "q1.csv": False},
        fixtures_planned=3,
        finish=False,
    )
    with pytest.raises(RunFileError, match="--summarize"):
        load_summary(interrupted)

    summarize_file(interrupted)
    compare_main([str(two_runs[0]), str(interrupted)])

    out = capsys.readouterr().out
    assert "| metric | baseline | cut (incomplete, 2/3 fixtures) |" in out
    assert "`d2.csv`: fail: delimiter -> not run" in out


def test_compare_script_as_main(
    monkeypatch: pytest.MonkeyPatch, two_runs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """The module's ``__main__`` guard reaches ``main``."""
    monkeypatch.setattr(sys, "argv", ["compare_runs.py", *map(str, two_runs)])

    runpy.run_path(str(SCRIPTS_DIR / "compare_runs.py"), run_name="__main__")

    assert "| metric |" in capsys.readouterr().out


def test_a_replay_is_labelled_and_its_source_shown(two_runs: tuple[Path, Path]) -> None:
    """A replayed column says so in its header and names the run it replayed."""
    baseline, candidate = (load_summary(path) for path in two_runs)
    candidate["replay_of"] = "runs/base.jsonl"

    labels = [
        run_label(path, summary)
        for path, summary in zip(two_runs, (baseline, candidate), strict=True)
    ]
    report = render(labels, [baseline, candidate])

    assert "| metric | baseline | candidate (replay) |" in report
    assert "| answers | live | replay of runs/base.jsonl |" in report
