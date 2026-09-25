"""Tests of ``eval_harness.replay`` and ``--replay``: recorded answers, no model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from csv_inspector import (
    CSVInspectionResult,
    LLMBackend,
    Settings,
)
from csv_inspector import _inspect as inspect_module
from eval_harness.cli import main
from eval_harness.evaluation import evaluate_file
from eval_harness.replay import load_replay
from eval_harness.runs import (
    run_info,
)
from harness_support import _read_run

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
        load_replay(without_raw, n_bytes=None, tail_bytes=None)
    with pytest.raises(ValueError, match="no run line"):
        load_replay(legacy, n_bytes=None, tail_bytes=None)


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
