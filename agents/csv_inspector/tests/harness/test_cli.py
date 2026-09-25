"""Tests of ``eval_harness.cli``: ``eval_samples.py`` end to end, with fake models."""

from __future__ import annotations

import re
import runpy
import sys
from pathlib import Path

import pytest

from csv_inspector import _prompt as prompt_module
from csv_inspector.cli import DEFAULT_CLI_TIMEOUT_SECONDS
from eval_harness.cli import main, parse_args
from eval_harness.guards import SUBSETS
from harness_support import FIXTURES, _answer, _fake_inspect, _main_args, _read_run, _usage

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], DEFAULT_CLI_TIMEOUT_SECONDS), (["--timeout", "30"], 30.0), (["--timeout", "0"], None)],
)
def test_timeout_option_matches_the_cli(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expected: float | None
) -> None:
    """``--timeout`` defaults to the CLI's budget, and 0 disables it (issue #54)."""
    monkeypatch.setattr(sys, "argv", ["eval_samples.py", *argv])

    assert parse_args().timeout == expected


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
        runpy.run_path(str(SCRIPTS_DIR / "eval_samples.py"), run_name="__main__")
    assert exit_info.value.code == 0


def test_subset_selects_its_fixtures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``--subset cloud`` runs exactly the cloud list."""
    calls = _fake_inspect(monkeypatch, _answer())

    main(["--no-env-file", "--log-level", "ERROR", "--subset", "cloud"])

    assert sorted(Path(str(call["source"])).name for call in calls) == sorted(SUBSETS["cloud"])


def test_the_dry_run_estimate_uses_the_library_chars_per_token(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One measured constant sizes Ollama's window and the dry run (#138)."""
    main(["--no-env-file", "--dry-run", "--fixture", "delimiter_comma.csv"])

    match = re.search(r"(\d+) chars, ~(\d+) tokens", capsys.readouterr().out)
    assert match is not None
    chars, tokens = match.groups()
    assert int(tokens) == round(int(chars) / prompt_module.CHARS_PER_TOKEN)
