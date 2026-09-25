"""Unit tests for the csv_inspector command-line helpers."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from csv_inspector import (
    BackendConfigurationError,
    CredentialsNotConfiguredError,
    LLMBackend,
    Settings,
)
from csv_inspector._sampling import MAX_SAMPLE_BYTES
from csv_inspector.cli import (
    DEFAULT_CLI_TIMEOUT_SECONDS,
    HEAD_BYTES,
    TAIL_BYTES,
    add_log_level_argument,
    bounded_int,
    load_cli_settings,
    main,
    resolve_backend,
    timeout_budget,
)
from fakes import install_fake_ollama, ollama_reply


@pytest.mark.parametrize(
    ("converter", "raw", "expected"),
    [(HEAD_BYTES, "1", 1), (HEAD_BYTES, "16384", 16384), (TAIL_BYTES, "0", 0)],
)
def test_window_converters_accept_their_range(
    converter: Callable[[str], int], raw: str, expected: int
) -> None:
    """--bytes takes 1..MAX_SAMPLE_BYTES and --tail-bytes 0..MAX_SAMPLE_BYTES."""
    assert converter(raw) == expected


@pytest.mark.parametrize(
    ("converter", "raw", "message"),
    [
        (HEAD_BYTES, "0", ">= 1"),
        (TAIL_BYTES, "-1", ">= 0"),
        (HEAD_BYTES, "16385", "<= 16384"),
        (TAIL_BYTES, "x", "expected an integer"),
        (HEAD_BYTES, "1.5", "expected an integer"),
    ],
)
def test_window_converters_reject_everything_else(
    converter: Callable[[str], int], raw: str, message: str
) -> None:
    """Out-of-range and non-integer values are argparse errors naming the bound."""
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        converter(raw)


def test_bounded_int_without_a_maximum() -> None:
    """The upper bound is optional."""
    assert bounded_int(5)("1000000") == 1_000_000


def test_add_log_level_argument_defaults_to_info_and_restricts_choices() -> None:
    """The shared ``--log-level`` option defaults to INFO and rejects unknown levels."""
    parser = argparse.ArgumentParser()
    add_log_level_argument(parser)

    assert parser.parse_args([]).log_level == "INFO"
    assert parser.parse_args(["--log-level", "DEBUG"]).log_level == "DEBUG"
    with pytest.raises(SystemExit):
        parser.parse_args(["--log-level", "TRACE"])


# ---------------------------------------------------------------------
# Settings loading, backend resolution and a full in-process run
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"), [("0.5", 0.5), ("30", 30.0), ("0", None), ("0.0", None)]
)
def test_timeout_budget_accepts_seconds_and_zero_for_no_limit(
    raw: str, expected: float | None
) -> None:
    """--timeout takes any number > 0, and 0 turns the limit off."""
    assert timeout_budget(raw) == expected


@pytest.mark.parametrize("raw", ["-1", "nan", "soon"])
def test_timeout_budget_rejects_everything_else(raw: str) -> None:
    """Negatives, NaN and non-numbers are argparse errors."""
    with pytest.raises(argparse.ArgumentTypeError):
        timeout_budget(raw)


def test_cli_reads_dotenv_in_the_working_directory_by_default(tmp_path: Path) -> None:
    """As an application, the CLI opts in to ./.env when it exists, on any install."""
    (tmp_path / ".env").write_text("OLLAMA_MODEL=from-dotenv\n")

    assert load_cli_settings(None, no_env_file=False).ollama_model == "from-dotenv"


def test_cli_no_env_file_ignores_dotenv(tmp_path: Path) -> None:
    """--no-env-file keeps ./.env out of the picture."""
    (tmp_path / ".env").write_text("OLLAMA_MODEL=from-dotenv\n")

    assert load_cli_settings(None, no_env_file=True).ollama_model == Settings().ollama_model


def test_cli_explicit_env_file_must_exist(tmp_path: Path) -> None:
    """A mistyped --env-file is an error, not a silent fallback to defaults."""
    with pytest.raises(BackendConfigurationError, match="does not exist"):
        load_cli_settings(tmp_path / "missing.env", no_env_file=False)


def test_resolve_backend_prefers_the_flag_over_settings() -> None:
    """--backend wins over LLM_BACKEND from the settings."""
    assert resolve_backend("local", Settings(llm_backend=LLMBackend.API)) is LLMBackend.LOCAL


def test_resolve_backend_checks_cloud_credentials_up_front() -> None:
    """Selecting the api backend without credentials fails before any work."""
    pytest.importorskip("google.genai")

    with pytest.raises(CredentialsNotConfiguredError):
        resolve_backend(None, Settings(llm_backend=LLMBackend.API))


def test_cli_main_prints_the_result_and_passes_the_timeout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A full CLI run: JSON on stdout, and --timeout reaches the Ollama client."""
    target = tmp_path / "data.csv"
    target.write_bytes(b"a;b\n1;2\n")
    answer = json.dumps(
        {
            "encoding": "utf-8",
            "delimiter": ";",
            "header_row_index": 0,
            "columns": ["a"],
            "confidence": 0.9,
        }
    )
    fake = install_fake_ollama(monkeypatch, lambda **kwargs: ollama_reply(answer))

    main([str(target), "--timeout", "5", "--no-env-file", "--log-level", "ERROR"])

    assert json.loads(capsys.readouterr().out)["delimiter"] == ";"
    assert 0 < fake.client_kwargs[0]["timeout"] <= 5


@pytest.mark.parametrize("option", ["--bytes", "--tail-bytes"])
def test_cli_rejects_sample_budgets_above_the_maximum(tmp_path: Path, option: str) -> None:
    """Oversized sample windows are a usage error, not a traceback."""
    target = tmp_path / "data.csv"
    target.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main([str(target), option, str(MAX_SAMPLE_BYTES + 1), "--no-env-file"])

    assert excinfo.value.code == 2


_CLI_ANSWER = json.dumps(
    {
        "encoding": "utf-8",
        "delimiter": ";",
        "header_row_index": 0,
        "columns": ["a"],
        "confidence": 0.9,
    }
)


def test_cli_passes_the_fallback_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """--fallback-model is the model tried after --model fails."""
    target = tmp_path / "data.csv"
    target.write_bytes(b"a;b\n1;2\n")

    def chat(**kwargs: Any) -> Any:
        if kwargs["model"] == "first":
            raise RuntimeError("model unavailable")
        return ollama_reply(_CLI_ANSWER)

    fake = install_fake_ollama(monkeypatch, chat)

    main([str(target), "--model", "first", "--fallback-model", "second", "--no-env-file"])

    assert [request["model"] for request in fake.requests] == ["first", "second"]
    assert json.loads(capsys.readouterr().out)["delimiter"] == ";"


def test_cli_prints_the_same_json_as_before(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """model_dump_json keeps non-ASCII text and the indented layout of json.dumps (#103)."""
    target = tmp_path / "data.csv"
    target.write_text("Año;Descripción\n2024;Señal\n", encoding="utf-8")
    answer = json.loads(_CLI_ANSWER) | {
        "columns": [
            "Año",
            "Descripción",
        ]
    }
    install_fake_ollama(monkeypatch, lambda **kwargs: ollama_reply(json.dumps(answer)))

    main([str(target), "--no-env-file", "--log-level", "ERROR"])

    out = capsys.readouterr().out
    assert "Descripción" in out
    assert out == json.dumps(json.loads(out), indent=2, ensure_ascii=False) + "\n"


def test_cli_applies_a_default_timeout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Without --timeout the Ollama client is still bounded by the CLI default."""
    target = tmp_path / "data.csv"
    target.write_bytes(b"a;b\n1;2\n")
    fake = install_fake_ollama(monkeypatch, lambda **kwargs: ollama_reply(_CLI_ANSWER))

    main([str(target), "--no-env-file", "--log-level", "ERROR"])

    capsys.readouterr()
    assert 0 < fake.client_kwargs[0]["timeout"] <= DEFAULT_CLI_TIMEOUT_SECONDS


def test_cli_timeout_zero_disables_the_limit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """--timeout 0 leaves the Ollama client without a timeout."""
    target = tmp_path / "data.csv"
    target.write_bytes(b"a;b\n1;2\n")
    fake = install_fake_ollama(monkeypatch, lambda **kwargs: ollama_reply(_CLI_ANSWER))

    main([str(target), "--timeout", "0", "--no-env-file", "--log-level", "ERROR"])

    capsys.readouterr()
    assert fake.client_kwargs[0].get("timeout") is None


def test_cli_explains_how_to_change_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Running out of time exits 1 with a hint about --timeout."""
    target = tmp_path / "data.csv"
    target.write_bytes(b"a;b\n1;2\n")
    install_fake_ollama(monkeypatch, lambda **kwargs: ollama_reply(_CLI_ANSWER), delay_seconds=0.5)

    with pytest.raises(SystemExit) as excinfo:
        main([str(target), "--timeout", "0.05", "--no-env-file"])

    assert excinfo.value.code == 1
    assert "--timeout 0" in caplog.text
