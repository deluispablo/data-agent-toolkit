"""Unit tests for the csv_inspector command-line helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from fakes import install_fake_ollama, ollama_reply

from csv_inspector import (
    BackendConfigurationError,
    CredentialsNotConfiguredError,
    LLMBackend,
    Settings,
)
from csv_inspector._sampling import MAX_SAMPLE_BYTES
from csv_inspector.cli import (
    add_log_level_argument,
    load_cli_settings,
    main,
    non_negative_int,
    positive_float,
    positive_int,
    resolve_backend,
)


@pytest.mark.parametrize(("raw", "expected"), [("1", 1), ("4096", 4096)])
def test_positive_int_accepts_integers_from_one(raw: str, expected: int) -> None:
    """Integers >= 1 are parsed as-is."""
    assert positive_int(raw) == expected


@pytest.mark.parametrize("raw", ["0", "-5", "abc", "1.5"])
def test_positive_int_rejects_everything_else(raw: str) -> None:
    """Zero, negatives and non-integers are rejected with an argparse error."""
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(raw)


@pytest.mark.parametrize(("raw", "expected"), [("0", 0), ("8192", 8192)])
def test_non_negative_int_accepts_zero_and_up(raw: str, expected: int) -> None:
    """Zero is a valid value (e.g. to disable tail sampling)."""
    assert non_negative_int(raw) == expected


@pytest.mark.parametrize("raw", ["-1", "x"])
def test_non_negative_int_rejects_negatives_and_non_integers(raw: str) -> None:
    """Negatives and non-integers are rejected with an argparse error."""
    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_int(raw)


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


@pytest.mark.parametrize(("raw", "expected"), [("0.5", 0.5), ("30", 30.0)])
def test_positive_float_accepts_positive_numbers(raw: str, expected: float) -> None:
    """Budgets like --timeout accept any number > 0."""
    assert positive_float(raw) == expected


@pytest.mark.parametrize("raw", ["0", "-1", "nan", "soon"])
def test_positive_float_rejects_everything_else(raw: str) -> None:
    """Zero, negatives, NaN and non-numbers are argparse errors."""
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float(raw)


def test_cli_reads_dotenv_in_the_working_directory_by_default(tmp_path: Path) -> None:
    """As an application, the CLI opts in to ./.env when it exists."""
    pytest.importorskip("pydantic_settings")
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
            "columns": [{"name": "a", "inferred_type": "string"}],
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
