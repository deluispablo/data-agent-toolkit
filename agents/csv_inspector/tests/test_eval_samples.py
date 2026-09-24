"""Unit tests for the LLM-free scoring logic of the manual evaluation harness."""

from __future__ import annotations

import sys
from typing import Any

import pytest

import eval_samples
from csv_inspector import (
    ColumnSchema,
    CSVInspectionResult,
    InspectionTimeoutError,
    LLMBackend,
    Settings,
)
from csv_inspector.cli import DEFAULT_CLI_TIMEOUT_SECONDS
from eval_samples import (
    FileEvaluation,
    _column_diagnostics,
    _format_file_line,
    _matches_encoding,
    _parse_args,
    evaluate_file,
)


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
    """Build a minimal inspection result reporting ``names`` as its columns."""
    return CSVInspectionResult(
        encoding="utf-8",
        delimiter=",",
        header_row_index=0,
        columns=[ColumnSchema(name=name, inferred_type="string") for name in names],
        confidence=0.9,
    )


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
