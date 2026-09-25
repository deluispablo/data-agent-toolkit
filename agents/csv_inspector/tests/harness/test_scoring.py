"""Tests of ``eval_harness.scoring``: a result against the manifest, votes over repeats."""

from __future__ import annotations

import pytest

from eval_harness.report import (
    format_file_line,
)
from eval_harness.scoring import column_diagnostics, matches_encoding, repeat_stats
from harness_support import _evaluate_with, _evaluation, _result_with_columns


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
    assert matches_encoding(expected, actual)


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
    assert not matches_encoding(expected, actual)


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
    assert "[columns recall 67%, count match]" in format_file_line(evaluation)


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
    assert "count mismatch" in format_file_line(evaluation)


def test_column_diagnostics_are_none_without_expected_columns() -> None:
    """No ground truth means no diagnostics, and an empty expectation has no recall."""
    assert column_diagnostics(None, ["a"]) == (None, None)
    assert column_diagnostics([], []) == (None, True)


def test_column_recall_counts_duplicated_names_once_per_occurrence() -> None:
    """A name expected twice is only fully recalled when reported twice."""
    assert column_diagnostics(["Fecha", "Fecha"], ["Fecha", "Otra"]) == (0.5, True)


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


def test_a_null_escapechar_is_scored_while_other_nulls_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``escapechar: null`` is an answer to check (issue #158); a null header row is not."""
    result = _result_with_columns("a").model_copy(update={"escapechar": "\\"})

    evaluation = _evaluate_with(
        monkeypatch, result, {"escapechar": None, "header_row_index": None, "delimiter": ","}
    )

    assert evaluation.mismatched_fields == [("escapechar", None, "\\")]
    assert evaluation.matched_fields == ["delimiter"]
    assert "header_row_index" in evaluation.skipped_fields
