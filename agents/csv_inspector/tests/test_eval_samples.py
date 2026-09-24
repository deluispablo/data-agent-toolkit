"""Unit tests for the LLM-free scoring logic of the manual evaluation harness."""

from __future__ import annotations

import pytest

from eval_samples import FileEvaluation, _matches_encoding


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
