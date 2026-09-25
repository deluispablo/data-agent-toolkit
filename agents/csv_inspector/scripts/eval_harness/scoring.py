"""Scoring: a result against the manifest's ground truth, and votes over repeats.

Standard library only (the result is duck-typed). Column names are scored
three ways: the exact list match (names as written in the file, the value
that counts in the score), plus two diagnostics: per-name recall (so a
paraphrase like ``Monto`` for ``Importe`` shows) and whether the column count
matches.
"""

from __future__ import annotations

import codecs
import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from csv_inspector import CSVInspectionResult

    from .evaluation import FileEvaluation

# CSVInspectionResult fields the harness knows how to score against the manifest.
RESULT_FIELDS: tuple[str, ...] = (
    "encoding",
    "delimiter",
    "quotechar",
    "escapechar",
    "doublequote",
    "has_header",
    "header_row_index",
    "footer_lines",
    "footer_rows_to_skip",
    "columns",
)
# The manifest's ``expected_error`` is scored like a field: matched when the
# inspection raised that exception class.
EXPECTED_ERROR = "expected_error"
COMPARABLE_FIELDS: tuple[str, ...] = (*RESULT_FIELDS, EXPECTED_ERROR)
# An expected ``null`` is normally "not scored" (``header_row_index`` of a
# header-less file); for these fields it is the answer to score.
_NULL_SCORED_FIELDS = frozenset({"escapechar"})
_ERROR_ANSWER = "<error>"
_MATCH_ANSWER = "<match>"


def _normalize_encoding(name: str) -> str:
    """Python's canonical codec name for a label (``latin-1`` and ``ISO-8859-1`` agree).

    An unknown label is returned lower-cased.
    """
    try:
        return codecs.lookup(name.strip()).name
    except LookupError:
        return name.strip().lower()


def matches_encoding(expected: str, actual: str) -> bool:
    """Whether a free-text expected encoding (``"latin-1 or cp1252 (not utf-8)"``) names ``actual``.

    Parenthesized remarks are ignored, and each ``or`` alternative is
    compared by canonical codec name, since chardet's exact label can vary.
    """
    without_remarks = re.sub(r"\([^)]*\)", "", expected)
    alternatives = [alt for alt in without_remarks.split(" or ") if alt.strip()]
    actual_codec = _normalize_encoding(actual)
    return any(_normalize_encoding(alt) == actual_codec for alt in alternatives)


def column_diagnostics(
    expected: list[str] | None, actual: list[str]
) -> tuple[float | None, bool | None]:
    """``(recall, count_match)`` of the column names beyond the exact list match.

    Recall counts each expected name at most once per occurrence, ignoring
    surrounding spaces, so it measures paraphrased or missing names rather
    than padding. Recall is ``None`` when no names are expected; both are
    ``None`` without ground truth.
    """
    if expected is None:
        return None, None
    count_match = len(expected) == len(actual)
    if not expected:
        return None, count_match
    found = Counter(name.strip() for name in expected) & Counter(name.strip() for name in actual)
    return sum(found.values()) / len(expected), count_match


def compare(
    expected: dict[str, Any], result: CSVInspectionResult
) -> tuple[list[str], list[tuple[str, Any, Any]], list[str]]:
    """Compare a result against one fixture's manifest ``expected`` mapping.

    A field missing from ``expected``, or ``null`` there, is skipped, except
    an ``escapechar`` of ``null``, which is scored (no escape character).
    Footer lines are compared stripped, so a stray carriage return is no miss.

    Returns:
        ``(matched, mismatched, skipped)``: field names, ``(field, expected,
        actual)`` triples, field names.
    """
    matched: list[str] = []
    mismatched: list[tuple[str, Any, Any]] = []
    skipped: list[str] = []
    for field_name in RESULT_FIELDS:
        if field_name not in expected or (
            expected[field_name] is None and field_name not in _NULL_SCORED_FIELDS
        ):
            skipped.append(field_name)
            continue
        expected_value = expected[field_name]
        actual_value = getattr(result, field_name)
        if field_name == "encoding":
            is_match = matches_encoding(str(expected_value), str(actual_value))
        elif field_name == "footer_lines":
            is_match = [line.strip() for line in expected_value] == [
                line.strip() for line in actual_value
            ]
        else:
            is_match = expected_value == actual_value
        if is_match:
            matched.append(field_name)
        else:
            mismatched.append((field_name, expected_value, actual_value))
    return matched, mismatched, skipped


def mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or ``None`` for no values."""
    return sum(values) / len(values) if values else None


def group_by_fixture(evaluations: list[FileEvaluation]) -> dict[str, list[FileEvaluation]]:
    """The evaluations of each fixture (its repeats), in run order."""
    groups: dict[str, list[FileEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        groups[evaluation.filename].append(evaluation)
    return dict(groups)


def _answer_key(evaluation: FileEvaluation, field_name: str) -> str | None:
    """One repeat's answer for a field, as a comparable key.

    A match is one key, a mismatch its actual value in canonical JSON, a
    failed inspection an error key, and a field without ground truth ``None``.
    """
    if evaluation.error:
        return _ERROR_ANSWER
    if field_name in evaluation.matched_fields:
        return _MATCH_ANSWER
    for name, _, actual in evaluation.mismatched_fields:
        if name == field_name:
            return json.dumps(actual, sort_keys=True, default=str)
    return None


def _field_votes(group: list[FileEvaluation]) -> dict[str, list[str]]:
    """Each compared field's answers over the repeats of one fixture.

    A field is included when at least one repeat compared it; a repeat that
    failed votes the error key for every such field.
    """
    compared = {
        name
        for evaluation in group
        for name in [*evaluation.matched_fields, *(m[0] for m in evaluation.mismatched_fields)]
    }
    votes: dict[str, list[str]] = {}
    for name in COMPARABLE_FIELDS:
        if name in compared:
            keys = [_answer_key(evaluation, name) for evaluation in group]
            votes[name] = [key for key in keys if key is not None]
    return votes


def _majority_matches(votes: list[str]) -> bool:
    """Whether strictly more than half of the repeats matched the ground truth."""
    return votes.count(_MATCH_ANSWER) * 2 > len(votes)


def _agreement(votes: list[str]) -> float:
    """Share of the repeats that gave the most common answer."""
    return Counter(votes).most_common(1)[0][1] / len(votes)


def _verdict(group: list[FileEvaluation], votes: dict[str, list[str]]) -> str:
    """Verdict by majority vote: ``pass``, ``fail: <fields>``, ``error`` or ``unscored``."""
    if all(evaluation.error for evaluation in group):
        return "error"
    if not votes:
        return "unscored"
    failing = [name for name, field_votes in votes.items() if not _majority_matches(field_votes)]
    return f"fail: {', '.join(failing)}" if failing else "pass"


@dataclass
class RepeatStats:
    """What running each fixture more than once shows.

    Known-limitation fixtures, and fixtures whose every repeat failed, are
    left out of the scores (as from the aggregate score) but keep a verdict.

    Attributes:
        majority_score: Mean, over fixtures, of the share of compared fields
            that matched by majority vote.
        field_majority: Per field, the share of fixtures where it matched by
            majority vote.
        field_agreement: Per field, the mean share of repeats that gave the
            most common answer (1.0 means the repeats never disagreed).
        disagreeing: Fixtures whose answers differ between repeats (a failed
            repeat next to a successful one counts).
        verdicts: Each fixture's verdict.
    """

    majority_score: float | None
    field_majority: dict[str, float]
    field_agreement: dict[str, float]
    disagreeing: list[str]
    verdicts: dict[str, str]


def repeat_stats(evaluations: list[FileEvaluation]) -> RepeatStats:
    """Majority-vote accuracy and agreement per field over the repeats of each fixture."""
    majority: dict[str, list[float]] = defaultdict(list)
    agreement: dict[str, list[float]] = defaultdict(list)
    fixture_scores: list[float] = []
    disagreeing: list[str] = []
    verdicts: dict[str, str] = {}
    for filename, group in group_by_fixture(evaluations).items():
        votes = _field_votes(group)
        verdicts[filename] = _verdict(group, votes)
        if any(len(set(field_votes)) > 1 for field_votes in votes.values()):
            disagreeing.append(filename)
        if group[0].known_limitation or verdicts[filename] == "error" or not votes:
            continue
        matches = [_majority_matches(field_votes) for field_votes in votes.values()]
        fixture_scores.append(sum(matches) / len(matches))
        for name, field_votes in votes.items():
            majority[name].append(float(_majority_matches(field_votes)))
            agreement[name].append(_agreement(field_votes))
    return RepeatStats(
        majority_score=mean(fixture_scores),
        # Keyed in COMPARABLE_FIELDS order, whatever order the fixtures came in.
        field_majority={
            name: sum(majority[name]) / len(majority[name])
            for name in COMPARABLE_FIELDS
            if name in majority
        },
        field_agreement={
            name: sum(agreement[name]) / len(agreement[name])
            for name in COMPARABLE_FIELDS
            if name in agreement
        },
        disagreeing=disagreeing,
        verdicts=verdicts,
    )
