"""Manual evaluation harness for csv_inspector against the samples/ catalog.

This script is deliberately **not** part of the ``pytest`` suite and is
**not** run in CI. It invokes a real local LLM (via Ollama) against every
fixture in ``samples/`` and compares the model's inferred dialect/schema
against the hand-authored ground truth in ``samples/manifest.json``.

Because LLM output is not fully deterministic — even at ``temperature=0``,
results can drift across model versions and quantizations — this produces a
per-file and aggregate *score*, not a pass/fail assertion. Use it to:

- Sanity-check a new model against a broad catalog of dialect edge cases.
- Compare two candidate models (e.g. ``qwen2.5-coder:7b`` vs ``qwen3:8b``)
  before switching ``DEFAULT_MODEL``.
- Notice regressions after a prompt change, by eye.

Column names are scored three ways: the exact list match (names as written
in the file, the value that counts in the score), plus two diagnostics shown
next to each file: per-name recall (how many expected names the model
reported, so a paraphrase like ``Monto`` for ``Importe`` shows) and whether
the column count matches.

Fixtures flagged ``known_limitation`` in the manifest (e.g. a quoted field
with an embedded real newline, which byte-window sampling cannot reliably
parse) are reported separately and excluded from the aggregate score: they
are not expected to pass today.

Prerequisites:
    - The package installed: ``uv sync`` at the repository root (or
      ``pip install -e ./agents/csv_inspector``).
    - Ollama running locally (``ollama serve``).
    - The target model pulled locally, e.g. ``ollama pull qwen2.5-coder:7b``.

Usage:
    python eval_samples.py
    python eval_samples.py --model qwen3:8b --bytes 8192 --category encoding
    python eval_samples.py --backend api --model gemini-3.6-flash
"""

from __future__ import annotations

import argparse
import codecs
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from csv_inspector import (
    DEFAULT_SAMPLE_BYTES,
    DEFAULT_TAIL_BYTES,
    CSVInspectionResult,
    CSVInspectorError,
    LLMBackend,
    Settings,
    inspect_csv,
)
from csv_inspector.cli import (
    DEFAULT_CLI_TIMEOUT_SECONDS,
    HEAD_BYTES,
    TAIL_BYTES,
    add_backend_argument,
    add_log_level_argument,
    add_settings_arguments,
    configure_cli,
    load_cli_settings,
    resolve_backend,
    timeout_budget,
)

logger = logging.getLogger(__name__)

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
MANIFEST_PATH = SAMPLES_DIR / "manifest.json"

# CSVInspectionResult fields the harness knows how to score against the manifest.
_COMPARABLE_FIELDS: tuple[str, ...] = (
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


@dataclass
class FileEvaluation:
    """The comparison outcome for a single fixture.

    Attributes:
        filename: Name of the evaluated fixture.
        category: The fixture's manifest category.
        known_limitation: Whether the manifest flags this as a documented
            limitation, excluded from the aggregate score.
        matched_fields: Names of fields where the model's output matched
            the manifest's ground truth.
        mismatched_fields: ``(field, expected, actual)`` triples for fields
            that did not match.
        skipped_fields: Fields with no ground truth to compare against
            (e.g. ``header_row_index`` for a file with no real header).
        error: The exception message, if the inspection pipeline itself
            failed (file unreadable, every model failed, etc.).
        columns_recall: Diagnostic, not part of the score: the fraction of
            expected column names (surrounding spaces ignored) the model
            reported, or ``None`` when there is nothing to compare.
        columns_count_match: Diagnostic, not part of the score: whether the
            model reported as many columns as expected, or ``None`` when
            there is nothing to compare.
    """

    filename: str
    category: str
    known_limitation: bool
    matched_fields: list[str] = field(default_factory=list)
    mismatched_fields: list[tuple[str, Any, Any]] = field(default_factory=list)
    skipped_fields: list[str] = field(default_factory=list)
    error: str | None = None
    columns_recall: float | None = None
    columns_count_match: bool | None = None

    @property
    def comparable_count(self) -> int:
        """Number of fields that had ground truth to compare against."""
        return len(self.matched_fields) + len(self.mismatched_fields)

    @property
    def score(self) -> float | None:
        """Fraction of comparable fields that matched, or ``None`` if unscoreable."""
        if self.error or self.comparable_count == 0:
            return None
        return len(self.matched_fields) / self.comparable_count


def _normalize_encoding(name: str) -> str:
    """Map an encoding label to Python's canonical codec name.

    Different tools spell the same codec differently (``latin-1`` vs
    ``ISO-8859-1``, ``windows-1252`` vs ``cp1252``, ``UTF-16LE`` vs
    ``utf_16_le``); resolving through :func:`codecs.lookup` makes those
    compare equal.

    Args:
        name: An encoding label.

    Returns:
        The canonical codec name, or the lower-cased label if Python does
        not recognize it.
    """
    try:
        return codecs.lookup(name.strip()).name
    except LookupError:
        return name.strip().lower()


def _matches_encoding(expected: str, actual: str) -> bool:
    """Match a free-text expected encoding description against the actual value.

    Manifest entries sometimes describe encoding as alternatives (e.g.
    ``"latin-1 or cp1252 (not utf-8)"``) since chardet's exact label can
    vary. Parenthesized remarks are ignored, and each alternative is
    compared to ``actual`` by canonical codec name.

    Args:
        expected: The manifest's expected encoding description.
        actual: The model's reported encoding.

    Returns:
        True if any alternative in ``expected`` names the same codec as
        ``actual``.
    """
    without_remarks = re.sub(r"\([^)]*\)", "", expected)
    alternatives = [alt for alt in without_remarks.split(" or ") if alt.strip()]
    actual_codec = _normalize_encoding(actual)
    return any(_normalize_encoding(alt) == actual_codec for alt in alternatives)


def _normalize_lines(lines: list[str]) -> list[str]:
    """Strip surrounding whitespace from each line, so a stray carriage return is not a miss."""
    return [line.strip() for line in lines]


def _column_diagnostics(
    expected: list[str] | None, actual: list[str]
) -> tuple[float | None, bool | None]:
    """Score the column names beyond the exact list match.

    Recall counts each expected name at most once per occurrence, ignoring
    surrounding spaces, so it measures paraphrased or missing names rather
    than padding (the exact match already covers that).

    Args:
        expected: The manifest's expected column names, if any.
        actual: The column names the model reported.

    Returns:
        A ``(recall, count_match)`` tuple. Recall is ``None`` when no
        names are expected; both are ``None`` without ground truth.
    """
    if expected is None:
        return None, None
    count_match = len(expected) == len(actual)
    if not expected:
        return None, count_match
    found = Counter(name.strip() for name in expected) & Counter(name.strip() for name in actual)
    return sum(found.values()) / len(expected), count_match


def _compare(
    expected: dict[str, Any], result: CSVInspectionResult
) -> tuple[list[str], list[tuple[str, Any, Any]], list[str]]:
    """Compare a model result against the manifest's expected fields.

    Args:
        expected: The manifest's ``expected`` mapping for one fixture.
        result: The validated inspection result returned by the model.

    Returns:
        A ``(matched, mismatched, skipped)`` tuple: matched and skipped are
        field-name lists, mismatched is a list of
        ``(field, expected_value, actual_value)`` triples.
    """
    matched: list[str] = []
    mismatched: list[tuple[str, Any, Any]] = []
    skipped: list[str] = []

    for field_name in _COMPARABLE_FIELDS:
        if field_name not in expected or expected[field_name] is None:
            skipped.append(field_name)
            continue

        expected_value = expected[field_name]
        if field_name == "columns":
            actual_value = [column.name for column in result.columns]
        else:
            actual_value = getattr(result, field_name)
        if field_name == "encoding":
            is_match = _matches_encoding(str(expected_value), str(actual_value))
        elif field_name == "footer_lines":
            is_match = _normalize_lines(expected_value) == _normalize_lines(actual_value)
        else:
            is_match = expected_value == actual_value

        if is_match:
            matched.append(field_name)
        else:
            mismatched.append((field_name, expected_value, actual_value))

    return matched, mismatched, skipped


def evaluate_file(
    filename: str,
    entry: dict[str, Any],
    *,
    backend: LLMBackend,
    settings: Settings,
    model: str,
    fallback_model: str,
    n_bytes: int,
    tail_bytes: int,
    timeout_seconds: float | None = None,
) -> FileEvaluation:
    """Run the real inspection pipeline against one fixture and score it.

    Args:
        filename: Name of the fixture under ``samples/``.
        entry: This fixture's manifest entry.
        backend: The LLM backend to evaluate.
        settings: Settings for the backend (credentials, models).
        model: Primary model to evaluate.
        fallback_model: Fallback model.
        n_bytes: Head sample size, in bytes.
        tail_bytes: Tail sample size, in bytes.
        timeout_seconds: Time budget for this fixture's model calls, or
            ``None`` for no limit. A fixture that runs out is reported as
            errored, like any other pipeline error.

    Returns:
        The resulting :class:`FileEvaluation`.
    """
    evaluation = FileEvaluation(
        filename=filename,
        category=entry["category"],
        known_limitation=entry["known_limitation"],
    )
    try:
        result = inspect_csv(
            SAMPLES_DIR / filename,
            backend=backend,
            settings=settings,
            model=model,
            fallback_model=fallback_model,
            n_bytes=n_bytes,
            tail_bytes=tail_bytes,
            timeout_seconds=timeout_seconds,
        )
    except CSVInspectorError as exc:
        evaluation.error = f"{type(exc).__name__}: {exc}"
        return evaluation

    matched, mismatched, skipped = _compare(entry["expected"], result)
    evaluation.matched_fields = matched
    evaluation.mismatched_fields = mismatched
    evaluation.skipped_fields = skipped
    evaluation.columns_recall, evaluation.columns_count_match = _column_diagnostics(
        entry["expected"].get("columns"), [column.name for column in result.columns]
    )
    return evaluation


def _format_file_line(evaluation: FileEvaluation) -> str:
    """Format one evaluation as a single human-readable report line."""
    prefix = f"[{evaluation.category}] {evaluation.filename}:"
    if evaluation.error:
        return f"{prefix} ERROR — {evaluation.error}"
    score = evaluation.score
    if score is None:
        return f"{prefix} no comparable fields in manifest"

    matched = len(evaluation.matched_fields)
    line = f"{prefix} {matched}/{evaluation.comparable_count} ({score:.0%})"
    if evaluation.columns_recall is not None:
        line += f" [columns recall {evaluation.columns_recall:.0%}"
        line += ", count match]" if evaluation.columns_count_match else ", count mismatch]"
    if evaluation.mismatched_fields:
        details = ", ".join(
            f"{name} (expected={expected!r}, got={actual!r})"
            for name, expected, actual in evaluation.mismatched_fields
        )
        line += f" — mismatched: {details}"
    return line


def print_report(
    evaluations: list[FileEvaluation],
    *,
    backend: LLMBackend,
    model: str,
    fallback_model: str,
) -> None:
    """Print the full evaluation report to stdout.

    Args:
        evaluations: Per-fixture evaluation results.
        backend: The backend that was evaluated (for the report header).
        model: The primary model that was evaluated (for the report header).
        fallback_model: The fallback model that was evaluated.
    """
    regular = [e for e in evaluations if not e.known_limitation and not e.error]
    known_limitations = [e for e in evaluations if e.known_limitation]
    errored = [e for e in evaluations if e.error and not e.known_limitation]

    print(
        f"=== csv_inspector eval report (backend={backend.value!r}, model={model!r}, "
        f"fallback={fallback_model!r}) ===\n"
    )

    for evaluation in regular:
        print(_format_file_line(evaluation))

    if errored:
        print("\n--- Pipeline errors (excluded from aggregate) ---")
        for evaluation in errored:
            print(_format_file_line(evaluation))

    if known_limitations:
        print("\n--- Known limitations (excluded from aggregate) ---")
        for evaluation in known_limitations:
            print(_format_file_line(evaluation))

    scores = [e.score for e in regular if e.score is not None]
    if scores:
        aggregate = sum(scores) / len(scores)
        print(
            f"\n=== Aggregate score: {aggregate:.1%} across {len(scores)} scoreable file(s) "
            f"(excluding {len(known_limitations)} known-limitation "
            f"and {len(errored)} errored file(s)) ==="
        )
    else:
        print("\n=== No scoreable files ===")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation harness."""
    parser = argparse.ArgumentParser(description="csv_inspector manual evaluation harness")
    add_backend_argument(parser)
    parser.add_argument(
        "--model", default=None, help="Primary model (default: the backend's configured model)."
    )
    parser.add_argument(
        "--fallback-model",
        default=None,
        help="Fallback model (default: the backend's configured fallback).",
    )
    parser.add_argument(
        "--bytes",
        type=HEAD_BYTES,
        default=DEFAULT_SAMPLE_BYTES,
        help="Head sample size, in bytes.",
    )
    parser.add_argument(
        "--tail-bytes",
        type=TAIL_BYTES,
        default=DEFAULT_TAIL_BYTES,
        help="Tail sample size, in bytes (0 disables).",
    )
    parser.add_argument(
        "--timeout",
        type=timeout_budget,
        default=DEFAULT_CLI_TIMEOUT_SECONDS,
        help=(
            "Time budget for each fixture's model calls, in seconds "
            f"(default: {DEFAULT_CLI_TIMEOUT_SECONDS:g}; 0 disables the limit)."
        ),
    )
    parser.add_argument(
        "--category", default=None, help="Restrict the run to one manifest category."
    )
    add_settings_arguments(parser)
    add_log_level_argument(parser)
    return parser.parse_args()


def main() -> None:
    """Run the evaluation harness against the full (or filtered) sample catalog."""
    args = _parse_args()
    configure_cli(args.log_level)

    try:
        settings = load_cli_settings(args.env_file, no_env_file=args.no_env_file)
        backend = resolve_backend(args.backend, settings)
        model = args.model or settings.model_for(backend)
        fallback_model = args.fallback_model or settings.fallback_model_for(backend)
    except CSVInspectorError as exc:
        # Fail once, up front, instead of reporting the same error per fixture.
        logger.error("Cannot run the evaluation: %s", exc)
        sys.exit(1)

    manifest: dict[str, dict[str, Any]] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    evaluations: list[FileEvaluation] = []
    for filename, entry in sorted(manifest.items()):
        if args.category and entry["category"] != args.category:
            continue
        logger.info("Evaluating '%s' (category=%s)...", filename, entry["category"])
        evaluations.append(
            evaluate_file(
                filename,
                entry,
                backend=backend,
                settings=settings,
                model=model,
                fallback_model=fallback_model,
                n_bytes=args.bytes,
                tail_bytes=args.tail_bytes,
                timeout_seconds=args.timeout,
            )
        )

    print_report(evaluations, backend=backend, model=model, fallback_model=fallback_model)


if __name__ == "__main__":
    main()
