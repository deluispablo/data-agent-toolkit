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

Fixtures flagged ``known_limitation`` in the manifest (e.g. a quoted field
with an embedded real newline, which byte-window sampling cannot reliably
parse) are reported separately and excluded from the aggregate score: they
are not expected to pass today.

Prerequisites:
    - Ollama running locally (``ollama serve``).
    - The target model pulled locally, e.g. ``ollama pull qwen2.5-coder:7b``.

Usage:
    python eval_samples.py
    python eval_samples.py --model qwen3:8b --bytes 8192 --category encoding
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from exceptions import CSVInspectorError
from inspector import DEFAULT_MODEL, DEFAULT_SAMPLE_BYTES, DEFAULT_TAIL_BYTES, FALLBACK_MODEL, inspect_csv
from models import CSVInspectionResult

logger = logging.getLogger(__name__)

SAMPLES_DIR = Path(__file__).parent / "samples"
MANIFEST_PATH = SAMPLES_DIR / "manifest.json"

# CSVInspectionResult fields the harness knows how to score against the manifest.
_COMPARABLE_FIELDS: tuple[str, ...] = (
    "encoding",
    "delimiter",
    "quotechar",
    "escapechar",
    "doublequote",
    "header_row_index",
    "footer_rows_to_skip",
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
    """

    filename: str
    category: str
    known_limitation: bool
    matched_fields: list[str] = field(default_factory=list)
    mismatched_fields: list[tuple[str, Any, Any]] = field(default_factory=list)
    skipped_fields: list[str] = field(default_factory=list)
    error: str | None = None

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


def _matches_encoding(expected: str, actual: str) -> bool:
    """Loosely match a free-text expected encoding description against the actual value.

    Manifest entries sometimes describe encoding as alternatives (e.g.
    ``"latin-1 or cp1252 (not utf-8)"``) since chardet's exact label can
    vary. This checks whether any alternative is a substring of the
    model's reported encoding, case- and separator-insensitively.

    Args:
        expected: The manifest's expected encoding description.
        actual: The model's reported encoding.

    Returns:
        True if any alternative in ``expected`` matches ``actual``.
    """
    expected_normalized = expected.lower().replace(" (not utf-8)", "")
    actual_normalized = actual.lower().replace("_", "-")
    alternatives = [alt.strip(" ()") for alt in expected_normalized.split(" or ")]
    return any(alt and alt in actual_normalized for alt in alternatives)


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
        actual_value = getattr(result, field_name)
        is_match = (
            _matches_encoding(str(expected_value), str(actual_value))
            if field_name == "encoding"
            else expected_value == actual_value
        )

        if is_match:
            matched.append(field_name)
        else:
            mismatched.append((field_name, expected_value, actual_value))

    return matched, mismatched, skipped


def evaluate_file(
    filename: str,
    entry: dict[str, Any],
    *,
    model: str,
    fallback_model: str,
    n_bytes: int,
    tail_bytes: int,
) -> FileEvaluation:
    """Run the real inspection pipeline against one fixture and score it.

    Args:
        filename: Name of the fixture under ``samples/``.
        entry: This fixture's manifest entry.
        model: Primary Ollama model to evaluate.
        fallback_model: Fallback Ollama model.
        n_bytes: Head sample size, in bytes.
        tail_bytes: Tail sample size, in bytes.

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
            model=model,
            fallback_model=fallback_model,
            n_bytes=n_bytes,
            tail_bytes=tail_bytes,
        )
    except CSVInspectorError as exc:
        evaluation.error = f"{type(exc).__name__}: {exc}"
        return evaluation

    matched, mismatched, skipped = _compare(entry["expected"], result)
    evaluation.matched_fields = matched
    evaluation.mismatched_fields = mismatched
    evaluation.skipped_fields = skipped
    return evaluation


def _format_file_line(evaluation: FileEvaluation) -> str:
    """Format one evaluation as a single human-readable report line."""
    tag = f"[{evaluation.category}]"
    if evaluation.error:
        return f"{tag} {evaluation.filename}: ERROR — {evaluation.error}"
    if evaluation.comparable_count == 0:
        return f"{tag} {evaluation.filename}: no comparable fields in manifest"

    pct = evaluation.score * 100  # type: ignore[operator]
    line = f"{tag} {evaluation.filename}: {len(evaluation.matched_fields)}/{evaluation.comparable_count} ({pct:.0f}%)"
    if evaluation.mismatched_fields:
        details = ", ".join(
            f"{name} (expected={expected!r}, got={actual!r})"
            for name, expected, actual in evaluation.mismatched_fields
        )
        line += f" — mismatched: {details}"
    return line


def print_report(evaluations: list[FileEvaluation], *, model: str, fallback_model: str) -> None:
    """Print the full evaluation report to stdout.

    Args:
        evaluations: Per-fixture evaluation results.
        model: The primary model that was evaluated (for the report header).
        fallback_model: The fallback model that was evaluated.
    """
    regular = [e for e in evaluations if not e.known_limitation and not e.error]
    known_limitations = [e for e in evaluations if e.known_limitation]
    errored = [e for e in evaluations if e.error and not e.known_limitation]

    print(f"=== csv_inspector eval report (model={model!r}, fallback={fallback_model!r}) ===\n")

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

    scoreable = [e for e in regular if e.score is not None]
    if scoreable:
        aggregate = sum(e.score for e in scoreable) / len(scoreable)  # type: ignore[misc]
        print(
            f"\n=== Aggregate score: {aggregate * 100:.1f}% across {len(scoreable)} scoreable file(s) "
            f"(excluding {len(known_limitations)} known-limitation and {len(errored)} errored file(s)) ==="
        )
    else:
        print("\n=== No scoreable files ===")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation harness."""
    parser = argparse.ArgumentParser(description="csv_inspector manual evaluation harness")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Primary Ollama model to evaluate.")
    parser.add_argument("--fallback-model", default=FALLBACK_MODEL, help="Fallback Ollama model to evaluate.")
    parser.add_argument("--bytes", type=int, default=DEFAULT_SAMPLE_BYTES, help="Head sample size, in bytes.")
    parser.add_argument("--tail-bytes", type=int, default=DEFAULT_TAIL_BYTES, help="Tail sample size, in bytes.")
    parser.add_argument("--category", default=None, help="Restrict the run to one manifest category.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for progress messages.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the evaluation harness against the full (or filtered) sample catalog."""
    args = _parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

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
                model=args.model,
                fallback_model=args.fallback_model,
                n_bytes=args.bytes,
                tail_bytes=args.tail_bytes,
            )
        )

    print_report(evaluations, model=args.model, fallback_model=args.fallback_model)


if __name__ == "__main__":
    main()
