"""One inspection of one fixture, scored: :class:`FileEvaluation` and its run-file line.

``--keep-raw`` records the answers through :mod:`.recording`, the harness's
one seam into the library's built-in invoker.
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

from csv_inspector import CSVInspectorError, LLMBackend, Settings, _invokers, inspect_csv
from csv_inspector._invokers import _RETRY_WARNING
from csv_inspector._models import Usage

from .guards import SAMPLES_DIR, calls_made
from .recording import recording_invoker
from .replay import RawAnswers, replaying_invoker
from .scoring import EXPECTED_ERROR, RESULT_FIELDS, column_diagnostics, compare


@dataclass
class FileEvaluation:
    """The outcome of one (fixture, repeat): the one representation of a run-file line.

    Attributes:
        filename: Name of the evaluated fixture.
        category: The fixture's manifest category.
        known_limitation: Whether the manifest flags this as a documented
            limitation, excluded from the aggregate score.
        matched_fields: Fields where the result matched the manifest.
        mismatched_fields: ``(field, expected, actual)`` of the fields that did not.
        skipped_fields: Fields with no ground truth to compare against
            (e.g. ``header_row_index`` for a file with no real header).
        error: ``"Class: message"`` when the inspection itself failed.
        columns_recall: Diagnostic, not scored: the fraction of expected
            column names reported (see :func:`.scoring.column_diagnostics`).
        columns_count_match: Diagnostic, not scored: whether the column
            count matched.
        repeat: Which repeat of the fixture this is, from 1.
        usage: What the model phase cost (``result.usage``), or ``None``
            when the inspection failed.
        latency_seconds: Wall time of the whole inspection as the harness
            measured it (sampling and grounding included, failures too).
        calls: Model calls charged against ``--max-calls``; see
            :func:`.guards.calls_made`.
        attempt_errors: For a failed inspection, ``"Class: message"`` of
            each model attempt that failed, keyed by model.
        raw_responses: With ``--keep-raw``, ``{"model", "text"}`` for every
            model call that answered, in call order; else ``None``.
        retries: Transient-error retries: from ``Usage`` when the inspection
            succeeded, else counted from the library's retry warnings.
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
    repeat: int = 1
    usage: Usage | None = None
    latency_seconds: float | None = None
    calls: int = 0
    attempt_errors: dict[str, str] = field(default_factory=dict)
    raw_responses: RawAnswers | None = None
    retries: int = 0

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

    def to_record(self) -> dict[str, Any]:
        """The run-file line: verdicts, cost and raw answers of this (fixture, repeat)."""
        usage = self.usage
        record: dict[str, Any] = {
            "fixture": self.filename,
            "category": self.category,
            "repeat": self.repeat,
            "known_limitation": self.known_limitation,
            "model_used": usage.model if usage is not None else None,
            "matched": self.matched_fields,
            "mismatched": [
                {"field": name, "expected": expected, "actual": actual}
                for name, expected, actual in self.mismatched_fields
            ],
            "skipped": self.skipped_fields,
            "score": self.score,
            "columns_recall": self.columns_recall,
            "columns_count_match": self.columns_count_match,
            "usage": usage.model_dump() if usage is not None else None,
            "latency_seconds": self.latency_seconds,
            "calls": self.calls,
            "error": self.error,
            "attempt_errors": self.attempt_errors,
            "retries": self.retries,
        }
        if self.raw_responses is not None:
            record["raw_response"] = self.raw_responses
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> FileEvaluation:
        """Rebuild an evaluation from its :meth:`to_record` line."""
        usage = record.get("usage")
        return cls(
            filename=record["fixture"],
            category=record["category"],
            known_limitation=record["known_limitation"],
            matched_fields=list(record["matched"]),
            mismatched_fields=[
                (m["field"], m["expected"], m["actual"]) for m in record["mismatched"]
            ],
            skipped_fields=list(record["skipped"]),
            error=record["error"],
            columns_recall=record["columns_recall"],
            columns_count_match=record["columns_count_match"],
            repeat=record["repeat"],
            usage=Usage.model_validate(usage) if usage is not None else None,
            latency_seconds=record["latency_seconds"],
            calls=record["calls"],
            attempt_errors=dict(record["attempt_errors"]),
            raw_responses=record.get("raw_response"),
            retries=record.get("retries", 0),
        )


class _RetryCounter(logging.Handler):
    """Counts the library's retry warnings (``_RETRY_WARNING``) while attached.

    A failed inspection has no ``Usage``, so this is the only place its
    cloud retries show. It needs WARNING enabled on the library's logger,
    which is the default. Use it as a context manager.
    """

    def __init__(self) -> None:
        """Start at zero retries."""
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        """Count one retry warning; ignore every other record."""
        if record.msg == _RETRY_WARNING:
            self.count += 1

    def __enter__(self) -> _RetryCounter:
        logging.getLogger(_invokers.__name__).addHandler(self)
        return self

    def __exit__(self, *exc_info: object) -> None:
        logging.getLogger(_invokers.__name__).removeHandler(self)


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
    repeat: int = 1,
    keep_raw: bool = False,
    replayed: RawAnswers | None = None,
) -> FileEvaluation:
    """Run the real inspection pipeline against one fixture and score it.

    ``timeout_seconds`` bounds the fixture's model calls (``None``: no
    limit); running out is reported as an error, like any pipeline error.
    ``keep_raw`` records the raw text of every answer. ``replayed`` answers
    with these recorded attempts instead of calling a model, and records the
    ones used. When the manifest names an ``expected_error`` and the
    inspection raises it, the fixture matches on that pseudo-field and is
    charged no call (those errors, such as ``EmptySampleError``, come before
    any model call).
    """
    evaluation = FileEvaluation(
        filename=filename,
        category=entry["category"],
        known_limitation=entry["known_limitation"],
        repeat=repeat,
    )
    expected_error: str | None = entry.get(EXPECTED_ERROR)
    raw: RawAnswers = []
    invoker = replaying_invoker(replayed, raw) if replayed is not None else None
    keep_raw = keep_raw or invoker is not None
    counter = _RetryCounter()
    started = time.monotonic()
    try:
        with (
            counter,
            recording_invoker(
                lambda model, _, answer: raw.append({"model": model, "text": answer.text})
            )
            if keep_raw and invoker is None
            else nullcontext(),
        ):
            result = inspect_csv(
                SAMPLES_DIR / filename,
                backend=backend,
                settings=settings,
                model=model,
                fallback_model=fallback_model,
                n_bytes=n_bytes,
                tail_bytes=tail_bytes,
                timeout_seconds=timeout_seconds,
                model_invoker=invoker,
            )
    except CSVInspectorError as exc:
        if type(exc).__name__ == expected_error:
            evaluation.matched_fields = [EXPECTED_ERROR]
            evaluation.skipped_fields = list(RESULT_FIELDS)
            return evaluation
        evaluation.error = f"{type(exc).__name__}: {exc}"
        attempts: dict[str, Exception] = getattr(exc, "attempts", None) or {}
        evaluation.attempt_errors = {
            name: f"{type(error).__name__}: {error}" for name, error in attempts.items()
        }
        result = None
    finally:
        evaluation.latency_seconds = time.monotonic() - started
        if keep_raw:
            evaluation.raw_responses = list(raw)

    evaluation.usage = result.usage if result is not None else None
    evaluation.retries = evaluation.usage.retries if evaluation.usage else counter.count
    evaluation.calls = calls_made(evaluation.usage, backend, model, fallback_model)
    if result is None:
        return evaluation

    matched, mismatched, skipped = compare(entry["expected"], result)
    if expected_error is not None:
        mismatched.append((EXPECTED_ERROR, expected_error, None))
    evaluation.matched_fields = matched
    evaluation.mismatched_fields = mismatched
    evaluation.skipped_fields = skipped
    evaluation.columns_recall, evaluation.columns_count_match = column_diagnostics(
        entry["expected"].get("columns"), result.columns
    )
    return evaluation
