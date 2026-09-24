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

Machine-readable runs: ``--out`` writes one JSON line per (fixture, repeat)
with the verdicts, the ``Usage`` of the inspection and the harness-measured
latency, plus a final ``{"summary": {...}}`` line with the aggregates;
``scripts/compare_runs.py`` turns two or more such files into a Markdown
table. ``--repeat N`` runs each fixture N times and reports majority-vote
accuracy and agreement per field, since answers drift even at
``temperature=0``. Quota guards (``--rpm``, ``--max-calls``,
``--max-fixtures``, ``--fixture``, ``--dry-run``) keep a cloud run from
burning a day's free-tier quota by accident. See ``docs/evaluation.md``.

Private imports: this script lives in the repository, not in a host, so it
may import private names. ``--dry-run`` builds prompts with
``csv_inspector._sampling.sample_source`` and ``csv_inspector._prompt``, and
``--keep-raw`` wraps ``csv_inspector._inspect.builtin_invoker`` for the
duration of each inspection. The public ``model_invoker`` returns plain
text, so going through it would drop the token counts from ``Usage`` and
change which errors count as a failed attempt; wrapping the built-in
invoker factory keeps both exactly as in a normal run.

Prerequisites:
    - The package installed: ``uv sync`` at the repository root (or
      ``pip install -e ./agents/csv_inspector``).
    - Ollama running locally (``ollama serve``).
    - The target model pulled locally, e.g. ``ollama pull qwen2.5-coder:7b``.

Usage:
    python eval_samples.py
    python eval_samples.py --model qwen3:8b --bytes 8192 --category encoding
    python eval_samples.py --repeat 3 --out runs/
    python eval_samples.py --model a --model b --out "runs/{model}.jsonl"
    python eval_samples.py --backend api --model gemini-3.6-flash --max-calls 20 --rpm 5
"""

from __future__ import annotations

import argparse
import codecs
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from csv_inspector import _inspect, _prompt
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
    bounded_int,
    add_log_level_argument,
    add_settings_arguments,
    configure_cli,
    load_cli_settings,
    resolve_backend,
    timeout_budget,
)
from csv_inspector._invokers import InvokerResponse
from csv_inspector._models import Usage
from csv_inspector._prompt import SYSTEM_PROMPT, build_prompt
from csv_inspector._sampling import sample_source

logger = logging.getLogger(__name__)

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
MANIFEST_PATH = SAMPLES_DIR / "manifest.json"

HARNESS_VERSION = "1"
"""Version of the JSONL run format; bump it when a line or summary field changes meaning."""

CHARS_PER_TOKEN = 2
"""``--dry-run`` token estimate: numeric CSV text tokenizes poorly (see ``_ollama_num_ctx``)."""

_TRANSIENT_STATUS = re.compile(r"\b(429|503)\b")
_ERROR_ANSWER = "<error>"
_MATCH_ANSWER = "<match>"

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
        repeat: Which repeat of the fixture this is, from 1.
        usage: What the model phase cost (``result.usage``), or ``None``
            when the inspection failed.
        latency_seconds: Wall time of the whole inspection as the harness
            measured it (sampling and grounding included, failures too).
        calls: Model calls charged against ``--max-calls``; see
            :func:`_calls_made`.
        attempt_errors: For a failed inspection, ``"Class: message"`` of
            each model attempt that failed, keyed by model.
        raw_responses: With ``--keep-raw``, ``{"model", "text"}`` for every
            model call that answered, in call order; else ``None``.
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
    raw_responses: list[dict[str, str]] | None = None

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


_SyncCall = Callable[[str, str, float | None], InvokerResponse]
# The factory inspect_csv looks up in its module when no model_invoker is given.
_SEAM = "builtin_invoker"


@contextmanager
def _recording_builtin_invoker(raw: list[dict[str, str]]) -> Iterator[None]:
    """Record the raw text of every built-in model call made inside the block.

    Replaces ``csv_inspector._inspect.builtin_invoker`` (the factory
    ``inspect_csv`` calls when no ``model_invoker`` is given) with one whose
    invokers append ``{"model", "text"}`` to ``raw`` and return the
    :class:`InvokerResponse` unchanged, so ``Usage`` keeps its token counts.
    The harness is sequential, so swapping a module attribute is safe here.

    Args:
        raw: The list the answers are appended to.
    """
    original: Callable[[LLMBackend, Settings], _SyncCall] = getattr(_inspect, _SEAM)

    def factory(backend: LLMBackend, settings: Settings) -> _SyncCall:
        call = original(backend, settings)

        def recording(prompt: str, model: str, timeout: float | None) -> InvokerResponse:
            response = call(prompt, model, timeout)
            raw.append({"model": model, "text": response.text})
            return response

        return recording

    setattr(_inspect, _SEAM, factory)
    try:
        yield
    finally:
        setattr(_inspect, _SEAM, original)


@contextmanager
def _no_recording() -> Iterator[None]:
    """A do-nothing stand-in for :func:`_recording_builtin_invoker`."""
    yield


def _worst_case_calls(backend: LLMBackend, model: str, fallback_model: str) -> int:
    """The most model requests one inspection can make.

    One per candidate model (the fallback is skipped when it equals the
    primary), doubled on the cloud backend, which retries a 429/503 once.
    """
    candidates = 1 if model == fallback_model else 2
    return candidates * (2 if backend is LLMBackend.API else 1)


def _calls_made(usage: Usage | None, backend: LLMBackend, model: str, fallback_model: str) -> int:
    """Model requests one inspection is charged against ``--max-calls``.

    A success reports its attempts and transient retries in ``Usage``. A
    failure reports no usage (and a failed cloud attempt hides whether it
    was retried), so it is charged the worst case, never less than it
    could have cost.
    """
    if usage is not None:
        return usage.attempts + usage.retries
    return _worst_case_calls(backend, model, fallback_model)


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
        repeat: Which repeat of the fixture this is, from 1.
        keep_raw: Record the raw text of every model answer.

    Returns:
        The resulting :class:`FileEvaluation`.
    """
    evaluation = FileEvaluation(
        filename=filename,
        category=entry["category"],
        known_limitation=entry["known_limitation"],
        repeat=repeat,
    )
    raw: list[dict[str, str]] = []
    started = time.monotonic()
    try:
        with _recording_builtin_invoker(raw) if keep_raw else _no_recording():
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
    evaluation.calls = _calls_made(evaluation.usage, backend, model, fallback_model)
    if result is None:
        return evaluation

    matched, mismatched, skipped = _compare(entry["expected"], result)
    evaluation.matched_fields = matched
    evaluation.mismatched_fields = mismatched
    evaluation.skipped_fields = skipped
    evaluation.columns_recall, evaluation.columns_count_match = _column_diagnostics(
        entry["expected"].get("columns"), [column.name for column in result.columns]
    )
    return evaluation


def _format_file_line(evaluation: FileEvaluation, *, show_repeat: bool = False) -> str:
    """Format one evaluation as a single human-readable report line."""
    repeat = f" #{evaluation.repeat}" if show_repeat else ""
    prefix = f"[{evaluation.category}] {evaluation.filename}{repeat}:"
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


# ---------------------------------------------------------------------
# Aggregation: per-line scores, repeats, summary
# ---------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or ``None`` for no values."""
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    """Linearly interpolated percentile (``fraction`` in [0, 1]), or ``None`` for no values."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _answer_key(evaluation: FileEvaluation, field_name: str) -> str | None:
    """One repeat's answer for a field, as a comparable key.

    A match is one key (the expected value), a mismatch is its actual value
    in canonical JSON, a failed inspection is an error key, and a field
    without ground truth has no answer (``None``).
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
    for name in _COMPARABLE_FIELDS:
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
    """One fixture's verdict: ``pass``, ``fail: <fields>``, ``error`` or ``unscored``.

    Fields are judged by majority vote over the repeats.
    """
    if all(evaluation.error for evaluation in group):
        return "error"
    if not votes:
        return "unscored"
    failing = [name for name, field_votes in votes.items() if not _majority_matches(field_votes)]
    return f"fail: {', '.join(failing)}" if failing else "pass"


def _group_by_fixture(evaluations: list[FileEvaluation]) -> dict[str, list[FileEvaluation]]:
    """The evaluations of each fixture (its repeats), in run order."""
    groups: dict[str, list[FileEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        groups[evaluation.filename].append(evaluation)
    return dict(groups)


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
        verdicts: Each fixture's :func:`_verdict`.
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
    for filename, group in _group_by_fixture(evaluations).items():
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
        majority_score=_mean(fixture_scores),
        field_majority={name: sum(v) / len(v) for name, v in majority.items()},
        field_agreement={name: sum(v) / len(v) for name, v in agreement.items()},
        disagreeing=disagreeing,
        verdicts=verdicts,
    )


def _regular(evaluations: list[FileEvaluation]) -> list[FileEvaluation]:
    """Evaluations that count in the aggregate: no known limitation, no pipeline error."""
    return [e for e in evaluations if not e.known_limitation and not e.error]


def _error_class(message: str) -> str:
    """The exception class name of a ``"Class: message"`` string."""
    return message.split(":", 1)[0]


def summarize(evaluations: list[FileEvaluation], info: dict[str, Any]) -> dict[str, Any]:
    """Aggregate a run into the JSONL summary line's payload.

    Scores follow :func:`print_report`: the mean of the per-line scores over
    lines with no known limitation and no pipeline error. Per-field scores
    are matched / compared over the same lines. Token means are over the
    lines that reported tokens; latency percentiles over every line.

    Args:
        evaluations: Every (fixture, repeat) evaluation of one model.
        info: The run's settings (versions, backend, models, sampling,
            timeout, repeat, start time, early stop), copied first.

    Returns:
        The summary as a JSON-serializable dict.
    """
    regular = _regular(evaluations)
    scores = [e.score for e in regular if e.score is not None]
    by_category: dict[str, list[float]] = defaultdict(list)
    for evaluation in regular:
        if evaluation.score is not None:
            by_category[evaluation.category].append(evaluation.score)
    field_counts: dict[str, list[int]] = {}
    for evaluation in regular:
        for name in evaluation.matched_fields:
            field_counts.setdefault(name, [0, 0])[0] += 1
        for name, _, _ in evaluation.mismatched_fields:
            field_counts.setdefault(name, [0, 0])[1] += 1

    usages = [e.usage for e in evaluations if e.usage is not None]
    prompt_tokens = [u.prompt_tokens for u in usages if u.prompt_tokens is not None]
    completion_tokens = [u.completion_tokens for u in usages if u.completion_tokens is not None]
    latencies = [e.latency_seconds for e in evaluations if e.latency_seconds is not None]
    attempt_messages = [m for e in evaluations for m in e.attempt_errors.values()]
    statuses = Counter(
        match.group(1) for m in attempt_messages if (match := _TRANSIENT_STATUS.search(m))
    )
    repeats = repeat_stats(evaluations)
    recalls = [e.columns_recall for e in regular if e.columns_recall is not None]
    count_matches = [e.columns_count_match for e in regular if e.columns_count_match is not None]

    return {
        **info,
        "fixtures_run": len(_group_by_fixture(evaluations)),
        "lines": len(evaluations),
        "scored_lines": len(scores),
        "known_limitation_lines": sum(e.known_limitation for e in evaluations),
        "errored_lines": sum(bool(e.error) and not e.known_limitation for e in evaluations),
        "aggregate_score": _mean(scores),
        "category_scores": {cat: sum(v) / len(v) for cat, v in sorted(by_category.items())},
        "field_scores": {
            name: counts[0] / (counts[0] + counts[1])
            for name in _COMPARABLE_FIELDS
            if (counts := field_counts.get(name))
        },
        "majority_score": repeats.majority_score,
        "field_majority_scores": repeats.field_majority,
        "field_agreement": repeats.field_agreement,
        "disagreeing_fixtures": repeats.disagreeing,
        "fixture_verdicts": repeats.verdicts,
        "columns_recall_mean": _mean(recalls),
        "columns_count_match_rate": _mean([float(m) for m in count_matches]),
        "tokens": {
            "prompt_total": sum(prompt_tokens),
            "completion_total": sum(completion_tokens),
            "prompt_mean": _mean(prompt_tokens),
            "completion_mean": _mean(completion_tokens),
        },
        "latency_seconds": {
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
        },
        "calls": sum(e.calls for e in evaluations),
        "fallback_used": sum(u.attempts > 1 for u in usages),
        "retries": sum(u.retries for u in usages),
        "errors": dict(Counter(_error_class(e.error) for e in evaluations if e.error)),
        "attempt_errors": dict(Counter(_error_class(m) for m in attempt_messages)),
        "http_429": statuses.get("429", 0),
        "http_503": statuses.get("503", 0),
    }


def run_info(
    *,
    backend: LLMBackend,
    model: str,
    fallback_model: str,
    n_bytes: int,
    tail_bytes: int,
    timeout_seconds: float | None,
    repeat: int,
    started_at: str,
    stopped_early: str | None,
) -> dict[str, Any]:
    """The settings of one model's run, recorded first in its summary line."""
    return {
        "harness_version": HARNESS_VERSION,
        # PROMPT_VERSION arrives with issue #127; None until then.
        "prompt_version": getattr(_prompt, "PROMPT_VERSION", None),
        "backend": backend.value,
        "model": model,
        "fallback_model": fallback_model,
        "n_bytes": n_bytes,
        "tail_bytes": tail_bytes,
        "timeout_seconds": timeout_seconds,
        "repeat": repeat,
        "started_at": started_at,
        "stopped_early": stopped_early,
    }


def line_record(evaluation: FileEvaluation) -> dict[str, Any]:
    """One JSONL line: the verdicts, cost and raw answers of one (fixture, repeat)."""
    usage = evaluation.usage
    record: dict[str, Any] = {
        "fixture": evaluation.filename,
        "category": evaluation.category,
        "repeat": evaluation.repeat,
        "known_limitation": evaluation.known_limitation,
        "model_used": usage.model if usage is not None else None,
        "matched": evaluation.matched_fields,
        "mismatched": [
            {"field": name, "expected": expected, "actual": actual}
            for name, expected, actual in evaluation.mismatched_fields
        ],
        "skipped": evaluation.skipped_fields,
        "score": evaluation.score,
        "columns_recall": evaluation.columns_recall,
        "columns_count_match": evaluation.columns_count_match,
        "usage": usage.model_dump() if usage is not None else None,
        "latency_seconds": evaluation.latency_seconds,
        "calls": evaluation.calls,
        "error": evaluation.error,
        "attempt_errors": evaluation.attempt_errors,
    }
    if evaluation.raw_responses is not None:
        record["raw_response"] = evaluation.raw_responses
    return record


def write_run(path: Path, evaluations: list[FileEvaluation], summary: dict[str, Any]) -> None:
    """Write one JSON line per evaluation, then the ``{"summary": ...}`` line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [line_record(evaluation) for evaluation in evaluations]
    lines.append({"summary": summary})
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------
# Human report
# ---------------------------------------------------------------------


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


def print_report(
    evaluations: list[FileEvaluation],
    *,
    backend: LLMBackend,
    model: str,
    fallback_model: str,
    summary: dict[str, Any] | None = None,
) -> None:
    """Print the full evaluation report to stdout.

    Args:
        evaluations: Per-fixture evaluation results.
        backend: The backend that was evaluated (for the report header).
        model: The primary model that was evaluated (for the report header).
        fallback_model: The fallback model that was evaluated.
        summary: The run's :func:`summarize` output; when given, cost,
            latency and (with repeats) majority-vote lines are added.
    """
    repeated = any(e.repeat > 1 for e in evaluations)
    regular = _regular(evaluations)
    known_limitations = [e for e in evaluations if e.known_limitation]
    errored = [e for e in evaluations if e.error and not e.known_limitation]

    print(
        f"=== csv_inspector eval report (backend={backend.value!r}, model={model!r}, "
        f"fallback={fallback_model!r}) ===\n"
    )

    for evaluation in regular:
        print(_format_file_line(evaluation, show_repeat=repeated))

    if errored:
        print("\n--- Pipeline errors (excluded from aggregate) ---")
        for evaluation in errored:
            print(_format_file_line(evaluation, show_repeat=repeated))

    if known_limitations:
        print("\n--- Known limitations (excluded from aggregate) ---")
        for evaluation in known_limitations:
            print(_format_file_line(evaluation, show_repeat=repeated))

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

    if summary is not None:
        _print_summary_extras(summary, repeated=repeated)


def _print_summary_extras(summary: dict[str, Any], *, repeated: bool) -> None:
    """Print the cost, latency and repeat lines of a run summary."""
    tokens = summary["tokens"]
    latency = summary["latency_seconds"]
    print(
        f"Model calls: {summary['calls']} (fallback used {summary['fallback_used']}x, "
        f"{summary['retries']} retries); tokens: {tokens['prompt_total']} prompt, "
        f"{tokens['completion_total']} completion; latency p50 {_seconds(latency['p50'])}, "
        f"p95 {_seconds(latency['p95'])}"
    )
    if repeated:
        agreement = ", ".join(
            f"{name} {value:.0%}" for name, value in summary["field_agreement"].items()
        )
        print(f"Majority-vote score: {_percent(summary['majority_score'])}")
        print(f"Agreement between repeats: {agreement or 'n/a'}")
        disagreeing = summary["disagreeing_fixtures"]
        print(f"Answers differ between repeats: {', '.join(disagreeing) or 'none'}")
    if summary["stopped_early"]:
        print(f"Stopped early: {summary['stopped_early']}")


# ---------------------------------------------------------------------
# Quota guards and the run loop
# ---------------------------------------------------------------------


class CallBudget:
    """The ``--max-calls`` hard stop, shared by every model of an invocation.

    A fixture only starts when its worst case still fits, so the run never
    makes more than ``max_calls`` model requests.
    """

    def __init__(self, max_calls: int | None) -> None:
        """Start with no call made; ``None`` means no limit."""
        self.max_calls = max_calls
        self.used = 0

    def allows(self, planned: int) -> bool:
        """Whether ``planned`` more calls fit in the budget."""
        return self.max_calls is None or self.used + planned <= self.max_calls

    def charge(self, calls: int) -> None:
        """Record ``calls`` model requests as made."""
        self.used += calls


class RateLimiter:
    """The ``--rpm`` guard: a sliding one-minute window of model requests.

    Before a fixture starts, it waits until the fixture's worst case fits
    in the last minute's window. Calls are stamped when the fixture ends,
    which is later than they happened, so the real rate only errs low.
    """

    WINDOW_SECONDS = 60.0

    def __init__(
        self,
        rpm: int | None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Limit to ``rpm`` requests per minute (``None``: no limit); clock and sleep are injectable."""
        self.rpm = rpm
        self._clock = clock
        self._sleep = sleep
        self._stamps: list[float] = []

    def wait(self, planned: int) -> None:
        """Sleep until ``planned`` more calls stay within ``rpm`` over the last minute."""
        if self.rpm is None:
            return
        planned = min(planned, self.rpm)
        while True:
            now = self._clock()
            self._stamps = [s for s in self._stamps if s > now - self.WINDOW_SECONDS]
            if len(self._stamps) + planned <= self.rpm:
                return
            delay = self._stamps[0] + self.WINDOW_SECONDS - now
            logger.info("--rpm %d: waiting %.1fs.", self.rpm, delay)
            self._sleep(delay)

    def record(self, calls: int) -> None:
        """Stamp ``calls`` model requests as made now."""
        self._stamps.extend([self._clock()] * calls)


def run_model(
    fixtures: list[tuple[str, dict[str, Any]]],
    *,
    backend: LLMBackend,
    settings: Settings,
    model: str,
    fallback_model: str,
    n_bytes: int,
    tail_bytes: int,
    timeout_seconds: float | None,
    repeat: int,
    keep_raw: bool,
    budget: CallBudget,
    limiter: RateLimiter,
) -> tuple[list[FileEvaluation], str | None]:
    """Evaluate every fixture ``repeat`` times with one model, within the quota guards.

    Returns:
        The evaluations, and why the run stopped early (``None`` if it did not).
    """
    worst_case = _worst_case_calls(backend, model, fallback_model)
    evaluations: list[FileEvaluation] = []
    for filename, entry in fixtures:
        for index in range(1, repeat + 1):
            if not budget.allows(worst_case):
                reason = (
                    f"--max-calls {budget.max_calls} reached ({budget.used} call(s) made, "
                    f"the next fixture may need {worst_case})"
                )
                logger.warning("Stopping: %s.", reason)
                return evaluations, reason
            limiter.wait(worst_case)
            logger.info(
                "Evaluating '%s' (category=%s, repeat %d/%d)...",
                filename,
                entry["category"],
                index,
                repeat,
            )
            evaluation = evaluate_file(
                filename,
                entry,
                backend=backend,
                settings=settings,
                model=model,
                fallback_model=fallback_model,
                n_bytes=n_bytes,
                tail_bytes=tail_bytes,
                timeout_seconds=timeout_seconds,
                repeat=index,
                keep_raw=keep_raw,
            )
            budget.charge(evaluation.calls)
            limiter.record(evaluation.calls)
            evaluations.append(evaluation)
    return evaluations, None


def dry_run(
    fixtures: list[tuple[str, dict[str, Any]]],
    *,
    n_bytes: int,
    tail_bytes: int,
    planned_calls: int,
) -> None:
    """Print each fixture's prompt size and the planned call count; no model is called.

    The character count covers the system prompt plus the built prompt, and
    the token estimate assumes :data:`CHARS_PER_TOKEN` characters per token.
    """
    total = 0
    for filename, entry in fixtures:
        prefix = f"[{entry['category']}] {filename}:"
        try:
            samples = sample_source(SAMPLES_DIR / filename, n_bytes, tail_bytes)
        except CSVInspectorError as exc:
            print(f"{prefix} cannot sample — {type(exc).__name__}: {exc}")
            continue
        prompt = build_prompt(
            samples.head_text,
            samples.encoding,
            tail_sample=samples.tail_text,
            covers_whole_file=samples.covers_whole_file,
        )
        chars = len(SYSTEM_PROMPT) + len(prompt)
        total += chars
        print(f"{prefix} {chars} chars, ~{chars // CHARS_PER_TOKEN} tokens")
    print(
        f"\n=== Dry run: {len(fixtures)} fixture(s), {total} prompt chars "
        f"(~{total // CHARS_PER_TOKEN} tokens) per pass; up to {planned_calls} model "
        "call(s) planned; no model was called ==="
    )


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------


def select_fixtures(
    manifest: dict[str, dict[str, Any]],
    *,
    category: str | None,
    names: list[str] | None,
    max_fixtures: int | None,
) -> list[tuple[str, dict[str, Any]]]:
    """The fixtures to run, sorted by name, after ``--category``/``--fixture``/``--max-fixtures``.

    Raises:
        ValueError: If a ``--fixture`` name is not in the manifest, or
            nothing is left to run.
    """
    if names:
        unknown = sorted(set(names) - manifest.keys())
        if unknown:
            raise ValueError(f"Unknown --fixture: {', '.join(unknown)}.")
    selected = [
        (filename, entry)
        for filename, entry in sorted(manifest.items())
        if (category is None or entry["category"] == category) and (not names or filename in names)
    ]
    if max_fixtures is not None:
        selected = selected[:max_fixtures]
    if not selected:
        raise ValueError("No fixture matches the filters.")
    return selected


def output_path(out: str | None, model: str, *, several_models: bool, stamp: str) -> Path | None:
    """Where one model's run is written.

    ``out`` containing ``{model}`` is a template (the model name made
    file-safe); a path ending in ``.jsonl`` is the file itself (one model
    only); anything else is a directory that receives
    ``<UTC timestamp>-<model>.jsonl``.

    Raises:
        ValueError: If a ``.jsonl`` path is given for several models, or the
            file already exists (a run never overwrites another).
    """
    if out is None:
        return None
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model)
    if "{model}" in out:
        path = Path(out.replace("{model}", safe_model))
    elif out.endswith(".jsonl"):
        if several_models:
            raise ValueError("With several --model, --out must be a directory or contain {model}.")
        path = Path(out)
    else:
        path = Path(out) / f"{stamp}-{safe_model}.jsonl"
    if path.exists():
        raise ValueError(f"'{path}' already exists; choose another --out.")
    return path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the evaluation harness."""
    parser = argparse.ArgumentParser(description="csv_inspector manual evaluation harness")
    add_backend_argument(parser)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Primary model (default: the backend's configured model). Repeatable: "
        "models run in sequence, each into its own --out file.",
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
    parser.add_argument(
        "--fixture",
        action="append",
        default=None,
        help="Restrict the run to this fixture (repeatable).",
    )
    parser.add_argument(
        "--max-fixtures",
        type=bounded_int(1),
        default=None,
        help="Run at most this many fixtures (the first, by name, after the filters).",
    )
    parser.add_argument(
        "--repeat", type=bounded_int(1), default=1, help="Run each fixture this many times."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write a JSONL run: a .jsonl file, a directory (receives "
        "<UTC timestamp>-<model>.jsonl), or a template containing {model}.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Store each model answer's raw text in the JSONL lines (raw_response).",
    )
    parser.add_argument(
        "--rpm",
        type=bounded_int(1),
        default=None,
        help="Never exceed this many model requests per minute (harness-side sleep).",
    )
    parser.add_argument(
        "--max-calls",
        type=bounded_int(1),
        default=None,
        help="Hard stop: never make more model requests than this, fallbacks and "
        "retries included, over all models.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the prompts and print their sizes and the planned calls; call no model.",
    )
    add_settings_arguments(parser)
    add_log_level_argument(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the evaluation harness against the full (or filtered) sample catalog."""
    args = _parse_args(argv)
    configure_cli(args.log_level)

    try:
        settings = load_cli_settings(args.env_file, no_env_file=args.no_env_file)
        if args.dry_run:
            # No model is called, so the backend needs no credentials.
            backend = LLMBackend(args.backend) if args.backend else settings.llm_backend
        else:
            backend = resolve_backend(args.backend, settings)
        models: list[str] = args.model or [settings.model_for(backend)]
        fallback_model = args.fallback_model or settings.fallback_model_for(backend)
    except CSVInspectorError as exc:
        # Fail once, up front, instead of reporting the same error per fixture.
        logger.error("Cannot run the evaluation: %s", exc)
        sys.exit(1)

    manifest: dict[str, dict[str, Any]] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        fixtures = select_fixtures(
            manifest, category=args.category, names=args.fixture, max_fixtures=args.max_fixtures
        )
        paths = [
            output_path(args.out, model, several_models=len(models) > 1, stamp=stamp)
            for model in models
        ]
    except ValueError as exc:
        logger.error("Cannot run the evaluation: %s", exc)
        sys.exit(2)

    planned = (
        len(fixtures)
        * args.repeat
        * sum(_worst_case_calls(backend, model, fallback_model) for model in models)
    )
    if args.dry_run:
        dry_run(fixtures, n_bytes=args.bytes, tail_bytes=args.tail_bytes, planned_calls=planned)
        return
    if backend is LLMBackend.API and args.max_calls is None:
        print(
            f"WARNING: --backend api without --max-calls: this run may make up to {planned} "
            "model request(s) against your quota. Pass --max-calls to cap it.",
            file=sys.stderr,
        )

    budget = CallBudget(args.max_calls)
    limiter = RateLimiter(args.rpm)
    for model, path in zip(models, paths, strict=True):
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        evaluations, stopped = run_model(
            fixtures,
            backend=backend,
            settings=settings,
            model=model,
            fallback_model=fallback_model,
            n_bytes=args.bytes,
            tail_bytes=args.tail_bytes,
            timeout_seconds=args.timeout,
            repeat=args.repeat,
            keep_raw=args.keep_raw,
            budget=budget,
            limiter=limiter,
        )
        summary = summarize(
            evaluations,
            run_info(
                backend=backend,
                model=model,
                fallback_model=fallback_model,
                n_bytes=args.bytes,
                tail_bytes=args.tail_bytes,
                timeout_seconds=args.timeout,
                repeat=args.repeat,
                started_at=started_at,
                stopped_early=stopped,
            ),
        )
        print_report(
            evaluations,
            backend=backend,
            model=model,
            fallback_model=fallback_model,
            summary=summary,
        )
        if path is not None:
            write_run(path, evaluations, summary)
            print(f"Wrote {len(evaluations)} line(s) and the summary to {path}\n")
        if stopped:
            break


if __name__ == "__main__":
    main()
