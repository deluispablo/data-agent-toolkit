"""Run files and the guarded run loop.

``--out`` writes a ``{"run": {...}}`` line with the run's settings, then one
line per (fixture, repeat) (:meth:`.evaluation.FileEvaluation.to_record`),
appended and flushed as each inspection finishes, and a final
``{"summary": {...}}`` line. A file without a summary line is an
interrupted run; ``--summarize`` recomputes it from the finished lines.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from csv_inspector import LLMBackend, Settings
from csv_inspector._prompt import PROMPT_VERSION

from .evaluation import FileEvaluation, evaluate_file
from .guards import CallBudget, Fixtures, RateLimiter, worst_case_calls
from .replay import RawAnswers, read_records
from .report import print_report
from .scoring import COMPARABLE_FIELDS, group_by_fixture, mean, repeat_stats

logger = logging.getLogger(__name__)

HARNESS_VERSION = "2"
"""Version of the JSONL run format; bump it when a line or summary field changes meaning."""

_TRANSIENT_STATUS = re.compile(r"\b(429|503)\b")

FOOTPRINT_KEYS = ("model_size_bytes", "model_vram_bytes")
"""Summary keys of a local run's loaded model size (:func:`model_footprint`)."""


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
    fixtures_planned: int,
) -> dict[str, Any]:
    """The settings of one model's run: its ``run`` line, and the start of its summary.

    ``incomplete`` is ``False`` here; ``--summarize`` sets it on the summary
    it recomputes from an interrupted run's lines.
    """
    return {
        "harness_version": HARNESS_VERSION,
        "prompt_version": PROMPT_VERSION,
        "backend": backend.value,
        "model": model,
        "fallback_model": fallback_model,
        "n_bytes": n_bytes,
        "tail_bytes": tail_bytes,
        "timeout_seconds": timeout_seconds,
        "repeat": repeat,
        "started_at": started_at,
        "stopped_early": stopped_early,
        "fixtures_planned": fixtures_planned,
        "incomplete": False,
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    """Linearly interpolated percentile (``fraction`` in [0, 1]), or ``None`` for no values."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _error_class(message: str) -> str:
    """The exception class name of a ``"Class: message"`` string."""
    return message.split(":", 1)[0]


def _field_scores(regular: list[FileEvaluation]) -> dict[str, float]:
    """Per field, matched / compared over the scored lines, in ``COMPARABLE_FIELDS`` order."""
    counts: dict[str, list[int]] = {}
    for evaluation in regular:
        for name in evaluation.matched_fields:
            counts.setdefault(name, [0, 0])[0] += 1
        for name, _, _ in evaluation.mismatched_fields:
            counts.setdefault(name, [0, 0])[1] += 1
    return {
        name: pair[0] / (pair[0] + pair[1])
        for name in COMPARABLE_FIELDS
        if (pair := counts.get(name))
    }


def summarize(evaluations: list[FileEvaluation], info: dict[str, Any]) -> dict[str, Any]:
    """Aggregate a run into its summary line, ``info`` (the run's settings) first.

    Scores are the mean of the per-line scores over lines with no known
    limitation and no pipeline error; per-field scores are matched /
    compared over the same lines. Token means are over the lines that
    reported tokens; latency percentiles over every line; model load times
    over the lines that reported one (none on a replay or a cloud run).
    """
    regular = [e for e in evaluations if not e.known_limitation and not e.error]
    scores = [e.score for e in regular if e.score is not None]
    by_category: dict[str, list[float]] = defaultdict(list)
    for evaluation in regular:
        if evaluation.score is not None:
            by_category[evaluation.category].append(evaluation.score)
    usages = [e.usage for e in evaluations if e.usage is not None]
    prompt_tokens = [u.prompt_tokens for u in usages if u.prompt_tokens is not None]
    completion_tokens = [u.completion_tokens for u in usages if u.completion_tokens is not None]
    latencies = [e.latency_seconds for e in evaluations if e.latency_seconds is not None]
    loads = [u.load_seconds for u in usages if u.load_seconds is not None]
    attempt_messages = [m for e in evaluations for m in e.attempt_errors.values()]
    statuses = Counter(
        match.group(1) for m in attempt_messages if (match := _TRANSIENT_STATUS.search(m))
    )
    repeats = repeat_stats(evaluations)
    recalls = [e.columns_recall for e in regular if e.columns_recall is not None]
    count_matches = [e.columns_count_match for e in regular if e.columns_count_match is not None]
    return {
        **info,
        "fixtures_run": len(group_by_fixture(evaluations)),
        "lines": len(evaluations),
        "scored_lines": len(scores),
        "known_limitation_lines": sum(e.known_limitation for e in evaluations),
        "errored_lines": sum(bool(e.error) and not e.known_limitation for e in evaluations),
        "aggregate_score": mean(scores),
        "category_scores": {cat: sum(v) / len(v) for cat, v in sorted(by_category.items())},
        "field_scores": _field_scores(regular),
        "majority_score": repeats.majority_score,
        "field_majority_scores": repeats.field_majority,
        "field_agreement": repeats.field_agreement,
        "disagreeing_fixtures": repeats.disagreeing,
        "fixture_verdicts": repeats.verdicts,
        "columns_recall_mean": mean(recalls),
        "columns_count_match_rate": mean([float(m) for m in count_matches]),
        "tokens": {
            "prompt_total": sum(prompt_tokens),
            "completion_total": sum(completion_tokens),
            "prompt_mean": mean(prompt_tokens),
            "completion_mean": mean(completion_tokens),
        },
        "latency_seconds": {
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
        },
        "load_seconds": {
            "max": max(loads, default=None),
            "p50": _percentile(loads, 0.5),
        },
        "calls": sum(e.calls for e in evaluations),
        "fallback_used": sum(u.attempts > 1 for u in usages),
        "retries": sum(e.retries for e in evaluations),
        "errors": dict(Counter(_error_class(e.error) for e in evaluations if e.error)),
        "attempt_errors": dict(Counter(_error_class(m) for m in attempt_messages)),
        "http_429": statuses.get("429", 0),
        "http_503": statuses.get("503", 0),
    }


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str) + "\n"


class RunWriter:
    """Streams one model's run to its JSONL file as it happens.

    The ``run`` line is written on creation, each evaluation line is
    appended and flushed as soon as its inspection finishes, and the summary
    comes last, so an interrupted run keeps every finished line.
    """

    def __init__(self, path: Path, info: dict[str, Any]) -> None:
        """Create ``path`` and write the ``run`` line; ``FileExistsError`` if it exists."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lines = 0
        self._handle = path.open("x", encoding="utf-8", newline="\n")
        self._write({"run": info})

    def _write(self, payload: dict[str, Any]) -> None:
        self._handle.write(_json_line(payload))
        self._handle.flush()

    def add(self, evaluation: FileEvaluation) -> None:
        """Append one evaluation's line."""
        self._write(evaluation.to_record())
        self.lines += 1

    def finish(self, summary: dict[str, Any]) -> None:
        """Append the summary line and close the file."""
        self._write({"summary": summary})
        self.close()

    def close(self) -> None:
        """Close the file, keeping whatever was written."""
        self._handle.close()


def summarize_file(path: Path) -> tuple[list[FileEvaluation], dict[str, Any]]:
    """Recompute and append the summary of an interrupted run (``--summarize``).

    The summary is marked ``"incomplete": true``; ``fixtures_planned`` (from
    the ``run`` line) and ``fixtures_run`` tell how far it got.

    Raises:
        ValueError: If the file has a summary already, or no ``run`` line
            (written by harness version 1, or not a run file).
    """
    records = read_records(path)
    if any("summary" in record for record in records):
        raise ValueError(f"'{path}' already has a summary line.")
    if not records or "run" not in records[0]:
        raise ValueError(f"'{path}' has no run line; only harness version 2 runs can be recovered.")
    info: dict[str, Any] = records[0]["run"]
    evaluations = [FileEvaluation.from_record(record) for record in records[1:]]
    summary = summarize(evaluations, {**info, "stopped_early": "interrupted", "incomplete": True})
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_json_line({"summary": summary}))
    return evaluations, summary


def model_footprint(model: str, host: str | None) -> dict[str, int | None]:
    """How much memory ``model`` takes once loaded, from Ollama's ``/api/ps``.

    Best effort: the harness asks once per local run, right after the first
    answer, while the model is still loaded. Any failure, or a model the
    server does not list, gives ``None`` for both sizes.

    Returns:
        ``model_size_bytes`` (the loaded size) and ``model_vram_bytes`` (the
        part of it in GPU memory; ``0`` on a CPU-only server).
    """
    footprint: dict[str, int | None] = dict.fromkeys(FOOTPRINT_KEYS)
    try:
        import ollama  # noqa: PLC0415 - lazily imported, as the local invoker does.

        with ollama.Client(host=host) as client:
            loaded = client.ps().models
    except Exception as exc:  # noqa: BLE001 - best effort: a size is never worth failing a run.
        logger.info("Model footprint unavailable: %s", exc)
        return footprint
    names = {model, model if ":" in model else f"{model}:latest"}
    for entry in loaded:
        if entry.model in names or entry.name in names:
            return {"model_size_bytes": entry.size, "model_vram_bytes": entry.size_vram}
    logger.info("Model footprint unavailable: '%s' is not loaded.", model)
    return footprint


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


def run_model(
    fixtures: Fixtures,
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
    on_evaluation: Callable[[FileEvaluation], None] | None = None,
    replay: dict[tuple[str, int], RawAnswers] | None = None,
) -> tuple[list[FileEvaluation], str | None]:
    """Evaluate every fixture ``repeat`` times with one model, within the quota guards.

    ``on_evaluation`` gets each evaluation as soon as it finishes (the run
    file's writer). With ``replay`` (raw answers by fixture and repeat), each
    (fixture, repeat) is answered from it, and one it does not hold is skipped.

    Returns:
        The evaluations, and why the run stopped early (``None`` if it did not).
    """
    worst_case = worst_case_calls(backend, model, fallback_model)
    evaluations: list[FileEvaluation] = []
    for filename, entry in fixtures:
        for index in range(1, repeat + 1):
            if replay is not None and (filename, index) not in replay:
                continue
            if not budget.allows(worst_case):
                reason = (
                    f"--max-calls {budget.max_calls} reached ({budget.used} call(s) made, "
                    f"the next fixture may need {worst_case})"
                )
                logger.warning("Stopping: %s.", reason)
                return evaluations, reason
            limiter.wait(worst_case)
            msg = "Evaluating '%s' (category=%s, repeat %d/%d)..."
            logger.info(msg, filename, entry["category"], index, repeat)
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
                replayed=replay[filename, index] if replay is not None else None,
            )
            budget.charge(evaluation.calls)
            limiter.record(evaluation.calls)
            evaluations.append(evaluation)
            if on_evaluation is not None:
                on_evaluation(evaluation)
    return evaluations, None


def run_and_report(
    fixtures: Fixtures,
    info: dict[str, Any],
    path: Path | None,
    *,
    settings: Settings,
    keep_raw: bool,
    budget: CallBudget,
    limiter: RateLimiter,
    replay: dict[tuple[str, int], RawAnswers] | None = None,
) -> str | None:
    """Run one model over ``fixtures`` as ``info`` describes, report it, write its file.

    A live ``local`` run also records the loaded model's size
    (:func:`model_footprint`) in its summary; a replay or a cloud run does not.

    Returns:
        Why the run stopped early, or ``None``.
    """
    writer = RunWriter(path, info) if path is not None else None
    measure = info["backend"] == LLMBackend.LOCAL.value and replay is None
    footprint: dict[str, int | None] = dict.fromkeys(FOOTPRINT_KEYS) if measure else {}

    def record(evaluation: FileEvaluation) -> None:
        nonlocal measure
        if writer is not None:
            writer.add(evaluation)
        if measure and evaluation.usage is not None:
            measure = False
            footprint.update(model_footprint(info["model"], settings.ollama_host))

    try:
        evaluations, stopped = run_model(
            fixtures,
            backend=LLMBackend(info["backend"]),
            settings=settings,
            model=info["model"],
            fallback_model=info["fallback_model"],
            n_bytes=info["n_bytes"],
            tail_bytes=info["tail_bytes"],
            timeout_seconds=info["timeout_seconds"],
            repeat=info["repeat"],
            keep_raw=keep_raw,
            budget=budget,
            limiter=limiter,
            on_evaluation=record,
            replay=replay,
        )
    except KeyboardInterrupt:
        if writer is not None:
            writer.close()
            msg = "Interrupted: %d finished line(s) kept in %s; recover the summary with "
            msg += "--summarize %s"
            logger.error(msg, writer.lines, writer.path, writer.path)
        sys.exit(130)
    summary = summarize(evaluations, {**info, **footprint, "stopped_early": stopped})

    print_report(
        evaluations,
        backend=info["backend"],
        model=info["model"],
        fallback_model=info["fallback_model"],
        summary=summary,
    )
    if writer is not None:
        writer.finish(summary)
        print(f"Wrote {len(evaluations)} line(s) and the summary to {writer.path}\n")
    return stopped
