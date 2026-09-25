"""Every rendering of a run: the per-run report and the multi-run Markdown table.

Standard library only, so ``compare_runs.py`` runs without the package
installed. The multi-run table reads only each run's final
``{"summary": {...}}`` line: a run recovered with ``--summarize`` is labelled
"(incomplete, N/M fixtures)", a replay "(replay)".
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .evaluation import FileEvaluation


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


def _text(value: object) -> str:
    return "n/a" if value is None else str(value).replace("|", "\\|")


# --- The per-run report ---


def format_file_line(evaluation: FileEvaluation, *, show_repeat: bool = False) -> str:
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


def print_report(
    evaluations: list[FileEvaluation],
    *,
    backend: str,
    model: str,
    fallback_model: str,
    summary: dict[str, Any] | None = None,
) -> None:
    """Print a run's report: a line per evaluation, the aggregate, then the summary's extras.

    With a ``summary`` (see :func:`.runs.summarize`), cost, latency and (with
    repeats) majority-vote lines are added. Known limitations and pipeline
    errors are listed apart and excluded from the aggregate.
    """
    repeated = any(e.repeat > 1 for e in evaluations)
    regular = [e for e in evaluations if not e.known_limitation and not e.error]
    known_limitations = [e for e in evaluations if e.known_limitation]
    errored = [e for e in evaluations if e.error and not e.known_limitation]
    print(
        f"=== csv_inspector eval report (backend={backend!r}, model={model!r}, "
        f"fallback={fallback_model!r}) ===\n"
    )
    for evaluation in regular:
        print(format_file_line(evaluation, show_repeat=repeated))
    for title, group in (("Pipeline errors", errored), ("Known limitations", known_limitations)):
        if group:
            print(f"\n--- {title} (excluded from aggregate) ---")
            for evaluation in group:
                print(format_file_line(evaluation, show_repeat=repeated))
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


# --- The multi-run table (compare_runs.py) ---


class RunFileError(Exception):
    """A run file is missing, unreadable, or has no summary line."""


def load_summary(path: Path) -> dict[str, Any]:
    """The payload of a run file's ``{"summary": ...}`` line (the last one wins).

    Raises:
        RunFileError: If the file cannot be read or parsed, or has no summary.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RunFileError(f"Cannot read '{path}': {exc}") from exc
    summary: dict[str, Any] | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunFileError(f"'{path}' line {number} is not JSON: {exc}") from exc
        if isinstance(record, dict) and isinstance(record.get("summary"), dict):
            summary = record["summary"]
    if summary is None:
        raise RunFileError(
            f"'{path}' has no summary line; was the run interrupted? Recover it with "
            f"'eval_samples.py --summarize {path}'."
        )
    return summary


def _answers(summary: dict[str, Any]) -> str:
    """Where a run's answers came from: a live model, or a replayed run file."""
    source = summary.get("replay_of")
    return "live" if source is None else f"replay of {_text(source)}"


def _union(summaries: Sequence[dict[str, Any]], key: str) -> list[str]:
    """The keys of a per-category or per-field mapping over every run, in first-seen order."""
    names: dict[str, None] = {}
    for summary in summaries:
        names.update(dict.fromkeys(summary.get(key) or {}))
    return list(names)


def render_table(labels: Sequence[str], summaries: Sequence[dict[str, Any]]) -> str:
    """Render the runs side by side as a Markdown table, one row per metric."""
    rows: list[tuple[str, list[str]]] = [
        ("model", [_text(s.get("model")) for s in summaries]),
        ("fallback model", [_text(s.get("fallback_model")) for s in summaries]),
        ("prompt version", [_text(s.get("prompt_version")) for s in summaries]),
        ("backend", [_text(s.get("backend")) for s in summaries]),
        ("answers", [_answers(s) for s in summaries]),
        (
            "fixtures x repeat",
            [f"{s.get('fixtures_run', 0)} x {s.get('repeat', 1)}" for s in summaries],
        ),
        ("prompt tokens (mean)", [_number(s["tokens"]["prompt_mean"]) for s in summaries]),
        ("completion tokens (mean)", [_number(s["tokens"]["completion_mean"]) for s in summaries]),
        ("latency p50", [_seconds(s["latency_seconds"]["p50"]) for s in summaries]),
        ("latency p95", [_seconds(s["latency_seconds"]["p95"]) for s in summaries]),
        ("model calls", [_text(s.get("calls")) for s in summaries]),
        ("errored lines", [_text(s.get("errored_lines")) for s in summaries]),
        ("**accuracy**", [f"**{_percent(s.get('aggregate_score'))}**" for s in summaries]),
        ("majority-vote accuracy", [_percent(s.get("majority_score")) for s in summaries]),
    ]
    for key, title in (("category_scores", "category"), ("field_scores", "field")):
        for name in _union(summaries, key):
            values = [_percent((s.get(key) or {}).get(name)) for s in summaries]
            rows.append((f"{title} `{name}`", values))
    lines = [
        "| metric | " + " | ".join(_text(label) for label in labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    lines += [f"| {name} | " + " | ".join(values) + " |" for name, values in rows]
    return "\n".join(lines)


def verdict_changes(first: dict[str, Any], later: dict[str, Any]) -> list[str]:
    """One ``fixture: before -> after`` line per verdict that differs, sorted by fixture.

    A fixture run only once shows ``not run`` on the other side.
    """
    before: dict[str, str] = first.get("fixture_verdicts") or {}
    after: dict[str, str] = later.get("fixture_verdicts") or {}
    changes = []
    for fixture in sorted(before.keys() | after.keys()):
        old = before.get(fixture, "not run")
        new = after.get(fixture, "not run")
        if old != new:
            changes.append(f"`{fixture}`: {old} -> {new}")
    return changes


def run_label(path: Path, summary: dict[str, Any]) -> str:
    """A run's column header: the file stem, flagged when the run is a replay or incomplete.

    A replay called no model: its tokens and latency are not a live run's.
    """
    label = f"{path.stem} (replay)" if summary.get("replay_of") else path.stem
    if not summary.get("incomplete"):
        return label
    run = summary.get("fixtures_run", 0)
    return f"{label} (incomplete, {run}/{summary.get('fixtures_planned', '?')} fixtures)"


def render(labels: Sequence[str], summaries: Sequence[dict[str, Any]]) -> str:
    """The full Markdown report: the table, then the verdict changes of each later run."""
    parts = [render_table(labels, summaries)]
    for label, summary in zip(labels[1:], summaries[1:], strict=True):
        changes = verdict_changes(summaries[0], summary)
        parts.append(f"\nVerdict changes, {labels[0]} -> {label}:")
        parts.append("\n".join(f"- {change}" for change in changes) if changes else "- none")
    return "\n".join(parts)


def compare_main(argv: Sequence[str] | None = None) -> None:
    """``compare_runs.py``: print the comparison of the run files given on the command line."""
    parser = argparse.ArgumentParser(description="Compare eval_samples.py JSONL runs.")
    parser.add_argument("runs", nargs="+", type=Path, help="Two or more JSONL run files.")
    args = parser.parse_args(argv)
    if len(args.runs) < 2:  # noqa: PLR2004 - a comparison needs two sides
        parser.error("give at least two run files")
    try:
        summaries = [load_summary(path) for path in args.runs]
    except RunFileError as exc:
        print(f"compare_runs: {exc}", file=sys.stderr)
        sys.exit(1)
    labels = [run_label(path, summary) for path, summary in zip(args.runs, summaries, strict=True)]
    print(render(labels, summaries))
