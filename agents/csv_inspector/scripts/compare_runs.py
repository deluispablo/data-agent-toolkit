"""Compare two or more evaluation runs written by ``eval_samples.py --out``.

Prints a Markdown table with one column per run (model, prompt version,
token means, latency percentiles, overall, per-category and per-field
accuracy) and, for every run after the first, the fixtures whose verdict
changed against the first run. Paste the output into the pull request of
any change that could move the numbers; ``docs/evaluation.md`` describes
the ritual.

Reads only each run's final ``{"summary": {...}}`` line. A run recovered
with ``eval_samples.py --summarize`` (an interrupted run) is labelled
"(incomplete, N/M fixtures)", and a replay of recorded answers
(``eval_samples.py --replay``, no model called) "(replay)". Standard
library only, so it runs without the package installed.

Usage:
    python compare_runs.py runs/baseline.jsonl runs/candidate.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class RunFileError(Exception):
    """A run file is missing, unreadable, or has no summary line."""


def load_summary(path: Path) -> dict[str, Any]:
    """Read the summary of one JSONL run file.

    Args:
        path: A file written by ``eval_samples.py --out``.

    Returns:
        The payload of its ``{"summary": ...}`` line (the last one wins).

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


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


def _text(value: object) -> str:
    return "n/a" if value is None else str(value).replace("|", "\\|")


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
    """Render the runs side by side as a Markdown table.

    Args:
        labels: One column header per run (e.g. the file stem).
        summaries: The runs' summaries, in the same order.

    Returns:
        The table, one row per metric.
    """
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
    for category in _union(summaries, "category_scores"):
        rows.append(
            (
                f"category `{category}`",
                [_percent((s.get("category_scores") or {}).get(category)) for s in summaries],
            )
        )
    for name in _union(summaries, "field_scores"):
        rows.append(
            (
                f"field `{name}`",
                [_percent((s.get("field_scores") or {}).get(name)) for s in summaries],
            )
        )
    lines = [
        "| metric | " + " | ".join(_text(label) for label in labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    lines += [f"| {name} | " + " | ".join(values) + " |" for name, values in rows]
    return "\n".join(lines)


def verdict_changes(first: dict[str, Any], later: dict[str, Any]) -> list[str]:
    """The fixtures whose verdict differs between two runs.

    Args:
        first: The reference run's summary.
        later: Another run's summary.

    Returns:
        One ``fixture: before -> after`` line per change, sorted by fixture;
        a fixture run only once shows ``not run`` on the other side.
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

    A replay (``eval_samples.py --replay``) called no model: its tokens and
    latency are not a live run's.
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


def main(argv: Sequence[str] | None = None) -> None:
    """Print the comparison of the run files given on the command line."""
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


if __name__ == "__main__":
    main()
