"""The command line of ``scripts/eval_samples.py``: live runs, dry runs, replays, summaries."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from csv_inspector import (
    DEFAULT_SAMPLE_BYTES,
    DEFAULT_TAIL_BYTES,
    CSVInspectorError,
    LLMBackend,
    Settings,
)
from csv_inspector._prompt import CHARS_PER_TOKEN, SYSTEM_PROMPT, build_prompt
from csv_inspector._sampling import sample_source
from csv_inspector.cli import (
    DEFAULT_CLI_TIMEOUT_SECONDS,
    HEAD_BYTES,
    TAIL_BYTES,
    add_backend_argument,
    add_log_level_argument,
    add_settings_arguments,
    bounded_int,
    configure_cli,
    load_cli_settings,
    resolve_backend,
    timeout_budget,
)

from .guards import (
    MANIFEST_PATH,
    SAMPLES_DIR,
    SUBSETS,
    CallBudget,
    Fixtures,
    RateLimiter,
    select_fixtures,
    worst_case_calls,
)
from .replay import load_replay
from .report import print_report
from .runs import output_path, run_and_report, run_info, summarize_file

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
        default=None,
        help=f"Head sample size, in bytes (default: {DEFAULT_SAMPLE_BYTES}; with --replay, "
        "the run's).",
    )
    parser.add_argument(
        "--tail-bytes",
        type=TAIL_BYTES,
        default=None,
        help=f"Tail sample size, in bytes, 0 disables (default: {DEFAULT_TAIL_BYTES}; with "
        "--replay, the run's).",
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
        "--subset",
        choices=sorted(SUBSETS),
        default=None,
        help="Run a documented fixture list: 'quick' to iterate, 'cloud' for a free-tier "
        "day (combines with --fixture).",
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
        "--summarize",
        type=Path,
        default=None,
        metavar="RUN",
        help="Recompute and append the summary of an interrupted --out file; call no model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the prompts and print their sizes and the planned calls; call no model.",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        metavar="RUN",
        help="Answer each fixture with the raw answers recorded in a --keep-raw run instead "
        "of calling a model (valid for grounding, validation and parsing changes only).",
    )
    add_settings_arguments(parser)
    add_log_level_argument(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the evaluation harness against the full (or filtered) sample catalog."""
    args = parse_args(argv)
    configure_cli(args.log_level)
    if args.summarize is not None:
        run_summarize(args.summarize)
    elif args.replay is not None:
        run_replay(args)
    else:
        run_live(args)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _selected_fixtures(args: argparse.Namespace) -> Fixtures:
    """The manifest fixtures picked by ``--category``/``--fixture``/``--subset``/``--max-fixtures``.

    Raises:
        ValueError: As :func:`.guards.select_fixtures`.
    """
    manifest: dict[str, dict[str, Any]] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    names = [*(args.fixture or []), *(SUBSETS[args.subset] if args.subset else ())]
    return select_fixtures(
        manifest, category=args.category, names=names, max_fixtures=args.max_fixtures
    )


def run_live(args: argparse.Namespace) -> None:
    """Run every ``--model`` over the selected fixtures (or ``--dry-run`` them)."""
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
    stamp = _stamp()
    try:
        fixtures = _selected_fixtures(args)
        several = len(models) > 1
        paths = [output_path(args.out, m, several_models=several, stamp=stamp) for m in models]
    except ValueError as exc:
        logger.error("Cannot run the evaluation: %s", exc)
        sys.exit(2)
    per_pass = sum(worst_case_calls(backend, model, fallback_model) for model in models)
    planned = len(fixtures) * args.repeat * per_pass
    n_bytes = DEFAULT_SAMPLE_BYTES if args.bytes is None else args.bytes
    tail_bytes = DEFAULT_TAIL_BYTES if args.tail_bytes is None else args.tail_bytes
    if args.dry_run:
        run_dry(fixtures, n_bytes=n_bytes, tail_bytes=tail_bytes, planned_calls=planned)
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
        info = run_info(
            backend=backend,
            model=model,
            fallback_model=fallback_model,
            n_bytes=n_bytes,
            tail_bytes=tail_bytes,
            timeout_seconds=args.timeout,
            repeat=args.repeat,
            started_at=_now(),
            stopped_early=None,
            fixtures_planned=len(fixtures),
        )
        stopped = run_and_report(
            fixtures,
            info,
            path,
            settings=settings,
            keep_raw=args.keep_raw,
            budget=budget,
            limiter=limiter,
        )
        if stopped:
            break


def run_dry(fixtures: Fixtures, *, n_bytes: int, tail_bytes: int, planned_calls: int) -> None:
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
        print(f"{prefix} {chars} chars, ~{round(chars / CHARS_PER_TOKEN)} tokens")
    print(
        f"\n=== Dry run: {len(fixtures)} fixture(s), {total} prompt chars "
        f"(~{round(total / CHARS_PER_TOKEN)} tokens) per pass; up to {planned_calls} model "
        "call(s) planned; no model was called ==="
    )


def run_replay(args: argparse.Namespace) -> None:
    """``--replay``: re-run the pipeline on a run's recorded answers; call no model."""
    flags = (
        ("--backend", args.backend),
        ("--rpm", args.rpm),
        ("--max-calls", args.max_calls),
        ("--dry-run", args.dry_run or None),
    )
    rejected = [flag for flag, value in flags if value is not None]
    try:
        if rejected:
            raise ValueError(f"--replay calls no model; drop {', '.join(rejected)}.")
        if args.model is not None and len(args.model) > 1:
            raise ValueError("--replay takes at most one --model.")
        source, answers = load_replay(args.replay, n_bytes=args.bytes, tail_bytes=args.tail_bytes)
        selected = _selected_fixtures(args)
        held = {fixture for fixture, _ in answers}
        fixtures = [(name, entry) for name, entry in selected if name in held]
        if not fixtures:
            raise ValueError(f"'{args.replay}' holds none of the selected fixtures.")
        model = args.model[0] if args.model else source["model"]
        path = output_path(args.out, model, several_models=False, stamp=_stamp())
    except (OSError, ValueError) as exc:
        logger.error("Cannot replay the run: %s", exc)
        sys.exit(2)
    skipped = len(selected) - len(fixtures)
    if skipped:
        logger.warning("Skipping %d selected fixture(s) the replayed run does not hold.", skipped)
    info = {
        **run_info(
            backend=LLMBackend(source["backend"]),
            model=model,
            fallback_model=args.fallback_model or source["fallback_model"],
            n_bytes=source["n_bytes"],
            tail_bytes=source["tail_bytes"],
            timeout_seconds=source["timeout_seconds"],
            repeat=source["repeat"],
            started_at=_now(),
            stopped_early=None,
            fixtures_planned=len(fixtures),
        ),
        # The answers came from the source run's prompt, not this checkout's.
        "prompt_version": source["prompt_version"],
        "replay_of": str(args.replay),
        "replay_skipped": skipped,
    }
    run_and_report(
        fixtures,
        info,
        path,
        settings=Settings(),
        keep_raw=True,
        budget=CallBudget(None),
        limiter=RateLimiter(None),
        replay=answers,
    )


def run_summarize(path: Path) -> None:
    """``--summarize``: append the summary of an interrupted run and print its report."""
    try:
        evaluations, summary = summarize_file(path)
    except (OSError, ValueError) as exc:
        logger.error("Cannot summarize the run: %s", exc)
        sys.exit(2)
    print_report(
        evaluations,
        backend=summary["backend"],
        model=summary["model"],
        fallback_model=summary["fallback_model"],
        summary=summary,
    )
    print(
        f"Appended an incomplete summary to {path} ({summary['fixtures_run']} of "
        f"{summary['fixtures_planned']} fixture(s) run).\n"
    )
