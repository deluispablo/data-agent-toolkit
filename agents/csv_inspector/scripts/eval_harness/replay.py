"""Replay: a ``--keep-raw`` run's recorded answers, fed back through the pipeline.

No model is called: each (fixture, repeat) is inspected again with a
``model_invoker`` that answers with the text recorded for it. Valid for
grounding, validation and parsing changes only (``docs/evaluation.md``).
Standard library only.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

RawAnswers = list[dict[str, str]]
"""A line's ``raw_response``: ``{"model", "text"}`` per model call that answered."""


class ReplayMissError(RuntimeError):
    """The replayed run holds no (more) answers from the model the pipeline asked."""


def replaying_invoker(recorded: RawAnswers, consumed: RawAnswers) -> Callable[[str, str], str]:
    """A ``model_invoker`` that answers with a run's recorded raw texts, in call order.

    Each call returns the first recorded attempt not yet used whose model is
    the one asked, and appends it to ``consumed``. An attempt that timed out
    or failed to answer left no text, so its model finds nothing and the
    call raises :class:`ReplayMissError`, which the pipeline counts as a
    failed attempt, as it did live.
    """
    pending = list(recorded)

    def invoke(prompt: str, model: str) -> str:
        for index, attempt in enumerate(pending):
            if attempt["model"] == model:
                consumed.append(pending.pop(index))
                return attempt["text"]
        raise ReplayMissError(f"The replayed run holds no answer left from '{model}'.")

    return invoke


def read_records(path: Path) -> list[dict[str, Any]]:
    """Every JSON line of a run file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_replay(
    path: Path, *, n_bytes: int | None, tail_bytes: int | None
) -> tuple[dict[str, Any], dict[tuple[str, int], RawAnswers]]:
    """Read a ``--keep-raw`` run: its ``run`` line and each (fixture, repeat)'s raw answers.

    ``n_bytes`` and ``tail_bytes`` are the requested windows (``None``: the
    run's), which may only repeat the run's.

    Raises:
        ValueError: If the file has no ``run`` line or no raw answers, or a
            window differs from the run's (its answers were given for the
            samples of those windows).
    """
    records = read_records(path)
    if not records or "run" not in records[0]:
        raise ValueError(f"'{path}' has no run line; only harness version 2 runs can be replayed.")
    info: dict[str, Any] = records[0]["run"]
    windows = (("--bytes", n_bytes, "n_bytes"), ("--tail-bytes", tail_bytes, "tail_bytes"))
    for flag, value, key in windows:
        if value is not None and value != info[key]:
            raise ValueError(
                f"{flag} {value} differs from the run's {info[key]}: its answers were given "
                "for other samples."
            )
    answers = {
        (record["fixture"], record["repeat"]): record["raw_response"]
        for record in records[1:]
        if "fixture" in record and "raw_response" in record
    }
    if not answers:
        raise ValueError(
            f"'{path}' holds no raw answers; replay needs a run written with --keep-raw."
        )
    return info, answers
