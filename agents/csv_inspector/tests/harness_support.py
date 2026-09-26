"""Helpers shared by the harness tests: fake inspections, answers, run files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from csv_inspector import (
    CSVInspectionResult,
    LLMBackend,
    Settings,
)
from csv_inspector._models import Usage
from csv_inspector._prompt import PROMPT_VERSION
from eval_harness import evaluation as evaluation_module
from eval_harness import runs as runs_module
from eval_harness.evaluation import FileEvaluation, evaluate_file


def _result_with_columns(*names: str) -> CSVInspectionResult:
    """Build a minimal inspection result reporting ``names`` as its columns.

    The names are set as written, surrounding spaces included, like grounding
    does after validation (the validator strips a model's names).
    """
    result = CSVInspectionResult(
        encoding="utf-8",
        delimiter=",",
        header_row_index=0,
        columns=list(names),
        confidence=0.9,
    )
    return result.model_copy(update={"columns": list(names)})


def _evaluate_with(
    monkeypatch: pytest.MonkeyPatch, result: CSVInspectionResult, expected: dict[str, Any]
) -> FileEvaluation:
    """Run ``evaluate_file`` with ``inspect_csv`` faked to return ``result``."""

    def fake_inspect_csv(source: object, /, **kwargs: Any) -> CSVInspectionResult:
        return result

    monkeypatch.setattr(evaluation_module, "inspect_csv", fake_inspect_csv)
    return evaluate_file(
        "f.csv",
        {"category": "c", "known_limitation": False, "expected": expected},
        backend=LLMBackend.LOCAL,
        settings=Settings(),
        model="primary",
        fallback_model="fallback",
        n_bytes=4096,
        tail_bytes=4096,
    )


FIXTURES = ["delimiter_comma.csv", "delimiter_pipe.csv", "delimiter_semicolon.csv"]


def _usage(**overrides: Any) -> Usage:
    """A ``Usage`` with small, recognizable counters."""
    values: dict[str, Any] = {
        "model": "primary",
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "latency_seconds": 1.5,
        "attempts": 1,
        "prompt_version": PROMPT_VERSION,
    }
    values.update(overrides)
    return Usage(**values)


def _answer(usage: Usage | None = None) -> CSVInspectionResult:
    """A comma-delimited result, with ``usage`` attached."""
    return _result_with_columns("a", "b").model_copy(update={"usage": usage or _usage()})


def _evaluation(
    filename: str = "f.csv",
    *,
    matched: list[str] | None = None,
    mismatched: list[tuple[str, Any, Any]] | None = None,
    **overrides: Any,
) -> FileEvaluation:
    """A synthetic evaluation, as a fake run would produce it."""
    values: dict[str, Any] = {"filename": filename, "category": "c", "known_limitation": False}
    values.update(overrides)
    return FileEvaluation(
        matched_fields=matched or [], mismatched_fields=mismatched or [], **values
    )


FOOTPRINT = {"model_size_bytes": 2_500_000_000, "model_vram_bytes": 0}
"""What the faked Ollama ``/api/ps`` lookup reports for every run (see :func:`_fake_inspect`)."""


def _fake_inspect(monkeypatch: pytest.MonkeyPatch, answer: Any) -> list[dict[str, Any]]:
    """Replace ``inspect_csv`` by ``answer`` (a result, an exception or a callable).

    The loaded-model lookup of a local run is faked too (:data:`FOOTPRINT`):
    there is no Ollama server in the tests.
    """
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(runs_module, "model_footprint", lambda model, host: dict(FOOTPRINT))

    def fake_inspect_csv(source: object, /, **kwargs: Any) -> CSVInspectionResult:
        calls.append({"source": source, **kwargs})
        if isinstance(answer, Exception):
            raise answer
        return answer(**kwargs) if callable(answer) else answer  # type: ignore[no-any-return]

    monkeypatch.setattr(evaluation_module, "inspect_csv", fake_inspect_csv)
    return calls


def _read_run(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _main_args(*extra: str) -> list[str]:
    fixtures = [arg for name in FIXTURES for arg in ("--fixture", name)]
    return ["--no-env-file", "--log-level", "ERROR", *fixtures, *extra]
