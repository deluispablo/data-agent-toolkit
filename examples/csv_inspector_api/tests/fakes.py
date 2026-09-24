"""Fake model invokers: every inspection in the tests runs without a model.

A deliberately small copy of the idea in the agent's own test fakes, which
are not shipped with the library (and whose module name would collide).
The library accepts any async ``(prompt, model) -> text`` callable as
``model_invoker``, so these stand in for Ollama or Gemini.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

SAMPLE_CSV = Path(__file__).resolve().parents[3] / "agents" / "csv_inspector" / "sample.csv"
"""The agent's demo file: two preamble lines, a ``;``-delimited header, five rows."""

SAMPLE_ANSWER = json.dumps(
    {
        "encoding": "utf-8",
        "delimiter": ";",
        "quotechar": '"',
        "escapechar": None,
        "doublequote": True,
        "header_row_index": 2,
        "footer_lines": [],
        "columns": [
            {"name": "Fecha", "inferred_type": "date", "nullable": False},
            {"name": "Cliente", "inferred_type": "string", "nullable": False},
            {"name": "Descripción", "inferred_type": "string", "nullable": False},
            {"name": "Importe", "inferred_type": "float", "nullable": False},
            {"name": "Observaciones", "inferred_type": "string", "nullable": True},
        ],
        "confidence": 0.9,
        "notes": "Two preamble lines before the header.",
    }
)
"""A valid model answer for :data:`SAMPLE_CSV`."""


class FakeInvoker:
    """Async model invoker with a scripted behaviour; records every call.

    Attributes:
        answer: Text returned to the library (valid JSON by default).
        error: Exception raised instead of answering, if set.
        delay: Seconds to sleep before answering, to exceed a time budget.
        calls: ``(prompt, model)`` of every call, in order.
    """

    def __init__(
        self,
        answer: str = SAMPLE_ANSWER,
        *,
        error: Exception | None = None,
        delay: float = 0,
    ) -> None:
        """Script the invoker.

        Args:
            answer: Text to return; not JSON drives the ``ResponseParsingError`` path.
            error: Exception to raise on every call instead of answering.
            delay: Seconds to sleep before answering or raising.
        """
        self.answer = answer
        self.error = error
        self.delay = delay
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, prompt: str, model: str) -> str:
        """Answer like a model would: ``(prompt, model) -> raw text``."""
        self.calls.append((prompt, model))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.answer
