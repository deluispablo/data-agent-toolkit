"""Capture one real inspection of the README demo file for the walkthrough.

Runs ``inspect_csv`` on ``docs/assets/demo_sales.csv`` with the local
backend and the default models, records the exact prompt and the raw
model answer, and writes ``docs/assets/walkthrough.json``: the decoded
samples, the prompt, the answer, the fields grounding changed, the final
result and the usage. ``scripts/render_walkthrough.py`` turns that file
into the README's "How it works" storyboard and ``<details>`` blocks.

The sample windows are shrunk to 224 and 80 bytes so that the 380-byte
demo file shows both a head and a mid-line tail, with bytes in between
that are never read; the library's defaults are 4 KiB each.

The prompt and the raw answer are recorded by wrapping
``csv_inspector._inspect.builtin_invoker``, the seam
``eval_samples.py --keep-raw`` uses. Rerun after a prompt, grounding or
default-model change, then rerun ``render_walkthrough.py``. Needs
``ollama serve`` with ``qwen2.5-coder:7b`` (and ``:3b``) pulled.

Usage:
    python scripts/capture_walkthrough.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from csv_inspector import LLMBackend, Settings, _inspect, inspect_csv
from csv_inspector._invokers import InvokerResponse
from csv_inspector._prompt import parse_and_validate
from csv_inspector._sampling import sample_source

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
DEMO_FILE = ASSETS / "demo_sales.csv"
OUTPUT = ASSETS / "walkthrough.json"
HEAD_BYTES = 224
TAIL_BYTES = 80
TIMEOUT_SECONDS = 300.0

# Fields the model answers and the result keeps under the same name.
_SHARED_FIELDS = (
    "encoding",
    "delimiter",
    "quotechar",
    "escapechar",
    "doublequote",
    "has_header",
    "header_row_index",
    "columns",
)

_SyncCall = Callable[[str, str, float | None], InvokerResponse]
# The factory inspect_csv looks up in its module when no model_invoker is given.
_SEAM = "builtin_invoker"


@contextmanager
def _recording_calls(calls: list[dict[str, str]]) -> Iterator[None]:
    """Record the prompt and raw answer of every built-in model call in the block.

    Args:
        calls: The list ``{"model", "prompt", "answer"}`` entries are appended to.
    """
    original: Callable[[LLMBackend, Settings], _SyncCall] = getattr(_inspect, _SEAM)

    def factory(backend: LLMBackend, settings: Settings) -> _SyncCall:
        call = original(backend, settings)

        def recording(prompt: str, model: str, timeout: float | None) -> InvokerResponse:
            response = call(prompt, model, timeout)
            calls.append({"model": model, "prompt": prompt, "answer": response.text})
            return response

        return recording

    setattr(_inspect, _SEAM, factory)
    try:
        yield
    finally:
        setattr(_inspect, _SEAM, original)


def grounding_diff(answer: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every field whose final value differs from the model's answer.

    The model names only the first footer line (``footer_first_line``);
    it is compared, as a one-line list, with the final ``footer_lines``.

    Args:
        answer: The parsed model answer.
        result: The final result, as ``CSVInspectionResult.model_dump(mode="json")``.

    Returns:
        ``{"field", "model", "result"}`` entries, in field order.
    """
    changes = [
        {"field": field, "model": answer[field], "result": result[field]}
        for field in _SHARED_FIELDS
        if answer[field] != result[field]
    ]
    first = answer["footer_first_line"]
    model_footer = [] if first is None else [first]
    if model_footer != result["footer_lines"]:
        changes.append(
            {"field": "footer_lines", "model": model_footer, "result": result["footer_lines"]}
        )
    return changes


def capture(source: Path, head_bytes: int, tail_bytes: int, settings: Settings) -> dict[str, Any]:
    """Inspect ``source`` for real and return everything the walkthrough shows.

    Args:
        source: The demo file.
        head_bytes: The head window (``n_bytes``).
        tail_bytes: The tail window.
        settings: Explicit library settings (the default is the local backend).

    Returns:
        The walkthrough data, ready for :func:`write_capture`.
    """
    calls: list[dict[str, str]] = []
    with _recording_calls(calls):
        result = inspect_csv(
            source,
            settings=settings,
            n_bytes=head_bytes,
            tail_bytes=tail_bytes,
            timeout_seconds=TIMEOUT_SECONDS,
        )
    kept = calls[-1]  # the answer inspect_csv kept: the fallback's when the primary failed
    samples = sample_source(source, head_bytes, tail_bytes)
    size = source.stat().st_size
    head = min(head_bytes, size)
    tail = 0 if samples.tail_text is None else min(tail_bytes, size - head)
    answer = parse_and_validate(kept["answer"], kept["model"]).model_dump(mode="json")
    final = result.model_dump(mode="json")
    usage = result.usage
    assert usage is not None  # every successful inspection records its usage
    return {
        "file": {"name": source.name, "size_bytes": size},
        "samples": {
            "encoding": samples.encoding,
            "head_bytes": head,
            "tail_bytes": tail,
            "unread_bytes": size - head - tail,
            "head_text": samples.head_text,
            "tail_text": samples.tail_text,
        },
        "prompt": {
            "text": kept["prompt"],
            "version": usage.prompt_version,
            "tokens": usage.prompt_tokens,
        },
        "answer": {"model": kept["model"], "text": kept["answer"], "fields": answer},
        "grounding": grounding_diff(answer, final),
        "result": final,
        "usage": {
            "model": usage.model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "latency_seconds": round(usage.latency_seconds, 1),
            "attempts": usage.attempts,
        },
    }


def write_capture(data: dict[str, Any], path: Path) -> None:
    """Write the walkthrough data as indented UTF-8 JSON with LF line endings."""
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> None:
    """Capture the walkthrough and write it to ``--out``."""
    parser = argparse.ArgumentParser(description="Capture the README walkthrough data.")
    parser.add_argument("--out", type=Path, default=OUTPUT, help="Output JSON file.")
    args = parser.parse_args(argv)
    data = capture(DEMO_FILE, HEAD_BYTES, TAIL_BYTES, Settings())
    write_capture(data, args.out)
    changed = ", ".join(d["field"] for d in data["grounding"]) or "nothing"
    print(f"Wrote {args.out}: model {data['usage']['model']}, grounding changed {changed}.")


if __name__ == "__main__":
    main()
