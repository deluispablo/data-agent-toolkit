"""Tests for the README walkthrough capture (no model is called: Ollama is faked)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from capture_walkthrough import (
    DEMO_FILE,
    HEAD_BYTES,
    TAIL_BYTES,
    capture,
    grounding_diff,
    write_capture,
)
from csv_inspector import Settings
from fakes import install_fake_ollama, ollama_reply

# A plausible answer that gets the header row wrong and, as asked, names only
# the first footer line: grounding fixes the first and extends the second.
ANSWER: dict[str, Any] = {
    "encoding": "Windows-1252",
    "delimiter": ";",
    "quotechar": '"',
    "escapechar": None,
    "doublequote": True,
    "has_header": True,
    "header_row_index": 2,
    "footer_first_line": "TOTAL;;;51;427,30;",
    "columns": ["Date", "Store", "Product", "Units", "Amount (€)", "Returned"],
    "confidence": 0.9,
}
FOOTER = ["TOTAL;;;51;427,30;", "*** End of report ***"]


def _replies(*contents: str) -> Iterator[Any]:
    """Fake Ollama replies with fixed token counts, one per call."""
    return iter(ollama_reply(c, prompt_eval_count=950, eval_count=100) for c in contents)


def test_grounding_diff_lists_changed_fields_and_the_extended_footer() -> None:
    """A corrected field and a footer extended past the model's first line are both listed."""
    result = {**ANSWER, "header_row_index": 3, "footer_lines": FOOTER}
    result.pop("footer_first_line")

    assert grounding_diff(ANSWER, result) == [
        {"field": "header_row_index", "model": 2, "result": 3},
        {"field": "footer_lines", "model": ["TOTAL;;;51;427,30;"], "result": FOOTER},
    ]


def test_grounding_diff_is_empty_when_the_answer_matches_the_bytes() -> None:
    """No entry when grounding kept every value the model answered."""
    answer = {**ANSWER, "header_row_index": 3, "footer_first_line": None}
    result = {**answer, "footer_lines": []}
    result.pop("footer_first_line")

    assert grounding_diff(answer, result) == []


def test_capture_records_samples_prompt_answer_and_grounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One inspection yields the samples, the exact prompt, the raw answer and the fixes."""
    replies = _replies(json.dumps(ANSWER))
    install_fake_ollama(monkeypatch, lambda **kwargs: next(replies))

    data = capture(DEMO_FILE, HEAD_BYTES, TAIL_BYTES, Settings())

    size = DEMO_FILE.stat().st_size
    assert data["file"] == {"name": "demo_sales.csv", "size_bytes": size}
    samples = data["samples"]
    assert (samples["head_bytes"], samples["tail_bytes"]) == (HEAD_BYTES, TAIL_BYTES)
    assert samples["unread_bytes"] == size - HEAD_BYTES - TAIL_BYTES
    assert samples["head_text"].startswith("Sales report")
    assert not samples["tail_text"].startswith("2026")  # the tail starts mid-line
    assert samples["head_text"].splitlines()[0] in data["prompt"]["text"]
    assert data["prompt"]["tokens"] == 950
    assert data["answer"]["model"] == "qwen2.5-coder:7b"
    assert data["answer"]["fields"]["header_row_index"] == 2
    assert data["result"]["header_row_index"] == 3
    assert data["result"]["footer_lines"] == FOOTER
    assert {d["field"] for d in data["grounding"]} == {"header_row_index", "footer_lines"}
    assert data["usage"]["attempts"] == 1


def test_capture_keeps_the_fallback_answer_when_the_primary_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorded answer is the one inspect_csv kept: the fallback's."""
    replies = _replies("not json", json.dumps(ANSWER))
    install_fake_ollama(monkeypatch, lambda **kwargs: next(replies))

    data = capture(DEMO_FILE, HEAD_BYTES, TAIL_BYTES, Settings())

    assert data["answer"]["model"] == "qwen2.5-coder:3b"
    assert data["usage"]["model"] == "qwen2.5-coder:3b"
    assert data["usage"]["attempts"] == 2


def test_capture_without_a_tail_reports_zero_tail_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With windows larger than the file there is no tail and nothing unread."""
    replies = _replies(json.dumps(ANSWER))
    install_fake_ollama(monkeypatch, lambda **kwargs: next(replies))

    data = capture(DEMO_FILE, 4096, 4096, Settings())

    assert data["samples"]["tail_text"] is None
    assert data["samples"]["tail_bytes"] == 0
    assert data["samples"]["unread_bytes"] == 0


def test_write_capture_writes_utf8_json_with_lf(tmp_path: Path) -> None:
    """The capture is UTF-8 JSON with LF line endings; strings keep their CRs."""
    path = tmp_path / "walkthrough.json"

    write_capture({"text": "Amount (€)\r\n"}, path)

    raw = path.read_bytes()
    assert b"\r\n" not in raw.replace(b"\\r\\n", b"")
    assert json.loads(raw.decode("utf-8")) == {"text": "Amount (€)\r\n"}
