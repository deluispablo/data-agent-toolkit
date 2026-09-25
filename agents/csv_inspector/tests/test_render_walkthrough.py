"""Tests for the README walkthrough renderer (pure functions, no model)."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from typing import Any

from render_readme_hero import DARK, LIGHT
from render_walkthrough import alt_text, clip, details_blocks, render

DATA: dict[str, Any] = {
    "file": {"name": "demo_sales.csv", "size_bytes": 380},
    "samples": {
        "encoding": "Windows-1252",
        "head_bytes": 224,
        "tail_bytes": 80,
        "unread_bytes": 76,
        "head_text": "Sales report – ACME Europe Ltd.\r\nGenerated: 2026-09-24 08:15\r\n\r\n"  # noqa: RUF001 (the demo's en dash)
        "Date;Store;Product;Units;Amount (€);Returned\r\n",
        "tail_text": "ançon;Ground coffee 1kg;9;107,55;no\r\nTOTAL;;;51;427,30;\r\n"
        "*** End of report ***\r\n",
    },
    "prompt": {
        "text": "Inspect this CSV sample.\nHEAD SAMPLE\n...",
        "version": "2026.09-m",
        "tokens": 842,
    },
    "answer": {
        "model": "qwen2.5-coder:7b",
        "text": '{"header_row_index": 0}',
        "fields": {
            "encoding": "Windows-1252",
            "delimiter": ";",
            "quotechar": '"',
            "escapechar": None,
            "doublequote": True,
            "has_header": True,
            "header_row_index": 0,
            "footer_first_line": "TOTAL;;;51;427,30;",
            "columns": ["Date", "Store", "Product", "Units", "Amount (€)", "Returned"],
            "confidence": 0.95,
        },
    },
    "grounding": [
        {"field": "header_row_index", "model": 0, "result": 3},
        {
            "field": "footer_lines",
            "model": ["TOTAL;;;51;427,30;"],
            "result": ["TOTAL;;;51;427,30;", "*** End of report ***"],
        },
    ],
    "result": {
        "encoding": "Windows-1252",
        "delimiter": ";",
        "quotechar": '"',
        "escapechar": None,
        "doublequote": True,
        "has_header": True,
        "header_row_index": 3,
        "footer_lines": ["TOTAL;;;51;427,30;", "*** End of report ***"],
        "columns": ["Date", "Store", "Product", "Units", "Amount (€)", "Returned"],
        "confidence": 0.95,
        "footer_rows_to_skip": 2,
    },
    "usage": {
        "model": "qwen2.5-coder:7b",
        "prompt_tokens": 842,
        "completion_tokens": 122,
        "latency_seconds": 2.0,
        "attempts": 1,
    },
}


def _texts(svg: str) -> list[str]:
    """Return the text of every ``<text>`` element; fails if the SVG is not well-formed."""
    root = ET.fromstring(svg)
    return ["".join(node.itertext()) for node in root.iter("{http://www.w3.org/2000/svg}text")]


def test_clip_drops_carriage_returns_and_marks_cut_lines() -> None:
    """Carriage returns go; a line longer than the width ends with an ellipsis."""
    assert clip("abc\r", 5) == "abc"
    assert clip("abcdefgh", 5) == "abcd…"
    assert len(clip("x" * 100, 35)) == 35


def test_render_is_well_formed_and_shows_the_real_values() -> None:
    """Both palettes draw the byte counts, tokens, the corrected field and the reader call."""
    for pal in (LIGHT, DARK):
        texts = " ".join(_texts(render(pal, DATA)))
        assert "head 224 B" in texts
        assert "76 B never read" in texts
        assert "842 tokens" in texts
        assert "header_row_index" in texts
        assert "*** End of report ***" in texts
        assert "pd.read_csv" in texts


def test_render_says_nothing_to_correct_without_grounding_changes() -> None:
    """An empty grounding diff is stated, not drawn as an empty panel."""
    data = copy.deepcopy(DATA)
    data["grounding"] = []

    assert any("nothing to correct" in t for t in _texts(render(LIGHT, data)))


def test_render_without_a_tail_says_the_head_covers_the_file() -> None:
    """With no tail sample, panel 1 says so and reports no unread bytes."""
    data = copy.deepcopy(DATA)
    data["samples"].update(tail_text=None, tail_bytes=0, unread_bytes=0)

    texts = " ".join(_texts(render(LIGHT, data)))
    assert "no tail: the head covers the file" in texts
    assert "never read" not in texts


def test_render_clips_long_values_and_escapes_markup() -> None:
    """Long values end with an ellipsis; & and < in the data keep the SVG well-formed."""
    data = copy.deepcopy(DATA)
    long_line = "TOTAL & <all> stores;" * 10
    data["grounding"][1]["result"] = [long_line]
    data["samples"]["head_text"] = "a\tb & <c>\r\n"

    texts = _texts(render(LIGHT, data))  # parses: & and < were escaped
    assert any(t.endswith("…") for t in texts)
    assert max(len(t) for t in texts) <= 140  # only the caption is long


def test_alt_text_names_the_five_steps_and_the_correction() -> None:
    """The alt text walks the five steps with the real correction."""
    alt = alt_text(DATA)

    for part in ("Sample", "Prompt", "Model answer", "Ground", "Result"):
        assert part in alt
    assert "header_row_index 0 corrected to 3" in alt


def test_details_blocks_hold_the_exact_data_of_each_step() -> None:
    """One block per step, holding that step's exact data without carriage returns."""
    blocks = details_blocks(DATA)

    assert len(blocks) == 5
    assert all(b.startswith("<details>") and b.endswith("</details>") for b in blocks)
    assert "Sales report" in blocks[0]
    assert "\r" not in blocks[0]
    assert "Inspect this CSV sample." in blocks[1]
    assert '{"header_row_index": 0}' in blocks[2]
    assert "| `header_row_index` | `0` | `3` |" in blocks[3]
    assert '"footer_rows_to_skip": 2' in blocks[4]


def test_details_table_escapes_pipes() -> None:
    """A pipe in a value does not split the Markdown table cell."""
    data = copy.deepcopy(DATA)
    data["grounding"] = [{"field": "delimiter", "model": ",", "result": "|"}]

    assert '| `delimiter` | `","` | `"\\|"` |' in details_blocks(data)[3]


def test_details_without_grounding_changes_says_nothing_changed() -> None:
    """An empty grounding diff reads as a sentence, not an empty table."""
    data = copy.deepcopy(DATA)
    data["grounding"] = []

    assert "Nothing: the model's answer matched the bytes." in details_blocks(data)[3]


def test_clip_counts_a_tab_as_the_three_characters_it_is_drawn_as() -> None:
    """A line of tabs is cut so that its drawn width, not its length, fits."""
    line = "SKU\tItem\tQty\tUnit price\tUpdated\tX\tY"  # 35 characters, 47 drawn

    clipped = clip(line, 35)

    assert clipped.endswith("…")
    assert len(clipped.replace("\t", " → ")) <= 35
    assert clip("a\tb", 5) == "a\tb"
