"""Tests of the prompt: ``build_prompt`` and the JSON Schema both backends send."""

from __future__ import annotations

import json

import pytest

from csv_inspector import (
    CSVInspectionResult,
)
from csv_inspector._prompt import build_prompt


def test_build_prompt_embeds_head_sample_and_encoding_hint() -> None:
    """The generated prompt must include the head sample and encoding hint."""
    prompt = build_prompt(head_sample="Fecha;Cliente\n2024-01-15;Acme", detected_encoding="utf-8")

    assert "Fecha;Cliente" in prompt
    assert "'utf-8'" in prompt


def test_build_prompt_omits_tail_section_when_tail_sample_is_none() -> None:
    """No tail section should appear in the prompt when there is no tail sample."""
    prompt = build_prompt(head_sample="a,b,c\n1,2,3\n", detected_encoding="utf-8", tail_sample=None)

    assert "TAIL SAMPLE START" not in prompt
    assert "contains the ENTIRE file" in prompt


def test_build_prompt_forbids_a_footer_when_the_end_was_not_sampled() -> None:
    """A truncated head with no tail must not be presented as the whole file (issue #10)."""
    prompt = build_prompt(
        head_sample="a,b,c\n1,2,3\n4,5",
        detected_encoding="utf-8",
        tail_sample=None,
        covers_whole_file=False,
    )

    assert "TAIL SAMPLE START" not in prompt
    assert "contains the ENTIRE file" not in prompt
    assert "its end was not sampled" in prompt
    assert '"footer_first_line" must be null' in prompt


def test_build_prompt_includes_tail_section_and_mid_line_caveat() -> None:
    """A provided tail sample must appear in its own labeled section with a caveat."""
    prompt = build_prompt(
        head_sample="a,b,c\n1,2,3\n",
        detected_encoding="utf-8",
        tail_sample=",99\nTOTAL,,999\n",
    )

    assert "TAIL SAMPLE START" in prompt
    assert ",99\nTOTAL,,999\n" in prompt
    assert "may start mid-line" in prompt


@pytest.mark.parametrize("tail_sample", [None, "2,3\nTOTAL,,5\n"])
def test_build_prompt_gives_concrete_footer_rules(tail_sample: str | None) -> None:
    """Footers get explicit rules and examples in both prompt variants.

    Regression test for issue #2: with only a metadata example in the prompt,
    the model filed end-of-report markers and timestamps under metadata.
    """
    prompt = build_prompt(
        head_sample="a,b\n1,2\n", detected_encoding="utf-8", tail_sample=tail_sample
    )

    assert "FOOTER (end of the file)" in prompt
    for example in ("TOTAL,,4241.25", "--- Fin del informe ---", "Generado el 2024-01-20"):
        assert example in prompt
    assert "footer lines are never preamble" in prompt


def test_build_prompt_reads_footer_only_from_the_tail_when_present() -> None:
    """With a tail sample, the head's (mid-file) last line must not be read as a footer."""
    prompt = build_prompt(head_sample="a,b\n1,2\n", detected_encoding="utf-8", tail_sample="9,9\n")

    assert "check the last lines of the TAIL sample" in prompt
    assert "The tail is the real end of the file" in prompt


def test_build_prompt_no_longer_requests_derived_or_removed_fields() -> None:
    """The model is asked for the first footer line only; the rest is read from the file."""
    prompt = build_prompt(head_sample="a,b\n1,2\n", detected_encoding="utf-8")

    assert '"footer_first_line"' in prompt
    assert "footer_lines" not in prompt
    assert "footer_rows_to_skip" not in prompt
    assert "metadata_lines" not in prompt


def test_json_schema_is_flat_with_no_column_objects_notes_or_usage() -> None:
    """The schema sent to Gemini lists names only: no $defs, notes or usage (issue #129)."""
    schema = CSVInspectionResult.model_json_schema()

    assert "$defs" not in schema
    assert "ColumnSchema" not in json.dumps(schema)
    assert "notes" not in schema["properties"]
    assert "usage" not in schema["properties"]
    assert schema["properties"]["columns"]["items"] == {"type": "string"}
    assert "columns" in schema["required"]


def test_prompt_asks_for_column_names_only() -> None:
    """The prompt asks for a list of names, with no types, examples or notes (issue #129)."""
    prompt = build_prompt("a\n1\n", "utf-8")

    assert '"columns" holds each name copied character for character' in prompt
    for removed in ("inferred_type", "nullable", "example_values", '"notes"', "boolean"):
        assert removed not in prompt
