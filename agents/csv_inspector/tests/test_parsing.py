"""Tests of ``parse_and_validate``: JSON extraction and the answer models.

The private ``_ModelAnswer`` is lenient with small-model spellings; the
public ``CSVInspectionResult`` is strict.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from csv_inspector import (
    CSVInspectionResult,
    ResponseParsingError,
)
from csv_inspector._models import _ModelAnswer
from csv_inspector._prompt import _extract_json_payload, parse_and_validate
from payloads import VALID_RESULT_PAYLOAD


def test_footer_rows_to_skip_is_derived_from_footer_lines() -> None:
    """The footer count always equals the number of footer lines, blank ones included."""
    payload = {**VALID_RESULT_PAYLOAD, "footer_lines": ["", "--- Fin ---", "Generado el X"]}

    result = CSVInspectionResult.model_validate(payload)

    assert result.footer_rows_to_skip == 3
    assert result.model_dump()["footer_rows_to_skip"] == 3


def test_contradictory_or_removed_model_fields_are_ignored() -> None:
    """A stale count or a legacy metadata_lines key from the model cannot leak into the result."""
    payload = {
        **VALID_RESULT_PAYLOAD,
        "footer_lines": ["TOTAL,,999"],
        "footer_rows_to_skip": 0,
        "metadata_lines": ["# banner"],
    }

    result = CSVInspectionResult.model_validate(payload)

    assert result.footer_rows_to_skip == 1
    assert "metadata_lines" not in result.model_dump()


@pytest.mark.parametrize(
    ("answer", "expected_index"),
    [
        pytest.param({"has_header": False, "header_row_index": None}, None, id="explicit"),
        pytest.param({"header_row_index": None}, None, id="null-infers-no-header"),
        pytest.param({"header_row_index": -1}, None, id="minus-one-means-null"),
        pytest.param({"header_row_index": 0}, 0, id="index-infers-header"),
        pytest.param({"has_header": True, "header_row_index": 3}, 3, id="explicit-header"),
    ],
)
def test_header_less_answers_validate(
    answer: dict[str, object], expected_index: int | None
) -> None:
    """Null (or -1) means "no header row"; has_header follows unless given."""
    result = _ModelAnswer.model_validate({**VALID_RESULT_PAYLOAD, **answer})

    assert result.header_row_index == expected_index
    assert result.has_header is (expected_index is not None)


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        pytest.param({"has_header": False, "header_row_index": 0}, "must be null when", id="index"),
        pytest.param({"header_row_index": -2}, "greater than or equal to 0", id="negative"),
    ],
)
def test_contradictory_header_answers_fail(answer: dict[str, object], message: str) -> None:
    """has_header and header_row_index must agree, with a clear message."""
    with pytest.raises(ValidationError, match=message):
        _ModelAnswer.model_validate({**VALID_RESULT_PAYLOAD, **answer})


@pytest.mark.parametrize("index", [None, -1])
def test_has_header_with_a_null_index_is_read_as_row_0(index: int | None) -> None:
    """Ambiguous, not a contradiction: grounding decides from the sample."""
    answer = _ModelAnswer.model_validate(
        {**VALID_RESULT_PAYLOAD, "has_header": True, "header_row_index": index}
    )

    assert (answer.has_header, answer.header_row_index) == (True, 0)


@pytest.mark.parametrize(("value", "expected"), [(90, 0.9), (100, 1.0), (0.8, 0.8), (1, 1)])
def test_a_percentage_confidence_is_read_as_a_fraction(value: float, expected: float) -> None:
    """Small models answer 90 or 100; the schema's maximum cannot stop them."""
    answer = _ModelAnswer.model_validate({**VALID_RESULT_PAYLOAD, "confidence": value})

    assert answer.confidence == pytest.approx(expected)


def test_a_missing_header_row_index_still_fails() -> None:
    """Omitting both fields is a malformed answer, not a header-less file."""
    payload = {k: v for k, v in VALID_RESULT_PAYLOAD.items() if k != "header_row_index"}

    with pytest.raises(ValidationError, match="required when has_header"):
        _ModelAnswer.model_validate(payload)


def test_extract_json_payload_strips_markdown_fence() -> None:
    """A JSON payload wrapped in a markdown code fence should be unwrapped."""
    fenced = '```json\n{"a": 1}\n```'

    assert _extract_json_payload(fenced) == '{"a": 1}'


def test_extract_json_payload_passes_through_bare_json() -> None:
    """A bare JSON payload with no fence should be returned unchanged (trimmed)."""
    bare = '  {"a": 1}  '

    assert _extract_json_payload(bare) == '{"a": 1}'


@pytest.mark.parametrize(
    "raw",
    [
        'Here is the result: {"a": {"b": 1}}',
        '{"a": {"b": 1}}\nLet me know if you need anything else.',
        'Sure!\n{"a": {"b": 1}}\nThe delimiter is ";".',
    ],
    ids=["leading-prose", "trailing-prose", "both"],
)
def test_extract_json_payload_drops_prose_around_an_unfenced_object(raw: str) -> None:
    """Without a fence, the object is taken from the first ``{`` to the last ``}``.

    Regression test for issue #23: such replies used to fail as invalid JSON.
    """
    assert _extract_json_payload(raw) == '{"a": {"b": 1}}'


def test_a_reply_without_any_object_is_still_invalid_json() -> None:
    """Text with no ``{...}`` span is parsed as is, and fails as invalid JSON."""
    with pytest.raises(ResponseParsingError, match="invalid JSON"):
        parse_and_validate("I could not determine the dialect.", "m")


def test_column_names_are_stripped_and_blank_and_duplicate_names_kept() -> None:
    """Surrounding whitespace goes; an empty name and a repeated name stay (issue #129)."""
    payload = {**VALID_RESULT_PAYLOAD, "columns": ["", " id ", "id", "\tvalue"]}

    result = _ModelAnswer.model_validate(payload)

    assert result.columns == ["", "id", "id", "value"]


@pytest.mark.parametrize(
    "columns",
    [[], "Fecha", [{"name": "Fecha", "inferred_type": "date"}], [1, 2], None],
    ids=["empty", "string", "objects", "numbers", "null"],
)
def test_columns_must_be_a_non_empty_list_of_names(columns: object) -> None:
    """Anything but a non-empty list of strings is a malformed answer (issue #129)."""
    with pytest.raises(ValidationError):
        _ModelAnswer.model_validate({**VALID_RESULT_PAYLOAD, "columns": columns})


@pytest.mark.parametrize(
    ("value", "expected"),
    [("\t", "\t"), ("\\t", "\t"), ("tab", "\t"), ("TAB", "\t"), (";", ";")],
)
def test_dialect_characters_accept_common_spellings_of_tab(value: str, expected: str) -> None:
    """A tab written as an escape sequence or a word becomes a real tab (issue #11)."""
    result = _ModelAnswer.model_validate({**VALID_RESULT_PAYLOAD, "delimiter": value})

    assert result.delimiter == expected


@pytest.mark.parametrize("value", ["", "null", "None", None])
def test_an_empty_escapechar_means_none(value: str | None) -> None:
    """``""``, ``"null"`` and ``"none"`` mean there is no escape character (issue #11)."""
    result = _ModelAnswer.model_validate({**VALID_RESULT_PAYLOAD, "escapechar": value})

    assert result.escapechar is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("delimiter", ""),
        ("delimiter", "null"),
        ("delimiter", ";;"),
        ("delimiter", "comma"),
        ("quotechar", "''"),
        ("escapechar", "\\\\"),
    ],
)
def test_dialect_characters_must_be_one_character(field: str, value: str) -> None:
    """Anything else that is not one character fails validation (issue #11)."""
    with pytest.raises(ValidationError, match="exactly one character"):
        _ModelAnswer.model_validate({**VALID_RESULT_PAYLOAD, field: value})


@pytest.mark.parametrize("value", ["", "null", "None", "NONE", " none ", None])
def test_no_quoting_spellings_map_quotechar_to_default(value: str | None) -> None:
    """Unquoted files: empty/null quotechar answers keep the inert default (issue #93)."""
    result = _ModelAnswer.model_validate({**VALID_RESULT_PAYLOAD, "quotechar": value})
    assert result.quotechar == '"'


@pytest.mark.parametrize(
    "dialect",
    [
        {"delimiter": ",", "quotechar": ","},
        {"delimiter": ",", "escapechar": ","},
        {"quotechar": "\n"},
        {"quotechar": "\r"},
    ],
)
def test_a_dialect_csv_cannot_read_fails_validation(dialect: dict[str, str]) -> None:
    """Characters that are valid alone but conflict fail validation (issue #48)."""
    with pytest.raises(ValidationError, match=r"line break|must differ"):
        _ModelAnswer.model_validate({**VALID_RESULT_PAYLOAD, **dialect})


@pytest.mark.parametrize("value", ["\n", "\r\n"])
def test_a_line_break_delimiter_in_the_answer_means_one_column(value: str) -> None:
    """A model reading a one-column file as one field per line answers a line break."""
    answer = _ModelAnswer.model_validate({**VALID_RESULT_PAYLOAD, "delimiter": value})

    assert answer.delimiter == ","
    with pytest.raises(ValidationError, match="line break"):
        CSVInspectionResult.model_validate({**VALID_RESULT_PAYLOAD, "delimiter": "\n"})


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        ("--- Fin ---\nGenerado el 2024-08-08\n\n", "--- Fin ---"),
        ("\r\n\r\nTOTAL;;1", "TOTAL;;1"),
        ("\n\n", ""),
        ("TOTAL;;1", "TOTAL;;1"),
    ],
)
def test_a_multi_line_footer_anchor_keeps_its_first_non_blank_line(
    anchor: str, expected: str
) -> None:
    """Asked for one line, a model that copies the whole footer still anchors on its first."""
    answer = _ModelAnswer.model_validate({**VALID_RESULT_PAYLOAD, "footer_first_line": anchor})

    assert answer.footer_first_line == expected


def test_an_escapechar_equal_to_the_quotechar_means_doubled_quotes() -> None:
    """``escapechar='"'`` describes RFC 4180 doubled quotes (issue #48)."""
    payload = {**VALID_RESULT_PAYLOAD, "escapechar": '"', "doublequote": False}

    result = _ModelAnswer.model_validate(payload)

    assert result.escapechar is None
    assert result.doublequote is True


def test_a_pre_0_4_footer_lines_answer_anchors_no_footer() -> None:
    """Since 0.6 a ``footer_lines`` list is ignored: only ``footer_first_line`` anchors a footer."""
    payload = {k: v for k, v in VALID_RESULT_PAYLOAD.items() if k != "footer_first_line"}

    answer = _ModelAnswer.model_validate({**payload, "footer_lines": ["", "TOTAL;;60.00"]})

    assert answer.footer_first_line is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"delimiter": "tab"},
        {"escapechar": ""},
        {"has_header": False, "header_row_index": -1},
        {"escapechar": '"'},
    ],
    ids=["tab-spelling", "empty-escape", "minus-one", "escaped-quote"],
)
def test_the_public_result_does_not_accept_model_spellings(overrides: dict[str, object]) -> None:
    """Leniency is for the model's answer; the result the library returns is strict."""
    payload = {**VALID_RESULT_PAYLOAD, "footer_lines": [], **overrides}

    _ModelAnswer.model_validate(payload)
    with pytest.raises(ValidationError):
        CSVInspectionResult.model_validate(payload)


def test_model_answer_is_private() -> None:
    """``_ModelAnswer`` is an internal type, never exported."""
    import csv_inspector  # noqa: PLC0415 - checked right here.

    assert "_ModelAnswer" not in csv_inspector.__all__
    assert not hasattr(csv_inspector, "_ModelAnswer")
