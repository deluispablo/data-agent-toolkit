"""Tests of grounding the header: header row, literal column names and header-less files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from csv_inspector import (
    CSVInspectionResult,
    Settings,
    inspect_csv,
)
from csv_inspector._grounding import (
    _field_shape,
    _first_row_is_data,
    ground_in_samples,
)
from csv_inspector._models import _ModelAnswer
from csv_inspector._prompt import build_prompt
from fakes import install_fake_ollama, ollama_reply
from generate_samples import CASES, SampleCase, derive_columns
from matrix import MATRIX, render
from payloads import _LEDGER, SAMPLE_CSV_PATH, VALID_RESULT_PAYLOAD, _sloppy_answer


def test_a_header_less_fixture_keeps_its_positional_columns() -> None:
    """End to end: grounding never renames positional columns or invents a header."""
    fixture = SAMPLE_CSV_PATH.parent / "samples" / "header_none_data_only.csv"
    answer = {
        **VALID_RESULT_PAYLOAD,
        "delimiter": ",",
        "has_header": False,
        "header_row_index": None,
        "columns": ["column_1", "column_2", "column_3"],
    }

    result = inspect_csv(fixture, model_invoker=lambda prompt, model: json.dumps(answer))

    assert result.has_header is False
    assert result.header_row_index is None
    assert result.columns == ["column_1", "column_2", "column_3"]
    assert '"has_header"' in build_prompt("a,b\n", "utf-8")


def test_grounding_detects_a_first_row_shaped_like_data() -> None:
    """A model that invents names for a data row is corrected to no header (#94, #131)."""
    fixture = SAMPLE_CSV_PATH.parent / "samples" / "header_none_data_only.csv"
    answer = {
        **VALID_RESULT_PAYLOAD,
        "delimiter": ",",
        "header_row_index": 0,
        "columns": ["date", "name", "amount"],
    }

    result = inspect_csv(fixture, model_invoker=lambda prompt, model: json.dumps(answer))

    assert (result.has_header, result.header_row_index) == (False, None)
    assert result.columns == ["column_1", "column_2", "column_3"]


def test_grounding_keeps_a_header_whose_names_are_not_examples(tmp_path: Path) -> None:
    """A real header row with paraphrased names is never taken for data."""
    target = tmp_path / "plain.csv"
    target.write_text("Fecha,Cliente,Importe\n2024-01-15,Acme,10\n", encoding="utf-8")
    answer = {
        **VALID_RESULT_PAYLOAD,
        "delimiter": ",",
        "header_row_index": 0,
        "columns": [
            "date",
            "client",
            "amount",
        ],
    }

    result = inspect_csv(target, model_invoker=lambda prompt, model: json.dumps(answer))

    assert (result.has_header, result.header_row_index) == (True, 0)


@pytest.mark.parametrize(
    ("value", "shape"),
    [
        ("", "empty"),
        ("  ", "empty"),
        ("42", "integer"),
        (" -7 ", "integer"),
        ("2023", "integer"),
        ("1250.50", "decimal"),
        ("1250,50", "decimal"),
        ("1.234,56", "decimal"),
        ("-0.5", "decimal"),
        ("2024-01-15", "date"),
        ("2024-01-15 08:30:00", "date"),
        ("15/01/2024", "date"),
        ("15.01.2024", "date"),
        ("1-2-24", "date"),
        ("Acme S.L.", "text"),
        ("A-0042", "text"),
        ("Importe", "text"),
    ],
)
def test_field_shape_classifies_a_field(value: str, shape: str) -> None:
    """Each field has exactly one shape: integer, decimal, date, empty or text."""
    assert _field_shape(value) == shape


def _ground_case(case: SampleCase, names: list[str]) -> CSVInspectionResult:
    """Ground a model answer claiming a header at row 0 with ``names`` in a fixture."""
    encoding = case.expected["encoding"]
    answer = {
        **VALID_RESULT_PAYLOAD,
        "encoding": encoding,
        "delimiter": case.expected["delimiter"],
        "header_row_index": 0,
        "columns": list(names),
    }
    head = case.raw_bytes.decode(encoding)
    return ground_in_samples(_ModelAnswer.model_validate(answer), head, None)


_HEADERLESS_CASES = [
    *(case for case in CASES if case.filename == "header_none_data_only.csv"),
    *(render(spec) for spec in MATRIX if spec.slug in {"headerless_narrow", "headerless_marker"}),
]


@pytest.mark.parametrize("case", _HEADERLESS_CASES, ids=lambda case: case.filename)
def test_grounding_reports_no_header_for_header_less_fixtures(case: SampleCase) -> None:
    """Invented names over a header-less fixture become positional columns (#131)."""
    width = len(derive_columns(case) or [])
    assert len(_HEADERLESS_CASES) == 3
    assert width >= 3

    result = _ground_case(case, [f"invented_{n}" for n in range(width)])

    assert (result.has_header, result.header_row_index) == (False, None)
    assert result.columns == derive_columns(case)


def test_grounding_keeps_an_all_text_header_it_cannot_anchor() -> None:
    """A header of names above typed data is kept, even with paraphrased names (#131)."""
    spec = next(
        spec
        for spec in MATRIX
        if spec.has_header and spec.preamble_lines == 0 and spec.encoding == "utf-8"
    )
    case = render(spec)
    width = len(derive_columns(case) or [])

    result = _ground_case(case, [f"paraphrased_{n}" for n in range(width)])

    assert (result.has_header, result.header_row_index) == (True, 0)


@pytest.mark.parametrize("names", [["2023", "2024", "2025"], ["y1", "y2", "y3"]])
def test_grounding_keeps_a_header_of_years(names: list[str]) -> None:
    """Integer years above decimal data stay a header, anchored or not (#131)."""
    case = next(case for case in CASES if case.filename == "header_years.csv")

    result = _ground_case(case, names)

    assert (result.has_header, result.header_row_index) == (True, 0)
    assert result.columns == names


def test_grounding_keeps_a_header_named_by_the_model_in_another_case() -> None:
    """A row-0 field equal to a model's name, case aside, keeps the header (#131)."""
    answer = {
        **VALID_RESULT_PAYLOAD,
        "delimiter": ",",
        "header_row_index": 0,
        "columns": [
            "fecha",
            "concepto",
        ],
    }
    head = "Fecha,Cliente\nAcme,Beta\nGamma,Delta\n"

    result = ground_in_samples(_ModelAnswer.model_validate(answer), head, None)

    assert (result.has_header, result.header_row_index) == (True, 0)


@pytest.mark.parametrize(
    ("head", "is_data"),
    [
        pytest.param("2024-01-15,Acme,10.5\n", False, id="one-line"),
        pytest.param("2024-01-15,Acme,10.5\n2024-01-16,Beta\n", False, id="ragged-second-row"),
        pytest.param("Fecha,Cliente,Importe\n2024-01-15,Acme,10.5\n", False, id="text-over-data"),
        pytest.param("2024-01-15,,10.5\n2024-01-16,Beta,3.5\n", True, id="empty-cell-matches"),
        pytest.param("2024-01-15,Acme,Norte\n2024-01-16,Beta,\n", False, id="mostly-text"),
    ],
)
def test_first_row_is_data_needs_a_second_row_of_the_same_shape(head: str, is_data: bool) -> None:
    """Only two rows of equal width and agreeing shapes flag row 0 as data."""
    answer = {
        **VALID_RESULT_PAYLOAD,
        "delimiter": ",",
        "header_row_index": 0,
        "columns": ["date", "client", "amount"],
    }
    result = CSVInspectionResult.model_validate(answer)

    assert _first_row_is_data(result, head) is is_data


def test_grounding_recovers_header_row_and_literal_column_names(tmp_path: Path) -> None:
    """A miscounted preamble and paraphrased names are corrected from the head."""
    target = tmp_path / "ledger.csv"
    target.write_text(_LEDGER, encoding="utf-8")

    result = inspect_csv(target, model_invoker=_sloppy_answer())

    assert result.header_row_index == 2
    assert result.columns == ["Fecha", "Cliente", "Importe"]


def test_grounded_column_names_keep_padding_like_csv_reader(tmp_path: Path) -> None:
    """Names match what csv/pandas read: surrounding spaces are kept, not stripped."""
    target = tmp_path / "padded.csv"
    target.write_text("Fecha; Cliente ;Importe\n2024-01-01;Acme;10.00\n", encoding="utf-8")
    columns = ["Fecha", "Cliente", "Importe"]

    result = inspect_csv(
        target, model_invoker=_sloppy_answer(columns=columns, footer_first_line=None)
    )

    assert result.columns == ["Fecha", " Cliente ", "Importe"]


def test_grounding_leaves_a_header_less_file_alone(tmp_path: Path) -> None:
    """Invented names that share nothing with the data never promote a data row to header.

    The first row has the shape of the second, so the header the model
    claimed at row 0 is dropped and the columns become positional (#131).
    """
    target = tmp_path / "no_header.csv"
    target.write_text("2024-01-01;Acme;10.00\n2024-01-02;Beta;20.00\n", encoding="utf-8")
    columns = [
        "col_1",
        "col_2",
        "col_3",
    ]

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(columns=columns, header_row_index=0, footer_first_line=None),
    )

    assert (result.has_header, result.header_row_index) == (False, None)
    assert result.columns == ["column_1", "column_2", "column_3"]
    assert result.footer_lines == []


def test_a_forty_column_answer_validates_and_grounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A wide file's answer fits the Ollama reply cap and is grounded (issues #129, #147).

    On 0.3.0 the per-column objects of a 40-column file ran past
    ``num_predict`` and every inspection failed with truncated JSON.
    """
    names = [f"Campo {number}" for number in range(1, 41)]
    row = ";".join(str(number) for number in range(40))
    target = tmp_path / "wide.csv"
    target.write_text(
        "# Export ERP\n" + ";".join(names) + "\n" + f"{row}\n" * 5 + "\nTOTAL;;\n",
        encoding="utf-8",
    )
    answer = json.dumps(
        {
            **VALID_RESULT_PAYLOAD,
            "header_row_index": 0,
            "columns": [
                name.replace("Campo", "Field") if number % 2 else name
                for number, name in enumerate(names)
            ],
            "footer_first_line": "TOTAL;;",
        }
    )
    fake = install_fake_ollama(monkeypatch, lambda **kwargs: ollama_reply(answer))

    result = inspect_csv(target, settings=Settings(), model="big", fallback_model="small")

    assert [request["model"] for request in fake.requests] == ["big"]
    # Even at a pessimistic 3 characters per token, the answer fits the cap.
    assert len(answer) / 3 < fake.requests[0]["options"]["num_predict"]
    assert result.header_row_index == 1
    assert result.columns == names
    assert result.footer_lines == ["", "TOTAL;;"]


def test_grounding_names_header_less_columns_positionally(tmp_path: Path) -> None:
    """A model that says "no header" but invents names gets column_1..N."""
    target = tmp_path / "rows.csv"
    target.write_text("2024-01-01,Acme,10.00\n2024-01-02,Beta,20.00\n", encoding="utf-8")

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(
            delimiter=",",
            has_header=False,
            header_row_index=None,
            columns=["Date", "Company", "Amount"],
            footer_first_line=None,
        ),
    )

    assert result.columns == ["column_1", "column_2", "column_3"]


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["Fecha", "Cliente", "Importe"], (True, 0, ["Fecha", "Cliente", "Importe"])),
        (["2024-01-01", "Acme", "10.00"], (False, None, ["column_1", "column_2", "column_3"])),
    ],
)
def test_grounding_finds_the_header_a_no_header_answer_named(
    tmp_path: Path, names: list[str], expected: tuple[bool, int | None, list[str]]
) -> None:
    """Names equal to a line above differently shaped data are a header; data values are not."""
    target = tmp_path / "ledger.csv"
    target.write_text(
        "Fecha,Cliente,Importe\n2024-01-01,Acme,10.00\n2024-01-02,Beta,20.00\n", encoding="utf-8"
    )
    if names[0] == "2024-01-01":
        target.write_text("2024-01-01,Acme,10.00\n2024-01-02,Beta,20.00\n", encoding="utf-8")

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(
            delimiter=",",
            has_header=False,
            header_row_index=None,
            columns=names,
            footer_first_line=None,
        ),
    )

    assert (result.has_header, result.header_row_index, result.columns) == expected


@pytest.mark.parametrize("has_header", [True, False])
def test_grounding_finds_the_header_of_a_one_column_file_listed_line_by_line(
    tmp_path: Path, has_header: bool
) -> None:
    """Every line listed as a column, header or not: the line of the first name is the header."""
    target = tmp_path / "names.csv"
    target.write_text("Cliente\nAcme S.L.\nBeta Corp\n", encoding="utf-8")

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(
            delimiter=",",
            has_header=has_header,
            header_row_index=0 if has_header else None,
            columns=["Cliente", "Acme S.L.", "Beta Corp"],
            footer_first_line=None,
        ),
    )

    assert (result.has_header, result.header_row_index, result.columns) == (True, 0, ["Cliente"])


def test_grounding_takes_a_header_only_file_for_a_header(tmp_path: Path) -> None:
    """A "no header" answer naming the file's only line: that line is the header."""
    target = tmp_path / "empty_table.csv"
    target.write_text("Fecha,Cliente,Importe\n", encoding="utf-8")

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(
            delimiter=",",
            has_header=False,
            header_row_index=None,
            columns=["Fecha", "Cliente", "Importe"],
            footer_first_line=None,
        ),
    )

    assert (result.has_header, result.header_row_index) == (True, 0)


def test_grounding_keeps_one_name_in_a_one_column_file(tmp_path: Path) -> None:
    """A model listing every line as a column of a one-column file gets the header only."""
    target = tmp_path / "names.csv"
    target.write_text("Cliente\nAcme S.L.\nBeta Corp\n", encoding="utf-8")

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(
            delimiter="\n", columns=["Cliente", "Acme S.L.", "Beta Corp"], footer_first_line=None
        ),
    )

    assert (result.delimiter, result.columns) == (",", ["Cliente"])


def test_grounding_anchors_a_header_with_a_blank_name(tmp_path: Path) -> None:
    """A blank name the model left out is restored from the header row (issue #153)."""
    target = tmp_path / "indexed.csv"
    target.write_text(",id,id,value\n0,1,2,a\n1,3,4,b\n", encoding="utf-8")
    columns = ["id", "id", "value"]

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(
            delimiter=",", columns=columns, header_row_index=1, footer_first_line=None
        ),
    )

    assert result.header_row_index == 0
    assert result.columns == ["", "id", "id", "value"]


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        # "2024,,4241.25": a numeric label, so a data row, never a footer.
        ("footer_like_data_row_numeric_label.csv", []),
        ("footer_summary_totals.csv", ["TOTAL,,,274887.10"]),
    ],
)
def test_grounding_of_totals_shaped_last_rows_in_the_catalog(
    fixture: str, expected: list[str]
) -> None:
    """Totals-shaped data rows are dropped and real totals rows kept (issue #153)."""
    lines = (SAMPLE_CSV_PATH.parent / "samples" / fixture).read_text(encoding="utf-8").splitlines()
    columns = lines[0].split(",")

    result = inspect_csv(
        SAMPLE_CSV_PATH.parent / "samples" / fixture,
        model_invoker=_sloppy_answer(delimiter=",", columns=columns, footer_first_line=lines[-1]),
    )

    assert result.footer_lines == expected


def test_grounding_counts_lines_like_csv_does(tmp_path: Path) -> None:
    """Characters str.splitlines() breaks on, but csv does not, never shift line indexes.

    Regression test for issue #16: stray form feeds and U+2028 in fields made
    the preamble look longer, and the blank "line" after a data row's form
    feed was taken as a footer separator, so a data row would be skipped.
    """
    target = tmp_path / "dirty.csv"
    target.write_text(
        "# Export\x0cv2\n"
        "Fecha;Cliente;Importe\n"
        "2024-01-01;Acme\u2028S.L.;10.00\n"
        "2024-01-02;Beta;20.00\x0c\n"
        "--- Fin del informe ---\n",
        encoding="utf-8",
    )

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(footer_first_line="--- Fin del informe ---"),
    )

    assert result.header_row_index == 1
    assert result.footer_lines == ["--- Fin del informe ---"]


def test_header_fallback_ignores_empty_names(tmp_path: Path) -> None:
    """An unnamed column never makes a data row with an empty cell look like the header.

    Regression test for issue #21: with columns ``["", "a", "b"]`` (a pandas
    index), the paraphrase fallback matched the first same-width data row
    that had an empty cell.
    """
    target = tmp_path / "index.csv"
    target.write_text("# Export\n,1,\n,2,3\n,a,b\n0,4,5\n1,6,7\n", encoding="utf-8")
    columns = ["", "x", "b"]

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(
            delimiter=",", columns=columns, header_row_index=0, footer_first_line=None
        ),
    )

    assert result.header_row_index == 3
    assert result.columns == ["", "a", "b"]
