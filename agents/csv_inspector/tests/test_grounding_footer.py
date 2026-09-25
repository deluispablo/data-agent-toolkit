"""Tests of grounding the footer: the anchor line, the verbatim footer and its bounds."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from csv_inspector import (
    CSVInspectionResult,
    inspect_csv,
)
from csv_inspector._grounding import (
    _extends_footer,
    _locate_footer_lines,
    _ParsedSample,
    ground_in_samples,
)
from csv_inspector._models import _ModelAnswer
from payloads import _LEDGER, SAMPLE_CSV_PATH, VALID_RESULT_PAYLOAD, _sloppy_answer


def test_grounding_recovers_skipped_footer_lines_verbatim(tmp_path: Path) -> None:
    """Lines after the reported footer and blank separators before it are recovered."""
    target = tmp_path / "ledger.csv"
    target.write_text(_LEDGER, encoding="utf-8")

    result = inspect_csv(target, model_invoker=_sloppy_answer())

    assert result.footer_lines == ["", "TOTAL;;60.00", "--- Fin del informe ---"]
    assert result.footer_rows_to_skip == 3


def test_grounding_reads_the_footer_from_the_tail_of_a_large_file() -> None:
    """On the real head/tail path, the footer is anchored in the tail sample."""
    fixture = SAMPLE_CSV_PATH.parent / "samples" / "header_and_footer_combined.csv"
    totals_row = fixture.read_text(encoding="utf-8").splitlines()[-2]

    # Mirrors what qwen2.5-coder:7b actually returned for this fixture.
    columns = ["Fecha", "Proveedor", "Descripción", "Monto"]

    result = inspect_csv(
        fixture, model_invoker=_sloppy_answer(columns=columns, footer_first_line=totals_row)
    )

    assert result.header_row_index == 2
    assert result.columns == ["Fecha", "Cliente", "Concepto", "Importe"]
    assert result.footer_lines == ["", totals_row, "--- Fin del informe ---"]


def test_grounding_anchors_the_footer_on_its_last_occurrence(tmp_path: Path) -> None:
    """Footer text that also appears in the data must not drag data rows into the footer."""
    target = tmp_path / "repeated.csv"
    target.write_text(
        "Fecha;Cliente;Importe\n2024-01-01;Acme;10.00\nRevisado\n2024-01-02;Beta;20.00\nRevisado\n",
        encoding="utf-8",
    )

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(footer_first_line="Revisado", header_row_index=0),
    )

    assert result.footer_lines == ["Revisado"]


def test_grounding_drops_a_footer_that_is_not_at_the_end_of_the_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A reported footer absent from the sampled end would drop real data rows (issue #52)."""
    target = tmp_path / "plain.csv"
    target.write_text("Fecha;Cliente;Importe\n2024-01-01;Acme;10.00\n", encoding="utf-8")

    result = inspect_csv(target, model_invoker=_sloppy_answer(footer_first_line="*** END ***"))

    assert result.footer_lines == []
    assert result.footer_rows_to_skip == 0
    assert "does not occur at the end of the file" in caplog.text


def test_grounding_drops_an_unanchored_footer_on_the_head_and_tail_path(tmp_path: Path) -> None:
    """The same applies when the end of the file comes from the tail sample (issue #52)."""
    target = tmp_path / "long.csv"
    rows = "".join(f"2024-01-{day % 28 + 1:02d};Cliente {day};{day}.00\n" for day in range(400))
    target.write_text("Fecha;Proveedor;Monto\n" + rows, encoding="utf-8")

    result = inspect_csv(
        target, n_bytes=512, tail_bytes=512, model_invoker=_sloppy_answer(footer_first_line="TOTAL")
    )

    assert result.footer_lines == []


def test_grounding_recovers_an_unreported_totals_row_above_the_footer(tmp_path: Path) -> None:
    """If the model only spots the closing marker, the totals row above it is still found."""
    target = tmp_path / "ledger.csv"
    target.write_text(_LEDGER, encoding="utf-8")

    result = inspect_csv(
        target, model_invoker=_sloppy_answer(footer_first_line="--- Fin del informe ---")
    )

    assert result.footer_lines == ["", "TOTAL;;60.00", "--- Fin del informe ---"]


def test_grounding_never_extends_the_footer_past_a_data_row(tmp_path: Path) -> None:
    """Backward extension stops at the first line that is neither blank nor a totals row."""
    target = tmp_path / "ledger.csv"
    target.write_text(
        "Fecha;Cliente;Importe\n"
        "2024-01-01;Total Care S.L.;10.00\n"
        "2024-01-02;Beta;20.00\n"
        "--- Fin del informe ---\n",
        encoding="utf-8",
    )

    result = inspect_csv(
        target, model_invoker=_sloppy_answer(footer_first_line="--- Fin del informe ---")
    )

    assert result.footer_lines == ["--- Fin del informe ---"]


def test_grounding_drops_a_footer_when_the_end_of_the_file_was_not_sampled(
    tmp_path: Path,
) -> None:
    """With tail sampling disabled, the truncated head's last rows are never a footer.

    Regression test for issue #10: the prompt claimed the head was the entire
    file, the model filed the last sampled rows as footer, and grounding
    copied them into ``footer_lines``, so consumers skipped real data.
    """
    target = tmp_path / "ledger.csv"
    target.write_text("Fecha;Cliente;Importe\n" + "2024-01-01;Acme;10.00\n" * 200, encoding="utf-8")
    prompts: list[str] = []
    answer = _sloppy_answer(footer_first_line="2024-01-01;Acme;10.00")

    def recording_invoker(prompt: str, model: str) -> str:
        prompts.append(prompt)
        return answer(prompt, model)

    result = inspect_csv(target, n_bytes=256, tail_bytes=0, model_invoker=recording_invoker)

    assert "contains the ENTIRE file" not in prompts[0]
    assert "its end was not sampled" in prompts[0]
    assert result.footer_lines == []
    assert result.footer_rows_to_skip == 0


@pytest.mark.parametrize(
    ("line", "delimiter", "expected"),
    [
        ("", ";", True),
        ("TOTAL;;;12.50", ";", True),
        ("Subtotal,,3", ",", True),
        ('"Total general",9', ",", True),
        ("Total registros: 250", ",", True),
        ("SUMA;;;1", ";", True),
        ("TOTAL;120;340;460", ";", True),
        ("Total ventas;;;460", ";", True),
        ("2024-01-01;Acme;10.00", ";", False),
        ("Totalmente nuevo,1,2", ",", False),
        ("Summary report", ",", False),
        ("--- Fin del informe ---", ",", False),
        ("Total Energies,2024-01-01,10.00", ",", False),
        ("Sum Holdings;Madrid;2024-01-01;10.00", ";", False),
    ],
)
def test_extends_footer_accepts_only_blank_and_totals_rows(
    line: str, delimiter: str, expected: bool
) -> None:
    """Only blank separators and totals rows, not data rows named "Total...", extend a footer."""
    assert _extends_footer(line, delimiter, '"') is expected


def test_grounding_keeps_a_data_row_named_like_a_totals_label_out_of_the_footer(
    tmp_path: Path,
) -> None:
    """A data row whose first field starts with "Total" is not pulled into the footer.

    Regression test for issue #20.
    """
    target = tmp_path / "companies.csv"
    target.write_text(
        "Empresa,Fecha,Importe\n"
        "Acme,2024-01-01,10.00\n"
        "Total Energies,2024-01-02,20.00\n"
        "--- Fin del informe ---\n",
        encoding="utf-8",
    )
    columns = ["Empresa", "Fecha", "Importe"]

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(
            delimiter=",", columns=columns, footer_first_line="--- Fin del informe ---"
        ),
    )

    assert result.footer_lines == ["--- Fin del informe ---"]


_MARKED_LEDGER = (
    "Fecha;Cliente;Importe\n"
    "2024-01-01;Acme;10.00\n"
    "2024-01-02;Beta;20.00\n"
    "\n"
    "--- Fin del informe ---\n"
    "Generado el 2024-08-08 10:00:00\n"
)


def test_grounding_drops_data_rows_the_model_reported_as_footer(tmp_path: Path) -> None:
    """A data row is never a footer line: the footer starts past it (issue #153)."""
    target = tmp_path / "ledger.csv"
    target.write_text(_MARKED_LEDGER, encoding="utf-8")

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(footer_first_line="2024-01-02;Beta;20.00"),
    )

    assert result.footer_lines == ["", "--- Fin del informe ---", "Generado el 2024-08-08 10:00:00"]


def test_grounding_discards_a_footer_made_only_of_data_rows(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The last data rows reported as a footer leave no footer at all (issue #153)."""
    target = tmp_path / "ledger.csv"
    target.write_text(
        "Fecha;Cliente;Importe\n2024-01-01;Acme;10.00\n2024-01-02;Beta;20.00\n", encoding="utf-8"
    )

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(footer_first_line="2024-01-01;Acme;10.00"),
    )

    assert result.footer_lines == []
    assert "does not occur at the end of the file" in caplog.text


def test_grounding_includes_a_marker_above_the_anchor(tmp_path: Path) -> None:
    """Non-data lines above the reported footer line belong to the footer (issue #153)."""
    target = tmp_path / "ledger.csv"
    target.write_text(_MARKED_LEDGER, encoding="utf-8")

    result = inspect_csv(
        target, model_invoker=_sloppy_answer(footer_first_line="Generado el 2024-08-08 10:00:00")
    )

    assert result.footer_lines == ["", "--- Fin del informe ---", "Generado el 2024-08-08 10:00:00"]


def test_grounding_keeps_a_totals_row_with_the_data_width(tmp_path: Path) -> None:
    """A fully filled totals row has the data shape but is not a data row (issue #153)."""
    target = tmp_path / "ledger.csv"
    target.write_text(
        "Fecha;Cliente;Importe\n2024-01-01;Acme;10.00\n2024-01-02;Beta;20.00\nTOTAL;2;30.00\n",
        encoding="utf-8",
    )

    result = inspect_csv(target, model_invoker=_sloppy_answer(footer_first_line="TOTAL;2;30.00"))

    assert result.footer_lines == ["TOTAL;2;30.00"]


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        (
            "2024-08-08 10:00:00",
            ["", "--- Fin del informe ---", "Generado el 2024-08-08 10:00:00"],
        ),
        ("Fin del", []),
    ],
)
def test_grounding_anchors_on_a_substring_of_the_footer_line(
    tmp_path: Path, reported: str, expected: list[str]
) -> None:
    """Reported text of 8 characters or more anchors the line it occurs in (issue #153)."""
    target = tmp_path / "ledger.csv"
    target.write_text(_MARKED_LEDGER, encoding="utf-8")

    result = inspect_csv(target, model_invoker=_sloppy_answer(footer_first_line=reported))

    assert result.footer_lines == expected


def test_grounding_ignores_trailing_empty_fields_in_an_anchor(tmp_path: Path) -> None:
    """A totals row copied with one trailing delimiter too few still anchors (issue #153)."""
    target = tmp_path / "ledger.csv"
    target.write_text(
        "Fecha;Cliente;Importe;Nota;Extra\n"
        "2024-01-01;Acme;10.00;a;b\n"
        "2024-01-02;Beta;20.00;c;d\n"
        "TOTAL;;30.00;;\n",
        encoding="utf-8",
    )

    result = inspect_csv(target, model_invoker=_sloppy_answer(footer_first_line="TOTAL;;30.00;"))

    assert result.footer_lines == ["TOTAL;;30.00;;"]


def test_grounding_starts_a_footer_after_the_last_data_row(tmp_path: Path) -> None:
    """A ragged data row the model points at, with a full row after it, is no footer."""
    target = tmp_path / "ledger.csv"
    target.write_text(
        "Fecha;Proveedor;Monto\n"
        "2024-01-01;Acme;10.00\n"
        "2024-01-02;Beta;20.00\n"
        "2024-01-03;Gamma\n"
        "2024-01-04;Delta;40.00\n",
        encoding="utf-8",
    )

    result = inspect_csv(target, model_invoker=_sloppy_answer(footer_first_line="2024-01-03;Gamma"))

    assert result.footer_lines == []


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (_MARKED_LEDGER, ["", "--- Fin del informe ---", "Generado el 2024-08-08 10:00:00"]),
        ("Fecha;Proveedor;Monto\n2024-01-01;Acme;10.00\n2024-01-02;Beta;20.00\n", []),
    ],
)
def test_grounding_reads_an_invented_data_row_as_the_end_of_the_data(
    tmp_path: Path, content: str, expected: list[str]
) -> None:
    """A made-up last data row anchors after the data: the footer if any, else none."""
    target = tmp_path / "ledger.csv"
    target.write_text(content, encoding="utf-8")

    result = inspect_csv(
        target, model_invoker=_sloppy_answer(footer_first_line="2024-01-09;Omega;99.00")
    )

    assert result.footer_lines == expected


def test_grounding_anchors_a_footer_copied_with_the_replaced_delimiter(tmp_path: Path) -> None:
    """The model copies the footer with its own delimiter; the grounded one still finds it."""
    target = tmp_path / "ledger.tsv"
    target.write_text(
        "Fecha\tProveedor\tMonto\n"
        "2024-01-01\tAcme\t10.00\n"
        "2024-01-02\tBeta\t20.00\n"
        "TOTAL\t\t30.00\n",
        encoding="utf-8",
    )

    result = inspect_csv(
        target, model_invoker=_sloppy_answer(delimiter=",", footer_first_line="TOTAL,,30.00")
    )

    assert result.delimiter == "\t"
    assert result.footer_lines == ["TOTAL\t\t30.00"]


def test_grounding_anchors_a_data_row_copied_with_spaces_for_the_delimiter(
    tmp_path: Path,
) -> None:
    """Separators squashed to spaces still designate the last data row."""
    target = tmp_path / "ledger.tsv"
    target.write_text(
        "Fecha\tProveedor\tMonto\n"
        "2024-01-01\tAcme\t10.00\n"
        "2024-01-02\tBeta S.L.\t20.00\n"
        "--- Fin del informe ---\n",
        encoding="utf-8",
    )

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(
            delimiter="\t", footer_first_line="2024-01-02 Beta S.L. 20.00"
        ),
    )

    assert result.footer_lines == ["--- Fin del informe ---"]


def _answer(**overrides: object) -> _ModelAnswer:
    """A validated model answer for the ``_MARKED_LEDGER`` file."""
    return _ModelAnswer.model_validate(
        {
            **VALID_RESULT_PAYLOAD,
            "header_row_index": 0,
            "columns": ["Fecha", "Cliente", "Importe"],
            **overrides,
        }
    )


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        pytest.param(
            "--- Fin del informe ---",
            ["", "--- Fin del informe ---", "Generado el 2024-08-08 10:00:00"],
            id="first-footer-line",
        ),
        pytest.param(
            "Generado el 2024-08-08 10:00:00",
            ["", "--- Fin del informe ---", "Generado el 2024-08-08 10:00:00"],
            id="last-footer-line",
        ),
        pytest.param(
            "2024-01-02;Beta;20.00",
            ["", "--- Fin del informe ---", "Generado el 2024-08-08 10:00:00"],
            id="last-data-row",
        ),
        pytest.param(
            "",
            ["", "--- Fin del informe ---", "Generado el 2024-08-08 10:00:00"],
            id="blank-separator",
        ),
        pytest.param("*** END ***", None, id="not-found"),
    ],
)
def test_locate_footer_lines_reads_the_footer_from_one_anchor(
    anchor: str, expected: list[str] | None
) -> None:
    """One anchor line is enough: the whole footer is read from the file."""
    assert _locate_footer_lines(anchor, _ParsedSample.parse(_MARKED_LEDGER, ";", '"')) == expected


def test_locate_footer_lines_finds_nothing_after_the_last_data_row() -> None:
    """A blank anchor over a file that ends with a data row anchors nothing."""
    assert _locate_footer_lines("", _ParsedSample.parse("a;b\n1;x\n2;y\n", ";", '"')) is None


def test_no_footer_line_with_the_end_sampled_gives_no_footer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``footer_first_line`` null means no footer, silently, even when the file has one."""
    with caplog.at_level(logging.WARNING):
        result = ground_in_samples(_answer(footer_first_line=None), _MARKED_LEDGER, None)

    assert result.footer_lines == []
    assert caplog.records == []


def test_an_unknown_footer_line_is_discarded_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An anchor that is not at the sampled end of the file leaves no footer, and says so."""
    with caplog.at_level(logging.WARNING, logger="csv_inspector"):
        result = ground_in_samples(_answer(footer_first_line="*** END ***"), _MARKED_LEDGER, None)

    assert result.footer_lines == []
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert "*** END ***" in record.getMessage()


def test_grounding_builds_the_public_result_from_the_answer() -> None:
    """The pipeline's only result constructor: an answer in, a CSVInspectionResult out."""
    answer = _answer(footer_first_line="--- Fin del informe ---")

    result = ground_in_samples(answer, _MARKED_LEDGER, None)

    assert type(result) is CSVInspectionResult
    assert result.footer_rows_to_skip == 3
    assert "footer_first_line" not in result.model_dump()
    assert result.model_dump().keys() == {
        "encoding",
        "delimiter",
        "quotechar",
        "escapechar",
        "doublequote",
        "has_header",
        "header_row_index",
        "footer_lines",
        "columns",
        "confidence",
        "footer_rows_to_skip",
    }
