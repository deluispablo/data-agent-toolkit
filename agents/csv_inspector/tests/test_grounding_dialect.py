"""Tests of grounding the dialect: delimiter, quote character, escaping and encoding."""

from __future__ import annotations

import codecs
from pathlib import Path

import pytest

from csv_inspector import (
    inspect_csv,
)
from csv_inspector._grounding import (
    _has_quoted_field,
)
from payloads import _LEDGER, _sloppy_answer


def test_grounding_replaces_a_delimiter_that_never_occurs(tmp_path: Path) -> None:
    """A ``,`` answer for a tab-separated file is sniffed from the head (issue #53)."""
    target = tmp_path / "ledger.tsv"
    target.write_text(_LEDGER.replace(";", "\t"), encoding="utf-8")

    result = inspect_csv(target, model_invoker=_sloppy_answer(delimiter=",", header_row_index=0))

    assert result.delimiter == "\t"
    # Header grounding ran with the grounded delimiter.
    assert result.columns == ["Fecha", "Cliente", "Importe"]
    assert result.header_row_index == 2


def test_grounding_keeps_a_delimiter_that_occurs(tmp_path: Path) -> None:
    """A delimiter found in the head is never second-guessed (issue #53)."""
    target = tmp_path / "ledger.csv"
    target.write_text(_LEDGER.replace(";", "\t", 1), encoding="utf-8")

    result = inspect_csv(target, model_invoker=_sloppy_answer())

    assert result.delimiter == ";"


def test_grounding_keeps_the_delimiter_of_a_one_column_file(tmp_path: Path) -> None:
    """With nothing to sniff, the model's delimiter is kept (issue #53)."""
    target = tmp_path / "ids.csv"
    target.write_text("id\n1\n2\n3\n", encoding="utf-8")

    result = inspect_csv(
        target, model_invoker=_sloppy_answer(delimiter=",", footer_first_line=None)
    )

    assert result.delimiter == ","


def test_grounding_replaces_a_delimiter_that_occurs_only_inside_a_value(
    tmp_path: Path,
) -> None:
    """One comma inside a TSV value does not make ``,`` the delimiter (issue #97)."""
    target = tmp_path / "ledger.tsv"
    target.write_text(_LEDGER.replace(";", "\t").replace("Acme", "Smith, John"), encoding="utf-8")

    result = inspect_csv(target, model_invoker=_sloppy_answer(delimiter=",", header_row_index=0))

    assert result.delimiter == "\t"
    assert result.columns == ["Fecha", "Cliente", "Importe"]


def test_grounding_replaces_a_delimiter_dominated_by_another(tmp_path: Path) -> None:
    """A ``,`` that splits a few TSV rows evenly still loses to the tab (#151)."""
    rows = "".join(
        f"2024-01-{day:02d}\tFernández, Asociados\t{day},50\n"
        if day % 4 == 0
        else f"2024-01-{day:02d}\tAcme\t{day}.00\n"
        for day in range(1, 21)
    )
    target = tmp_path / "ledger.tsv"
    target.write_text("Fecha\tCliente\tImporte\n" + rows, encoding="utf-8")

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(delimiter=",", header_row_index=0, footer_first_line=None),
    )

    assert result.delimiter == "\t"
    assert result.columns == ["Fecha", "Cliente", "Importe"]


def test_grounding_keeps_a_delimiter_that_is_not_clearly_dominated(tmp_path: Path) -> None:
    """A ``,`` splitting more than two thirds as many rows as the tab stays (#151)."""
    rows = "".join(
        f"2024-01-{day:02d}\tFernández, Asociados\t{day},50\n"
        if day % 5 != 0
        else f"2024-01-{day:02d}\tAcme\t{day}.00\n"
        for day in range(1, 21)
    )
    target = tmp_path / "ledger.tsv"
    target.write_text("Fecha\tCliente\tImporte\n" + rows, encoding="utf-8")

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(delimiter=",", header_row_index=0, footer_first_line=None),
    )

    assert result.delimiter == ","


def test_grounding_keeps_a_reported_delimiter_that_splits_the_rows(tmp_path: Path) -> None:
    """``;`` that really separates the fields stays, even with commas in values (#97)."""
    target = tmp_path / "ledger.csv"
    target.write_text(_LEDGER.replace("10.00", "10,00").replace("20.00", "20,00"), encoding="utf-8")

    result = inspect_csv(target, model_invoker=_sloppy_answer())

    assert result.delimiter == ";"


def test_grounding_keeps_the_reported_delimiter_when_candidates_tie(tmp_path: Path) -> None:
    """With no single clear winner, the model's answer is kept (#97)."""
    target = tmp_path / "ambiguous.csv"
    target.write_text("a,b;c|d\n1,2;3\n4,5;6\n7,8;9\n", encoding="utf-8")

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(delimiter="|", footer_first_line=None, columns=["a"]),
    )

    assert result.delimiter == "|"


def test_grounding_replaces_a_delimiter_dominated_one_and_a_half_times(tmp_path: Path) -> None:
    """Tab on 8 lines against ``,`` on 5, as in the 40-column fixtures: tab wins (#151)."""
    rows = "".join(
        f"2024-01-{day:02d}\tFernández, Asociados\t{day}.50\n"
        if day <= 5
        else f"2024-01-{day:02d}\tAcme\t{day}.00\n"
        for day in range(1, 8)
    )
    target = tmp_path / "ledger.tsv"
    target.write_text("Fecha\tCliente\tImporte\n" + rows, encoding="utf-8")

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(delimiter=",", header_row_index=0, footer_first_line=None),
    )

    assert result.delimiter == "\t"


@pytest.mark.parametrize(
    ("text", "delimiter", "quote", "found"),
    [
        ('Id\tNombre\r\n1\t"Arandela, zinc"\r\n', "\t", '"', True),
        ('"Id"|"Nombre"\n', "|", '"', True),
        ("Id;Nombre\n1;'O Grove'\n", ";", "'", True),
        ('Id,Nota\n1,dice "hola" ya\n', ",", '"', False),
        ('Id,Nota\n1,"abre\n2,cierra"\n', ",", '"', False),
        ("Id,Nota\n1,sin comillas\n", ",", '"', False),
    ],
)
def test_a_quoted_field_opens_and_closes_at_field_edges(
    text: str, delimiter: str, quote: str, found: bool
) -> None:
    """Only a whole field in quotes counts, whatever the delimiter, quote or line break."""
    assert _has_quoted_field(text, delimiter, quote) is found


@pytest.mark.parametrize(
    ("note", "answered", "expected"),
    [
        ('"dijo \\"hola\\" ya"', (None, True), ("\\", False)),
        ('"fin \\"hola\\""', (None, True), ("\\", False)),
        ('"dijo ""hola"" ya"', ("\\", False), (None, True)),
        ('"sin comillas"', ("\\", False), (None, False)),
        ('"Arandela, zinc"', ("\\", True), (None, True)),
        ("sin comillas", ("\\", False), ("\\", False)),
        ('dice "hola"', ("\\", True), ("\\", True)),
        ('"mezcla \\"a\\" y ""b"""', (None, True), (None, True)),
    ],
)
def test_grounding_reads_quote_escaping_from_the_samples(
    tmp_path: Path,
    note: str,
    answered: tuple[str | None, bool],
    expected: tuple[str | None, bool],
) -> None:
    """One escaping convention in the samples overrides the answer; both keep it.

    With neither, a quoted field means no escape character (issue #158); an
    unquoted sample, or a quote inside an unquoted field, keeps the answer.
    """
    target = tmp_path / "notes.csv"
    target.write_text(
        f"Fecha,Cliente,Nota\n2024-01-01,Acme,{note}\n2024-01-02,Beta,x\n", encoding="utf-8"
    )

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(
            delimiter=",",
            columns=["Fecha", "Cliente", "Nota"],
            escapechar=answered[0],
            doublequote=answered[1],
            footer_first_line=None,
        ),
    )

    assert (result.escapechar, result.doublequote) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("Fecha|Cliente\n2024-01-01|Acme\n2024-01-02|Beta\n", '"'),
        ("Fecha|Cliente\n2024-01-01|'Acme'\n2024-01-02|Beta\n", "'"),
    ],
)
def test_grounding_resets_a_quote_character_that_never_occurs(
    tmp_path: Path, content: str, expected: str
) -> None:
    """A guessed ``'`` that quotes nothing becomes the default; one in the file stays."""
    target = tmp_path / "pipes.csv"
    target.write_text(content, encoding="utf-8")

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(
            delimiter="|", quotechar="'", columns=["Fecha", "Cliente"], footer_first_line=None
        ),
    )

    assert result.quotechar == expected


def test_grounding_reports_the_default_for_a_delimiter_that_never_occurs(tmp_path: Path) -> None:
    """A tab guessed for a one-column file splits nothing: the default ``,`` is reported."""
    target = tmp_path / "names.csv"
    target.write_text("Cliente\nAcme S.L.\nBeta Corp\n", encoding="utf-8")

    result = inspect_csv(
        target,
        model_invoker=_sloppy_answer(delimiter="\t", columns=["Cliente"], footer_first_line=None),
    )

    assert (result.delimiter, result.columns) == (",", ["Cliente"])


@pytest.mark.parametrize(
    ("content", "reported", "expected"),
    [
        (codecs.BOM_UTF8 + _LEDGER.encode("utf-8"), "utf-8", "UTF-8-SIG"),
        (codecs.BOM_UTF16_LE + _LEDGER.encode("utf-16-le"), "utf-16-le", "UTF-16"),
        (_LEDGER.encode("utf-8"), "UTF-8 with BOM", "utf-8"),
        (codecs.BOM_UTF8 + _LEDGER.encode("utf-8"), "utf_8_sig", "utf_8_sig"),
        (_LEDGER.encode("cp1252"), "cp1252", "cp1252"),
    ],
    ids=["utf-8 BOM", "utf-16 BOM", "not a codec", "same codec, other spelling", "no BOM"],
)
def test_grounding_keeps_a_bom_encoding_and_rejects_unknown_codecs(
    tmp_path: Path, content: bytes, reported: str, expected: str
) -> None:
    """A detected BOM, or a valid codec name, always wins over the model's answer (issue #22)."""
    target = tmp_path / "ledger.csv"
    target.write_bytes(content)

    result = inspect_csv(target, model_invoker=_sloppy_answer(encoding=reported))

    assert result.encoding == expected
    assert result.columns[0] == "Fecha"
