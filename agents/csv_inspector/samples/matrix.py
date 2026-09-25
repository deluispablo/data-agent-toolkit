"""Parametric fixtures: a table of specs rendered into catalog entries.

The hand-written fixtures in ``generate_samples.py`` each isolate one
quirk. Real exports combine them (semicolon + decimal comma + cp1252 +
preamble + totals row) and vary in width and length. Each
:class:`FixtureSpec` in :data:`MATRIX` names such a combination;
:func:`render` turns it into a :class:`~generate_samples.SampleCase` with
deterministic bytes and its complete ground truth, and
``generate_samples.py`` appends the rendered cases to ``CASES``. The
files are written as ``gen_<slug>.csv`` and flagged ``"generated": true``
in ``manifest.json``.

Adding a fixture: hand-written vs matrix spec
---------------------------------------------

Write a hand-written ``SampleCase`` in ``generate_samples.py`` when the
fixture is a readable example of one quirk, or when its bytes need
something the renderer cannot express (a quoted newline, an escape
character, mixed line endings, a model-confusing literal). Its
``expected`` block is typed by hand.

Add a :class:`FixtureSpec` to :data:`MATRIX` when the fixture is a
combination of the dimensions below (delimiter, quote character,
encoding and BOM, line ending, preamble, header, footer kind, width,
length, decimal comma, quoted header, ragged rows). The renderer derives
every expected field from the spec, so the ground truth cannot drift from
the bytes. Then run ``python generate_samples.py`` and commit the new
file together with ``manifest.json``.

Values come from a ``random.Random`` seeded with a CRC-32 of the slug and
drawn only through ``random()``, whose output is identical on every
supported Python version, so regeneration is byte-for-byte reproducible.
"""

from __future__ import annotations

import csv
import io
import random
import zlib
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from generate_samples import UTF8_BOM, UTF16LE_BOM, SampleCase

FooterKind = Literal["none", "totals", "marker", "timestamp", "blank_totals"]

# (encoding, bom) -> the manifest's expected-encoding label. The first
# alternative of a label is the codec the tests decode the fixture with.
_ENCODING_LABELS: dict[tuple[str, bool], str] = {
    ("utf-8", False): "utf-8",
    ("utf-8", True): "utf-8-sig",
    ("cp1252", False): "cp1252 or latin-1 (not utf-8)",
    ("utf-16-le", True): "utf-16 or utf-16-le",
}
_DELIMITER_NAMES = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}

# Column kinds, cycled for wide files. The first three give a narrow file
# a date, a text and an amount column.
_COLUMN_KINDS = (
    "Fecha",
    "Cliente",
    "Importe",
    "Concepto",
    "Cantidad",
    "Región",
    "Código",
    "Estado",
)
_TEXT_KINDS = frozenset({"Cliente", "Concepto", "Región", "Estado"})
_CLIENTS = (
    "García S.L.",
    "Muñoz Hermanos",
    "Fernández, Asociados",
    "Núñez Ortega y Cía.",
    "Acme Distribuciones S.L.",
    'Taller "El Rápido"',
    "Peña & Hijos",
    "Ibáñez Logística",
)
_CONCEPTS = (
    "Compra de material informático",
    "Servicio de mantenimiento",
    "Consultoría técnica",
    "Formación interna",
    "Revisión anual de equipos",
    "Transporte de mercancía",
)
_REGIONS = ("Andalucía", "Aragón", "Cataluña", "Galicia", "Madrid", "País Vasco")
_STATES = ("Pagado", "Pendiente", "Anulado", "En revisión")
_PADDING_WORDS = ("detalle", "ampliado", "según", "contrato", "vigente", "del", "ejercicio")
# A ragged file drops trailing fields from one data row in every seven.
_RAGGED_PERIOD = 7
_PREAMBLE = (
    "# Exportado desde SistemaXYZ v3.2",
    "# Empresa: Distribuciones Ejemplo S.A.",
    "# Periodo: {start} a {end}",
    "# Moneda: EUR",
    "# Filas de datos: {rows}",
)


@dataclass(frozen=True)
class FixtureSpec:
    """One parametric fixture: the quirks it combines and its size.

    Attributes:
        slug: Unique name; the file is ``gen_<slug>.csv`` and the value
            generator is seeded from it.
        category: Manifest category: ``"combo"`` or the dominant quirk.
        delimiter: Field delimiter.
        quotechar: Quote character used where a field needs quoting.
        encoding: Python codec: ``"utf-8"``, ``"cp1252"`` or ``"utf-16-le"``.
        bom: Prefix a byte-order mark (``utf-8`` or ``utf-16-le`` only;
            required for ``utf-16-le``).
        newline: Line ending, LF or CRLF.
        preamble_lines: Comment lines before the header (0 to 5).
        has_header: Whether a header row precedes the data.
        footer_kind: Lines after the data: ``"none"``, a ``"totals"`` row,
            an end-of-report ``"marker"``, a generation ``"timestamp"`` or
            ``"blank_totals"`` (a blank line, then the totals row).
        n_columns: Number of columns.
        n_rows: Number of data rows.
        decimal_comma: Format amounts as ``1.234,56`` instead of ``1234.56``.
        quoted_header: Quote every header field.
        ragged: Drop trailing fields from every seventh data row.
        known_limitation: The fixture shows a documented limitation.
        notes: Manifest notes, e.g. the rule or issue the fixture guards.
        cell_padding: Minimum length of header names and text cells, to
            build very long lines.
    """

    slug: str
    category: str
    delimiter: str = ","
    quotechar: str = '"'
    encoding: str = "utf-8"
    bom: bool = False
    newline: str = "\n"
    preamble_lines: int = 0
    has_header: bool = True
    footer_kind: FooterKind = "none"
    n_columns: int = 5
    n_rows: int = 40
    decimal_comma: bool = False
    quoted_header: bool = False
    ragged: bool = False
    known_limitation: bool = False
    notes: str | None = None
    cell_padding: int = 0

    def __post_init__(self) -> None:
        """Reject combinations the renderer cannot give an exact ground truth for.

        Raises:
            ValueError: If the spec is inconsistent.
        """
        if (self.encoding, self.bom) not in _ENCODING_LABELS:
            raise ValueError(f"{self.slug}: unsupported encoding/BOM {self.encoding}/{self.bom}")
        if self.newline not in {"\n", "\r\n"}:
            raise ValueError(f"{self.slug}: newline must be LF or CRLF")
        if not 0 <= self.preamble_lines <= len(_PREAMBLE):
            raise ValueError(f"{self.slug}: preamble_lines out of range")
        # A header-less file with a preamble is a known limitation of its
        # own (#94), and ragged rows plus a preamble or footer break the
        # rectangular-body check of the docs recipe tests.
        if not self.has_header and self.preamble_lines:
            raise ValueError(f"{self.slug}: a header-less spec cannot have a preamble")
        if self.ragged and (self.preamble_lines or self.footer_kind != "none"):
            raise ValueError(f"{self.slug}: a ragged spec cannot have a preamble or footer")

    @property
    def filename(self) -> str:
        """The fixture's file name under ``samples/``."""
        return f"gen_{self.slug}.csv"


def _pick(rng: random.Random, n: int) -> int:
    """Return an index in ``range(n)`` using only the version-stable ``random()``."""
    return int(rng.random() * n)


def _pad(text: str, width: int) -> str:
    """Extend ``text`` with filler words until it is at least ``width`` characters long."""
    words = [text]
    index = 0
    while sum(len(word) + 1 for word in words) - 1 < width:
        words.append(_PADDING_WORDS[index % len(_PADDING_WORDS)])
        index += 1
    return " ".join(words)


def _column_names(spec: FixtureSpec) -> list[str]:
    """Build the header names: the kinds, numbered from their second cycle on."""
    names = []
    for index in range(spec.n_columns):
        kind = _COLUMN_KINDS[index % len(_COLUMN_KINDS)]
        cycle = index // len(_COLUMN_KINDS)
        name = kind if cycle == 0 else f"{kind}_{cycle + 1}"
        names.append(_pad(name, spec.cell_padding) if spec.cell_padding else name)
    return names


def _format_amount(cents: int, *, decimal_comma: bool) -> str:
    """Format integer cents as ``1234.56`` or, European style, ``1.234,56``."""
    units, rest = divmod(cents, 100)
    if not decimal_comma:
        return f"{units}.{rest:02d}"
    return f"{units:,}".replace(",", ".") + f",{rest:02d}"


def _data_rows(spec: FixtureSpec, rng: random.Random, start: date) -> tuple[list[list[str]], int]:
    """Generate the data rows and the sum, in cents, of every amount cell.

    Args:
        spec: The fixture spec.
        rng: The slug-seeded generator.
        start: Date of the first row.

    Returns:
        A ``(rows, total_cents)`` tuple; ragged rows are already trimmed.
    """
    rows: list[list[str]] = []
    total_cents = 0
    for row_index in range(spec.n_rows):
        row = []
        for column in range(spec.n_columns):
            kind = _COLUMN_KINDS[column % len(_COLUMN_KINDS)]
            if kind == "Fecha":
                value = (start + timedelta(days=row_index + column)).isoformat()
            elif kind == "Importe":
                cents = 100 + _pick(rng, 500_000)
                total_cents += cents
                value = _format_amount(cents, decimal_comma=spec.decimal_comma)
            elif kind == "Cantidad":
                value = str(1 + _pick(rng, 500))
            elif kind == "Código":
                value = f"{'ABCDEFGH'[_pick(rng, 8)]}-{_pick(rng, 10_000):04d}"
            else:
                pool = {
                    "Cliente": _CLIENTS,
                    "Concepto": _CONCEPTS,
                    "Región": _REGIONS,
                    "Estado": _STATES,
                }[kind]
                value = pool[_pick(rng, len(pool))]
            if spec.cell_padding and kind in _TEXT_KINDS:
                value = _pad(value, spec.cell_padding)
            row.append(value)
        if spec.ragged and row_index % _RAGGED_PERIOD == _RAGGED_PERIOD // 2:
            row = row[: spec.n_columns - 1 - row_index % 2]
        rows.append(row)
    return rows, total_cents


def _write_row(row: list[str], spec: FixtureSpec, *, quote_all: bool = False) -> str:
    """Serialize one row with the spec's dialect, without the line ending."""
    buffer = io.StringIO()
    csv.writer(
        buffer,
        delimiter=spec.delimiter,
        quotechar=spec.quotechar,
        lineterminator="",
        quoting=csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL,
    ).writerow(row)
    return buffer.getvalue()


def _footer_lines(spec: FixtureSpec, total_cents: int, end: date, rng: random.Random) -> list[str]:
    """Build the footer lines for the spec's ``footer_kind``."""
    if spec.footer_kind == "none":
        return []
    if spec.footer_kind == "marker":
        return [f"--- Fin del informe: {spec.n_rows} líneas ---"]
    if spec.footer_kind == "timestamp":
        clock = f"{8 + _pick(rng, 10):02d}:{_pick(rng, 60):02d}:{_pick(rng, 60):02d}"
        return [f"Informe generado el {end.isoformat()} a las {clock}"]
    totals = [""] * spec.n_columns
    totals[0] = "TOTAL"
    totals[_COLUMN_KINDS.index("Importe")] = _format_amount(
        total_cents, decimal_comma=spec.decimal_comma
    )
    line = _write_row(totals, spec)
    return ["", line] if spec.footer_kind == "blank_totals" else [line]


def _description(spec: FixtureSpec) -> str:
    """Summarize the spec in one line for the manifest."""
    newline = "CRLF" if spec.newline == "\r\n" else "LF"
    quirks = [
        f"{spec.preamble_lines}-line preamble" if spec.preamble_lines else "",
        "no header row" if not spec.has_header else "",
        "quoted header" if spec.quoted_header else "",
        "decimal comma" if spec.decimal_comma else "",
        "ragged rows" if spec.ragged else "",
        f"{spec.footer_kind.replace('_', ' + ')} footer" if spec.footer_kind != "none" else "",
        f"quotechar {spec.quotechar}" if spec.quotechar != '"' else "",
    ]
    listed = ", ".join(quirk for quirk in quirks if quirk)
    return (
        f"Generated {_DELIMITER_NAMES[spec.delimiter]}-delimited file, "
        f"{_ENCODING_LABELS[spec.encoding, spec.bom].split(' or ')[0]}, {newline}, "
        f"{spec.n_columns} columns x {spec.n_rows} rows" + (f"; {listed}." if listed else ".")
    )


def render(spec: FixtureSpec) -> SampleCase:
    """Render a spec into a fixture with deterministic bytes and full ground truth.

    Args:
        spec: The fixture to render.

    Returns:
        The catalog entry, ``generated`` set, with ``expected`` holding the
        encoding, dialect (quote escaping too, when a field is quoted),
        header position and footer, and ``columns`` the header names
        (``column_1..N`` without a header).
    """
    rng = random.Random(zlib.crc32(spec.slug.encode("utf-8")))
    start = date(2023, 1, 1) + timedelta(days=_pick(rng, 365))
    end = start + timedelta(days=spec.n_rows + spec.n_columns)
    names = _column_names(spec)
    rows, total_cents = _data_rows(spec, rng, start)
    footer = _footer_lines(spec, total_cents, end, rng)

    preamble = [
        line.format(start=start.isoformat(), end=end.isoformat(), rows=spec.n_rows)
        for line in _PREAMBLE[: spec.preamble_lines]
    ]
    header = [_write_row(names, spec, quote_all=spec.quoted_header)] if spec.has_header else []
    table = [*header, *(_write_row(row, spec) for row in rows), *footer]
    lines = [*preamble, *table]
    text = "".join(line + spec.newline for line in lines)

    bom = {"utf-8": UTF8_BOM, "utf-16-le": UTF16LE_BOM}.get(spec.encoding, b"") if spec.bom else b""
    expected: dict[str, object] = {
        "encoding": _ENCODING_LABELS[spec.encoding, spec.bom],
        "delimiter": spec.delimiter,
        "quotechar": spec.quotechar,
    }
    if any(spec.quotechar in line for line in table):
        # csv.writer quotes a field holding the delimiter or the quote and
        # doubles a quote inside it; it never writes an escape character.
        expected.update({"escapechar": None, "doublequote": True})
    if not spec.has_header:
        expected["has_header"] = False
    expected.update(
        {
            "header_row_index": spec.preamble_lines if spec.has_header else None,
            "footer_rows_to_skip": len(footer),
            "footer_lines": footer,
        }
    )
    return SampleCase(
        filename=spec.filename,
        category=spec.category,
        description=_description(spec),
        raw_bytes=bom + text.encode(spec.encoding),
        expected=expected,
        known_limitation=spec.known_limitation,
        notes=spec.notes,
        columns=names if spec.has_header else [f"column_{n}" for n in range(1, spec.n_columns + 1)],
        generated=True,
    )


_FOOTER_KINDS: tuple[FooterKind, ...] = ("none", "totals", "marker", "timestamp", "blank_totals")
_FOOTER_DELIMITERS = (",", ";", "\t", "|", ";")

MATRIX: list[FixtureSpec] = [
    # Every footer kind, wide (40 columns) and narrow (3 columns). Footer
    # files are larger than the default head + tail budget, so the footer
    # reaches the model through the tail window only.
    *(
        FixtureSpec(
            slug=f"footer_{kind}_wide",
            category="header_footer",
            delimiter=delimiter,
            footer_kind=kind,
            n_columns=40,
            n_rows=30,
        )
        for kind, delimiter in zip(_FOOTER_KINDS, _FOOTER_DELIMITERS, strict=True)
    ),
    *(
        FixtureSpec(
            slug=f"footer_{kind}_narrow",
            category="header_footer",
            delimiter=delimiter,
            newline="\r\n",
            footer_kind=kind,
            n_columns=3,
            n_rows=300,
        )
        for kind, delimiter in zip(_FOOTER_KINDS, reversed(_FOOTER_DELIMITERS), strict=True)
    ),
    # Every encoding x LF / CRLF. The marker footers carry accented text, so
    # the tail window must be decoded with the encoding found in the head.
    FixtureSpec(slug="encoding_utf8_lf", category="encoding", footer_kind="totals", n_rows=150),
    FixtureSpec(
        slug="encoding_utf8_crlf",
        category="encoding",
        delimiter=";",
        newline="\r\n",
        n_columns=8,
    ),
    FixtureSpec(slug="encoding_utf8_bom_lf", category="encoding", bom=True),
    FixtureSpec(
        slug="encoding_utf8_bom_crlf",
        category="encoding",
        bom=True,
        newline="\r\n",
        footer_kind="marker",
        n_columns=8,
        n_rows=120,
    ),
    FixtureSpec(
        slug="encoding_cp1252_lf",
        category="encoding",
        delimiter=";",
        encoding="cp1252",
        footer_kind="marker",
        n_columns=6,
        n_rows=150,
    ),
    FixtureSpec(
        slug="encoding_cp1252_crlf",
        category="encoding",
        delimiter=";",
        encoding="cp1252",
        newline="\r\n",
        footer_kind="timestamp",
        n_columns=6,
        n_rows=150,
    ),
    FixtureSpec(
        slug="encoding_utf16le_lf",
        category="encoding",
        delimiter="\t",
        encoding="utf-16-le",
        bom=True,
    ),
    FixtureSpec(
        slug="encoding_utf16le_crlf",
        category="encoding",
        delimiter="\t",
        encoding="utf-16-le",
        bom=True,
        newline="\r\n",
        footer_kind="marker",
        n_columns=6,
        n_rows=80,
    ),
    # Preamble lengths 1 and 5 (0 is every other spec).
    FixtureSpec(slug="preamble_1", category="header_footer", delimiter=";", preamble_lines=1),
    FixtureSpec(
        slug="preamble_5_crlf",
        category="header_footer",
        newline="\r\n",
        preamble_lines=5,
        n_columns=7,
    ),
    FixtureSpec(
        slug="preamble_5_footer",
        category="header_footer",
        delimiter="\t",
        preamble_lines=5,
        footer_kind="blank_totals",
        n_columns=6,
        n_rows=150,
    ),
    # Header-less files.
    FixtureSpec(slug="headerless_narrow", category="header_footer", has_header=False, n_columns=3),
    FixtureSpec(
        slug="headerless_marker",
        category="header_footer",
        delimiter=";",
        has_header=False,
        footer_kind="marker",
        n_columns=6,
        n_rows=150,
    ),
    # Quoting.
    FixtureSpec(slug="quoted_header", category="quoting", quoted_header=True, n_columns=8),
    FixtureSpec(
        slug="quotechar_single",
        category="quoting",
        quotechar="'",
        quoted_header=True,
        n_columns=6,
    ),
    # Ragged rows.
    FixtureSpec(slug="ragged_comma", category="structural", ragged=True, n_columns=6),
    FixtureSpec(
        slug="ragged_tab_wide",
        category="structural",
        delimiter="\t",
        ragged=True,
        n_columns=40,
    ),
    # Decimal comma, with and without a delimiter that forces quoting.
    FixtureSpec(
        slug="decimal_comma_semicolon", category="data_format", delimiter=";", decimal_comma=True
    ),
    FixtureSpec(
        slug="decimal_comma_comma", category="data_format", decimal_comma=True, n_columns=4
    ),
    # Larger than 64 KiB: the tail window is a real suffix far from the head.
    FixtureSpec(
        slug="large_ledger",
        category="structural",
        delimiter=";",
        preamble_lines=2,
        footer_kind="totals",
        n_rows=1200,
    ),
    FixtureSpec(
        slug="large_wide",
        category="structural",
        newline="\r\n",
        footer_kind="marker",
        n_columns=40,
        n_rows=160,
    ),
    # Combos: four or more quirks in one file.
    FixtureSpec(
        slug="combo_eu_legacy",
        category="combo",
        delimiter=";",
        encoding="cp1252",
        newline="\r\n",
        preamble_lines=2,
        footer_kind="blank_totals",
        decimal_comma=True,
        n_columns=6,
        n_rows=150,
    ),
    FixtureSpec(
        slug="combo_bom_crlf_preamble",
        category="combo",
        bom=True,
        newline="\r\n",
        preamble_lines=3,
        footer_kind="timestamp",
        quoted_header=True,
        n_columns=8,
        n_rows=120,
    ),
    FixtureSpec(
        slug="combo_pipe_ragged",
        category="combo",
        delimiter="|",
        bom=True,
        newline="\r\n",
        quoted_header=True,
        decimal_comma=True,
        ragged=True,
        n_columns=12,
    ),
    # Issue #125, case 9: a wide UTF-16 file; the tail read must start on
    # a code-unit boundary to decode.
    FixtureSpec(
        slug="utf16le_bom_crlf_wide",
        category="encoding",
        delimiter="\t",
        encoding="utf-16-le",
        bom=True,
        newline="\r\n",
        footer_kind="totals",
        n_columns=40,
        n_rows=30,
        notes="Guards _encoding.code_unit_size: the UTF-16 tail window must be aligned to "
        "2-byte code units, or the footer decodes as garbage.",
    ),
    # Issue #125, case 13: lines longer than the default 4096-byte head
    # window, which then holds less than one complete line.
    FixtureSpec(
        slug="very_wide_long_lines",
        category="structural",
        n_columns=200,
        n_rows=6,
        cell_padding=66,
        known_limitation=True,
        notes="Each line is ~8 KB, so the default head window holds less than one complete line; "
        "byte windows cannot show a header, line-based windows (#134) would.",
    ),
]
