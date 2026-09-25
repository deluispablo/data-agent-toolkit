"""Reproducible generator for the csv_inspector test fixture catalog.

Running this script (re)writes every fixture under ``samples/`` plus
``samples/manifest.json``, which records the ground truth (expected
dialect, header/footer position, column names and known limitations) for
each fixture. Column names are derived from the fixture bytes, not typed.
Fixtures are generated rather than hand-typed so the catalog stays
reviewable in a diff and easy to extend as new real-world "CSVs we didn't
expect" show up.

Most fixtures are small and illustrative. The footer fixtures are the
exception: they are production-sized (~14 KiB), larger than the default
head + tail sampling budget, because a footer only ever reaches the model
through the tail sample in real files. Multi-megabyte inputs (used to prove
reads stay bounded) are deliberately NOT generated here: they are created on
the fly inside pytest fixtures (``tmp_path``) so the repository never
carries large binaries.

Usage:
    python generate_samples.py
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# matrix.py imports SampleCase from this module. When this file runs as a
# script, register it under its import name so that import reuses this
# module instead of executing the file a second time.
sys.modules.setdefault("generate_samples", sys.modules[__name__])

SAMPLES_DIR = Path(__file__).parent
UTF8_BOM = b"\xef\xbb\xbf"
UTF16LE_BOM = b"\xff\xfe"


@dataclass(frozen=True)
class SampleCase:
    """A single generated fixture and the ground truth it is expected to yield.

    Attributes:
        filename: Name of the fixture file under ``samples/``.
        category: Short grouping label (e.g. ``"delimiter"``, ``"encoding"``).
        description: Human-readable explanation of what this fixture exercises.
        raw_bytes: The exact bytes to write to disk.
        expected: Ground-truth fields, matching (a subset of)
            ``CSVInspectionResult``, used by the manual evaluation harness
            (``eval_samples.py``) and by deterministic tests that check
            byte-level properties (size, presence of markers).
        known_limitation: True when this fixture demonstrates a documented
            limitation of the current byte-window sampling approach, rather
            than a case the agent is expected to fully solve today.
        notes: Optional free-text context surfaced in the manifest.
        columns: Expected column names, set only for fixtures the
            derivation in :func:`derive_columns` cannot be trusted to parse.
            Every other fixture gets ``expected["columns"]`` derived from
            its bytes by :func:`build_manifest`.
        generated: True for fixtures rendered from a ``matrix.py`` spec
            rather than written by hand.
        expected_error: The exception class ``inspect_csv`` must raise for
            this fixture (for example ``"EmptySampleError"``), or ``None``
            when it is expected to return a result. The harness scores the
            expected exception as a pass, not as a pipeline error.
    """

    filename: str
    category: str
    description: str
    raw_bytes: bytes
    expected: dict[str, Any] = field(default_factory=dict)
    known_limitation: bool = False
    notes: str | None = None
    columns: list[str] | None = None
    generated: bool = False
    expected_error: str | None = None


def _encode(text: str, encoding: str, *, bom: bytes = b"") -> bytes:
    """Encode text with an explicit encoding and an optional BOM prefix.

    Args:
        text: The text content to encode.
        encoding: The Python codec name to encode with.
        bom: Raw bytes to prepend (e.g. a UTF-8 or UTF-16LE byte-order mark).

    Returns:
        The encoded bytes, with ``bom`` prepended.
    """
    return bom + text.encode(encoding)


# ---------------------------------------------------------------------
# A. Delimiters
# ---------------------------------------------------------------------

_BASE_ROWS_PLAIN = [
    ("Fecha", "Cliente", "Descripcion", "Importe", "Observaciones"),
    ("2024-01-15", "Acme S.L.", "Compra de material informatico", "1250.50", "Sin observaciones"),
    ("2024-01-16", "Beta Corp", "Servicio de mantenimiento", "890.00", "Pendiente de aprobacion"),
    ("2024-01-17", "Gamma SA", "Consultoria tecnica", "2100.75", ""),
    ("2024-01-18", "Delta SL", "Formacion interna", "450.00", "Curso de 3 dias"),
    ("2024-01-19", "Epsilon SA", "Revision anual de equipos", "675.20", "Incluye traslado"),
]


def _rows_to_text(rows: Sequence[tuple[str, ...]], delimiter: str) -> str:
    """Join tabular rows into delimited text with a trailing newline."""
    return "\n".join(delimiter.join(row) for row in rows) + "\n"


# Production-like ledger used by the footer fixtures. Those fixtures must be
# larger than the default head + tail sampling budget (4 KiB + 4 KiB) so the
# real head/tail split is exercised, with the footer fully inside the tail.
_LEDGER_HEADER = ("Fecha", "Cliente", "Concepto", "Importe")
_LEDGER_ROW_COUNT = 220
_LEDGER_START = date(2024, 1, 1)
_LEDGER_CLIENTS = (
    "Acme Distribuciones S.L.",
    "Beta Consultores S.A.",
    "Gamma Logística",
    "Delta Servicios Técnicos",
    "Epsilon Hostelería",
    "Zeta Construcciones S.L.",
    "Eta Farmacia Central",
    "Theta Mobiliario",
)
_LEDGER_CONCEPTS = (
    "Compra de material informático",
    "Servicio de mantenimiento",
    "Consultoría técnica",
    "Formación interna",
    "Revisión anual de equipos",
    "Transporte de mercancía",
)


def _ledger_rows(count: int = _LEDGER_ROW_COUNT) -> tuple[list[tuple[str, ...]], str]:
    """Build deterministic ledger data rows and their formatted amount total.

    Values come from simple arithmetic rather than ``random`` so the bytes
    are identical on every Python version, which the fixture-drift tests
    rely on. Amounts are summed in integer cents to keep the total exact.

    Args:
        count: Number of data rows to generate.

    Returns:
        A ``(rows, total)`` tuple: the data rows (without header) and the sum
        of the ``Importe`` column formatted with two decimals.
    """
    rows: list[tuple[str, ...]] = []
    total_cents = 0
    for i in range(count):
        cents = 1_000 + (i * 7_919) % 250_000
        total_cents += cents
        rows.append(
            (
                (_LEDGER_START + timedelta(days=i)).isoformat(),
                _LEDGER_CLIENTS[(i * 3) % len(_LEDGER_CLIENTS)],
                _LEDGER_CONCEPTS[(i * 5) % len(_LEDGER_CONCEPTS)],
                f"{cents // 100}.{cents % 100:02d}",
            )
        )
    return rows, f"{total_cents // 100}.{total_cents % 100:02d}"


_LEDGER_ROWS, _LEDGER_TOTAL = _ledger_rows()


# Building blocks of the hard-case fixtures (issue #125).

# A quoted field holding a real line break, spliced into ledger rows.
_MULTILINE_CONCEPT = '"Pedido urgente\nentregado en dos partes"'


def _ledger_with_multiline_record(row: int) -> str:
    """Ledger text whose data row ``row`` has a quoted field spanning two lines."""
    rows = list(_LEDGER_ROWS)
    date_, client, _, amount = rows[row]
    rows[row] = (date_, client, _MULTILINE_CONCEPT, amount)
    return _rows_to_text([_LEDGER_HEADER, *rows], ",")


# Forty ~130-character notes: over 5 KiB, more than the default 4 KiB tail.
_LONG_FOOTER = [
    f"Nota {n:02d}: importes provisionales hasta el cierre contable del periodo; "
    f"revise la partida {n:02d} antes de conciliar con el extracto bancario"
    for n in range(1, 41)
]

_ASCII_CLIENTS = (
    "Acme Distribuciones S.L.",
    "Beta Consultores S.A.",
    "Gamma Logistica",
    "Delta Servicios Tecnicos",
    "Epsilon Hosteleria",
)
_ACCENTED_CLIENTS = (
    "Muñoz Hermanos",
    "García y Cía.",
    "Peña Logística",
    "Fernández Asociados",
    "Ibáñez Construcciones",
)


def _ascii_head_accented_tail() -> str:
    """Semicolon ledger whose accented client names occur only in the last rows."""
    rows = [
        (
            (_LEDGER_START + timedelta(days=i)).isoformat(),
            _ASCII_CLIENTS[i % len(_ASCII_CLIENTS)],
            "Servicio de mantenimiento",
            f"{100 + (i * 37) % 900}.{i % 100:02d}",
        )
        for i in range(200)
    ]
    rows += [
        ((_LEDGER_START + timedelta(days=200 + i)).isoformat(), client, "Consultoria", "450.00")
        for i, client in enumerate(_ACCENTED_CLIENTS)
    ]
    return _rows_to_text([("Fecha", "Cliente", "Concepto", "Importe"), *rows], ";")


def _exactly_head_sized() -> str:
    """ASCII CSV of exactly 4096 bytes (the default head window), newline-terminated."""
    size = 4096
    lines = ["Fecha,Cliente,Importe,Observaciones"]
    day = 0
    # Stop while there is room for one last row of at least ~60 bytes.
    while sum(len(line) + 1 for line in lines) < size - 120:
        lines.append(
            f"{(_LEDGER_START + timedelta(days=day)).isoformat()},"
            f"{_ASCII_CLIENTS[day % len(_ASCII_CLIENTS)]},{100 + day}.50,Sin observaciones"
        )
        day += 1
    prefix = f"{(_LEDGER_START + timedelta(days=day)).isoformat()},Acme,999.99,"
    room = size - sum(len(line) + 1 for line in lines) - len(prefix) - 1
    lines.append(prefix + ("Pendiente de revision " * 10)[: room - 1] + ".")
    return "\n".join(lines) + "\n"


CASES: list[SampleCase] = [
    SampleCase(
        filename="delimiter_comma.csv",
        category="delimiter",
        description="Standard comma-delimited file with no dialect quirks.",
        raw_bytes=_encode(_rows_to_text(_BASE_ROWS_PLAIN, ","), "utf-8"),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "quotechar": '"',
            "header_row_index": 0,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
    ),
    SampleCase(
        filename="delimiter_semicolon.csv",
        category="delimiter",
        description="Semicolon-delimited file with accented characters (common EU export format).",
        raw_bytes=_encode(
            "Fecha;Cliente;Descripción;Importe;Observaciones\n"
            "2024-01-15;García S.L.;Compra de material informático;1250.50;Sin observaciones\n"
            "2024-01-16;Muñoz Hermanos;Servicio de mantenimiento;890.00;Pendiente de aprobación\n"
            "2024-01-17;Fernández y Asociados;Consultoría técnica;2100.75;\n"
            "2024-01-18;López-Pérez S.A.;Formación interna;450.00;Curso de 3 días\n"
            "2024-01-19;Núñez Ortega y Cía.;Revisión anual de equipos;675.20;Incluye traslado\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ";",
            "quotechar": '"',
            "header_row_index": 0,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
    ),
    SampleCase(
        filename="delimiter_tab.tsv",
        category="delimiter",
        description="Tab-delimited (TSV) file; extension deliberately still generic to test content-based detection.",
        raw_bytes=_encode(_rows_to_text(_BASE_ROWS_PLAIN, "\t"), "utf-8"),
        expected={
            "encoding": "utf-8",
            "delimiter": "\t",
            "quotechar": '"',
            "header_row_index": 0,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
    ),
    SampleCase(
        filename="delimiter_pipe.csv",
        category="delimiter",
        description="Pipe-delimited file, common in legacy mainframe/EDI-style exports.",
        raw_bytes=_encode(_rows_to_text(_BASE_ROWS_PLAIN, "|"), "utf-8"),
        expected={
            "encoding": "utf-8",
            "delimiter": "|",
            "quotechar": '"',
            "header_row_index": 0,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
    ),
    SampleCase(
        filename="delimiter_inside_quoted_field.csv",
        category="delimiter",
        description="Comma-delimited file where a quoted field legitimately contains the delimiter character.",
        raw_bytes=_encode(
            "Fecha,Cliente,Descripcion,Importe\n"
            '2024-01-15,"Acme, S.L.","Compra de material, incluye instalación",1250.50\n'
            '2024-01-16,"Beta, Corp","Servicio de mantenimiento, incluye revisión",890.00\n'
            "2024-01-17,Gamma SA,Consultoria tecnica,2100.75\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "quotechar": '"',
            "escapechar": None,
            "doublequote": True,
            "header_row_index": 0,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
        notes="The model must not mistake the comma inside the quoted field for a fourth column.",
    ),
    # -------------------------------------------------------------
    # B. Encoding
    # -------------------------------------------------------------
    SampleCase(
        filename="encoding_utf8_bom.csv",
        category="encoding",
        description="UTF-8 file with a byte-order mark, as produced by Excel's 'CSV UTF-8' export.",
        raw_bytes=_encode(
            "Fecha,Cliente,Descripción,Importe\n"
            "2024-01-15,Acme S.L.,Compra de material informático,1250.50\n"
            "2024-01-16,Muñoz Hermanos,Servicio de mantenimiento,890.00\n",
            "utf-8",
            bom=UTF8_BOM,
        ),
        expected={
            "encoding": "utf-8-sig",
            "delimiter": ",",
            "header_row_index": 0,
        },
        notes="chardet typically reports this as 'UTF-8-SIG'; the BOM itself must not leak into the first column name.",
    ),
    SampleCase(
        filename="encoding_latin1.csv",
        category="encoding",
        description="Semicolon-delimited file encoded as Latin-1/cp1252, not valid UTF-8 (classic legacy Windows export).",
        raw_bytes=_encode(
            "Fecha;Cliente;Descripción;Importe\n"
            "2024-01-15;García S.L.;Compra de material informático;1250.50\n"
            "2024-01-16;Muñoz Hermanos;Instalación y revisión;890.00\n",
            "latin-1",
        ),
        expected={
            "encoding": "latin-1 or cp1252 (not utf-8)",
            "delimiter": ";",
            "header_row_index": 0,
        },
        notes="These bytes are not valid UTF-8; the agent must not silently mangle accented characters.",
    ),
    SampleCase(
        filename="encoding_utf16le_bom.csv",
        category="encoding",
        description="UTF-16 LE file with BOM, tab-delimited (classic old-Excel 'Unicode Text' export).",
        raw_bytes=_encode(_rows_to_text(_BASE_ROWS_PLAIN, "\t"), "utf-16-le", bom=UTF16LE_BOM),
        expected={
            "encoding": "utf-16 or utf-16-le",
            "delimiter": "\t",
            "header_row_index": 0,
        },
        notes="A classic 'unexpected file' gotcha: every other byte is 0x00 when misread as a single-byte encoding.",
    ),
    # -------------------------------------------------------------
    # C. Header / footer
    # -------------------------------------------------------------
    SampleCase(
        filename="header_metadata_banner.csv",
        category="header_footer",
        description="Two comment/metadata lines before the real header row (export banner style).",
        raw_bytes=_encode(
            "# Exportado desde SistemaXYZ v3.2\n"
            "# Fecha de generación: 2024-01-15\n"
            "Fecha;Cliente;Descripción;Importe;Observaciones\n"
            "2024-01-15;García S.L.;Compra de material informático;1250.50;Sin observaciones\n"
            "2024-01-16;Muñoz Hermanos;Servicio de mantenimiento;890.00;Pendiente de aprobación\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ";",
            "header_row_index": 2,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
    ),
    SampleCase(
        filename="header_none_data_only.csv",
        category="header_footer",
        description="No header row at all; data starts on the very first line.",
        raw_bytes=_encode(
            "2024-01-15,Acme S.L.,1250.50\n"
            "2024-01-16,Beta Corp,890.00\n"
            "2024-01-17,Gamma SA,2100.75\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "has_header": False,
            "header_row_index": None,
        },
        notes="There is no real header row: the agent should answer has_header=false and "
        "header_row_index=null, not assert a fabricated header row.",
    ),
    SampleCase(
        filename="header_duplicated_mid_file.csv",
        category="header_footer",
        description="The header row is repeated mid-file, as happens when exports get concatenated.",
        raw_bytes=_encode(
            "Fecha,Cliente,Importe\n"
            "2024-01-15,Acme,100\n"
            "2024-01-16,Beta,200\n"
            "Fecha,Cliente,Importe\n"
            "2024-01-17,Gamma,300\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "header_row_index": 0,
        },
        notes="The duplicated header mid-file is a structural anomaly the agent should flag in 'notes', not treat as a data row.",
    ),
    SampleCase(
        filename="footer_summary_totals.csv",
        category="header_footer",
        description="Production-sized ledger (~14 KiB) with a totals row appended after the data, sharing the same delimiter.",
        raw_bytes=_encode(
            _rows_to_text([_LEDGER_HEADER, *_LEDGER_ROWS, ("TOTAL", "", "", _LEDGER_TOTAL)], ","),
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "header_row_index": 0,
            "footer_rows_to_skip": 1,
            "footer_lines": [f"TOTAL,,,{_LEDGER_TOTAL}"],
        },
        notes="Larger than the default head + tail budget, so the footer is only visible in the tail sample.",
    ),
    SampleCase(
        filename="footer_end_marker.csv",
        category="header_footer",
        description="Production-sized ledger (~14 KiB) followed by a blank line, an 'end of report' marker and a generation timestamp.",
        raw_bytes=_encode(
            _rows_to_text([_LEDGER_HEADER, *_LEDGER_ROWS], ",")
            + "\n"
            + "--- Fin del informe ---\n"
            + "Generado el 2024-08-08 10:00:00\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "header_row_index": 0,
            "footer_rows_to_skip": 3,
            "footer_lines": ["", "--- Fin del informe ---", "Generado el 2024-08-08 10:00:00"],
        },
        notes="Regression case: the timestamp line used to be misclassified as a header/metadata line.",
    ),
    SampleCase(
        filename="header_and_footer_combined.csv",
        category="header_footer",
        description="Production-sized semicolon export (~14 KiB) with a two-line banner before the header and a blank line, totals row and end-of-report marker after the data.",
        raw_bytes=_encode(
            "# Exportado desde SistemaXYZ v3.2\n"
            "# Periodo: 2024-01-01 a 2024-08-07\n"
            + _rows_to_text([_LEDGER_HEADER, *_LEDGER_ROWS], ";")
            + "\n"
            + f"TOTAL;;;{_LEDGER_TOTAL}\n"
            + "--- Fin del informe ---\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ";",
            "header_row_index": 2,
            "footer_rows_to_skip": 3,
            "footer_lines": ["", f"TOTAL;;;{_LEDGER_TOTAL}", "--- Fin del informe ---"],
        },
        notes="The banner is only visible in the head sample and the footer only in the tail sample.",
    ),
    # -------------------------------------------------------------
    # D. Quoting and escaping
    # -------------------------------------------------------------
    SampleCase(
        filename="quoting_doubled_quotes.csv",
        category="quoting",
        description='Standard RFC 4180 escaping: an embedded double-quote is doubled ("").',
        raw_bytes=_encode(
            "Fecha,Cliente,Descripcion,Importe\n"
            '2024-01-17,"Fernández & Asociados","Consultoría técnica ""urgente"" solicitada por el cliente",2100.75\n',
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "quotechar": '"',
            "escapechar": None,
            "doublequote": True,
            "header_row_index": 0,
        },
    ),
    SampleCase(
        filename="quoting_backslash_escape.csv",
        category="quoting",
        description="Backslash-escaped quotes inside fields, as produced by some MySQL/legacy exporters.",
        raw_bytes=_encode(
            "Fecha,Cliente,Descripcion,Importe\n"
            '2024-01-17,"Fernandez & Asociados","Consultoria \\"urgente\\" solicitada",2100.75\n',
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "quotechar": '"',
            "escapechar": "\\",
            "doublequote": False,
            "header_row_index": 0,
        },
    ),
    SampleCase(
        filename="quoting_embedded_newline.csv",
        category="quoting",
        description="A quoted field spans a real physical newline (a genuinely multi-line CSV record).",
        raw_bytes=_encode(
            "Fecha,Cliente,Descripcion,Importe\n"
            '2024-01-17,Acme,"Linea 1 de la nota\nLinea 2 de la nota",2100.75\n'
            "2024-01-18,Beta,Nota simple,890.00\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "escapechar": None,
            "doublequote": True,
            "header_row_index": 0,
        },
        known_limitation=True,
        notes="Byte-prefix/suffix sampling cannot reliably parse multi-line quoted records: a head or tail window "
        "may cut this field mid-record. Documented as a known limitation rather than solved by this iteration.",
        # The manifest records no quotechar for the multi-line record, so the
        # names are given explicitly and checked against the derived value.
        columns=["Fecha", "Cliente", "Descripcion", "Importe"],
    ),
    SampleCase(
        filename="quoting_inconsistent.csv",
        category="quoting",
        description="Mixed quoting: some fields quoted, others not, within the same file.",
        raw_bytes=_encode(
            "Fecha,Cliente,Descripcion,Importe\n"
            '2024-01-15,Acme,"Compra de material, con detalle",1250.50\n'
            "2024-01-16,Beta,Servicio simple,890.00\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "quotechar": '"',
            "escapechar": None,
            "doublequote": True,
            "header_row_index": 0,
        },
    ),
    SampleCase(
        filename="quoting_trailing_empty_field.csv",
        category="quoting",
        description="A trailing delimiter leaves an empty last field on some rows.",
        raw_bytes=_encode(
            "Fecha,Cliente,Importe,Observaciones\n"
            "2024-01-15,Acme,1250.50,\n"
            "2024-01-16,Beta,890.00,Pendiente\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "header_row_index": 0,
        },
    ),
    # -------------------------------------------------------------
    # E. Structural anomalies
    # -------------------------------------------------------------
    SampleCase(
        filename="ragged_rows_inconsistent_columns.csv",
        category="structural",
        description="Rows have an inconsistent number of fields (some trailing values missing).",
        raw_bytes=_encode(
            "Fecha,Cliente,Importe,Observaciones\n"
            "2024-01-15,Acme,1250.50,Sin observaciones\n"
            "2024-01-16,Beta,890.00\n"
            "2024-01-17,Gamma,2100.75,Pendiente,Extra\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "header_row_index": 0,
        },
        notes="Row 2 is missing a field, row 3 has an extra one; the agent should note this rather than error out.",
    ),
    SampleCase(
        filename="line_endings_mixed_crlf_lf.csv",
        category="structural",
        description="Mixed line endings (\\r\\n and \\n) within the same file, from concatenated cross-OS exports.",
        raw_bytes=(
            b"Fecha,Cliente,Importe\r\n"
            b"2024-01-15,Acme,1250.50\r\n"
            b"2024-01-16,Beta,890.00\n"
            b"2024-01-17,Gamma,2100.75\n"
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "header_row_index": 0,
        },
    ),
    SampleCase(
        filename="no_trailing_newline_eof.csv",
        category="structural",
        description="The last line has no terminating newline character at end of file.",
        raw_bytes=_encode(
            "Fecha,Cliente,Importe\n2024-01-15,Acme,1250.50\n2024-01-16,Beta,890.00",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "header_row_index": 0,
        },
    ),
    SampleCase(
        filename="blank_lines_between_rows.csv",
        category="structural",
        description="Stray blank lines interspersed between data rows.",
        raw_bytes=_encode(
            "Fecha,Cliente,Importe\n"
            "2024-01-15,Acme,1250.50\n"
            "\n"
            "2024-01-16,Beta,890.00\n"
            "\n"
            "2024-01-17,Gamma,2100.75\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "header_row_index": 0,
        },
    ),
    SampleCase(
        filename="empty_file.csv",
        category="structural",
        description="A completely empty (zero-byte) file — extreme edge case for both head and tail reads.",
        raw_bytes=b"",
        expected={
            "encoding": None,
            "delimiter": None,
            "header_row_index": None,
        },
        notes="_sampling._PathReader.head and .tail must both return b'' without raising; "
        "inspect_csv then raises EmptySampleError before invoking any model, since there is "
        "nothing to infer from.",
        # No encoding or delimiter to parse with: there are no columns at all.
        columns=[],
        expected_error="EmptySampleError",
    ),
    SampleCase(
        filename="header_only_no_data.csv",
        category="structural",
        description="A header row is present but there are zero data rows.",
        raw_bytes=_encode("Fecha,Cliente,Importe\n", "utf-8"),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "header_row_index": 0,
        },
        # Structural edge case: given explicitly, and checked against the
        # derived value by the catalog tests.
        columns=["Fecha", "Cliente", "Importe"],
    ),
    # -------------------------------------------------------------
    # F. Data-format gotchas
    # -------------------------------------------------------------
    SampleCase(
        filename="numeric_european_format.csv",
        category="data_format",
        description="European decimal-comma / thousands-dot numeric formatting (e.g. '1.250,50').",
        raw_bytes=_encode(
            "Fecha;Cliente;Importe\n"
            "2024-01-15;Acme;1.250,50\n"
            "2024-01-16;Beta;890,00\n"
            "2024-01-17;Gamma;12.100,75\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ";",
            "header_row_index": 0,
        },
        notes="'Importe' should ideally be inferred as a numeric type despite the European formatting, not as a "
        "plain string; worth tracking separately as a column-typing quality metric.",
    ),
    SampleCase(
        filename="null_representations_mixed.csv",
        category="data_format",
        description="Inconsistent null markers across rows: empty string, NULL, N/A, '-', NaN.",
        raw_bytes=_encode(
            "Fecha,Cliente,Observaciones\n"
            "2024-01-15,Acme,\n"
            "2024-01-16,Beta,NULL\n"
            "2024-01-17,Gamma,N/A\n"
            "2024-01-18,Delta,-\n"
            "2024-01-19,Epsilon,NaN\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "header_row_index": 0,
        },
        notes="'Observaciones' should be flagged nullable=True; the mix of null markers is worth a mention in 'notes'.",
    ),
    SampleCase(
        filename="whitespace_padded_fields.csv",
        category="data_format",
        description="Fields padded with spaces around the delimiter, from a fixed-width-to-CSV conversion.",
        raw_bytes=_encode(
            "Fecha ; Cliente ; Importe\n"
            "2024-01-15 ; Acme       ; 1250.50\n"
            "2024-01-16 ; Beta Corp  ;  890.00\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ";",
            "header_row_index": 0,
        },
    ),
    # -------------------------------------------------------------
    # G. Hard cases (issue #125)
    # -------------------------------------------------------------
    SampleCase(
        filename="header_none_after_preamble.csv",
        category="header_footer",
        description="No header row, but two comment lines before the first data row.",
        raw_bytes=_encode(
            "# Exportado desde SistemaXYZ v3.2\n"
            "# Sin cabecera: fecha, cliente, importe\n"
            "2024-01-15,Acme S.L.,1250.50\n"
            "2024-01-16,Beta Corp,890.00\n"
            "2024-01-17,Gamma SA,2100.75\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "quotechar": '"',
            "has_header": False,
            "header_row_index": None,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
        known_limitation=True,
        notes="Guards _grounding._first_row_is_data: it checks line 0 only, and the result cannot "
        "describe preamble lines without a header row (#94).",
    ),
    SampleCase(
        filename="quoting_newline_in_head_window.csv",
        category="quoting",
        description="Production-sized ledger (~14 KiB) whose 11th data row has a quoted field "
        "spanning two physical lines, inside the default head window.",
        raw_bytes=_encode(_ledger_with_multiline_record(10), "utf-8"),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "quotechar": '"',
            "escapechar": None,
            "doublequote": True,
            "header_row_index": 0,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
        known_limitation=True,
        notes="Guards _grounding._ParsedSample.parse: grounding splits the head on every line break, "
        "so the record reads as two short rows.",
    ),
    SampleCase(
        filename="quoting_newline_in_tail_window.csv",
        category="quoting",
        description="Production-sized ledger (~14 KiB) whose fifth-to-last data row has a quoted "
        "field spanning two physical lines, inside the default tail window.",
        raw_bytes=_encode(_ledger_with_multiline_record(len(_LEDGER_ROWS) - 5), "utf-8"),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "quotechar": '"',
            "escapechar": None,
            "doublequote": True,
            "header_row_index": 0,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
        known_limitation=True,
        notes="Guards _grounding._ParsedSample.parse on the tail: the second half of the record "
        "looks like a short trailing line.",
    ),
    SampleCase(
        filename="footer_longer_than_tail_window.csv",
        category="header_footer",
        description="Production-sized ledger followed by a 40-line footer of ~130-character "
        "notes (~5 KiB), longer than the default 4 KiB tail window.",
        raw_bytes=_encode(
            _rows_to_text([_LEDGER_HEADER, *_LEDGER_ROWS], ",") + "\n".join(_LONG_FOOTER) + "\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "quotechar": '"',
            "header_row_index": 0,
            "footer_rows_to_skip": len(_LONG_FOOTER),
            "footer_lines": _LONG_FOOTER,
        },
        known_limitation=True,
        notes="Guards _grounding._locate_footer_lines: footer exceeds tail window, so only its "
        "last lines can be anchored; a larger tail_bytes is the workaround.",
    ),
    SampleCase(
        filename="footer_like_data_row_numeric_label.csv",
        category="header_footer",
        description="The last data row ('2024,,4241.25') is shaped like a totals row but has a "
        "numeric first field and belongs to the data.",
        raw_bytes=_encode(
            "Anio,Region,Importe\n"
            "2023,Norte,1180.20\n"
            "2023,Sur,1015.60\n"
            "2024,Norte,1520.00\n"
            "2024,Sur,1310.50\n"
            "2024,,4241.25\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "quotechar": '"',
            "header_row_index": 0,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
        notes="Guards _grounding._extends_footer: a year is not a totals label, so the row is "
        "never pulled into a footer.",
    ),
    SampleCase(
        filename="footer_after_total_energies_row.csv",
        category="header_footer",
        description="Production-sized ledger keyed by client whose last data row starts with "
        "'Total Energies SA', followed by an end-of-report marker.",
        raw_bytes=_encode(
            _rows_to_text(
                [
                    ("Cliente", "Fecha", "Concepto", "Importe"),
                    *(
                        (client, date_, concept, amount)
                        for date_, client, concept, amount in _LEDGER_ROWS
                    ),
                    ("Total Energies SA", "2024-08-08", "Suministro de combustible", "1234.56"),
                ],
                ",",
            )
            + "--- Fin del informe ---\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "quotechar": '"',
            "header_row_index": 0,
            "footer_rows_to_skip": 1,
            "footer_lines": ["--- Fin del informe ---"],
        },
        notes="Guards the _grounding._extends_footer label rule: 'Total Energies SA' is not a "
        "bare totals label and its other fields are filled, so the row stays data.",
    ),
    SampleCase(
        filename="delimiter_semicolon_decimal_comma.csv",
        category="delimiter",
        description="Semicolon-delimited amounts with decimal comma and thousands dot "
        "('1.234,56'); every data row also splits evenly on the comma.",
        raw_bytes=_encode(
            "Fecha;Cliente;Base;IVA;Total\n"
            "2024-01-15;Acme S.L.;1.234,56;259,26;1.493,82\n"
            "2024-01-16;Beta Corp;890,00;186,90;1.076,90\n"
            "2024-01-17;Gamma SA;12.100,75;2.541,16;14.641,91\n"
            "2024-01-18;Delta SL;450,00;94,50;544,50\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ";",
            "quotechar": '"',
            "header_row_index": 0,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
        notes="Guards _grounding._ground_delimiter: ',' agrees on every data row too, so a "
        "reported ',' is kept; only ';' splits the header.",
    ),
    SampleCase(
        filename="delimiter_tab_commas_quoted_header.tsv",
        category="delimiter",
        description="Tab-delimited file with a quoted header and unquoted values containing commas.",
        raw_bytes=_encode(
            '"Fecha"\t"Cliente"\t"Descripcion"\t"Importe"\n'
            "2024-01-15\tAcme, S.L.\tCompra de material, incluye instalacion\t1250.50\n"
            "2024-01-16\tBeta Corp\tServicio de mantenimiento\t890.00\n"
            "2024-01-17\tGamma, S.A.\tConsultoria tecnica, fase 1, urgente\t2100.75\n"
            "2024-01-18\tDelta SL\tFormacion interna, 3 dias\t450.00\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": "\t",
            "quotechar": '"',
            "escapechar": None,
            "doublequote": True,
            "header_row_index": 0,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
        notes="Guards _grounding._ground_delimiter: commas in the values must not beat the tab.",
    ),
    SampleCase(
        filename="delimiter_pipe_inside_quotes.csv",
        category="delimiter",
        description="Pipe-delimited file where quoted fields contain the pipe character.",
        raw_bytes=_encode(
            "Fecha|Cliente|Descripcion|Importe\n"
            '2024-01-15|"Acme | Filial Norte"|Compra de material|1250.50\n'
            '2024-01-16|Beta Corp|"Mantenimiento | revision anual"|890.00\n'
            '2024-01-17|"Gamma | Sur"|"Consultoria | fase 1 | fase 2"|2100.75\n'
            "2024-01-18|Delta SL|Formacion interna|450.00\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": "|",
            "quotechar": '"',
            "escapechar": None,
            "doublequote": True,
            "header_row_index": 0,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
        notes="Guards _grounding._modal_width: fields are split with the quotechar, so "
        "quoted pipes do not change the field count.",
    ),
    SampleCase(
        filename="encoding_cp1252_tail_only.csv",
        category="encoding",
        description="Semicolon ledger (~10 KiB) in cp1252 whose head is pure ASCII; accented "
        "client names occur only in the last rows.",
        raw_bytes=_encode(_ascii_head_accented_tail(), "cp1252"),
        expected={
            "encoding": "latin-1 or cp1252 (not utf-8)",
            "delimiter": ";",
            "quotechar": '"',
            "header_row_index": 0,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
        notes="Guards _sampling.sample_source tail re-detection: an ASCII head must not "
        "settle the encoding as utf-8.",
    ),
    SampleCase(
        filename="single_column.csv",
        category="structural",
        description="A single column: no delimiter occurs anywhere in the file.",
        raw_bytes=_encode(
            "Cliente\nAcme S.L.\nBeta Corp\nGamma SA\nDelta SL\nEpsilon SA\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "quotechar": '"',
            "header_row_index": 0,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
        notes="Guards _grounding._ground_delimiter boundary: no candidate splits a line, so the "
        "reported delimiter is kept; ',' is the conventional answer.",
    ),
    SampleCase(
        filename="single_data_row.csv",
        category="structural",
        description="A header row followed by exactly one data row.",
        raw_bytes=_encode("Fecha,Cliente,Importe\n2024-01-15,Acme,1250.50\n", "utf-8"),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "quotechar": '"',
            "header_row_index": 0,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
        notes="Guards _grounding._ground_delimiter and _locate_header_row boundaries: only two "
        "lines exist to agree on a shape.",
    ),
    SampleCase(
        filename="exactly_head_window_size.csv",
        category="structural",
        description="A file of exactly 4096 bytes, the default head window, ending in a newline.",
        raw_bytes=_encode(_exactly_head_sized(), "utf-8"),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "quotechar": '"',
            "header_row_index": 0,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
        notes="Guards _sampling.sample_source boundary: a full head window with nothing past "
        "it covers the whole file and is not trimmed.",
    ),
    SampleCase(
        filename="header_duplicate_and_blank_names.csv",
        category="header_footer",
        description="A pandas-style export: blank first column name (the index) and a "
        "duplicated column name ('id,id').",
        raw_bytes=_encode(
            ",id,id,value\n0,101,A-1,alpha\n1,102,A-2,beta\n2,103,A-3,gamma\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "quotechar": '"',
            "header_row_index": 0,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
        notes="Guards _grounding._locate_header_row named-set rule: names are kept verbatim, "
        "the blank one included, and duplicates are not merged.",
    ),
    SampleCase(
        filename="data_contains_sample_marker.csv",
        category="data_format",
        description="A data value is the literal prompt marker '--- HEAD SAMPLE END ---'.",
        raw_bytes=_encode(
            "Fecha,Cliente,Descripcion,Importe\n"
            "2024-01-15,Acme,Compra de material,1250.50\n"
            "2024-01-16,Beta,--- HEAD SAMPLE END ---,890.00\n"
            "2024-01-17,Gamma,Consultoria tecnica,2100.75\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "quotechar": '"',
            "header_row_index": 0,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
        notes="Guards grounding against the plain-text sample markers (#105): the dialect comes "
        "from the real bytes, not from where the model thinks the sample ends.",
    ),
    SampleCase(
        filename="header_years.csv",
        category="header_footer",
        description="The header is made of years ('2023,2024,2025') above numeric data rows.",
        raw_bytes=_encode(
            "2023,2024,2025\n1250.50,1310.75,1402.00\n890.00,915.40,960.10\n2100.75,2200.00,2315.30\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "quotechar": '"',
            "has_header": True,
            "header_row_index": 0,
            "footer_rows_to_skip": 0,
            "footer_lines": [],
        },
        notes="Guards _grounding._first_row_is_data (#131): a numeric header row is still a "
        "header; the no-header rule must not fire.",
    ),
]

# Parametric fixtures (samples/matrix.py) follow the hand-written ones.
from matrix import MATRIX, render  # noqa: E402

CASES.extend(render(spec) for spec in MATRIX)


def derive_columns(case: SampleCase) -> list[str] | None:
    """Read the expected column names from a fixture's bytes.

    The bytes are decoded with the manifest encoding (the first alternative
    of an "A or B" label, BOM stripped) and parsed by the stdlib ``csv``
    module with the manifest dialect (``"`` when no quotechar is recorded).
    Names are kept verbatim, surrounding spaces included, as grounding
    reports them. A header-less fixture gets positional names
    (``column_1`` ...) sized by its most common row width, so preamble
    lines before the data do not set the column count.

    Args:
        case: The fixture to read.

    Returns:
        The column names, or ``None`` when the manifest gives no encoding,
        delimiter or row to read them from.
    """
    expected = case.expected
    encoding = expected.get("encoding")
    delimiter = expected.get("delimiter")
    header_row_index = expected.get("header_row_index")
    has_header = expected.get("has_header", True)
    if not encoding or not delimiter:
        return None

    codec = encoding.split(" or ")[0].strip()
    text = case.raw_bytes.decode(codec).lstrip("﻿")
    rows = list(
        csv.reader(
            io.StringIO(text, newline=""),
            delimiter=delimiter,
            quotechar=expected.get("quotechar") or '"',
            escapechar=expected.get("escapechar"),
            doublequote=expected.get("doublequote", True) is not False,
        )
    )
    if not has_header:
        if not rows:
            return None
        width = Counter(len(row) for row in rows).most_common(1)[0][0]
        return [f"column_{number}" for number in range(1, width + 1)]
    if header_row_index is None or header_row_index >= len(rows):
        return None
    return rows[header_row_index]


def build_manifest(cases: Sequence[SampleCase]) -> dict[str, dict[str, Any]]:
    """Build the ``manifest.json`` payload describing ``cases``.

    ``expected["columns"]`` is derived by :func:`derive_columns`, or taken
    from :attr:`SampleCase.columns` when a fixture sets it.

    Args:
        cases: The fixtures to describe, typically :data:`CASES`.

    Returns:
        A mapping of fixture filename to its category, description,
        expected ground truth, known-limitation and generated flags,
        expected exception and notes.
    """
    return {
        case.filename: {
            "category": case.category,
            "description": case.description,
            "expected": {
                **case.expected,
                "columns": case.columns if case.columns is not None else derive_columns(case),
            },
            "known_limitation": case.known_limitation,
            "generated": case.generated,
            "expected_error": case.expected_error,
            "notes": case.notes,
        }
        for case in cases
    }


def main() -> None:
    """Write every fixture in ``CASES`` and the aggregated ``manifest.json``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    for case in CASES:
        (SAMPLES_DIR / case.filename).write_bytes(case.raw_bytes)
        logger.info(
            "Wrote %s (%d bytes, category=%s).", case.filename, len(case.raw_bytes), case.category
        )

    manifest = build_manifest(CASES)
    manifest_path = SAMPLES_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    logger.info("Wrote manifest.json with %d entries.", len(manifest))


if __name__ == "__main__":
    main()
