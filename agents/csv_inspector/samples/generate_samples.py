"""Reproducible generator for the csv_inspector test fixture catalog.

Running this script (re)writes every fixture under ``samples/`` plus
``samples/manifest.json``, which records the ground truth (expected
dialect, header/footer position, and known limitations) for each fixture.
Fixtures are generated rather than hand-typed so the catalog stays
reviewable in a diff and easy to extend as new real-world "CSVs we didn't
expect" show up.

This script writes only small, illustrative fixtures. Large files (used to
exercise ``read_tail_bytes`` against multi-megabyte inputs) are deliberately
NOT generated here: they are created on the fly inside pytest fixtures
(``tmp_path``) so the repository never carries multi-megabyte binaries.

Usage:
    python generate_samples.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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
    """

    filename: str
    category: str
    description: str
    raw_bytes: bytes
    expected: dict[str, Any] = field(default_factory=dict)
    known_limitation: bool = False
    notes: str | None = None


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


def _rows_to_text(rows: list[tuple[str, ...]], delimiter: str) -> str:
    """Join tabular rows into delimited text with a trailing newline."""
    return "\n".join(delimiter.join(row) for row in rows) + "\n"


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
            "metadata_lines": [],
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
            "metadata_lines": [],
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
            "metadata_lines": [],
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
            "metadata_lines": [],
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
            "header_row_index": 0,
            "metadata_lines": [],
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
            "metadata_lines": [
                "# Exportado desde SistemaXYZ v3.2",
                "# Fecha de generación: 2024-01-15",
            ],
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
            "header_row_index": None,
        },
        notes="Ground truth is intentionally null: there is no real header row. The agent should either say so in "
        "'notes' or return a low 'confidence', not confidently assert a fabricated header row.",
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
        description="A totals row appended after the data, sharing the same delimiter.",
        raw_bytes=_encode(
            "Fecha,Cliente,Importe\n"
            "2024-01-15,Acme,1250.50\n"
            "2024-01-16,Beta,890.00\n"
            "2024-01-17,Gamma,2100.75\n"
            "TOTAL,,4241.25\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "header_row_index": 0,
            "footer_rows_to_skip": 1,
            "footer_lines": ["TOTAL,,4241.25"],
        },
    ),
    SampleCase(
        filename="footer_end_marker.csv",
        category="header_footer",
        description="A blank line followed by an 'end of report' marker and a generation timestamp.",
        raw_bytes=_encode(
            "Fecha,Cliente,Importe\n"
            "2024-01-15,Acme,1250.50\n"
            "2024-01-16,Beta,890.00\n"
            "\n"
            "--- Fin del informe ---\n"
            "Generado el 2024-01-20 10:00:00\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "header_row_index": 0,
            "footer_rows_to_skip": 3,
            "footer_lines": ["", "--- Fin del informe ---", "Generado el 2024-01-20 10:00:00"],
        },
    ),
    SampleCase(
        filename="header_and_footer_combined.csv",
        category="header_footer",
        description="Both a metadata banner before the header and an end-of-report marker after the data.",
        raw_bytes=_encode(
            "# Exportado desde SistemaXYZ v3.2\n"
            "Fecha,Cliente,Importe\n"
            "2024-01-15,Acme,1250.50\n"
            "2024-01-16,Beta,890.00\n"
            "--- Fin del informe ---\n",
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "header_row_index": 1,
            "metadata_lines": ["# Exportado desde SistemaXYZ v3.2"],
            "footer_rows_to_skip": 1,
            "footer_lines": ["--- Fin del informe ---"],
        },
    ),
    # -------------------------------------------------------------
    # D. Quoting and escaping
    # -------------------------------------------------------------
    SampleCase(
        filename="quoting_doubled_quotes.csv",
        category="quoting",
        description="Standard RFC 4180 escaping: an embedded double-quote is doubled (\"\").",
        raw_bytes=_encode(
            "Fecha,Cliente,Descripcion,Importe\n"
            '2024-01-17,"Fernández & Asociados","Consultoría técnica ""urgente"" solicitada por el cliente",2100.75\n',
            "utf-8",
        ),
        expected={
            "encoding": "utf-8",
            "delimiter": ",",
            "quotechar": '"',
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
            "header_row_index": 0,
        },
        known_limitation=True,
        notes="Byte-prefix/suffix sampling cannot reliably parse multi-line quoted records: a head or tail window "
        "may cut this field mid-record. Documented as a known limitation rather than solved by this iteration.",
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
        notes="read_sample_bytes and read_tail_bytes must both return b'' without raising; the LLM step is expected "
        "to fail gracefully (there is nothing to infer from), which is a case for InspectionFailedError/manual review "
        "rather than a schema-conformant result.",
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
]


def main() -> None:
    """Write every fixture in ``CASES`` and the aggregated ``manifest.json``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    manifest: dict[str, dict[str, Any]] = {}
    for case in CASES:
        target = SAMPLES_DIR / case.filename
        target.write_bytes(case.raw_bytes)
        manifest[case.filename] = {
            "category": case.category,
            "description": case.description,
            "expected": case.expected,
            "known_limitation": case.known_limitation,
            "notes": case.notes,
        }
        logger.info("Wrote %s (%d bytes, category=%s).", case.filename, len(case.raw_bytes), case.category)

    manifest_path = SAMPLES_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote manifest.json with %d entries.", len(manifest))


if __name__ == "__main__":
    main()
