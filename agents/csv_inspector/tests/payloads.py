"""Payloads and helpers shared by the pipeline, prompt, parsing and grounding tests."""

from __future__ import annotations

import json
from pathlib import Path

from csv_inspector._invokers import ModelInvoker

SAMPLE_CSV_PATH = Path(__file__).resolve().parent.parent / "sample.csv"


VALID_RESULT_PAYLOAD: dict[str, object] = {
    "encoding": "utf-8",
    "delimiter": ";",
    "quotechar": '"',
    "escapechar": None,
    "doublequote": True,
    "header_row_index": 2,
    "footer_first_line": None,
    "columns": ["Fecha", "Importe"],
    "confidence": 0.95,
}


_LEDGER = (
    "# Exportado desde SistemaXYZ v3.2\n"
    "# Periodo: 2024-01-01 a 2024-01-03\n"
    "Fecha;Cliente;Importe\n"
    "2024-01-01;Acme;10.00\n"
    "2024-01-02;Beta;20.00\n"
    "2024-01-03;Gamma;30.00\n"
    "\n"
    "TOTAL;;60.00\n"
    "--- Fin del informe ---\n"
)


def _sloppy_answer(**overrides: object) -> ModelInvoker:
    """Fake model that recognizes the structure but miscounts, paraphrases and skips lines.

    By default it spots the totals row but drops the blank line before it
    and the end-of-report marker after it.
    """
    payload = {
        **VALID_RESULT_PAYLOAD,
        "header_row_index": 0,
        "columns": ["Fecha", "Proveedor", "Monto"],
        "footer_first_line": "TOTAL;;60.00",
        **overrides,
    }

    def fake_invoker(prompt: str, model: str) -> str:
        return json.dumps(payload)

    return fake_invoker
