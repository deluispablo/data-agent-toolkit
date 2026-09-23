"""Local execution demo for the csv_inspector agent.

Runs the agent against a sample CSV file and prints the full structured
JSON result. By default it uses a local Ollama model, entirely for free and
with no cloud credentials. ``--backend api`` opts in to Google Gemini (needs
the ``requirements-cloud.txt`` extra and credentials; see ``.env.example``).

Prerequisites (local backend):
    - Ollama running locally (``ollama serve``).
    - The target model pulled locally, e.g. ``ollama pull qwen2.5-coder:7b``.

Usage:
    python main_demo.py
    python main_demo.py --model qwen2.5-coder:7b --bytes 8192 --tail-bytes 8192 --log-level DEBUG
    python main_demo.py --backend api --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from cli_support import (
    add_backend_argument,
    add_log_level_argument,
    configure_cli,
    non_negative_int,
    positive_int,
    resolve_backend,
)
from exceptions import CSVInspectorError
from inspector import DEFAULT_SAMPLE_BYTES, DEFAULT_TAIL_BYTES, get_default_model, inspect_csv

logger = logging.getLogger(__name__)

SAMPLE_PATH = Path(__file__).parent / "sample.csv"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the demo script.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="csv_inspector demo")
    parser.add_argument(
        "--file", type=Path, default=SAMPLE_PATH, help="Path to the CSV file to inspect."
    )
    add_backend_argument(parser)
    parser.add_argument(
        "--model", default=None, help="Model to use. Defaults to the backend's configured model."
    )
    parser.add_argument(
        "--bytes",
        type=positive_int,
        default=DEFAULT_SAMPLE_BYTES,
        help="Number of leading (head) bytes to sample.",
    )
    parser.add_argument(
        "--tail-bytes",
        type=non_negative_int,
        default=DEFAULT_TAIL_BYTES,
        help="Number of trailing (tail) bytes to sample, for footer detection (0 disables).",
    )
    add_log_level_argument(parser)
    return parser.parse_args()


def main() -> None:
    """Run the csv_inspector agent against a sample file and print the result."""
    args = _parse_args()
    configure_cli(args.log_level)

    try:
        backend = resolve_backend(args.backend)
        model = args.model or get_default_model(backend)
        logger.info(
            "Inspecting '%s' with %s model '%s' (head=%d bytes, tail=%d bytes).",
            args.file,
            backend.value,
            model,
            args.bytes,
            args.tail_bytes,
        )
        result = inspect_csv(
            args.file,
            backend=backend,
            model=model,
            n_bytes=args.bytes,
            tail_bytes=args.tail_bytes,
        )
    except CSVInspectorError as exc:
        # Expected failure modes get a one-line message; the traceback is
        # only useful when debugging.
        logger.error("Inspection failed: %s", exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        sys.exit(1)

    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

    logger.info(
        "Summary: encoding=%s delimiter=%r header_row=%d columns=%s confidence=%.2f",
        result.encoding,
        result.delimiter,
        result.header_row_index,
        [column.name for column in result.columns],
        result.confidence,
    )
    if result.notes:
        logger.info("Notes: %s", result.notes)


if __name__ == "__main__":
    main()
