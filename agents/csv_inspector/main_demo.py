"""Local execution demo for the csv_inspector agent.

Runs the agent against a sample CSV file using a local Ollama model and
prints the full structured JSON result. Runs entirely for free against a
locally running Ollama instance; no cloud credentials required.

Prerequisites:
    - Ollama running locally (``ollama serve``).
    - The target model pulled locally, e.g. ``ollama pull qwen2.5-coder:7b``.

Usage:
    python main_demo.py
    python main_demo.py --model qwen2.5-coder:7b --bytes 8192 --tail-bytes 8192 --log-level DEBUG
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from exceptions import CSVInspectorError
from inspector import DEFAULT_MODEL, DEFAULT_SAMPLE_BYTES, DEFAULT_TAIL_BYTES, inspect_csv

logger = logging.getLogger(__name__)

SAMPLE_PATH = Path(__file__).parent / "sample.csv"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the demo script.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="csv_inspector local demo")
    parser.add_argument(
        "--file", default=str(SAMPLE_PATH), help="Path to the CSV file to inspect."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model to use.")
    parser.add_argument(
        "--bytes",
        type=int,
        default=DEFAULT_SAMPLE_BYTES,
        help="Number of leading (head) bytes to sample.",
    )
    parser.add_argument(
        "--tail-bytes",
        type=int,
        default=DEFAULT_TAIL_BYTES,
        help="Number of trailing (tail) bytes to sample, for footer detection.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the csv_inspector agent against a sample file and print the result."""
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(
        "Inspecting '%s' with model '%s' (head=%d bytes, tail=%d bytes).",
        args.file,
        args.model,
        args.bytes,
        args.tail_bytes,
    )

    try:
        result = inspect_csv(
            args.file,
            model=args.model,
            n_bytes=args.bytes,
            tail_bytes=args.tail_bytes,
        )
    except CSVInspectorError:
        logger.exception("Inspection failed.")
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
