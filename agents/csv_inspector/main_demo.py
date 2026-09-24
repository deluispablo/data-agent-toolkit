"""Repository demo: inspect the bundled ``sample.csv`` with the csv_inspector CLI.

A thin wrapper around the installed package's CLI (``csv-inspector``) that
defaults the file argument to ``sample.csv`` next to this script. All other
options are the CLI's; see ``csv-inspector --help``.

Prerequisites:
    - The package installed: ``uv sync`` at the repository root (or
      ``pip install -e ./agents/csv_inspector``).
    - Ollama running locally with the model pulled (local backend).

Usage:
    python main_demo.py
    python main_demo.py --model qwen2.5-coder:7b --bytes 8192 --tail-bytes 8192 --log-level DEBUG
    python main_demo.py --backend api --model gemini-2.5-flash
"""

from __future__ import annotations

from pathlib import Path

from csv_inspector.cli import main

SAMPLE_PATH = Path(__file__).parent / "sample.csv"

if __name__ == "__main__":
    main(default_file=SAMPLE_PATH)
