"""Manual evaluation harness for csv_inspector against the ``samples/`` catalog.

Not part of ``pytest`` or CI: it runs a real model (local Ollama by default)
against every fixture of ``samples/manifest.json`` and scores the answers
per field. The code is the ``eval_harness`` package next to this file; the
flags, the run files and the ritual are in ``docs/evaluation.md``.

Usage:
    python eval_samples.py --repeat 3 --keep-raw --out runs/
    python eval_samples.py --replay runs/baseline.jsonl --out runs/replay.jsonl
"""

from eval_harness.cli import main

if __name__ == "__main__":
    main()
