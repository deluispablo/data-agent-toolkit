"""Compare two or more evaluation runs written by ``eval_samples.py --out``.

Prints a Markdown table, one column per run, and the fixtures whose verdict
changed against the first run. Standard library only; the code is
``eval_harness.report`` (see ``docs/evaluation.md``).

Usage:
    python compare_runs.py runs/baseline.jsonl runs/candidate.jsonl
"""

from eval_harness.report import compare_main as main

if __name__ == "__main__":
    main()
