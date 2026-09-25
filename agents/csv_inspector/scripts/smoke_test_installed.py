"""Smoke test for an installed csv-inspector distribution (no repository needed).

CI installs the built wheel (and, separately, the sdist) into a fresh
virtual environment and runs this file with that environment's Python from
a directory outside the repository, so the package can only be found through
the installation itself: no ``sys.path`` tricks, no editable install. It
exercises the public API end to end with a fake model invoker (no Ollama, no
network) and checks that the ``csv-inspector`` console script runs.

Usage:
    python smoke_test_installed.py
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import csv_inspector
from csv_inspector import (
    CSVInspectionResult,
    InspectionTimeoutError,
    LLMBackend,
    Settings,
    Usage,
    ainspect_csv,
    ensure_backend_ready,
    inspect_csv,
)

CSV = b"# Export\nFecha;Cliente;Importe\n2024-01-01;Acme;10.00\n\nTOTAL;;10.00\n"
ANSWER = json.dumps(
    {
        "encoding": "utf-8",
        "delimiter": ";",
        "header_row_index": 0,
        "footer_first_line": "TOTAL;;10.00",
        "columns": ["Fecha", "Cliente", "Importe"],
        "confidence": 0.9,
    }
)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"SMOKE TEST FAILED: {message}")


def main() -> None:
    """Run the smoke checks and report success."""
    installed_at = Path(csv_inspector.__file__).resolve()
    _check(
        "site-packages" in installed_at.parts,
        f"not imported from site-packages: {installed_at}",
    )
    _check(
        csv_inspector.__version__ == importlib.metadata.version("csv-inspector"),
        "__version__ does not match the installed distribution",
    )
    _check((installed_at.parent / "py.typed").is_file(), "py.typed marker missing")
    ensure_backend_ready(LLMBackend.LOCAL, Settings())  # public readiness check; never raises

    def invoker(prompt: str, model: str) -> str:
        return ANSWER

    kwargs = {"model": "m", "fallback_model": "m", "settings": Settings()}
    from_bytes = inspect_csv(CSV, model_invoker=invoker, **kwargs)  # type: ignore[arg-type]
    from_stream = inspect_csv(io.BytesIO(CSV), model_invoker=invoker, **kwargs)  # type: ignore[arg-type]
    _check(isinstance(from_bytes, CSVInspectionResult), "no CSVInspectionResult returned")
    # model_dump() leaves out result.usage, whose latency differs between calls.
    _check(from_bytes.model_dump() == from_stream.model_dump(), "bytes and stream sources disagree")
    usage = from_bytes.usage
    _check(isinstance(usage, Usage) and usage.attempts == 1, "no usage attached to the result")
    _check(from_bytes.columns == ["Fecha", "Cliente", "Importe"], "column names not returned")
    # Grounding fixes the header row (the model said 0) and the blank separator.
    _check(from_bytes.header_row_index == 1, "header row was not grounded")
    _check(from_bytes.footer_lines == ["", "TOTAL;;10.00"], "footer was not grounded")

    async def async_invoker(prompt: str, model: str) -> str:
        return ANSWER

    async_result = asyncio.run(ainspect_csv(CSV, model_invoker=async_invoker, **kwargs))  # type: ignore[arg-type]
    _check(async_result.model_dump() == from_bytes.model_dump(), "async and sync results disagree")

    def slow_invoker(prompt: str, model: str) -> str:
        import time  # noqa: PLC0415

        time.sleep(5)
        return ANSWER

    try:
        inspect_csv(CSV, model_invoker=slow_invoker, timeout_seconds=0.2, **kwargs)  # type: ignore[arg-type]
    except InspectionTimeoutError:
        pass
    else:
        raise SystemExit("SMOKE TEST FAILED: the timeout was not enforced")

    # Console scripts live next to the interpreter of the environment under test.
    script = shutil.which("csv-inspector", path=str(Path(sys.executable).parent))
    if script is None:
        raise SystemExit("SMOKE TEST FAILED: the csv-inspector console script is not installed")
    completed = subprocess.run([script, "--help"], capture_output=True, check=False)
    _check(completed.returncode == 0, "csv-inspector --help failed")

    print(f"csv-inspector {csv_inspector.__version__} smoke test passed ({installed_at.parent})")


if __name__ == "__main__":
    main()
