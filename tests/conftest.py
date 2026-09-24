"""Shared pytest configuration for the data-agent-toolkit test suite.

The csv_inspector **library** is imported as an installed package
(``pip install -e ./agents/csv_inspector[cloud]``, via
``requirements-dev.txt``), exactly as any external host imports it; there
is no ``sys.path`` manipulation for it.

Only the repository's **tooling** that is deliberately not part of the
package (the fixture generator in ``samples/`` and the evaluation harness in
``scripts/``) is put on ``sys.path`` so tests can import it by name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CSV_INSPECTOR_DIR = _REPO_ROOT / "agents" / "csv_inspector"
_TOOLING_DIRS = (_CSV_INSPECTOR_DIR / "samples", _CSV_INSPECTOR_DIR / "scripts")

for _directory in _TOOLING_DIRS:
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))


# Every environment variable csv_inspector's settings read.
SETTINGS_ENV_VARS = (
    "LLM_BACKEND",
    "OLLAMA_MODEL",
    "OLLAMA_FALLBACK_MODEL",
    "GEMINI_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "CLOUD_MODEL",
    "CLOUD_FALLBACK_MODEL",
)


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep every test hermetic from the developer's real configuration.

    Clears the settings environment variables and runs each test from an
    empty directory, so neither exported variables nor a local ``.env``
    file can change a test's outcome (or leak a real credential into it).
    """
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
