"""Shared pytest configuration for the data-agent-toolkit test suite.

Each agent under ``agents/<agent_name>/`` is a self-contained, flat module
namespace (no package ``__init__.py``), mirroring how it is executed
directly with ``python agents/<agent_name>/main_demo.py``. This fixture adds
each agent's directory to ``sys.path`` so its modules can be imported by
name from the root-level ``tests/`` package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CSV_INSPECTOR_DIR = _REPO_ROOT / "agents" / "csv_inspector"
_IMPORTABLE_DIRS = (_CSV_INSPECTOR_DIR, _CSV_INSPECTOR_DIR / "samples")

for _directory in _IMPORTABLE_DIRS:
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
