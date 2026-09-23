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

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CSV_INSPECTOR_DIR = _REPO_ROOT / "agents" / "csv_inspector"

if str(_CSV_INSPECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(_CSV_INSPECTOR_DIR))
