"""The embedding rules the example demonstrates, checked on its source with ``ast``.

- Only ``main_demo.py`` may ``print()`` or call ``logging.basicConfig()``: the
  API logs through module loggers and leaves handlers to whoever runs it.
- Only the library's public API is used: no ``csv_inspector._*`` module and
  no name outside ``csv_inspector.__all__``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import csv_inspector
import pytest

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
DEMO = EXAMPLE_ROOT / "main_demo.py"
SOURCES = sorted((EXAMPLE_ROOT / "src").rglob("*.py"))
ALL_PYTHON = [*SOURCES, *sorted((EXAMPLE_ROOT / "tests").rglob("*.py")), DEMO]


def _relative(path: Path) -> str:
    """Path relative to the example, for readable test ids."""
    return path.relative_to(EXAMPLE_ROOT).as_posix()


def _tree(path: Path) -> ast.Module:
    """Parse a source file."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _forbidden_calls(tree: ast.Module) -> list[str]:
    """Names of ``print`` and ``basicConfig`` calls in ``tree``."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name in {"print", "basicConfig"}:
            found.append(f"{name}() at line {node.lineno}")
    return found


def test_sources_are_found() -> None:
    """The scan covers the package, the tests and the demo."""
    assert EXAMPLE_ROOT / "src" / "csv_inspector_api" / "app.py" in SOURCES
    assert DEMO.is_file()


@pytest.mark.parametrize("path", SOURCES, ids=_relative)
def test_no_print_or_basic_config_in_the_api(path: Path) -> None:
    """The API package never prints or configures logging."""
    assert _forbidden_calls(_tree(path)) == []


def test_the_rule_detects_violations() -> None:
    """The check itself catches both calls, bare and through a module."""
    tree = ast.parse("print('x')\nimport logging\nlogging.basicConfig()\n")

    assert _forbidden_calls(tree) == ["print() at line 1", "basicConfig() at line 3"]


def _private_uses(tree: ast.Module) -> list[str]:
    """Imports of ``csv_inspector`` internals in ``tree``."""
    public = set(csv_inspector.__all__)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names if alias.name.startswith("csv_inspector.")]
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            if node.module.startswith("csv_inspector."):
                found.append(node.module)
            elif node.module == "csv_inspector":
                found += [alias.name for alias in node.names if alias.name not in public]
    return found


@pytest.mark.parametrize("path", ALL_PYTHON, ids=_relative)
def test_only_the_public_library_api_is_imported(path: Path) -> None:
    """Every ``csv_inspector`` import names a public module and an exported name."""
    assert _private_uses(_tree(path)) == []


def test_the_import_rule_detects_violations() -> None:
    """The check catches private modules and names outside ``__all__``."""
    tree = ast.parse(
        "import csv_inspector._config\n"
        "from csv_inspector._inspect import ainspect_csv\n"
        "from csv_inspector import Settings, ensure_backend_ready\n"
        "from .csv_inspector import anything\n"
    )

    assert _private_uses(tree) == [
        "csv_inspector._config",
        "csv_inspector._inspect",
        "ensure_backend_ready",
    ]
