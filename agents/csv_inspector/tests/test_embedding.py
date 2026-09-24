"""Tests for csv_inspector as an embedded library: API surface, hygiene, isolation.

These guard the contract a host application relies on: a stable public API,
library-grade logging (no handlers, no output), no shared mutable state,
correct results under concurrency, and explicit settings that never read the
process environment.
"""

from __future__ import annotations

import ast
import importlib
import importlib.metadata
import json
import logging
import pkgutil
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import csv_inspector
from csv_inspector import LLMBackend, Settings, inspect_csv
from fakes import install_fake_ollama, ollama_reply

PACKAGE_DIR = Path(csv_inspector.__file__).resolve().parent
CLI_MODULES = {"cli.py", "__main__.py"}

EXPECTED_PUBLIC_API = {
    "DEFAULT_SAMPLE_BYTES",
    "DEFAULT_TAIL_BYTES",
    "MAX_SAMPLE_BYTES",
    "AsyncModelInvoker",
    "BackendConfigurationError",
    "CSVInspectionResult",
    "CSVInspectorError",
    "CSVSource",
    "ColumnSchema",
    "ColumnType",
    "CredentialsNotConfiguredError",
    "EmptySampleError",
    "FileSampleReadError",
    "InspectionFailedError",
    "InspectionTimeoutError",
    "LLMBackend",
    "ModelInvocationError",
    "ModelInvoker",
    "ModelTimeoutError",
    "ResponseParsingError",
    "SchemaValidationError",
    "Settings",
    "__version__",
    "ainspect_csv",
    "ensure_backend_ready",
    "inspect_csv",
    "load_settings",
}


def _package_modules() -> list[ModuleType]:
    """Import and return every module of the csv_inspector package."""
    modules = [csv_inspector]
    for info in pkgutil.iter_modules([str(PACKAGE_DIR)]):
        if info.name != "__main__":
            modules.append(importlib.import_module(f"csv_inspector.{info.name}"))
    return modules


def _library_sources() -> list[Path]:
    """Every package source file except the CLI layer."""
    return sorted(p for p in PACKAGE_DIR.glob("*.py") if p.name not in CLI_MODULES)


# ---------------------------------------------------------------------
# Installation and public API
# ---------------------------------------------------------------------


def test_the_package_is_installed_not_imported_from_a_path_hack() -> None:
    """csv_inspector resolves through installed distribution metadata."""
    distribution = importlib.metadata.distribution("csv-inspector")

    assert csv_inspector.__version__ == distribution.version
    assert re.fullmatch(r"\d+\.\d+\.\d+", csv_inspector.__version__)


def test_the_distribution_declares_the_cloud_extra_and_console_script() -> None:
    """The [cloud] extra and the csv-inspector entry point are part of the metadata."""
    distribution = importlib.metadata.distribution("csv-inspector")
    entry_points = {ep.name: ep.value for ep in distribution.entry_points}

    assert "cloud" in (distribution.metadata.get_all("Provides-Extra") or [])
    assert entry_points["csv-inspector"] == "csv_inspector.cli:main"


def test_the_package_ships_a_py_typed_marker() -> None:
    """PEP 561: type checkers use the package's inline annotations."""
    assert (PACKAGE_DIR / "py.typed").is_file()


def test_public_api_is_exactly_the_documented_set() -> None:
    """``__all__`` is the contract: nothing missing, nothing accidental."""
    assert set(csv_inspector.__all__) == EXPECTED_PUBLIC_API


def test_every_public_name_is_importable() -> None:
    """Each name in ``__all__`` resolves on the package."""
    for name in csv_inspector.__all__:
        assert getattr(csv_inspector, name) is not None, name


def test_exported_sample_limit_is_the_enforced_one() -> None:
    """MAX_SAMPLE_BYTES is the bound inspect_csv enforces, not a copy (issue #100)."""
    limit = csv_inspector.MAX_SAMPLE_BYTES
    assert 0 < csv_inspector.DEFAULT_SAMPLE_BYTES <= limit
    assert 0 <= csv_inspector.DEFAULT_TAIL_BYTES <= limit

    def inspect(n_bytes: int, tail_bytes: int) -> None:
        csv_inspector.inspect_csv(
            b"a,b\n1,2\n",
            model="m",
            fallback_model="m",
            model_invoker=lambda prompt, model: "{}",
            n_bytes=n_bytes,
            tail_bytes=tail_bytes,
        )

    with pytest.raises(ValueError, match="n_bytes"):
        inspect(limit + 1, 0)
    with pytest.raises(ValueError, match="tail_bytes"):
        inspect(limit, limit + 1)


def test_all_exceptions_share_the_domain_base() -> None:
    """Hosts can catch every domain failure with CSVInspectorError."""
    for name in csv_inspector.__all__:
        obj = getattr(csv_inspector, name)
        if isinstance(obj, type) and issubclass(obj, Exception):
            assert issubclass(obj, csv_inspector.CSVInspectorError), name


# ---------------------------------------------------------------------
# Library-grade logging and output
# ---------------------------------------------------------------------


@pytest.mark.parametrize("source", _library_sources(), ids=lambda path: path.name)
def test_library_modules_never_configure_logging_or_print(source: Path) -> None:
    """Only the CLI layer may call logging.basicConfig() or print()."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    offenders = [
        f"line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "print")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "basicConfig")
        )
    ]

    assert offenders == []


def test_importing_the_package_adds_only_a_null_handler() -> None:
    """The package logger has exactly one NullHandler."""
    package_logger = logging.getLogger("csv_inspector")

    assert [type(h) for h in package_logger.handlers] == [logging.NullHandler]


def test_importing_and_using_the_library_leaves_the_root_logger_alone(tmp_path: Path) -> None:
    """In a fresh process, importing and running the library adds no root handlers."""
    code = (
        "import json, logging\n"
        "from csv_inspector import inspect_csv\n"
        "answer = json.dumps({'encoding': 'utf-8', 'delimiter': ';', 'header_row_index': 0,\n"
        "                     'columns': [], 'confidence': 1.0})\n"
        "inspect_csv(b'a;b\\n1;2\\n', model='m', fallback_model='m',\n"
        "            model_invoker=lambda p, m: answer)\n"
        "print(len(logging.getLogger().handlers), logging.getLogger().level)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["0", str(logging.WARNING)]
    assert completed.stderr == ""


def test_module_loggers_live_under_the_package_namespace() -> None:
    """Hosts can configure every library log record through 'csv_inspector'."""
    for module in _package_modules():
        logger = getattr(module, "logger", None)
        if isinstance(logger, logging.Logger):
            assert logger.name.startswith("csv_inspector"), module.__name__


# ---------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------


def test_no_mutable_state_at_module_level() -> None:
    """No module-level list/dict/set/bytearray that calls could share."""
    mutable = (list, dict, set, bytearray)
    offenders = [
        f"{module.__name__}.{name}"
        for module in _package_modules()
        for name, value in vars(module).items()
        if isinstance(value, mutable) and not (name.startswith("__") and name.endswith("__"))
    ]

    assert offenders == []


def test_concurrent_inspections_from_threads_are_independent() -> None:
    """Parallel calls each get the answer for their own source."""
    delimiters = [",", ";", "|", "\t"] * 8

    def invoker_for(prompt: str, model: str) -> str:
        # Answer with whatever delimiter this call's own sample uses.
        delimiter = next(d for d in (",", ";", "|", "\t") if f"a{d}b" in prompt)
        return json.dumps(
            {
                "encoding": "utf-8",
                "delimiter": delimiter,
                "header_row_index": 0,
                "columns": [{"name": "a", "inferred_type": "string"}],
                "confidence": 0.9,
            }
        )

    def inspect_one(delimiter: str) -> str:
        source = f"a{delimiter}b\n1{delimiter}2\n".encode()
        result = inspect_csv(
            source,
            model="m",
            fallback_model="m",
            model_invoker=invoker_for,
            timeout_seconds=10,
        )
        return result.delimiter

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(inspect_one, delimiters))

    assert results == delimiters


# ---------------------------------------------------------------------
# Explicit settings isolate the host from the process environment
# ---------------------------------------------------------------------


def test_explicit_settings_ignore_process_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With settings=, OLLAMA_MODEL from the environment has no effect."""
    monkeypatch.setenv("OLLAMA_MODEL", "from-environment")
    fake = install_fake_ollama(monkeypatch, lambda **kwargs: ollama_reply(_ANSWER))

    inspect_csv(b"a;b\n1;2\n", settings=Settings(ollama_model="injected"))

    assert fake.requests[0]["model"] == "injected"


def test_explicit_settings_never_call_load_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Injecting settings means the environment loader is not even consulted."""

    def forbidden(**kwargs: Any) -> Settings:
        raise AssertionError("load_settings must not be called when settings are injected")

    monkeypatch.setattr("csv_inspector._config.load_settings", forbidden)
    install_fake_ollama(monkeypatch, lambda **kwargs: ollama_reply(_ANSWER))

    inspect_csv(b"a;b\n1;2\n", settings=Settings())


def test_without_settings_the_environment_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """Standalone mode (no settings=) still honours environment variables."""
    monkeypatch.setenv("OLLAMA_MODEL", "from-environment")
    fake = install_fake_ollama(monkeypatch, lambda **kwargs: ollama_reply(_ANSWER))

    inspect_csv(b"a;b\n1;2\n")

    assert fake.requests[0]["model"] == "from-environment"


def test_two_tenants_with_different_settings_do_not_interfere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent calls with different injected settings keep their own models."""
    fake = install_fake_ollama(monkeypatch, lambda **kwargs: ollama_reply(_ANSWER))
    tenants = [Settings(ollama_model=f"tenant-{i}") for i in range(6)]

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda s: inspect_csv(b"a;b\n1;2\n", settings=s), tenants))

    assert sorted(request["model"] for request in fake.requests) == sorted(
        f"tenant-{i}" for i in range(6)
    )


def test_injected_api_settings_are_checked_before_the_source_is_read() -> None:
    """A tenant without credentials fails fast, without consuming its stream."""

    class ExplodingStream:
        def read(self, size: int = -1) -> bytes:
            raise AssertionError("the source must not be read")

    with pytest.raises(csv_inspector.CredentialsNotConfiguredError):
        inspect_csv(ExplodingStream(), backend=LLMBackend.API, settings=Settings())


_ANSWER = json.dumps(
    {
        "encoding": "utf-8",
        "delimiter": ";",
        "header_row_index": 0,
        "columns": [{"name": "a", "inferred_type": "string"}],
        "confidence": 0.9,
    }
)
