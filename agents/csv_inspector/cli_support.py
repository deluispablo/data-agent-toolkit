"""Shared command-line plumbing for the csv_inspector scripts.

Keeps ``main_demo.py`` and ``eval_samples.py`` consistent in how they parse
byte budgets, select the LLM backend, configure logging, and write non-ASCII
output.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys

from backends import LLMBackend
from inspector import ensure_backend_ready, get_configured_backend

LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def positive_int(value: str) -> int:
    """Argparse ``type=`` converter accepting only integers >= 1.

    Args:
        value: The raw command-line string.

    Returns:
        The parsed integer.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not an integer >= 1.
    """
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {number}")
    return number


def non_negative_int(value: str) -> int:
    """Argparse ``type=`` converter accepting only integers >= 0.

    Args:
        value: The raw command-line string.

    Returns:
        The parsed integer.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not an integer >= 0.
    """
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if number < 0:
        raise argparse.ArgumentTypeError(f"expected an integer >= 0, got {number}")
    return number


def add_backend_argument(parser: argparse.ArgumentParser) -> None:
    """Add the standard ``--backend`` option to ``parser``.

    The option defaults to ``None`` so :func:`resolve_backend` can fall back
    to the ``LLM_BACKEND`` setting, and then to the local backend.

    Args:
        parser: The parser to extend.
    """
    parser.add_argument(
        "--backend",
        choices=[backend.value for backend in LLMBackend],
        default=None,
        help="LLM backend: 'local' (Ollama, default) or 'api' (Gemini; needs credentials). "
        "Defaults to LLM_BACKEND, then 'local'.",
    )


def resolve_backend(value: str | None) -> LLMBackend:
    """Resolve the backend to use and check it is ready, before any work starts.

    Precedence: the ``--backend`` value, then ``LLM_BACKEND``, then local.

    Args:
        value: The raw ``--backend`` value, or ``None`` when not given.

    Returns:
        The resolved backend.

    Raises:
        BackendConfigurationError: If the backend cannot be used as
            configured (e.g. the cloud extra is not installed).
        CredentialsNotConfiguredError: If the cloud backend has no usable
            credentials.
    """
    backend = LLMBackend(value) if value is not None else get_configured_backend()
    ensure_backend_ready(backend)
    return backend


def add_log_level_argument(parser: argparse.ArgumentParser) -> None:
    """Add the standard ``--log-level`` option to ``parser``.

    Args:
        parser: The parser to extend.
    """
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVELS,
        help="Logging verbosity.",
    )


def configure_cli(log_level: str) -> None:
    """Configure logging and force UTF-8 output for a command-line entry point.

    Windows consoles default to a legacy code page (e.g. cp1252), which
    mangles the accented characters common in real-world CSV exports, so
    stdout (results) and stderr (logs) are switched to UTF-8 when they are
    regular text streams.

    Args:
        log_level: One of :data:`LOG_LEVELS`.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=log_level, format=_LOG_FORMAT)
