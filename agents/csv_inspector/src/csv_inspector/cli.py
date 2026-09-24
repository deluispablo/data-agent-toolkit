"""Command-line interface for csv_inspector (the application layer).

This is the only module in the package that configures logging, writes to
stdout, or decides to read a ``.env`` file: those are application choices,
never library behaviour. Installed as the ``csv-inspector`` console script
and runnable as ``python -m csv_inspector``.

Usage:
    csv-inspector data.csv
    csv-inspector data.csv --model qwen2.5-coder:7b --bytes 8192 --timeout 30
    csv-inspector data.csv --backend api --model gemini-2.5-flash --env-file secrets.env
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from ._backends import LLMBackend
from ._config import Settings, load_settings
from ._exceptions import BackendConfigurationError, CSVInspectorError
from ._inspect import inspect_csv
from ._invokers import ensure_backend_ready
from ._sampling import DEFAULT_SAMPLE_BYTES, DEFAULT_TAIL_BYTES, MAX_SAMPLE_BYTES

logger = logging.getLogger(__name__)

LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_ENV_FILE = Path(".env")


def positive_int(value: str) -> int:
    """Argparse ``type=`` converter accepting only integers >= 1.

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


def positive_float(value: str) -> float:
    """Argparse ``type=`` converter accepting only numbers > 0.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not a number > 0.
    """
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not number > 0:
        raise argparse.ArgumentTypeError(f"expected a number > 0, got {number}")
    return number


def add_backend_argument(parser: argparse.ArgumentParser) -> None:
    """Add the standard ``--backend`` option (default: ``LLM_BACKEND``, then local)."""
    parser.add_argument(
        "--backend",
        choices=[backend.value for backend in LLMBackend],
        default=None,
        help="LLM backend: 'local' (Ollama, default) or 'api' (Gemini; needs credentials). "
        "Defaults to LLM_BACKEND, then 'local'.",
    )


def add_settings_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--env-file`` / ``--no-env-file``, controlling where settings come from."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Read settings from this .env file (environment variables still win). "
        "Default: ./.env if it exists.",
    )
    group.add_argument(
        "--no-env-file",
        action="store_true",
        help="Read settings from environment variables only; ignore ./.env.",
    )


def add_log_level_argument(parser: argparse.ArgumentParser) -> None:
    """Add the standard ``--log-level`` option."""
    parser.add_argument(
        "--log-level", default="INFO", choices=LOG_LEVELS, help="Logging verbosity."
    )


def load_cli_settings(env_file: Path | None, *, no_env_file: bool) -> Settings:
    """Load settings for a CLI run: environment variables plus an optional ``.env``.

    Unlike the library, the CLI reads ``./.env`` by default when it exists:
    the user running the command chose the working directory.

    Args:
        env_file: An explicit ``.env`` path (``--env-file``), or ``None``.
        no_env_file: Whether ``--no-env-file`` was given.

    Returns:
        The loaded settings; built-in defaults when ``pydantic-settings`` is
        not installed and no ``.env`` file was requested.

    Raises:
        BackendConfigurationError: If an explicitly requested ``.env`` file is
            missing, needs the ``[cloud]`` extra, or holds invalid values.
    """
    if env_file is not None and not env_file.is_file():
        raise BackendConfigurationError(f"Settings file '{env_file}' does not exist.")
    if env_file is None and not no_env_file and _DEFAULT_ENV_FILE.is_file():
        env_file = _DEFAULT_ENV_FILE
    if env_file is None and importlib.util.find_spec("pydantic_settings") is None:
        return Settings()
    return load_settings(env_file=env_file)


def resolve_backend(value: str | None, settings: Settings) -> LLMBackend:
    """Resolve the backend (flag, then settings) and check it is ready.

    Raises:
        BackendConfigurationError: If the backend cannot be used as configured.
        CredentialsNotConfiguredError: If the cloud backend has no credentials.
    """
    backend = LLMBackend(value) if value is not None else settings.llm_backend
    ensure_backend_ready(backend, settings)
    return backend


def configure_cli(log_level: str) -> None:
    """Configure logging and force UTF-8 output for a command-line entry point.

    Windows consoles default to a legacy code page (e.g. cp1252), which
    mangles the accented characters common in real-world CSV exports, so
    stdout (results) and stderr (logs) are switched to UTF-8 when they are
    regular text streams.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=log_level, format=_LOG_FORMAT)


def _parse_args(argv: Sequence[str] | None, default_file: Path | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="csv-inspector",
        description="Infer the dialect, header/footer layout and schema of a CSV/TSV file.",
    )
    parser.add_argument(
        "file",
        type=Path,
        nargs="?" if default_file is not None else None,
        default=default_file,
        help="Path to the CSV/TSV file to inspect.",
    )
    add_backend_argument(parser)
    parser.add_argument(
        "--model", default=None, help="Model to use. Defaults to the backend's configured model."
    )
    parser.add_argument(
        "--bytes",
        type=positive_int,
        default=DEFAULT_SAMPLE_BYTES,
        help=f"Number of leading (head) bytes to sample (at most {MAX_SAMPLE_BYTES}).",
    )
    parser.add_argument(
        "--tail-bytes",
        type=non_negative_int,
        default=DEFAULT_TAIL_BYTES,
        help=(
            "Number of trailing (tail) bytes to sample, for footer detection "
            f"(0 disables; at most {MAX_SAMPLE_BYTES})."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=None,
        help="Overall time budget for the model calls, in seconds (default: no limit).",
    )
    add_settings_arguments(parser)
    add_log_level_argument(parser)
    args = parser.parse_args(argv)
    for option, value in (("--bytes", args.bytes), ("--tail-bytes", args.tail_bytes)):
        if value > MAX_SAMPLE_BYTES:
            parser.error(f"{option}: expected an integer <= {MAX_SAMPLE_BYTES}, got {value}")
    return args


def main(argv: Sequence[str] | None = None, *, default_file: Path | None = None) -> None:
    """Run the CLI: inspect one file and print the result as JSON.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).
        default_file: File to inspect when none is given on the command line
            (used by the repository's ``main_demo.py``); when ``None``, the
            file argument is required.
    """
    args = _parse_args(argv, default_file)
    configure_cli(args.log_level)

    try:
        settings = load_cli_settings(args.env_file, no_env_file=args.no_env_file)
        backend = resolve_backend(args.backend, settings)
        model = args.model or settings.model_for(backend)
        logger.info(
            "Inspecting '%s' with %s model '%s' (head=%d bytes, tail=%d bytes).",
            args.file,
            backend.value,
            model,
            args.bytes,
            args.tail_bytes,
        )
        result = inspect_csv(
            args.file,
            backend=backend,
            settings=settings,
            model=model,
            n_bytes=args.bytes,
            tail_bytes=args.tail_bytes,
            timeout_seconds=args.timeout,
        )
    except CSVInspectorError as exc:
        # Expected failure modes get a one-line message; the traceback is
        # only useful when debugging.
        logger.error("Inspection failed: %s", exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        sys.exit(1)

    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

    logger.info(
        "Summary: encoding=%s delimiter=%r header_row=%d columns=%s confidence=%.2f",
        result.encoding,
        result.delimiter,
        result.header_row_index,
        [column.name for column in result.columns],
        result.confidence,
    )
    if result.notes:
        logger.info("Notes: %s", result.notes)


if __name__ == "__main__":  # pragma: no cover
    main()
