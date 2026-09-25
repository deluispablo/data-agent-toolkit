"""Command-line interface for csv_inspector (the application layer).

This is the only module in the package that configures logging, writes to
stdout, or decides to read a ``.env`` file: those are application choices,
never library behaviour. Installed as the ``csv-inspector`` console script
and runnable as ``python -m csv_inspector``.

Usage:
    csv-inspector data.csv
    csv-inspector data.csv --model qwen2.5-coder:7b --fallback-model qwen2.5-coder:3b
    csv-inspector data.csv --bytes 8192 --timeout 30
    csv-inspector data.csv --stats
    csv-inspector data.csv --backend api --model gemini-3.6-flash --env-file secrets.env
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from ._backends import LLMBackend
from ._config import Settings, ensure_backend_ready, load_settings
from ._exceptions import BackendConfigurationError, CSVInspectorError, InspectionTimeoutError
from ._inspect import inspect_csv
from ._sampling import DEFAULT_SAMPLE_BYTES, DEFAULT_TAIL_BYTES, MAX_SAMPLE_BYTES

logger = logging.getLogger(__name__)

LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_ENV_FILE = Path(".env")
# The library has no default budget (hosts set their own), but a person at a
# shell should not wait forever on a stalled Ollama. 300 s covers a cold 7B
# load on CPU.
DEFAULT_CLI_TIMEOUT_SECONDS = 300.0


def bounded_int(minimum: int, maximum: int | None = None) -> Callable[[str], int]:
    """An argparse ``type=`` for integers in ``[minimum, maximum]`` (``None``: unbounded).

    Anything else raises :class:`argparse.ArgumentTypeError`, a usage error
    naming the option.
    """

    def convert(value: str) -> int:
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
        if number < minimum:
            raise argparse.ArgumentTypeError(f"expected an integer >= {minimum}, got {number}")
        if maximum is not None and number > maximum:
            raise argparse.ArgumentTypeError(f"expected an integer <= {maximum}, got {number}")
        return number

    return convert


HEAD_BYTES = bounded_int(1, MAX_SAMPLE_BYTES)
"""``--bytes`` converter: a head window of 1 to ``MAX_SAMPLE_BYTES`` bytes."""

TAIL_BYTES = bounded_int(0, MAX_SAMPLE_BYTES)
"""``--tail-bytes`` converter: a tail window of 0 (off) to ``MAX_SAMPLE_BYTES`` bytes."""


def timeout_budget(value: str) -> float | None:
    """Argparse ``type=`` for ``--timeout``: seconds, or ``None`` (no limit) for 0.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not a number >= 0.
    """
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not number >= 0:
        raise argparse.ArgumentTypeError(f"expected a number >= 0, got {number}")
    return number or None


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
    """Load settings for a CLI run: the environment plus ``--env-file``, else ``./.env``.

    Unlike the library, the CLI reads ``./.env`` by default (unless
    ``--no-env-file``): the user running it chose the working directory.

    Raises:
        BackendConfigurationError: If an explicitly requested ``.env`` file is
            missing or unreadable, or a setting holds an invalid value.
    """
    if env_file is not None and not env_file.is_file():
        raise BackendConfigurationError(f"Settings file '{env_file}' does not exist.")
    if env_file is None and not no_env_file and _DEFAULT_ENV_FILE.is_file():
        env_file = _DEFAULT_ENV_FILE
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
    """Configure logging and force UTF-8 stdout and stderr for a command-line entry point.

    Windows consoles default to a legacy code page (e.g. cp1252), which
    mangles the accented characters common in real-world CSV exports.
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
        "--fallback-model",
        default=None,
        help="Model to try if --model fails. Defaults to the backend's configured fallback model.",
    )
    parser.add_argument(
        "--bytes",
        type=HEAD_BYTES,
        default=DEFAULT_SAMPLE_BYTES,
        help=f"Number of leading (head) bytes to sample (at most {MAX_SAMPLE_BYTES}).",
    )
    parser.add_argument(
        "--tail-bytes",
        type=TAIL_BYTES,
        default=DEFAULT_TAIL_BYTES,
        help=(
            "Number of trailing (tail) bytes to sample, for footer detection "
            f"(0 disables; at most {MAX_SAMPLE_BYTES})."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=timeout_budget,
        default=DEFAULT_CLI_TIMEOUT_SECONDS,
        help=(
            "Overall time budget for the model calls, in seconds "
            f"(default: {DEFAULT_CLI_TIMEOUT_SECONDS:g}; 0 disables the limit)."
        ),
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="After the result, print the inspection's usage (model, tokens, latency, "
        "attempts, retries) to stderr as JSON; stdout stays the result only.",
    )
    add_settings_arguments(parser)
    add_log_level_argument(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, default_file: Path | None = None) -> None:
    """Run the CLI: inspect one file and print the result as JSON.

    ``default_file`` (``main_demo.py``'s) is inspected when the command line
    names none; without it the file argument is required.
    """
    args = _parse_args(argv, default_file)
    configure_cli(args.log_level)

    try:
        settings = load_cli_settings(args.env_file, no_env_file=args.no_env_file)
        backend = resolve_backend(args.backend, settings)
        model = args.model or settings.model_for(backend)
        fallback_model = args.fallback_model or settings.fallback_model_for(backend)
        logger.info(
            "Inspecting '%s' with %s model '%s', fallback '%s' (head=%d bytes, tail=%d bytes).",
            args.file,
            backend.value,
            model,
            fallback_model,
            args.bytes,
            args.tail_bytes,
        )
        result = inspect_csv(
            args.file,
            backend=backend,
            settings=settings,
            model=model,
            fallback_model=fallback_model,
            n_bytes=args.bytes,
            tail_bytes=args.tail_bytes,
            timeout_seconds=args.timeout,
        )
    except CSVInspectorError as exc:
        # Expected failure modes get a one-line message; the traceback is
        # only useful when debugging.
        logger.error("Inspection failed: %s", exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        if isinstance(exc, InspectionTimeoutError):
            logger.error("Raise --timeout, or pass --timeout 0 to wait without a limit.")
        sys.exit(1)

    print(result.model_dump_json(indent=2))
    if args.stats and result.usage is not None:
        print(result.usage.model_dump_json(indent=2), file=sys.stderr)

    logger.info(
        "Summary: encoding=%s delimiter=%r header_row=%s columns=%s confidence=%.2f",
        result.encoding,
        result.delimiter,
        result.header_row_index if result.has_header else "none",
        result.columns,
        result.confidence,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
