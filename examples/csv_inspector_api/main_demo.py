"""Demo of the csv-inspector API: start it, inspect one file over HTTP, print the result.

Starts the API in-process with uvicorn, waits for ``GET /health``, uploads a
file to ``POST /inspect`` and prints the JSON answer. With ``--keep-running``
the server stays up afterwards for ``curl`` or the interactive docs.

Run from the repository root, with ``ollama serve`` running::

    uv run examples/csv_inspector_api/main_demo.py

This is the only module of the example that may ``print()`` or call
``logging.basicConfig()``: it is the application, the API package is not.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from csv_inspector import LLMBackend

# The example is never installed: put its package on the path, as pytest and mypy do.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from csv_inspector_api import create_app
from csv_inspector_api.request_id import RequestIdFilter
from csv_inspector_api.settings import ApiSettings

logger = logging.getLogger("csv_inspector_api.demo")

DEFAULT_FILE = Path(__file__).resolve().parents[2] / "agents" / "csv_inspector" / "sample.csv"
STARTUP_TIMEOUT_SECONDS = 15
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the demo's arguments, named like the csv-inspector CLI.

    Args:
        argv: Arguments without the program name; ``None`` reads ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on.")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="CSV/TSV file to upload.")
    parser.add_argument(
        "--backend",
        choices=[backend.value for backend in LLMBackend],
        default=None,
        help="LLM backend: 'local' (Ollama, default) or 'api' (Gemini; needs credentials).",
    )
    parser.add_argument(
        "--model", default=None, help="Primary model. Defaults to the backend's configured model."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Time budget of the inspection, in seconds. Defaults to the API's default.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=LOG_LEVELS, help="Logging verbosity."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Read CSV_INSPECTOR_API_* settings from this file (variables still win). "
        "Default: ./.env, if it exists.",
    )
    group.add_argument(
        "--no-env-file", action="store_true", help="Read settings from environment variables only."
    )
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Leave the server up after the demo request, until Ctrl+C.",
    )
    return parser.parse_args(argv)


def build_settings(args: argparse.Namespace) -> ApiSettings:
    """Build the API settings from the environment, the env file and the arguments.

    Args:
        args: Parsed arguments.

    Returns:
        The settings; ``--backend`` and ``--model`` override the environment.
    """
    overrides: dict[str, Any] = {}
    if args.backend is not None:
        overrides["llm_backend"] = LLMBackend(args.backend)
    env_file = None if args.no_env_file else args.env_file
    settings = ApiSettings(_env_file=env_file, **overrides)
    if args.model is not None:
        field = "cloud_model" if settings.llm_backend is LLMBackend.API else "ollama_model"
        settings = settings.model_copy(update={field: args.model})
    return settings


def wait_until_healthy(base_url: str, server_thread: threading.Thread) -> dict[str, Any]:
    """Poll ``GET /health`` until the server answers.

    Args:
        base_url: Root URL of the server.
        server_thread: The thread running the server, to stop waiting if it dies.

    Returns:
        The health body.

    Raises:
        RuntimeError: If the server stops or does not answer in time.
    """
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline and server_thread.is_alive():
        try:
            response = httpx.get(f"{base_url}/health", timeout=1)
        except httpx.TransportError:
            time.sleep(0.1)
            continue
        response.raise_for_status()
        health: dict[str, Any] = response.json()
        return health
    msg = f"the API did not become healthy at {base_url} within {STARTUP_TIMEOUT_SECONDS} s"
    raise RuntimeError(msg)


def main(argv: list[str] | None = None) -> int:
    """Run the demo.

    Args:
        argv: Arguments without the program name; ``None`` reads ``sys.argv``.

    Returns:
        Process exit code: 0 when the inspection succeeded, 1 otherwise.
    """
    args = parse_args(argv)
    # Windows consoles default to a legacy code page that mangles accented
    # characters, which are common in real CSV exports: print UTF-8, as the
    # csv-inspector CLI does.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    # The request-id filter goes on the handler, so every record it emits,
    # the csv_inspector library's included, carries the id of its request.
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    logging.basicConfig(
        level=args.log_level,
        format="%(levelname)s [%(request_id)s] %(name)s: %(message)s",
        handlers=[handler],
    )
    settings = build_settings(args)

    base_url = f"http://127.0.0.1:{args.port}"
    config = uvicorn.Config(
        create_app(settings),
        host="127.0.0.1",
        port=args.port,
        log_level=args.log_level.lower(),
        # The API logs one access line per request, with its id (csv_inspector_api.access).
        access_log=False,
    )
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    server_thread.start()
    try:
        health = wait_until_healthy(base_url, server_thread)
        logger.info("API up at %s: %s", base_url, health)

        params = {} if args.timeout is None else {"timeout_seconds": args.timeout}
        # The client waits a little longer than the server-side budget, so the
        # API's own 504 comes back instead of a client timeout.
        client_timeout = (args.timeout or settings.default_timeout_seconds) + 30
        with args.file.open("rb") as upload:
            response = httpx.post(
                f"{base_url}/inspect",
                params=params,
                files={"file": (args.file.name, upload, "text/csv")},
                timeout=client_timeout,
            )
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        if response.is_error:
            logger.error("POST /inspect answered %d", response.status_code)

        if args.keep_running:
            print(f"Serving at {base_url} (docs at {base_url}/docs); Ctrl+C to stop.")
            while server_thread.is_alive():
                server_thread.join(0.5)
        return 1 if response.is_error else 0
    except KeyboardInterrupt:
        return 0
    finally:
        server.should_exit = True
        server_thread.join()


if __name__ == "__main__":
    sys.exit(main())
