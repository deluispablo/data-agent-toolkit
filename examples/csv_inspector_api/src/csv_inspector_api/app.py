"""Application factory of the API."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI

from .errors import register_exception_handlers
from .settings import ApiSettings

# The example is never built or installed, so there is no package metadata to
# read the version from: keep it in step with pyproject.toml by hand.
__version__ = "0.0.0"

AsyncModelInvoker = Callable[[str, str], Awaitable[str]]
"""Async ``(prompt, model) -> raw response`` callable, as ``ainspect_csv`` accepts."""


def create_app(
    settings: ApiSettings | None = None,
    *,
    model_invoker: AsyncModelInvoker | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    Serve it with ``uvicorn --app-dir src csv_inspector_api.app:create_app --factory``.

    Args:
        settings: API settings. Defaults to :class:`ApiSettings` read from the
            ``CSV_INSPECTOR_API_*`` environment variables.
        model_invoker: Replacement for the built-in model client, forwarded to
            ``csv_inspector.ainspect_csv``. A test seam: with it, requests never
            reach Ollama or a cloud API.

    Returns:
        The application, with ``settings``, ``library_settings`` and
        ``model_invoker`` stored on ``app.state``.
    """
    settings = settings if settings is not None else ApiSettings()
    app = FastAPI(
        title="csv-inspector API",
        version=__version__,
        description="Example HTTP host that embeds the csv-inspector agent.",
    )
    app.state.settings = settings
    app.state.library_settings = settings.to_library_settings()
    app.state.model_invoker = model_invoker
    register_exception_handlers(app)
    return app
