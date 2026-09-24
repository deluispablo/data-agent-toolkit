"""Application factory of the API."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from csv_inspector import AsyncModelInvoker
from fastapi import FastAPI

from .errors import register_exception_handlers
from .request_id import RequestIdMiddleware
from .routes import health
from .routes.inspect import build_inspect_router
from .settings import ApiSettings
from .sources.gcs import GcsClient, GcsNotInstalledError, create_client

logger = logging.getLogger(__name__)

# The example is never built or installed, so there is no package metadata to
# read the version from: keep it in step with pyproject.toml by hand.
__version__ = "0.0.0"

_REPOSITORY = "https://github.com/deluispablo/data-agent-toolkit"

_DESCRIPTION = f"""Example HTTP host that embeds the **csv-inspector** agent: upload a CSV or TSV
file and get back its encoding, dialect, header row, footer lines and a
preliminary column schema, inferred by a model (a local Ollama by default) from
a bounded head and tail sample. It is executable documentation of how to embed
the agent, not a product: see the
[example's README]({_REPOSITORY}/tree/main/examples/csv_inspector_api) and the
[embedding guide]({_REPOSITORY}/blob/main/agents/csv_inspector/docs/embedding.md).
"""

_TAGS = [
    {"name": "inspection", "description": "Inspect CSV/TSV files."},
    {"name": "meta", "description": "Health and configuration of the service."},
]


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the Cloud Storage client once, at startup, unless one was injected.

    A failure does not stop the API, whose other routes need no Cloud
    Storage: ``POST /inspect/gcs`` tries again on each request and answers
    with the error's problem response until it succeeds.
    """
    if app.state.gcs_client is None:
        try:
            app.state.gcs_client = create_client(app.state.settings.google_cloud_project)
        except GcsNotInstalledError as exc:
            logger.info("POST /inspect/gcs disabled: %s", exc)
        except Exception as exc:  # noqa: BLE001 - e.g. DefaultCredentialsError; retried per request
            logger.warning("no Cloud Storage client at startup: %s: %s", type(exc).__name__, exc)
    yield


def create_app(
    settings: ApiSettings | None = None,
    *,
    model_invoker: AsyncModelInvoker | None = None,
    gcs_client: GcsClient | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    Serve it with ``uvicorn --app-dir src csv_inspector_api.app:create_app --factory``.

    Args:
        settings: API settings. Defaults to :class:`ApiSettings` read from the
            ``CSV_INSPECTOR_API_*`` environment variables.
        model_invoker: Replacement for the built-in model client, forwarded to
            ``csv_inspector.ainspect_csv``. A test seam: with it, requests never
            reach Ollama or a cloud API.
        gcs_client: Cloud Storage client of ``POST /inspect/gcs``. Defaults to
            a ``google.cloud.storage.Client`` with Application Default
            Credentials, built at startup (the ``[gcs]`` extra). A test seam too.

    Returns:
        The application, with ``settings``, ``library_settings``,
        ``model_invoker`` and ``gcs_client`` stored on ``app.state``.
    """
    settings = settings if settings is not None else ApiSettings()
    app = FastAPI(
        title="csv-inspector API example",
        version=__version__,
        summary="Infer the dialect, header, footer and schema of an uploaded CSV/TSV file.",
        description=_DESCRIPTION,
        openapi_tags=_TAGS,
        lifespan=_lifespan,
    )
    app.state.settings = settings
    app.state.library_settings = settings.to_library_settings()
    app.state.model_invoker = model_invoker
    app.state.gcs_client = gcs_client
    register_exception_handlers(app)
    app.add_middleware(RequestIdMiddleware)
    app.include_router(build_inspect_router(settings))
    app.include_router(health.router)
    return app
