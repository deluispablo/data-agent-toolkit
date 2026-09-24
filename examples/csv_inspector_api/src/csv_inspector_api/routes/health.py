"""``GET /health``: liveness plus the backend and models the API is configured with."""

from __future__ import annotations

from typing import Annotated, Literal

import csv_inspector
from csv_inspector import LLMBackend, Settings, ensure_backend_ready
from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from ..errors import problem_responses

router = APIRouter(tags=["meta"])


class HealthResponse(BaseModel):
    """Body of ``GET /health``.

    Carries no secret and no cloud project or location: only what a caller
    needs to know which models answer its requests.

    Attributes:
        status: Always ``ok``: the process is up and serving.
        csv_inspector_version: Version of the embedded library.
        api_version: Version of this example API.
        backend: Model backend used for inspections.
        model: Primary model.
        fallback_model: Model tried when the primary one fails.
    """

    status: Literal["ok"] = "ok"
    csv_inspector_version: str
    api_version: str
    backend: LLMBackend
    model: str
    fallback_model: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and configuration",
    responses=problem_responses(503),
)
async def health(
    request: Request,
    probe: Annotated[
        bool,
        Query(
            description="Also check the backend configuration (credentials, SDK), "
            "without a network or model call; 503 when it is unusable."
        ),
    ] = False,
) -> HealthResponse:
    """Report that the API is up, and which backend and models it uses.

    The model backend is never contacted: a health check must be cheap and
    free. With ``probe=true`` the library's own readiness check runs, the one
    every inspection runs first; a failure is a 503 through the error handler.
    It is skipped when the app runs with a custom model invoker, as the
    library skips it then. It makes no network call, so it does not prove
    Ollama is reachable: for the ``local`` backend it always passes. Use
    ``/health`` as the liveness probe and ``/health?probe=true`` as the
    readiness or startup probe. No API key, project or location is reported.
    """
    settings: Settings = request.app.state.library_settings
    backend = settings.llm_backend
    if probe and request.app.state.model_invoker is None:
        ensure_backend_ready(backend, settings)
    return HealthResponse(
        csv_inspector_version=csv_inspector.__version__,
        api_version=request.app.version,
        backend=backend,
        model=settings.model_for(backend),
        fallback_model=settings.fallback_model_for(backend),
    )
