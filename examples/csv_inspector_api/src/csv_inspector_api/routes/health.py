"""``GET /health``: liveness plus the backend and models the API is configured with."""

from __future__ import annotations

from typing import Literal

import csv_inspector
from csv_inspector import LLMBackend, Settings
from fastapi import APIRouter, Request
from pydantic import BaseModel

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


@router.get("/health", response_model=HealthResponse, summary="Liveness and configuration")
async def health(request: Request) -> HealthResponse:
    """Report that the API is up, and which backend and models it uses.

    The model backend is never contacted: a health check must be cheap and
    free. A reachability probe needs ``ensure_backend_ready``, which
    ``csv_inspector`` does not export publicly, so there is none.
    """
    settings: Settings = request.app.state.library_settings
    backend = settings.llm_backend
    return HealthResponse(
        csv_inspector_version=csv_inspector.__version__,
        api_version=request.app.version,
        backend=backend,
        model=settings.model_for(backend),
        fallback_model=settings.fallback_model_for(backend),
    )
