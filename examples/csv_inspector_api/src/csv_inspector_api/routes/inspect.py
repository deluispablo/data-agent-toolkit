"""``POST /inspect``, ``/inspect/raw`` and ``/inspect/gcs``: inspect a CSV/TSV file.

No ``from __future__ import annotations``: FastAPI evaluates the annotations.
``/inspect`` takes ``multipart/form-data`` (browsers, ``curl -F``): Starlette
receives the whole upload first (spooled to disk past 1 MiB), so its ``413``
caps what is inspected, not what is received: cap the request size at the
reverse proxy too. ``/inspect/raw`` takes the file as the body (scripts,
pipes, ``curl --data-binary``), inspected as it streams with ``n_bytes +
tail_bytes`` plus one chunk in memory and nothing on disk; its ``413`` comes
from ``Content-Length`` or while streaming. Past 64 MiB after the head the
body is not read to its end: no tail, no footer. A client silent for
``timeout_seconds`` fails the read (``422``); one that disconnects releases
the worker thread at once. Never send ``multipart/form-data`` to it: the
envelope would be inspected as the file.

At most ``max_concurrent_inspections`` inspections run at once in a process;
the others wait for a slot up to ``queue_timeout_seconds``, before reading any
byte, then get ``503`` with ``Retry-After`` (``GET /health`` is never gated).
Content types are not checked: the library detects what the bytes are.
Invalid query parameters are FastAPI's own ``422``; every other error is a
problem response (``errors.py``). Model names are plain tokens: they appear
in log lines.
"""

import asyncio
import functools
import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Annotated, Any, cast

from csv_inspector import (
    DEFAULT_SAMPLE_BYTES,
    DEFAULT_TAIL_BYTES,
    MAX_SAMPLE_BYTES,
    CSVInspectionResult,
    CSVSource,
    FileSampleReadError,
    LLMBackend,
    Settings,
    ainspect_csv,
)
from fastapi import APIRouter, Depends, File, Query, Request, Response, UploadFile

from ..errors import (
    BackendOverrideDisabledError,
    ServerBusyError,
    UploadTooLargeError,
    problem_responses,
)
from ..settings import ApiSettings
from ..sources.gcs import GcsInspectRequest, chunk_size_for, create_client, open_gcs_object
from ..streaming import AsyncIteratorReader

logger = logging.getLogger("csv_inspector_api.inspect")

# API policy, not a library limit: a head this small rarely holds a header.
MIN_HEAD_BYTES = 512
# Model names ("qwen2.5-coder:7b", "org/model:tag") are short tokens: no forged log lines.
MAX_MODEL_NAME_LENGTH = 200
MODEL_NAME_PATTERN = r"^[A-Za-z0-9._:/@+-]+$"
_OVERRIDE_OFF = "cloud calls cost money, and CSV_INSPECTOR_API_ALLOW_BACKEND_OVERRIDE is off"


@dataclass(frozen=True)
class InspectParams:
    """Query parameters shared by the routes; ``None`` keeps the configured backend or model."""

    n_bytes: int
    tail_bytes: int
    timeout_seconds: float
    backend: LLMBackend | None = None
    model: str | None = None
    fallback_model: str | None = None

    def check_override(self, settings: ApiSettings) -> None:
        """Raise ``BackendOverrideDisabledError`` for a billed cloud call, overrides off."""
        if settings.allow_backend_override:
            return
        configured = settings.llm_backend
        if self.backend is LLMBackend.API and configured is not LLMBackend.API:
            msg = "backend=api would move this local deployment to the paid cloud backend"
            raise BackendOverrideDisabledError(f"{msg}; {_OVERRIDE_OFF}")
        if (self.backend or configured) is LLMBackend.API and (self.model or self.fallback_model):
            msg = "model and fallback_model would pick the (billed) models of a cloud call"
            raise BackendOverrideDisabledError(f"{msg}; {_OVERRIDE_OFF}")


def _model_name(description: str) -> object:
    """The ``Query`` of a model name: a plain token of at most 200 characters."""
    return Query(
        min_length=1,
        max_length=MAX_MODEL_NAME_LENGTH,
        pattern=MODEL_NAME_PATTERN,
        description=f"{description} for this request; default: the configured one. "
        "On the cloud backend, a 403 unless overrides are allowed.",
    )


_HEAD = Query(
    ge=MIN_HEAD_BYTES, le=MAX_SAMPLE_BYTES, description="Bytes sampled from the start of the file."
)
_TAIL = Query(
    ge=0,
    le=MAX_SAMPLE_BYTES,
    description="Bytes sampled from the end of the file; 0 skips the tail.",
)
_BACKEND = Query(
    description="Backend for this request; default: the configured one. "
    "`api` (paid) on a local deployment is a 403 unless overrides are allowed.",
)


def _params_dependency(settings: ApiSettings) -> Callable[..., None]:
    """The routes' query parameters, declared on the router with this deployment's bounds.

    The OpenAPI schema so shows the real limits; the routes read it through :func:`_params`.
    """

    def inspect_params(  # noqa: PLR0913, PLR0917 - one argument per query parameter
        request: Request,
        n_bytes: Annotated[int, _HEAD] = DEFAULT_SAMPLE_BYTES,
        tail_bytes: Annotated[int, _TAIL] = DEFAULT_TAIL_BYTES,
        timeout_seconds: Annotated[
            float,
            Query(
                ge=1,
                le=settings.max_timeout_seconds,
                description="Time budget of the whole model phase, in seconds.",
            ),
        ] = settings.default_timeout_seconds,
        backend: Annotated[LLMBackend | None, _BACKEND] = None,
        model: Annotated[str | None, _model_name("Primary model")] = None,
        fallback_model: Annotated[str | None, _model_name("Fallback model")] = None,
    ) -> None:
        params = InspectParams(n_bytes, tail_bytes, timeout_seconds, backend, model, fallback_model)
        params.check_override(settings)
        request.state.inspect_params = params

    return inspect_params


def _params(request: Request) -> InspectParams:
    """The query parameters of the request (none when invalid: the route then never runs)."""
    return cast("InspectParams", getattr(request.state, "inspect_params", None))


Params = Annotated[InspectParams, Depends(_params)]


async def _inspect(
    request: Request, response: Response, source: CSVSource, params: InspectParams, *, label: str
) -> CSVInspectionResult:
    """Run one inspection in a slot of ``app.state.inspection_slots``, log it, set usage headers.

    No free slot within ``queue_timeout_seconds`` is a ``ServerBusyError``. The
    model and its token counts (when reported) go in ``X-Inspection-*`` headers.
    """
    state = request.app.state
    settings: ApiSettings = state.settings
    try:
        await asyncio.wait_for(state.inspection_slots.acquire(), settings.queue_timeout_seconds)
    except asyncio.TimeoutError:
        slots, wait = settings.max_concurrent_inspections, settings.queue_timeout_seconds
        msg = f"all {slots} inspection slots stayed busy for {wait:g} s"
        raise ServerBusyError(msg, retry_after_seconds=max(1, math.ceil(wait))) from None
    library_settings: Settings = state.library_settings
    backend = params.backend or library_settings.llm_backend
    started = time.perf_counter()
    try:
        options = {**asdict(params), "backend": backend, "model_invoker": state.model_invoker}
        result = await ainspect_csv(source, settings=library_settings, **options)
    finally:
        state.inspection_slots.release()
    model = params.model or library_settings.model_for(backend)
    msg = "inspected %s with %s/%s in %.2f s, confidence %.2f"
    logger.info(msg, label, backend.value, model, time.perf_counter() - started, result.confidence)
    if (usage := result.usage) is not None:
        response.headers["X-Inspection-Model"] = usage.model
        for kind, count in (
            ("Prompt", usage.prompt_tokens),
            ("Completion", usage.completion_tokens),
        ):
            if count is not None:
                response.headers[f"X-Inspection-{kind}-Tokens"] = str(count)
    return result


def _too_large(size: int, settings: ApiSettings) -> UploadTooLargeError:
    """The 413 error for a body of ``size`` bytes."""
    msg = f"upload of {size} bytes exceeds the {settings.max_upload_bytes}-byte limit"
    return UploadTooLargeError(msg)


async def inspect_upload(
    request: Request,
    response: Response,
    file: Annotated[UploadFile, File(description="The CSV/TSV file, in any encoding.")],
    params: Params,
) -> CSVInspectionResult:
    """Infer the encoding, dialect, header, footer and column names of the upload.

    Only a bounded head and tail of the file are read and sent to the model.
    The content type of the upload is not checked: browsers and tools send
    anything from ``text/csv`` to ``application/vnd.ms-excel``, and the
    library detects what the bytes are.
    """
    settings: ApiSettings = request.app.state.settings
    # The body is already received: this caps what is inspected, not received.
    if file.size is not None and file.size > settings.max_upload_bytes:
        raise _too_large(file.size, settings)
    # A seekable SpooledTemporaryFile: only the sampled windows are read.
    label = f"{file.filename!r} ({file.size} bytes)"
    return await _inspect(request, response, file.file, params, label=label)


async def inspect_raw(request: Request, response: Response, params: Params) -> CSVInspectionResult:
    """Infer the same as ``POST /inspect``, streaming the body with bounded memory.

    The body is the file itself (``Content-Type: application/octet-stream``
    or ``text/csv``; not checked), with or without ``Content-Length``. It
    is read once, as it arrives: only the head, the last ``tail_bytes``
    and one chunk are held in memory, and nothing is written to disk.
    """
    settings: ApiSettings = request.app.state.settings
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > settings.max_upload_bytes:
        raise _too_large(int(declared), settings)
    # Consumed once in the library's worker thread, each read() pulling a chunk from this loop.
    reader = AsyncIteratorReader(
        request.stream(),
        asyncio.get_running_loop(),
        max_bytes=settings.max_upload_bytes,
        read_timeout_seconds=params.timeout_seconds,
    )
    try:
        return await _inspect(request, response, reader, params, label="request body")
    except FileSampleReadError as exc:
        if reader.limit_exceeded:
            raise _too_large(reader.bytes_read, settings) from exc
        raise
    finally:
        # Releases the worker thread if it still waits for a chunk (client gone, or a failure).
        reader.close()


async def inspect_gcs(
    request: Request, response: Response, body: GcsInspectRequest, params: Params
) -> CSVInspectionResult:
    """Infer the same as ``POST /inspect`` for a ``gs://`` object, never downloading it.

    The object is read like a local file: one metadata request, then one
    ranged read per sampled window, whatever its size. The body of the
    answer is the one of ``POST /inspect``; the object's size and the
    generation read are in the ``X-Object-Size`` and
    ``X-Object-Generation`` headers.
    """
    state = request.app.state
    if state.gcs_client is None:
        # Retried on each request until it works; credentials may need the network.
        project = state.settings.google_cloud_project
        state.gcs_client = await asyncio.to_thread(create_client, project)
    chunk_size = chunk_size_for(params.n_bytes, params.tail_bytes)
    gcs_object = open_gcs_object(
        body.uri, client=state.gcs_client, generation=body.generation, chunk_size=chunk_size
    )
    try:
        # Sampled in the library's worker thread: the ranged reads never block the loop.
        result = await _inspect(request, response, gcs_object.reader, params, label=repr(body.uri))
    finally:
        gcs_object.reader.close()
    if gcs_object.blob.size is not None:
        response.headers["X-Object-Size"] = str(gcs_object.blob.size)
    if gcs_object.blob.generation is not None:
        response.headers["X-Object-Generation"] = str(gcs_object.blob.generation)
    return result


def _header(description: str, schema_type: str) -> dict[str, Any]:
    return {"description": description, "schema": {"type": schema_type}}


_USAGE_HEADERS = {
    "X-Inspection-Model": _header(
        "Model whose answer was kept (the fallback when the primary failed).", "string"
    ),
    "X-Inspection-Prompt-Tokens": _header(
        "Prompt tokens over every model attempt; absent when not reported.", "integer"
    ),
    "X-Inspection-Completion-Tokens": _header(
        "Completion tokens over every model attempt; absent when not reported.", "integer"
    ),
}
_OBJECT_HEADERS = {
    "X-Object-Size": _header("Size of the object, in bytes.", "integer"),
    "X-Object-Generation": _header("Generation of the object that was read.", "integer"),
}
# Shown in OpenAPI as the 200 example; the library model itself stays untouched.
_EXAMPLE_RESULT = {
    "encoding": "utf-8",
    "delimiter": ";",
    "quotechar": '"',
    "escapechar": None,
    "doublequote": True,
    "has_header": True,
    "header_row_index": 2,
    "footer_lines": [],
    "columns": ["Fecha", "Importe"],
    "confidence": 0.9,
}
_OK = {"content": {"application/json": {"example": _EXAMPLE_RESULT}}, "headers": _USAGE_HEADERS}
_RAW = {"schema": {"type": "string", "format": "binary"}}
_RAW_CONTENT = {"application/octet-stream": _RAW, "text/csv": _RAW}


def build_inspect_router(settings: ApiSettings) -> APIRouter:
    """The router of ``POST /inspect``, ``/inspect/raw`` and ``/inspect/gcs`` for ``settings``."""
    router = APIRouter(tags=["inspection"], dependencies=[Depends(_params_dependency(settings))])
    post = functools.partial(router.post, response_model=CSVInspectionResult)
    responses = {200: _OK, **problem_responses(403, 413, 422, 502, 503, 504, gated=True)}
    post("/inspect", summary="Inspect an uploaded CSV/TSV file", responses=responses)(
        inspect_upload
    )
    post(
        "/inspect/raw",
        summary="Inspect a CSV/TSV file sent as the raw request body",
        responses=responses,
        openapi_extra={"requestBody": {"required": True, "content": _RAW_CONTENT}},
    )(inspect_raw)
    post(
        "/inspect/gcs",
        summary="Inspect a Cloud Storage object with ranged reads",
        responses={
            200: {**_OK, "headers": {**_USAGE_HEADERS, **_OBJECT_HEADERS}},
            **problem_responses(403, 404, 422, 429, 502, 503, 504, gcs=True, gated=True),
        },
    )(inspect_gcs)
    return router
