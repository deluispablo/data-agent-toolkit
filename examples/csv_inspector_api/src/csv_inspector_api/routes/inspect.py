"""``POST /inspect``, ``/inspect/raw`` and ``/inspect/gcs``: inspect a CSV/TSV file.

No ``from __future__ import annotations`` here: FastAPI evaluates the route's
annotations, and the ``Query`` bounds refer to the factory's ``settings``,
which is not a module global.

``/inspect`` or ``/inspect/raw``? ``/inspect`` takes ``multipart/form-data``
(browsers, ``curl -F``). Starlette receives the whole upload first, spooled
to a temporary file past 1 MiB, and the library samples it as a seekable
stream, tail included whatever the size. Its ``413`` comes after the body
was received, so it caps what is inspected, not what is received: cap the
request size at the reverse proxy too. ``/inspect/raw`` takes the file
bytes as the body (scripts, pipes, ``curl --data-binary``). It is inspected
as it streams, with ``n_bytes + tail_bytes`` plus one chunk in memory and
nothing on disk. Its ``413`` comes before reading (``Content-Length``) or
while streaming. A body over 64 MiB past the head is not read to its end,
so there is no tail and no footer. A client that stops sending for
``timeout_seconds`` fails the read (``422``); one that disconnects cancels
the request and releases the worker thread at once. Never send
``multipart/form-data`` to ``/inspect/raw``: the envelope would be
inspected as the file.

Content types are not checked: clients send anything from ``text/csv`` to
``application/octet-stream``, and the library detects what the bytes are.
Invalid query parameters are FastAPI's own ``422`` (``application/json``);
every other error is a problem response (see ``errors.py``).

Model names must be plain tokens (letters, digits and ``._:/@+-``, at most
200 characters) because they appear in log lines; an unknown model is the
library's normal failure path (``502`` after the fallback, or ``503``).
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Annotated, Any

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

from ..errors import BackendOverrideDisabledError, UploadTooLargeError, problem_responses
from ..settings import ApiSettings
from ..sources.gcs import (
    GcsClient,
    GcsInspectRequest,
    chunk_size_for,
    create_client,
    open_gcs_object,
)
from ..streaming import AsyncIteratorReader

logger = logging.getLogger("csv_inspector_api.inspect")

# API policy, not a library limit: a head this small rarely holds a header.
MIN_HEAD_BYTES = 512
# Model names ("qwen2.5-coder:7b", "gemini-2.5-flash", "org/model:tag") are short
# tokens: the bounds keep a client-chosen name from forging or flooding log lines.
MAX_MODEL_NAME_LENGTH = 200
MODEL_NAME_PATTERN = r"^[A-Za-z0-9._:/@+-]+$"


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
    "columns": [
        {
            "name": "Fecha",
            "inferred_type": "date",
            "nullable": False,
            "example_values": ["2024-01-15", "2024-01-16"],
        },
        {
            "name": "Importe",
            "inferred_type": "float",
            "nullable": False,
            "example_values": ["1250.50", "890.00"],
        },
    ],
    "confidence": 0.9,
    "notes": "Two preamble lines precede the header row.",
}


@dataclass(frozen=True)
class InspectParams:
    """Query parameters shared by the inspection routes.

    Attributes:
        n_bytes: Bytes sampled from the start of the file.
        tail_bytes: Bytes sampled from the end of the file; 0 skips the tail.
        timeout_seconds: Time budget of the whole model phase, in seconds.
        backend: Backend requested for this call; ``None`` keeps the configured one.
        model: Primary model for this call; ``None`` keeps the backend's configured one.
        fallback_model: Fallback model for this call; ``None`` keeps the configured one.
    """

    n_bytes: int
    tail_bytes: int
    timeout_seconds: float
    backend: LLMBackend | None = None
    model: str | None = None
    fallback_model: str | None = None


async def _inspect(
    request: Request, source: CSVSource, params: InspectParams, *, label: str
) -> CSVInspectionResult:
    """Run one inspection with the app's settings and log its outcome.

    Library errors propagate to the handler in ``errors.py``.

    Args:
        request: The current request; its app holds the settings and the invoker.
        source: What ``ainspect_csv`` samples.
        params: The request's query parameters.
        label: Log-safe description of the source, for the log line.

    Returns:
        The library's inspection result.
    """
    library_settings: Settings = request.app.state.library_settings
    backend = params.backend or library_settings.llm_backend
    started = time.perf_counter()
    result = await ainspect_csv(
        source,
        backend=backend,
        settings=library_settings,
        model=params.model,
        fallback_model=params.fallback_model,
        n_bytes=params.n_bytes,
        tail_bytes=params.tail_bytes,
        timeout_seconds=params.timeout_seconds,
        model_invoker=request.app.state.model_invoker,
    )
    logger.info(
        "inspected %s with %s/%s in %.2f s, confidence %.2f",
        label,
        backend.value,
        params.model or library_settings.model_for(backend),
        time.perf_counter() - started,
        result.confidence,
    )
    return result


def _too_large(size: int, settings: ApiSettings) -> UploadTooLargeError:
    """The 413 error for a body of ``size`` bytes."""
    msg = f"upload of {size} bytes exceeds the {settings.max_upload_bytes}-byte limit"
    return UploadTooLargeError(msg)


async def _gcs_client(request: Request) -> GcsClient:
    """The app's Cloud Storage client.

    Built at startup; when that failed (the ``[gcs]`` extra is missing, no
    credentials), each request tries again, so a fixed deployment recovers,
    and the failure is answered by its error handler.

    Args:
        request: The current request; its app holds the client.

    Returns:
        The client.
    """
    client: GcsClient | None = request.app.state.gcs_client
    if client is None:
        project = request.app.state.settings.google_cloud_project
        # Finding credentials may query the metadata server: off the event loop.
        client = await asyncio.to_thread(create_client, project)
        request.app.state.gcs_client = client
    return client


_OBJECT_HEADERS = {
    "X-Object-Size": {
        "description": "Size of the object, in bytes.",
        "schema": {"type": "integer"},
    },
    "X-Object-Generation": {
        "description": "Generation of the object that was read.",
        "schema": {"type": "integer"},
    },
}
_RAW_BODY_SCHEMA = {"type": "string", "format": "binary"}
_OVERRIDE_OFF = "cloud calls cost money, and CSV_INSPECTOR_API_ALLOW_BACKEND_OVERRIDE is off"


def build_inspect_router(settings: ApiSettings) -> APIRouter:
    """Build the router of ``POST /inspect``, ``/inspect/raw`` and ``/inspect/gcs``.

    The time budget bounds come from ``settings``, so the router is built per
    application and the OpenAPI schema shows the deployment's real limits.

    Args:
        settings: The API settings of the application being built.

    Returns:
        A router with the inspection routes.
    """
    router = APIRouter(tags=["inspection"])

    def inspect_params(  # noqa: PLR0913, PLR0917 - one argument per query parameter
        n_bytes: Annotated[
            int,
            Query(
                ge=MIN_HEAD_BYTES,
                le=MAX_SAMPLE_BYTES,
                description="Bytes sampled from the start of the file.",
            ),
        ] = DEFAULT_SAMPLE_BYTES,
        tail_bytes: Annotated[
            int,
            Query(
                ge=0,
                le=MAX_SAMPLE_BYTES,
                description="Bytes sampled from the end of the file; 0 skips the tail.",
            ),
        ] = DEFAULT_TAIL_BYTES,
        timeout_seconds: Annotated[
            float,
            Query(
                ge=1,
                le=settings.max_timeout_seconds,
                description="Time budget of the whole model phase, in seconds.",
            ),
        ] = settings.default_timeout_seconds,
        backend: Annotated[
            LLMBackend | None,
            Query(
                description="Backend for this request; default: the configured one. "
                "`api` (paid) on a local deployment is a 403 unless overrides are allowed.",
            ),
        ] = None,
        model: Annotated[
            str | None,
            Query(
                min_length=1,
                max_length=MAX_MODEL_NAME_LENGTH,
                pattern=MODEL_NAME_PATTERN,
                description="Primary model for this request; default: the configured one. "
                "On the cloud backend, a 403 unless overrides are allowed.",
            ),
        ] = None,
        fallback_model: Annotated[
            str | None,
            Query(
                min_length=1,
                max_length=MAX_MODEL_NAME_LENGTH,
                pattern=MODEL_NAME_PATTERN,
                description="Fallback model for this request; default: the configured one. "
                "On the cloud backend, a 403 unless overrides are allowed.",
            ),
        ] = None,
    ) -> InspectParams:
        """Collect the query parameters shared by both routes (a FastAPI dependency).

        Raises:
            BackendOverrideDisabledError: If, while ``settings.allow_backend_override``
                is off, the request would switch a local deployment to the cloud
                backend, or pick the models of a cloud call.
        """
        if not settings.allow_backend_override:
            configured = settings.llm_backend
            if backend is LLMBackend.API and configured is not LLMBackend.API:
                msg = "backend=api would move this local deployment to the paid cloud backend"
                raise BackendOverrideDisabledError(f"{msg}; {_OVERRIDE_OFF}")
            if (backend or configured) is LLMBackend.API and (model or fallback_model):
                msg = "model and fallback_model would pick the (billed) models of a cloud call"
                raise BackendOverrideDisabledError(f"{msg}; {_OVERRIDE_OFF}")
        return InspectParams(n_bytes, tail_bytes, timeout_seconds, backend, model, fallback_model)

    responses: dict[int | str, dict[str, Any]] = {
        200: {"content": {"application/json": {"example": _EXAMPLE_RESULT}}},
        **problem_responses(403, 413, 422, 502, 503, 504),
    }

    @router.post(
        "/inspect",
        response_model=CSVInspectionResult,
        summary="Inspect an uploaded CSV/TSV file",
        responses=responses,
    )
    async def inspect_upload(
        request: Request,
        file: Annotated[UploadFile, File(description="The CSV/TSV file, in any encoding.")],
        params: Annotated[InspectParams, Depends(inspect_params)],
    ) -> CSVInspectionResult:
        """Infer the encoding, dialect, header, footer and column schema of the upload.

        Only a bounded head and tail of the file are read and sent to the model.
        The content type of the upload is not checked: browsers and tools send
        anything from ``text/csv`` to ``application/vnd.ms-excel``, and the
        library detects what the bytes are.
        """
        # The multipart body is already received (spooled to disk past 1 MiB)
        # when the route runs: this caps what is inspected; cap what is
        # received at the reverse proxy.
        if file.size is not None and file.size > settings.max_upload_bytes:
            raise _too_large(file.size, settings)
        # UploadFile.file is a seekable SpooledTemporaryFile: passed as is, the
        # library reads only its sampled windows and restores the position.
        return await _inspect(
            request, file.file, params, label=f"{file.filename!r} ({file.size} bytes)"
        )

    @router.post(
        "/inspect/raw",
        response_model=CSVInspectionResult,
        summary="Inspect a CSV/TSV file sent as the raw request body",
        responses=responses,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/octet-stream": {"schema": _RAW_BODY_SCHEMA},
                    "text/csv": {"schema": _RAW_BODY_SCHEMA},
                },
            }
        },
    )
    async def inspect_raw(
        request: Request, params: Annotated[InspectParams, Depends(inspect_params)]
    ) -> CSVInspectionResult:
        """Infer the same as ``POST /inspect``, streaming the body with bounded memory.

        The body is the file itself (``Content-Type: application/octet-stream``
        or ``text/csv``; not checked), with or without ``Content-Length``. It
        is read once, as it arrives: only the head, the last ``tail_bytes``
        and one chunk are held in memory, and nothing is written to disk.
        """
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > settings.max_upload_bytes:
            raise _too_large(int(declared), settings)
        # A non-seekable stream: the library consumes it once in its worker
        # thread, each read() pulling the next chunk from this event loop.
        reader = AsyncIteratorReader(
            request.stream(),
            asyncio.get_running_loop(),
            max_bytes=settings.max_upload_bytes,
            read_timeout_seconds=params.timeout_seconds,
        )
        try:
            return await _inspect(request, reader, params, label="request body")
        except FileSampleReadError as exc:
            if reader.limit_exceeded:
                raise _too_large(reader.bytes_read, settings) from exc
            raise
        finally:
            # Releases the worker thread if it still waits for a chunk: the
            # request was cancelled (client gone) or failed before the end.
            reader.close()

    @router.post(
        "/inspect/gcs",
        response_model=CSVInspectionResult,
        summary="Inspect a Cloud Storage object with ranged reads",
        responses={
            200: {**responses[200], "headers": _OBJECT_HEADERS},
            **problem_responses(403, 404, 422, 429, 502, 503, 504, gcs=True),
        },
    )
    async def inspect_gcs(
        request: Request,
        response: Response,
        body: GcsInspectRequest,
        params: Annotated[InspectParams, Depends(inspect_params)],
    ) -> CSVInspectionResult:
        """Infer the same as ``POST /inspect`` for a ``gs://`` object, never downloading it.

        The object is read like a local file: one metadata request, then one
        ranged read per sampled window, whatever its size. The body of the
        answer is the one of ``POST /inspect``; the object's size and the
        generation read are in the ``X-Object-Size`` and
        ``X-Object-Generation`` headers.
        """
        client = await _gcs_client(request)
        chunk_size = chunk_size_for(params.n_bytes, params.tail_bytes)
        gcs_object = open_gcs_object(
            body.uri, client=client, generation=body.generation, chunk_size=chunk_size
        )
        try:
            # The library samples the seekable reader in its worker thread, so
            # the blocking ranged reads never run on the event loop.
            result = await _inspect(request, gcs_object.reader, params, label=repr(body.uri))
        finally:
            gcs_object.reader.close()
        blob = gcs_object.blob
        if blob.size is not None:
            response.headers["X-Object-Size"] = str(blob.size)
        if blob.generation is not None:
            response.headers["X-Object-Generation"] = str(blob.generation)
        return result

    return router
