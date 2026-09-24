"""``POST /inspect`` and ``POST /inspect/raw``: inspect a CSV/TSV file.

No ``from __future__ import annotations`` here: FastAPI evaluates the route's
annotations, and the ``Query`` bounds refer to the factory's ``settings``,
which is not a module global.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Annotated, Any

from csv_inspector import (
    CSVInspectionResult,
    CSVSource,
    FileSampleReadError,
    Settings,
    ainspect_csv,
)
from fastapi import APIRouter, Depends, File, Query, Request, UploadFile

from ..errors import UploadTooLargeError, problem_responses
from ..settings import ApiSettings
from ..streaming import AsyncIteratorReader

logger = logging.getLogger("csv_inspector_api.inspect")

# The library's sampling windows: 4 KiB by default, at most 16 KiB each.
DEFAULT_WINDOW_BYTES = 4096
MIN_HEAD_BYTES = 512
MAX_WINDOW_BYTES = 16384


# Shown in OpenAPI as the 200 example; the library model itself stays untouched.
_EXAMPLE_RESULT = {
    "encoding": "utf-8",
    "delimiter": ";",
    "quotechar": '"',
    "escapechar": None,
    "doublequote": True,
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
    """

    n_bytes: int
    tail_bytes: int
    timeout_seconds: float


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
    backend = library_settings.llm_backend
    started = time.perf_counter()
    result = await ainspect_csv(
        source,
        backend=backend,
        settings=library_settings,
        n_bytes=params.n_bytes,
        tail_bytes=params.tail_bytes,
        timeout_seconds=params.timeout_seconds,
        model_invoker=request.app.state.model_invoker,
    )
    logger.info(
        "inspected %s with %s/%s in %.2f s, confidence %.2f",
        label,
        backend.value,
        library_settings.model_for(backend),
        time.perf_counter() - started,
        result.confidence,
    )
    return result


def _too_large(size: int, settings: ApiSettings) -> UploadTooLargeError:
    """The 413 error for a body of ``size`` bytes."""
    msg = f"upload of {size} bytes exceeds the {settings.max_upload_bytes}-byte limit"
    return UploadTooLargeError(msg)


_RAW_BODY_SCHEMA = {"type": "string", "format": "binary"}


def build_inspect_router(settings: ApiSettings) -> APIRouter:
    """Build the router of ``POST /inspect`` and ``POST /inspect/raw``.

    The time budget bounds come from ``settings``, so the router is built per
    application and the OpenAPI schema shows the deployment's real limits.

    Args:
        settings: The API settings of the application being built.

    Returns:
        A router with the inspection routes.
    """
    router = APIRouter(tags=["inspection"])

    def inspect_params(
        n_bytes: Annotated[
            int,
            Query(
                ge=MIN_HEAD_BYTES,
                le=MAX_WINDOW_BYTES,
                description="Bytes sampled from the start of the file.",
            ),
        ] = DEFAULT_WINDOW_BYTES,
        tail_bytes: Annotated[
            int,
            Query(
                ge=0,
                le=MAX_WINDOW_BYTES,
                description="Bytes sampled from the end of the file; 0 skips the tail.",
            ),
        ] = DEFAULT_WINDOW_BYTES,
        timeout_seconds: Annotated[
            float,
            Query(
                ge=1,
                le=settings.max_timeout_seconds,
                description="Time budget of the whole model phase, in seconds.",
            ),
        ] = settings.default_timeout_seconds,
    ) -> InspectParams:
        """Collect the query parameters shared by both routes (a FastAPI dependency)."""
        return InspectParams(n_bytes, tail_bytes, timeout_seconds)

    responses: dict[int | str, dict[str, Any]] = {
        200: {"content": {"application/json": {"example": _EXAMPLE_RESULT}}},
        **problem_responses(413, 422, 502, 503, 504),
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

    return router
