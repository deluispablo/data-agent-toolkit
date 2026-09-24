"""``POST /inspect``: inspect an uploaded CSV/TSV file.

No ``from __future__ import annotations`` here: FastAPI evaluates the route's
annotations, and the ``Query`` bounds refer to the factory's ``settings``,
which is not a module global.
"""

import logging
import time
from typing import Annotated

from csv_inspector import CSVInspectionResult, Settings, ainspect_csv
from fastapi import APIRouter, File, Query, Request, UploadFile

from ..errors import UploadTooLargeError, problem_responses
from ..settings import ApiSettings

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


def build_inspect_router(settings: ApiSettings) -> APIRouter:
    """Build the router of ``POST /inspect``.

    The time budget bounds come from ``settings``, so the router is built per
    application and the OpenAPI schema shows the deployment's real limits.

    Args:
        settings: The API settings of the application being built.

    Returns:
        A router with the ``/inspect`` route.
    """
    router = APIRouter(tags=["inspection"])

    @router.post(
        "/inspect",
        response_model=CSVInspectionResult,
        summary="Inspect an uploaded CSV/TSV file",
        responses={
            200: {"content": {"application/json": {"example": _EXAMPLE_RESULT}}},
            **problem_responses(413, 422, 502, 503, 504),
        },
    )
    async def inspect_upload(
        request: Request,
        file: Annotated[UploadFile, File(description="The CSV/TSV file, in any encoding.")],
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
            msg = f"upload of {file.size} bytes exceeds the {settings.max_upload_bytes}-byte limit"
            raise UploadTooLargeError(msg)

        library_settings: Settings = request.app.state.library_settings
        backend = library_settings.llm_backend
        started = time.perf_counter()
        # UploadFile.file is a seekable SpooledTemporaryFile: passed as is, the
        # library reads only its sampled windows and restores the position.
        # Library errors propagate to the handler in errors.py.
        result = await ainspect_csv(
            file.file,
            backend=backend,
            settings=library_settings,
            n_bytes=n_bytes,
            tail_bytes=tail_bytes,
            timeout_seconds=timeout_seconds,
            model_invoker=request.app.state.model_invoker,
        )
        logger.info(
            "inspected %r (%s bytes) with %s/%s in %.2f s, confidence %.2f",
            file.filename,
            file.size,
            backend.value,
            library_settings.model_for(backend),
            time.perf_counter() - started,
            result.confidence,
        )
        return result

    return router
