"""Mapping of ``csv_inspector``, API and Cloud Storage errors to HTTP problem responses.

Every error escaping a route becomes an RFC 9457 ``application/problem+json``
response (``error`` is the class name) from the ordered tables below; the
Cloud Storage one needs the ``[gcs]`` extra, imported only then. ``ValueError``
and ``TypeError`` are host bugs: a plain 500. Server faults log at ``ERROR``.

- 403 ``BackendOverrideDisabledError``; Cloud Storage ``Forbidden`` (grant
  ``storage.objects.get``). 404 ``NotFound``: buckets are not told apart.
- 413 ``UploadTooLargeError``. 422 ``EmptySampleError``, ``FileSampleReadError``.
- 429 ``TooManyRequests``, ``Retry-After`` passed through. 500 any other
  ``CSVInspectorError``: report it.
- 502 ``InspectionFailedError``, ``ModelInvocationError``, ``ResponseParsingError``,
  ``SchemaValidationError`` (retry); any other ``GoogleAPICallError``, ``RetryError``.
- 503 ``ServerBusyError`` (retry after ``Retry-After``), ``GcsNotInstalledError``,
  ``BackendConfigurationError``, ``CredentialsNotConfiguredError``, ``Unauthorized``,
  ``DefaultCredentialsError``, ``RefreshError``: a misconfigured deployment.
- 504 ``InspectionTimeoutError``, matched before its parent ``InspectionFailedError``.

A Cloud Storage ``detail`` is fixed: the SDK's message names the bucket, so
it only reaches the log.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from csv_inspector import (
    BackendConfigurationError,
    CSVInspectorError,
    EmptySampleError,
    FileSampleReadError,
    InspectionFailedError,
    InspectionTimeoutError,
    ModelInvocationError,
    ResponseParsingError,
    SchemaValidationError,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .sources.gcs import GcsNotInstalledError

logger = logging.getLogger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"


class ProblemDetails(BaseModel):
    """Body of every error response, after RFC 9457 (Problem Details for HTTP APIs).

    Attributes:
        type: Problem type URI; always ``about:blank``, the status says it all.
        title: Short, human-readable summary of the problem class.
        status: HTTP status code, repeated for clients that only see the body.
        detail: Explanation of this occurrence (the exception message).
        error: Exception class name, so clients can branch without parsing ``detail``.
    """

    type: str = "about:blank"
    title: str
    status: int
    detail: str
    error: str = Field(examples=["InspectionTimeoutError"])


class UploadTooLargeError(Exception):
    """The upload exceeds ``ApiSettings.max_upload_bytes``; raised before the library runs."""


class BackendOverrideDisabledError(Exception):
    """A request asked for a billed cloud call while overrides are off."""


class ServerBusyError(Exception):
    """No inspection slot freed up in time; ``retry_after_seconds`` is sent as ``Retry-After``."""

    def __init__(self, message: str, *, retry_after_seconds: int) -> None:
        """Describe the refusal; ``message`` is sent as ``detail``."""
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class _Problem:
    """One row of an error table; ``detail`` replaces the exception message when given."""

    exceptions: tuple[type[Exception], ...]
    status: int
    title: str
    detail: str | None = None
    log_level: int = logging.WARNING


_MODEL_FAILED = (InspectionFailedError, ModelInvocationError, ResponseParsingError)
_MISCONFIGURED = "Model backend misconfigured on the server; retrying elsewhere may help"
# Checked in order: InspectionTimeoutError is an InspectionFailedError, and
# BackendConfigurationError a ModelInvocationError. The last row catches the rest.
_LIBRARY = (
    _Problem((EmptySampleError, FileSampleReadError), 422, "Unusable CSV input"),
    _Problem((InspectionTimeoutError,), 504, "Inspection timed out"),
    _Problem((BackendConfigurationError,), 503, _MISCONFIGURED, log_level=logging.ERROR),
    _Problem((*_MODEL_FAILED, SchemaValidationError), 502, "Model backend failed"),
    _Problem((CSVInspectorError,), 500, "Inspection error", log_level=logging.ERROR),
)
_NO_GCS = "Cloud Storage unavailable on the server"
_API = (
    _TOO_LARGE := _Problem((UploadTooLargeError,), 413, "Upload too large"),
    _OVERRIDE_OFF := _Problem((BackendOverrideDisabledError,), 403, "Backend override disabled"),
    _Problem((ServerBusyError,), 503, "Server busy; retry after Retry-After seconds"),
    _Problem((GcsNotInstalledError,), 503, _NO_GCS, log_level=logging.ERROR),
)
_GCS: dict[int, tuple[str, str]] = {
    404: ("Object not found", "The object does not exist, or its bucket does not exist."),
    403: (
        "Access to the object denied",
        "The server's service account may not read this object (storage.objects.get).",
    ),
    503: (
        "Cloud Storage credentials misconfigured on the server",
        "The server has no usable Cloud Storage credentials.",
    ),
    429: (
        "Cloud Storage rate limit reached",
        "Cloud Storage rate-limited the request; retry later.",
    ),
    502: ("Cloud Storage failed", "Cloud Storage failed or is unavailable; retry later."),
}


def status_for(exc: CSVInspectorError) -> int:
    """The HTTP status of a library error (the first matching row of the library table)."""
    return next(row for row in _LIBRARY if isinstance(exc, row.exceptions)).status


def _retry_after(exc: Exception, status: int) -> dict[str, str] | None:
    """``Retry-After`` of a busy server, or passed through from a rate-limited Cloud Storage."""
    if isinstance(exc, ServerBusyError):
        return {"Retry-After": str(exc.retry_after_seconds)}
    if status == 429:  # noqa: PLR2004 - the status code itself
        retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("Retry-After")
        return {"Retry-After": retry_after} if retry_after else None
    return None


def _handler(
    rows: tuple[_Problem, ...], *, api: bool = False
) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
    """The handler of one table: log once, answer with the first matching row's problem."""

    async def handle(request: Request, exc: Exception) -> JSONResponse:
        row = next(row for row in rows if isinstance(exc, row.exceptions))
        name, where = type(exc).__name__, (request.method, request.url.path, row.status)
        if api:
            server_fault = row.log_level >= logging.ERROR
            msg = "%s %s failed with %d: %s" if server_fault else "%s %s rejected with %d: %s"
            logger.log(row.log_level, msg, *where, exc)
        else:
            trace = exc if row is _LIBRARY[-1] else None
            logger.log(
                row.log_level, "%s %s failed with %d %s: %s", *where, name, exc, exc_info=trace
            )
        detail = str(exc) if row.detail is None else row.detail
        problem = ProblemDetails(title=row.title, status=row.status, detail=detail, error=name)
        headers = _retry_after(exc, row.status)
        return JSONResponse(
            problem.model_dump(mode="json"), row.status, headers, PROBLEM_MEDIA_TYPE
        )

    return handle


def register_exception_handlers(app: FastAPI) -> None:
    """Install the API's error handlers on ``app`` (the one ``create_app`` builds)."""
    app.add_exception_handler(CSVInspectorError, _handler(_LIBRARY))
    for row in _API:
        app.add_exception_handler(row.exceptions[0], _handler((row,), api=True))
    try:
        from google.api_core import exceptions as api  # noqa: PLC0415 - the optional [gcs] extra
        from google.auth import exceptions as auth  # noqa: PLC0415
    except ImportError:
        return  # No google.* error can reach a route; /inspect/gcs answers GcsNotInstalledError.
    auth_errors = (auth.DefaultCredentialsError, auth.RefreshError)
    matches: tuple[tuple[tuple[type[Exception], ...], int, int], ...] = (
        ((api.NotFound,), 404, logging.WARNING),  # The specific GoogleAPICallError first.
        ((api.Forbidden,), 403, logging.WARNING),
        ((api.Unauthorized, *auth_errors), 503, logging.ERROR),
        ((api.TooManyRequests,), 429, logging.WARNING),
        ((api.GoogleAPICallError, api.RetryError), 502, logging.WARNING),
    )
    rows = tuple(_Problem(types, status, *_GCS[status], level) for types, status, level in matches)
    handler = _handler(rows)
    for error_type in (api.GoogleAPICallError, api.RetryError, *auth_errors):
        app.add_exception_handler(error_type, handler)


_VALIDATION_ERROR_SCHEMA: dict[str, Any] = {
    "title": "HTTPValidationError",
    "type": "object",
    "properties": {"detail": {"type": "array", "items": {"type": "object"}}},
}
"""Shape of FastAPI's own 422 body for invalid parameters or a missing field."""

_GCS_DESCRIPTIONS = {
    403: f"{_OVERRIDE_OFF.title}, or {_GCS[403][0].lower()}",
    404: f"{_GCS[404][0]} (or its bucket: not told apart)",
    429: f"{_GCS[429][0]}; honour Retry-After when present",
    502: f"Model backend or {_GCS[502][0]}",
    503: "Model backend, Cloud Storage credentials or the [gcs] extra misconfigured on the server",
}
"""OpenAPI descriptions of the statuses of ``POST /inspect/gcs``."""


def problem_responses(
    *statuses: int, gcs: bool = False, gated: bool = False
) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses`` entries for error statuses with a ``ProblemDetails`` body.

    ``gcs`` adds the Cloud Storage errors of ``POST /inspect/gcs``; ``gated``, the busy 503.
    """
    titles = {row.status: row.title for row in (*_LIBRARY, _TOO_LARGE, _OVERRIDE_OFF)}
    if gcs:
        titles.update(_GCS_DESCRIPTIONS)
    if gated:
        titles[503] += "; or the server is busy (honour Retry-After)"
    schema = ProblemDetails.model_json_schema()
    responses: dict[int | str, dict[str, Any]] = {
        status: {"description": titles[status], "content": {PROBLEM_MEDIA_TYPE: {"schema": schema}}}
        for status in statuses
    }
    if 422 in responses:  # noqa: PLR2004 - the status code itself
        # Declaring 422 replaces FastAPI's own entry for invalid parameters,
        # which keep FastAPI's body: document both.
        responses[422]["description"] += " (problem+json), or invalid request (application/json)"
        responses[422]["content"]["application/json"] = {"schema": _VALIDATION_ERROR_SCHEMA}
    return responses
