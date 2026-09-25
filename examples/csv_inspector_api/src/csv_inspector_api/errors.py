"""Mapping of ``csv_inspector`` and Cloud Storage errors to HTTP problem responses.

Every :class:`csv_inspector.CSVInspectorError` that escapes a route becomes
an RFC 9457-style ``application/problem+json`` response through one
exception handler, so routes never catch library errors themselves. The
errors of ``POST /inspect/gcs`` (``google.api_core`` and ``google.auth``
exceptions, raised while the library samples the object) get a second
handler, registered only when the ``[gcs]`` extra is installed: this module
imports ``google.*`` inside that registration, never at module level.

``ValueError`` and ``TypeError`` from the library (a byte budget out of
range, an unsupported source) are programming errors in this host, not
domain failures: they are deliberately not handled here and surface as a
plain 500 through FastAPI's default handling.

Status per exception (``error`` in the body is the class name, so clients
can branch without parsing ``detail``):

- 403 ``BackendOverrideDisabledError`` (API): drop ``backend=api`` or the
  model overrides, or ask the operator.
- 413 ``UploadTooLargeError`` (API): send a smaller file.
- 503 ``ServerBusyError`` (API): every inspection slot stayed taken for
  ``queue_timeout_seconds``; retry after ``Retry-After`` seconds.
- 503 ``GcsNotInstalledError`` (API): install the ``[gcs]`` extra.
- 404 ``NotFound`` (Cloud Storage): a missing bucket and a missing object
  answer the same, so buckets cannot be enumerated.
- 403 ``Forbidden``: grant ``storage.objects.get`` to the service account.
- 503 ``Unauthorized``, ``DefaultCredentialsError``, ``RefreshError``: the
  deployment's credentials are missing or unusable, not the request.
- 429 ``TooManyRequests``: ``Retry-After`` is passed through when sent.
- 502 any other ``GoogleAPICallError`` or ``RetryError``.
- 422 ``EmptySampleError``, ``FileSampleReadError``: fix the input.
- 504 ``InspectionTimeoutError``, matched **before** its parent
  ``InspectionFailedError``: retry with a larger budget or smaller windows.
- 503 ``CredentialsNotConfiguredError``, ``BackendConfigurationError``: a
  misconfigured deployment, not a bad request; another instance may work.
- 502 ``InspectionFailedError``, ``ModelInvocationError``,
  ``ResponseParsingError``, ``SchemaValidationError``: retry, maybe with
  another model.
- 500 any other ``CSVInspectorError``: report it.

For Cloud Storage errors ``detail`` is a fixed sentence per status. The
SDK's message (which names the bucket and whether it exists) only reaches
the server log.
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


@dataclass(frozen=True)
class _Rule:
    """One row of the error table: the exceptions it matches and their response.

    Server-side faults (503, 500) log at ``ERROR``; client input, timeouts and
    model failures at ``WARNING``.
    """

    exceptions: tuple[type[CSVInspectorError], ...]
    status: int
    title: str
    log_level: int = logging.WARNING


# Checked in order, so a subclass row must come before its parent's row:
# InspectionTimeoutError is an InspectionFailedError, and
# BackendConfigurationError is a ModelInvocationError.
_RULES = (
    _Rule((EmptySampleError, FileSampleReadError), 422, "Unusable CSV input"),
    _Rule((InspectionTimeoutError,), 504, "Inspection timed out"),
    _Rule(
        (BackendConfigurationError,),
        503,
        "Model backend misconfigured on the server; retrying elsewhere may help",
        logging.ERROR,
    ),
    _Rule(
        (InspectionFailedError, ModelInvocationError, ResponseParsingError, SchemaValidationError),
        502,
        "Model backend failed",
    ),
)
_UPLOAD_TOO_LARGE_TITLE = "Upload too large"
_OVERRIDE_DISABLED_TITLE = "Backend override disabled"
_SERVER_BUSY_TITLE = "Server busy; retry after Retry-After seconds"
_GCS_UNAVAILABLE_TITLE = "Cloud Storage unavailable on the server"
_GCS_NOT_FOUND_TITLE = "Object not found"
_GCS_FORBIDDEN_TITLE = "Access to the object denied"
_GCS_CREDENTIALS_TITLE = "Cloud Storage credentials misconfigured on the server"
_GCS_RATE_LIMITED_TITLE = "Cloud Storage rate limit reached"
_GCS_FAILED_TITLE = "Cloud Storage failed"
_FALLBACK = _Rule((CSVInspectorError,), 500, "Inspection error", logging.ERROR)


def _rule_for(exc: CSVInspectorError) -> _Rule:
    """Return the first rule of the error table that matches ``exc``."""
    return next((rule for rule in _RULES if isinstance(exc, rule.exceptions)), _FALLBACK)


def status_for(exc: CSVInspectorError) -> int:
    """Return the HTTP status code for a library error.

    Args:
        exc: The error raised by ``csv_inspector``.

    Returns:
        422 for unusable input, 504 for a timeout, 503 for a misconfigured
        backend, 502 for a failed or nonsensical model answer, 500 otherwise.
    """
    return _rule_for(exc).status


class UploadTooLargeError(Exception):
    """The upload exceeds ``ApiSettings.max_upload_bytes``; answered with 413.

    Not a library error: the API raises it before calling ``csv_inspector``.
    """


class BackendOverrideDisabledError(Exception):
    """A request asked for the cloud backend while overrides are off; answered with 403.

    Not a library error: the API raises it before calling ``csv_inspector``.
    """


class ServerBusyError(Exception):
    """No inspection slot freed up within the queue timeout; answered with 503.

    Not a library error: the API raises it before calling ``csv_inspector``.

    Attributes:
        retry_after_seconds: Whole seconds the client should wait before
            retrying, sent as ``Retry-After``.
    """

    def __init__(self, message: str, *, retry_after_seconds: int) -> None:
        """Describe the refusal.

        Args:
            message: Explanation sent as ``detail``.
            retry_after_seconds: Value of the ``Retry-After`` header.
        """
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def problem_response(
    status: int,
    title: str,
    exc: Exception,
    *,
    detail: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build an ``application/problem+json`` response for ``exc``.

    Args:
        status: HTTP status code.
        title: Short summary of the problem class.
        exc: The error; its class name becomes ``error``, and its message
            ``detail`` unless ``detail`` is given.
        detail: Explanation to send instead of the exception message, when
            that message must not reach the client.
        headers: Extra response headers.

    Returns:
        The problem response.
    """
    problem = ProblemDetails(
        title=title,
        status=status,
        detail=str(exc) if detail is None else detail,
        error=type(exc).__name__,
    )
    return JSONResponse(
        problem.model_dump(mode="json"),
        status_code=status,
        media_type=PROBLEM_MEDIA_TYPE,
        headers=headers,
    )


async def _handle_inspector_error(request: Request, exc: Exception) -> JSONResponse:
    """Turn a library error into a problem response and log it once."""
    assert isinstance(exc, CSVInspectorError)  # registered for this class only
    rule = _rule_for(exc)
    logger.log(
        rule.log_level,
        "%s %s failed with %d %s: %s",
        request.method,
        request.url.path,
        rule.status,
        type(exc).__name__,
        exc,
        # A traceback only for the unexpected: the other rows are understood failures.
        exc_info=exc if rule is _FALLBACK else None,
    )
    return problem_response(rule.status, rule.title, exc)


async def _handle_upload_too_large(request: Request, exc: Exception) -> JSONResponse:
    """Answer an oversize upload with 413."""
    logger.warning("%s %s rejected with 413: %s", request.method, request.url.path, exc)
    return problem_response(413, _UPLOAD_TOO_LARGE_TITLE, exc)


async def _handle_override_disabled(request: Request, exc: Exception) -> JSONResponse:
    """Answer a refused backend override with 403."""
    logger.warning("%s %s rejected with 403: %s", request.method, request.url.path, exc)
    return problem_response(403, _OVERRIDE_DISABLED_TITLE, exc)


async def _handle_server_busy(request: Request, exc: Exception) -> JSONResponse:
    """Answer a request that found no free inspection slot with 503 and ``Retry-After``."""
    assert isinstance(exc, ServerBusyError)  # registered for this class only
    logger.warning("%s %s rejected with 503: %s", request.method, request.url.path, exc)
    headers = {"Retry-After": str(exc.retry_after_seconds)}
    return problem_response(503, _SERVER_BUSY_TITLE, exc, headers=headers)


async def _handle_gcs_not_installed(request: Request, exc: Exception) -> JSONResponse:
    """Answer ``POST /inspect/gcs`` with 503 when the ``[gcs]`` extra is missing."""
    logger.error("%s %s failed with 503: %s", request.method, request.url.path, exc)
    return problem_response(503, _GCS_UNAVAILABLE_TITLE, exc)


@dataclass(frozen=True)
class _GcsRule:
    """One row of the Cloud Storage error table.

    ``detail`` replaces the SDK's message, which names the bucket and the
    object and says which of the two is missing: clients must not learn
    which buckets exist.
    """

    exceptions: tuple[type[Exception], ...]
    status: int
    title: str
    detail: str
    log_level: int = logging.WARNING


def _gcs_error_handler(
    rules: tuple[_GcsRule, ...],
) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
    """Build the handler that answers a Cloud Storage error from ``rules``."""

    async def _handle_gcs_error(request: Request, exc: Exception) -> JSONResponse:
        rule = next(rule for rule in rules if isinstance(exc, rule.exceptions))
        logger.log(
            rule.log_level,
            "%s %s failed with %d %s: %s",
            request.method,
            request.url.path,
            rule.status,
            type(exc).__name__,
            exc,
        )
        headers = None
        if rule.status == 429:  # noqa: PLR2004 - the status code itself
            response = getattr(exc, "response", None)
            retry_after = getattr(response, "headers", {}).get("Retry-After")
            headers = {"Retry-After": retry_after} if retry_after else None
        return problem_response(rule.status, rule.title, exc, detail=rule.detail, headers=headers)

    return _handle_gcs_error


def _register_gcs_handlers(app: FastAPI) -> None:
    """Install the Cloud Storage error handler, when the ``[gcs]`` extra is installed.

    Without the extra no ``google.*`` exception can reach a route, and
    ``POST /inspect/gcs`` answers 503 through :class:`GcsNotInstalledError`.
    """
    try:
        from google.api_core import exceptions as api  # noqa: PLC0415 - the optional [gcs] extra
        from google.auth import exceptions as auth  # noqa: PLC0415
    except ImportError:
        return
    # Checked in order: the specific GoogleAPICallError subclasses first.
    rules = (
        _GcsRule(
            (api.NotFound,),
            404,
            _GCS_NOT_FOUND_TITLE,
            "The object does not exist, or its bucket does not exist.",
        ),
        _GcsRule(
            (api.Forbidden,),
            403,
            _GCS_FORBIDDEN_TITLE,
            "The server's service account may not read this object (storage.objects.get).",
        ),
        _GcsRule(
            (api.Unauthorized, auth.DefaultCredentialsError, auth.RefreshError),
            503,
            _GCS_CREDENTIALS_TITLE,
            "The server has no usable Cloud Storage credentials.",
            logging.ERROR,
        ),
        _GcsRule(
            (api.TooManyRequests,),
            429,
            _GCS_RATE_LIMITED_TITLE,
            "Cloud Storage rate-limited the request; retry later.",
        ),
        _GcsRule(
            (api.GoogleAPICallError, api.RetryError),
            502,
            _GCS_FAILED_TITLE,
            "Cloud Storage failed or is unavailable; retry later.",
        ),
    )
    handler = _gcs_error_handler(rules)
    for error_type in (
        api.GoogleAPICallError,
        api.RetryError,
        auth.DefaultCredentialsError,
        auth.RefreshError,
    ):
        app.add_exception_handler(error_type, handler)


def register_exception_handlers(app: FastAPI) -> None:
    """Install the API's error handlers on ``app``.

    Args:
        app: The application being built by ``create_app``.
    """
    app.add_exception_handler(CSVInspectorError, _handle_inspector_error)
    app.add_exception_handler(UploadTooLargeError, _handle_upload_too_large)
    app.add_exception_handler(BackendOverrideDisabledError, _handle_override_disabled)
    app.add_exception_handler(ServerBusyError, _handle_server_busy)
    app.add_exception_handler(GcsNotInstalledError, _handle_gcs_not_installed)
    _register_gcs_handlers(app)


_VALIDATION_ERROR_SCHEMA: dict[str, Any] = {
    "title": "HTTPValidationError",
    "type": "object",
    "properties": {"detail": {"type": "array", "items": {"type": "object"}}},
}
"""Shape of FastAPI's own 422 body for invalid parameters or a missing field."""


_GCS_DESCRIPTIONS = {
    403: f"{_OVERRIDE_DISABLED_TITLE}, or {_GCS_FORBIDDEN_TITLE.lower()}",
    404: f"{_GCS_NOT_FOUND_TITLE} (or its bucket: not told apart)",
    429: f"{_GCS_RATE_LIMITED_TITLE}; honour Retry-After when present",
    502: f"Model backend or {_GCS_FAILED_TITLE}",
    503: "Model backend, Cloud Storage credentials or the [gcs] extra misconfigured on the server",
}
"""OpenAPI descriptions of the statuses of ``POST /inspect/gcs``."""


def problem_responses(
    *statuses: int, gcs: bool = False, gated: bool = False
) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses`` entries for error statuses with a ``ProblemDetails`` body.

    Args:
        *statuses: The error statuses a route can return.
        gcs: Describe the statuses of ``POST /inspect/gcs``, which add the
            Cloud Storage errors to the library's ones.
        gated: The route waits for an inspection slot, so its 503 may also
            mean the server is busy.

    Returns:
        A mapping for a route decorator's ``responses=`` argument.
    """
    titles = {rule.status: rule.title for rule in (*_RULES, _FALLBACK)}
    titles[413] = _UPLOAD_TOO_LARGE_TITLE
    titles[403] = _OVERRIDE_DISABLED_TITLE
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
