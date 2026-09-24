"""Mapping of ``csv_inspector`` errors to HTTP problem responses.

Every :class:`csv_inspector.CSVInspectorError` that escapes a route becomes
an RFC 9457-style ``application/problem+json`` response through one
exception handler, so routes never catch library errors themselves.

``ValueError`` and ``TypeError`` from the library (a byte budget out of
range, an unsupported source) are programming errors in this host, not
domain failures: they are deliberately not handled here and surface as a
plain 500 through FastAPI's default handling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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


def problem_response(status: int, title: str, exc: Exception) -> JSONResponse:
    """Build an ``application/problem+json`` response for ``exc``.

    Args:
        status: HTTP status code.
        title: Short summary of the problem class.
        exc: The error; its message becomes ``detail`` and its class name ``error``.

    Returns:
        The problem response.
    """
    problem = ProblemDetails(title=title, status=status, detail=str(exc), error=type(exc).__name__)
    return JSONResponse(
        problem.model_dump(mode="json"), status_code=status, media_type=PROBLEM_MEDIA_TYPE
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


def register_exception_handlers(app: FastAPI) -> None:
    """Install the API's error handlers on ``app``.

    Args:
        app: The application being built by ``create_app``.
    """
    app.add_exception_handler(CSVInspectorError, _handle_inspector_error)
