"""Request ids and the access log.

:class:`RequestIdMiddleware` gives every HTTP request an id: the client's
``X-Request-ID`` when it is a sane token, a new ``uuid4`` otherwise. The id
is echoed in the response's ``X-Request-ID`` header and stored in a
:class:`~contextvars.ContextVar` for the duration of the request, which
``asyncio`` copies into the request's tasks and ``asyncio.to_thread``
workers, so it follows the library into its sampling thread.

:class:`RequestIdFilter` copies that id onto log records as
``record.request_id``. Attach it to a **handler**, not a logger: a logger's
filters only see records logged on that exact logger, while a handler's see
every record it emits, including those of the ``csv_inspector`` library.
Configuring handlers is the application's job (``main_demo.py``), never this
package's.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from contextvars import ContextVar

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

HEADER = "X-Request-ID"
CLIENT_CLOSED_REQUEST = 499
"""Status logged for a request cancelled before its response (nginx's convention)."""
NO_REQUEST_ID = "-"
"""The ``request_id`` of records logged outside a request."""

request_id_var: ContextVar[str] = ContextVar("request_id", default=NO_REQUEST_ID)
"""The id of the request being handled; :data:`NO_REQUEST_ID` outside one."""

# Printable ASCII without spaces, bounded: a client-chosen id ends up in log
# lines, so anything that could forge or break a line is replaced.
_VALID_ID = re.compile(r"[\x21-\x7e]{1,128}")

access_logger = logging.getLogger("csv_inspector_api.access")


class RequestIdFilter(logging.Filter):
    """Set ``record.request_id`` from the current request; never drops a record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Annotate ``record`` with the current request id.

        Args:
            record: The record about to be emitted.

        Returns:
            Always ``True``.
        """
        record.request_id = request_id_var.get()
        return True


class RequestIdMiddleware:
    """ASGI middleware: request id in, request id out, one access log line.

    A plain ASGI middleware rather than Starlette's ``BaseHTTPMiddleware``, so
    the request body still streams to ``/inspect/raw`` untouched and a
    cancelled request is not wrapped in another task.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap ``app``.

        Args:
            app: The ASGI application to wrap.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection; only HTTP requests get an id.

        Args:
            scope: The connection scope.
            receive: The ASGI receive channel.
            send: The ASGI send channel.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        claimed = dict(scope["headers"]).get(HEADER.lower().encode(), b"").decode("latin-1")
        request_id = claimed if _VALID_ID.fullmatch(claimed) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        status = 500  # what the client gets if the app fails before responding
        started = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message)[HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except asyncio.CancelledError:
            status = CLIENT_CLOSED_REQUEST
            raise
        finally:
            access_logger.info(
                "%s %s %d %.1f ms",
                scope["method"],
                scope["path"],
                status,
                (time.perf_counter() - started) * 1000,
            )
            request_id_var.reset(token)
