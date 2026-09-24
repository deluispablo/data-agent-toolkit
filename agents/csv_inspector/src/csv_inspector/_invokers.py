"""Model invokers (sync and async) for the local and cloud backends.

Every invoker builds its own client per call and closes it afterwards, so
there is no shared, mutable client state and calls are thread-safe. SDKs are
imported lazily: the local backend needs only ``ollama``, the cloud backend
only ``google-genai`` (``[cloud]`` extra).
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

from pydantic import SecretStr

from ._backends import LLMBackend
from ._config import CLOUD_EXTRA_HINT, CloudCredentials, Settings, resolve_settings
from ._exceptions import (
    BackendConfigurationError,
    CredentialsNotConfiguredError,
    ModelInvocationError,
    ModelTimeoutError,
)
from ._models import CSVInspectionResult
from ._prompt import SYSTEM_PROMPT

if TYPE_CHECKING:
    from google.genai import Client as GenaiClient

logger = logging.getLogger(__name__)

ModelInvoker = Callable[[str, str], str]
"""A callable that sends ``prompt`` to ``model`` and returns the raw response text."""

AsyncModelInvoker = Callable[[str, str], Awaitable[str]]
"""An async callable that sends ``prompt`` to ``model`` and returns the raw text."""


class InvokerResponse(NamedTuple):
    """One model call's raw text plus what the backend reported about its cost.

    Attributes:
        text: The raw response text, exactly as the model returned it.
        prompt_tokens: Prompt tokens, or ``None`` when not reported.
        completion_tokens: Completion tokens, or ``None`` when not reported.
        retries: Transient errors retried before this answer (cloud only).
        load_seconds: Time Ollama spent loading the model, or ``None``.
    """

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    retries: int = 0
    load_seconds: float | None = None


class _OllamaMessage(Protocol):
    """The part of an Ollama chat message this module reads."""

    @property
    def content(self) -> str | None:
        """The message text."""
        ...


class _OllamaChatResponse(Protocol):
    """The part of an Ollama chat response this module reads."""

    @property
    def message(self) -> _OllamaMessage:
        """The assistant message."""
        ...


class _GenaiResponse(Protocol):
    """The part of a google-genai response this module reads."""

    @property
    def text(self) -> str | None:
        """The concatenated response text."""
        ...

    @property
    def prompt_feedback(self) -> object | None:
        """Why the prompt was blocked, if it was (has a ``block_reason``)."""
        ...

    @property
    def candidates(self) -> Sequence[object] | None:
        """The answer candidates (each has a ``finish_reason``)."""
        ...


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------


def _is_timeout(exc: BaseException) -> bool:
    """Whether ``exc`` is a client-side timeout (built-in or ``httpx``)."""
    if isinstance(exc, TimeoutError):
        return True
    try:
        import httpx  # noqa: PLC0415 - transport of both SDKs; only needed on error.
    except ImportError:  # pragma: no cover - both SDKs depend on httpx.
        return False
    return isinstance(exc, httpx.TimeoutException)


def _count(value: object) -> int | None:
    """A token count read from an SDK response, or ``None`` when absent or malformed."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _redact(message: str, secret: SecretStr | None) -> str:
    """Remove a secret's value from an error message before it is logged or raised."""
    if secret is None:
        return message
    value = secret.get_secret_value()
    return message.replace(value, "***") if value else message


# ---------------------------------------------------------------------
# Local backend (Ollama)
# ---------------------------------------------------------------------


def _import_ollama() -> ModuleType:
    """Import the ``ollama`` package lazily.

    Raises:
        BackendConfigurationError: If it is not installed.
    """
    try:
        import ollama  # noqa: PLC0415 - lazily imported: only this backend needs it.
    except ImportError as exc:
        raise BackendConfigurationError(
            "The 'ollama' package is required to use the local Ollama backend. "
            "Install it with 'pip install ollama' or inject a custom model_invoker."
        ) from exc
    return ollama


_OLLAMA_MIN_NUM_CTX = 4096
_OLLAMA_MAX_NUM_CTX = 32768
_OLLAMA_RESPONSE_TOKENS = 1024


def _ollama_num_ctx(prompt: str) -> int:
    """Context window (tokens) large enough for the system prompt, ``prompt`` and the reply.

    Ollama's default window is small and it silently drops the *start* of an
    overflowing prompt (the instructions and head sample), so the window is
    sized from the prompt. Numeric CSV text tokenizes poorly, so this assumes
    ~2 characters per token. The result is rounded up to a power of two to
    limit model reloads (Ollama reloads a model when ``num_ctx`` changes) and
    capped; the sampling limits keep built-in prompts under the cap.
    """
    needed = (len(SYSTEM_PROMPT) + len(prompt)) // 2 + _OLLAMA_RESPONSE_TOKENS
    if needed > _OLLAMA_MAX_NUM_CTX:
        logger.warning(
            "Prompt needs ~%d tokens; capping Ollama num_ctx at %d, so it may be truncated.",
            needed,
            _OLLAMA_MAX_NUM_CTX,
        )
        return _OLLAMA_MAX_NUM_CTX
    return max(_OLLAMA_MIN_NUM_CTX, 1 << (needed - 1).bit_length())


def _ollama_request(prompt: str, model: str) -> dict[str, Any]:
    """Keyword arguments for an Ollama chat request."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "format": "json",
        # num_predict caps the reply at the budget num_ctx reserves for it.
        # Without it a model stuck repeating (e.g. an endless example_values
        # list) generates until the timeout, or forever when there is none;
        # a capped reply is truncated JSON, which fails parsing and moves on
        # to the fallback model.
        "options": {
            "temperature": 0.0,
            "num_ctx": _ollama_num_ctx(prompt),
            "num_predict": _OLLAMA_RESPONSE_TOKENS,
        },
    }


def _ollama_error(model: str, exc: Exception) -> ModelInvocationError:
    """Map an Ollama client failure to a domain error."""
    if _is_timeout(exc):
        return ModelTimeoutError(f"Model '{model}' timed out: {exc}")
    return ModelInvocationError(f"Model '{model}' failed to respond: {exc}")


def _ollama_response(response: _OllamaChatResponse, model: str) -> InvokerResponse:
    """Extract the non-empty message content and the usage counters of an Ollama reply.

    ``prompt_eval_count``, ``eval_count`` and ``load_duration`` (nanoseconds)
    are optional in the SDK, so each one missing is ``None``.
    """
    content = response.message.content
    if not content:
        raise ModelInvocationError(f"Model '{model}' returned an empty response.")
    load_ns = _count(getattr(response, "load_duration", None))
    return InvokerResponse(
        content,
        prompt_tokens=_count(getattr(response, "prompt_eval_count", None)),
        completion_tokens=_count(getattr(response, "eval_count", None)),
        load_seconds=None if load_ns is None else load_ns / 1e9,
    )


def _invoke_ollama(
    prompt: str, model: str, *, host: str | None, timeout_seconds: float | None
) -> InvokerResponse:
    """:func:`invoke_ollama_model`, returning the usage counters with the text."""
    ollama = _import_ollama()
    try:
        with ollama.Client(host=host, timeout=timeout_seconds) as client:
            response = client.chat(**_ollama_request(prompt, model))
    except Exception as exc:
        raise _ollama_error(model, exc) from exc
    return _ollama_response(response, model)


async def _ainvoke_ollama(
    prompt: str, model: str, *, host: str | None, timeout_seconds: float | None
) -> InvokerResponse:
    """:func:`ainvoke_ollama_model`, returning the usage counters with the text."""
    ollama = _import_ollama()
    try:
        async with ollama.AsyncClient(host=host, timeout=timeout_seconds) as client:
            response = await client.chat(**_ollama_request(prompt, model))
    except Exception as exc:
        raise _ollama_error(model, exc) from exc
    return _ollama_response(response, model)


def invoke_ollama_model(
    prompt: str, model: str, *, host: str | None = None, timeout_seconds: float | None = None
) -> str:
    """Send a prompt to a local Ollama model and return its raw text response.

    Args:
        prompt: The fully-built prompt to send.
        model: Name of the Ollama model to invoke (e.g. ``"qwen2.5-coder:7b"``).
        host: Base URL of the Ollama server, or ``None`` for the SDK's
            default (which honours ``OLLAMA_HOST``).
        timeout_seconds: Client-side timeout for the request, or ``None``
            for the client's default (no timeout).

    Returns:
        The raw text content of the model's response.

    Raises:
        BackendConfigurationError: If the ``ollama`` package is not installed.
        ModelTimeoutError: If the request times out.
        ModelInvocationError: If the backend cannot be reached, the model is
            not available locally, or the model returns an empty message.
    """
    return _invoke_ollama(prompt, model, host=host, timeout_seconds=timeout_seconds).text


async def ainvoke_ollama_model(
    prompt: str, model: str, *, host: str | None = None, timeout_seconds: float | None = None
) -> str:
    """Async variant of :func:`invoke_ollama_model`, using ``ollama.AsyncClient``.

    Args:
        prompt: The fully-built prompt to send.
        model: Name of the Ollama model to invoke.
        host: Base URL of the Ollama server, or ``None`` for the SDK's default.
        timeout_seconds: Client-side timeout for the request, or ``None``.

    Returns:
        The raw text content of the model's response.

    Raises:
        BackendConfigurationError: If the ``ollama`` package is not installed.
        ModelTimeoutError: If the request times out.
        ModelInvocationError: If the request fails or the response is empty.
    """
    response = await _ainvoke_ollama(prompt, model, host=host, timeout_seconds=timeout_seconds)
    return response.text


# ---------------------------------------------------------------------
# Cloud backend (Google Gemini via google-genai)
# ---------------------------------------------------------------------


# One retry of the same request for these transient statuses: 429
# RESOURCE_EXHAUSTED (rate limit) and 503 UNAVAILABLE (high demand). The
# fallback model stays the strategy for persistent failures.
_TRANSIENT_STATUS_CODES = frozenset({429, 503})
_RETRY_DELAY_SECONDS = 1.0
_RETRY_JITTER = 0.2
# A Retry-After longer than this means a quota, not a blip: fall back instead.
_MAX_RETRY_AFTER_SECONDS = 10.0
# The retried request needs at least this much of the model's budget left.
_MIN_RETRY_BUDGET_SECONDS = 1.0


def _to_milliseconds(timeout_seconds: float) -> int:
    """Convert seconds to the whole milliseconds ``google-genai`` expects (at least 1)."""
    return max(1, math.ceil(timeout_seconds * 1000))


def _transient_status(exc: Exception) -> int | None:
    """Return the HTTP status of a transient ``google-genai`` error, else ``None``."""
    code = getattr(exc, "code", None)
    return code if code in _TRANSIENT_STATUS_CODES else None


def _retry_after_seconds(exc: Exception) -> float | None:
    """Read a numeric ``Retry-After`` header from the error's HTTP response, if any."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    value = headers.get("Retry-After") if headers is not None else None
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


class _CloudCall:
    """Everything needed for one Gemini request, prepared before any network use."""

    def __init__(self, settings: Settings | None, timeout_seconds: float | None) -> None:
        self.credentials: CloudCredentials = resolve_settings(settings).cloud_credentials()
        try:
            # Lazily imported: only this backend needs them. google-auth ships with google-genai.
            from google import genai  # noqa: PLC0415
            from google.auth import exceptions as google_auth_exceptions  # noqa: PLC0415
            from google.genai import types  # noqa: PLC0415
        except ImportError as exc:
            raise BackendConfigurationError(
                f"The 'google-genai' package is required for the 'api' backend. {CLOUD_EXTRA_HINT}"
            ) from exc
        self._genai = genai
        self._types = types
        self._auth_errors = google_auth_exceptions
        self._timeout_seconds = timeout_seconds
        self._started = time.monotonic()
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
            response_json_schema=CSVInspectionResult.model_json_schema(),
            # No tools are passed; disabling AFC stops the SDK from logging an
            # "AFC is enabled" INFO line and a WARNING on every call.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        self._client_kwargs: dict[str, Any] = (
            {"api_key": self.credentials.api_key.get_secret_value()}
            if self.credentials.api_key is not None
            else {
                "vertexai": True,
                "project": self.credentials.project,
                "location": self.credentials.location,
            }
        )
        if timeout_seconds is not None:
            self._set_request_timeout(timeout_seconds)

    def _set_request_timeout(self, timeout_seconds: float) -> None:
        """Set the request timeout of the clients created from now on."""
        # google-genai takes the request timeout in MILLISECONDS.
        self._client_kwargs["http_options"] = self._types.HttpOptions(
            timeout=_to_milliseconds(timeout_seconds)
        )

    def client(self) -> GenaiClient:
        """Create the ``google-genai`` client for this call."""
        return self._genai.Client(**self._client_kwargs)

    def retry_delay(self, model: str, exc: Exception) -> float | None:
        """Decide whether a failed request is retried once, and after how long.

        Only a 429 or a 503 is retried. The wait is the ``Retry-After``
        header when present (a longer one than ``_MAX_RETRY_AFTER_SECONDS``
        means a quota: no retry), else about one second with jitter. The
        retry is skipped when the wait would leave the model's time budget
        less than ``_MIN_RETRY_BUDGET_SECONDS``; otherwise the retried
        request's timeout is cut to what remains of the budget.

        Args:
            model: The model that failed, for the log line.
            exc: The exception the request raised.

        Returns:
            The seconds to wait before the one retry, or ``None`` to give up.
        """
        status = _transient_status(exc)
        if status is None:
            return None
        retry_after = _retry_after_seconds(exc)
        if retry_after is None:
            jitter = random.uniform(1 - _RETRY_JITTER, 1 + _RETRY_JITTER)
            delay = _RETRY_DELAY_SECONDS * jitter
        elif retry_after <= _MAX_RETRY_AFTER_SECONDS:
            delay = retry_after
        else:
            return None
        if self._timeout_seconds is not None:
            remaining = self._timeout_seconds - (time.monotonic() - self._started) - delay
            if remaining < _MIN_RETRY_BUDGET_SECONDS:
                return None
            self._set_request_timeout(remaining)
        logger.warning(
            "Cloud model '%s' answered %d; retrying once in %.1f s.", model, status, delay
        )
        return delay

    def error(self, model: str, exc: Exception) -> ModelInvocationError:
        """Map a cloud failure to a domain error whose text never contains the key."""
        message = _redact(str(exc), self.credentials.api_key)
        if isinstance(exc, self._auth_errors.DefaultCredentialsError):
            return CredentialsNotConfiguredError(
                "Vertex AI could not find Application Default Credentials; run "
                f"'gcloud auth application-default login' or set GEMINI_API_KEY. ({message})"
            )
        if _is_timeout(exc):
            return ModelTimeoutError(f"Cloud model '{model}' timed out: {message}")
        return ModelInvocationError(f"Cloud model '{model}' failed to respond: {message}")


def _empty_response_reason(response: _GenaiResponse) -> str:
    """Say why a Gemini response has no text: a blocked prompt or the finish reason."""
    block_reason = getattr(response.prompt_feedback, "block_reason", None)
    if block_reason:
        return f" (prompt blocked: {getattr(block_reason, 'value', block_reason)})"
    candidates = response.candidates or []
    finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    if finish_reason:
        return f" (finish reason: {getattr(finish_reason, 'value', finish_reason)})"
    return ""


def _cloud_response(response: _GenaiResponse, model: str, *, retries: int) -> InvokerResponse:
    """Extract the non-empty text and the token counts of a Gemini response."""
    text = response.text
    if not text:
        raise ModelInvocationError(
            f"Cloud model '{model}' returned an empty response{_empty_response_reason(response)}."
        )
    usage = getattr(response, "usage_metadata", None)
    return InvokerResponse(
        text,
        prompt_tokens=_count(getattr(usage, "prompt_token_count", None)),
        completion_tokens=_count(getattr(usage, "candidates_token_count", None)),
        retries=retries,
    )


def _invoke_cloud(
    prompt: str, model: str, *, settings: Settings | None, timeout_seconds: float | None
) -> InvokerResponse:
    """:func:`invoke_cloud_model`, returning the usage counters with the text."""
    call = _CloudCall(settings, timeout_seconds)
    logger.debug("Calling cloud model '%s' via %s.", model, call.credentials.describe())
    retries = 0
    while True:
        try:
            with call.client() as client:
                response = client.models.generate_content(
                    model=model, contents=prompt, config=call.config
                )
            break
        except Exception as exc:  # noqa: BLE001 - varied SDK/transport errors; see below.
            delay = None if retries else call.retry_delay(model, exc)
            if delay is None:
                # Re-raised "from None" on purpose: the original exception (and its
                # traceback) could carry the API key, so only a redacted message is kept.
                raise call.error(model, exc) from None
        retries += 1
        time.sleep(delay)
    return _cloud_response(response, model, retries=retries)


async def _ainvoke_cloud(
    prompt: str, model: str, *, settings: Settings | None, timeout_seconds: float | None
) -> InvokerResponse:
    """:func:`ainvoke_cloud_model`, returning the usage counters with the text."""
    call = _CloudCall(settings, timeout_seconds)
    logger.debug("Calling cloud model '%s' via %s (async).", model, call.credentials.describe())
    retries = 0
    while True:
        try:
            with call.client() as client:
                async with client.aio as aio:
                    response = await aio.models.generate_content(
                        model=model, contents=prompt, config=call.config
                    )
            break
        except Exception as exc:  # noqa: BLE001 - varied SDK/transport errors; see below.
            delay = None if retries else call.retry_delay(model, exc)
            if delay is None:
                # Re-raised "from None" on purpose; see _invoke_cloud.
                raise call.error(model, exc) from None
        retries += 1
        await asyncio.sleep(delay)
    return _cloud_response(response, model, retries=retries)


def invoke_cloud_model(
    prompt: str,
    model: str,
    *,
    settings: Settings | None = None,
    timeout_seconds: float | None = None,
) -> str:
    """Send a prompt to a Google Gemini model and return its raw text response.

    Authenticates with ``gemini_api_key`` (Gemini Developer API) or, failing
    that, ``google_cloud_project`` + ``google_cloud_location`` (Vertex AI with
    Application Default Credentials), from ``settings`` or the environment.
    Requests JSON constrained by the :class:`CSVInspectionResult` JSON
    Schema, at ``temperature=0.0``. Credentials are checked before any client
    is created, and the API key never appears in logs or raised errors.

    Args:
        prompt: The fully-built prompt to send.
        model: Name of the Gemini model to invoke (e.g. ``"gemini-3.6-flash"``).
        settings: Injected settings; when ``None``, read from the environment.
        timeout_seconds: Client-side timeout for the request, or ``None``.

    Returns:
        The raw text content of the model's response.

    Raises:
        BackendConfigurationError: If the ``[cloud]`` extra is missing or a
            setting is invalid.
        CredentialsNotConfiguredError: If no usable credentials are set.
        ModelTimeoutError: If the request times out.
        ModelInvocationError: If the service rejects or fails the request, or
            returns an empty response.
    """
    return _invoke_cloud(prompt, model, settings=settings, timeout_seconds=timeout_seconds).text


async def ainvoke_cloud_model(
    prompt: str,
    model: str,
    *,
    settings: Settings | None = None,
    timeout_seconds: float | None = None,
) -> str:
    """Async variant of :func:`invoke_cloud_model`, using ``client.aio``.

    Args:
        prompt: The fully-built prompt to send.
        model: Name of the Gemini model to invoke.
        settings: Injected settings; when ``None``, read from the environment.
        timeout_seconds: Client-side timeout for the request, or ``None``.

    Returns:
        The raw text content of the model's response.

    Raises:
        BackendConfigurationError: If the ``[cloud]`` extra is missing or a
            setting is invalid.
        CredentialsNotConfiguredError: If no usable credentials are set.
        ModelTimeoutError: If the request times out.
        ModelInvocationError: If the request fails or the response is empty.
    """
    response = await _ainvoke_cloud(
        prompt, model, settings=settings, timeout_seconds=timeout_seconds
    )
    return response.text


# ---------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------


def builtin_invoker(
    backend: LLMBackend, settings: Settings
) -> Callable[[str, str, float | None], InvokerResponse]:
    """Return the built-in sync invoker as ``(prompt, model, timeout_seconds) -> response``.

    This is the seam the inspection calls: ``response.text`` is the model's
    raw text, before any parsing, and the other fields feed ``Usage``. A
    wrapper around the returned callable (e.g. one keeping the raw text)
    must return the :class:`InvokerResponse` unchanged so usage keeps flowing.
    """
    if backend is LLMBackend.API:
        return lambda prompt, model, timeout: _invoke_cloud(
            prompt, model, settings=settings, timeout_seconds=timeout
        )
    return lambda prompt, model, timeout: _invoke_ollama(
        prompt, model, host=settings.ollama_host, timeout_seconds=timeout
    )


def builtin_async_invoker(
    backend: LLMBackend, settings: Settings
) -> Callable[[str, str, float | None], Awaitable[InvokerResponse]]:
    """Return the built-in async invoker; the async twin of :func:`builtin_invoker`."""
    if backend is LLMBackend.API:
        return lambda prompt, model, timeout: _ainvoke_cloud(
            prompt, model, settings=settings, timeout_seconds=timeout
        )
    return lambda prompt, model, timeout: _ainvoke_ollama(
        prompt, model, host=settings.ollama_host, timeout_seconds=timeout
    )
