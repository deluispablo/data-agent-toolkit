"""Model invokers (sync and async) for the local and cloud backends.

Every call builds its own client and closes it (thread-safe); SDKs are imported
lazily (``google-genai`` is the ``[cloud]`` extra); API keys never reach an error.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import random
import time
from collections.abc import Awaitable, Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

from pydantic import SecretStr

from ._backends import LLMBackend
from ._config import CLOUD_EXTRA_HINT, CloudCredentials, Settings, resolve_settings
from ._exceptions import (
    BackendConfigurationError,
    CredentialsNotConfiguredError,
    ModelInvocationError,
    ModelTimeoutError,
)
from ._prompt import CHARS_PER_TOKEN, SYSTEM_PROMPT, response_schema

if TYPE_CHECKING:
    from google.genai import Client as GenaiClient
    from google.genai.types import GenerateContentResponse
    from ollama import ChatResponse

logger = logging.getLogger(__name__)

_R = TypeVar("_R")

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


def _import_ollama() -> ModuleType:
    """Import the ``ollama`` package lazily, or raise :class:`BackendConfigurationError`."""
    try:
        import ollama  # noqa: PLC0415 - lazily imported: only this backend needs it.
    except ImportError as exc:
        raise BackendConfigurationError(
            "The 'ollama' package is required to use the local Ollama backend. "
            "Install it with 'pip install ollama' or inject a custom model_invoker."
        ) from exc
    return ollama


# Three context windows: few steps limit model reloads (Ollama reloads a
# model when num_ctx changes); the sampling limits keep prompts under the last.
_OLLAMA_NUM_CTX_STEPS = (8192, 16384, 32768)
# The reply cap: fixed keys plus one name per field, over a floor. In the M6-03
# runs (2026-09-25) answers for 12 fields or fewer had a p99 of 263 tokens
# (floor: x 1.5, rounded up to 64); wider ones up to 16.9 tokens per field.
_OLLAMA_MIN_RESPONSE_TOKENS = 448
_REPLY_BASE_TOKENS = 32
_REPLY_TOKENS_PER_FIELD = 20
# The status an Ollama server older than 0.5 answers to a schema ``format``.
_HTTP_BAD_REQUEST = 400


def ollama_reply_tokens(fields: int | None) -> int:
    """The reply cap (``num_predict``) for ``fields`` columns (``None``: the floor).

    A model stuck repeating is cut there: it fails fast, leaving the fallback time.
    """
    if fields is None:
        return _OLLAMA_MIN_RESPONSE_TOKENS
    return max(_OLLAMA_MIN_RESPONSE_TOKENS, _REPLY_BASE_TOKENS + _REPLY_TOKENS_PER_FIELD * fields)


def _ollama_num_ctx(prompt: str, reply_tokens: int) -> int:
    """Context window (tokens) large enough for the system prompt, ``prompt`` and the reply.

    Ollama silently drops the *start* of an overflowing prompt: the window is
    sized at :data:`CHARS_PER_TOKEN` characters per token, up to the last step.
    """
    needed = math.ceil((len(SYSTEM_PROMPT) + len(prompt)) / CHARS_PER_TOKEN) + reply_tokens
    for step in _OLLAMA_NUM_CTX_STEPS:
        if needed <= step:
            if step > _OLLAMA_NUM_CTX_STEPS[0]:
                logger.info("Prompt and reply need ~%d tokens; Ollama num_ctx %d.", needed, step)
            return step
    cap = _OLLAMA_NUM_CTX_STEPS[-1]
    msg = "Prompt needs ~%d tokens; capping Ollama num_ctx at %d, so it may be truncated."
    logger.warning(msg, needed, cap)
    return cap


def _ollama_request(
    prompt: str, model: str, *, reply_tokens: int, schema: bool = True
) -> dict[str, Any]:
    """Keyword arguments for an Ollama chat request (``schema=False``: plain JSON mode)."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        # The contract of the reply's shape; Ollama compiles it into a grammar.
        "format": response_schema() if schema else "json",
        # A model stuck repeating is capped (truncated JSON fails over to the fallback).
        "options": {
            "temperature": 0.0,
            "num_ctx": _ollama_num_ctx(prompt, reply_tokens),
            "num_predict": reply_tokens,
        },
    }


def _schema_rejected(ollama: ModuleType, model: str, exc: Exception) -> bool:
    """Whether a pre-0.5 server rejected the schema ``format`` (HTTP 400); warns when so."""
    rejected = (
        isinstance(exc, getattr(ollama, "ResponseError", ()))
        and getattr(exc, "status_code", None) == _HTTP_BAD_REQUEST
        and "format" in str(exc).lower()
    )
    if rejected:
        msg = "Ollama rejected the response schema for model '%s' (%s); retrying with "
        msg += "format='json'. Upgrade the Ollama server (0.5 or later) for structured outputs."
        logger.warning(msg, model, exc)
    return rejected


def _ollama_error(model: str, exc: Exception) -> ModelInvocationError:
    """Map an Ollama client failure to a domain error."""
    if _is_timeout(exc):
        return ModelTimeoutError(f"Model '{model}' timed out: {exc}")
    return ModelInvocationError(f"Model '{model}' failed to respond: {exc}")


def _ollama_response(response: ChatResponse, model: str) -> InvokerResponse:
    """The non-empty message content and the (optional) usage counters of an Ollama reply."""
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
    prompt: str,
    model: str,
    timeout_seconds: float | None = None,
    *,
    host: str | None = None,
    reply_tokens: int = _OLLAMA_MIN_RESPONSE_TOKENS,
) -> InvokerResponse:
    """Send ``prompt`` to Ollama; a pre-0.5 server rejecting the schema is asked in JSON mode.

    Raises:
        BackendConfigurationError: If the ``ollama`` package is not installed.
        ModelTimeoutError: If the request times out.
        ModelInvocationError: If the request fails or the response is empty.
    """
    ollama = _import_ollama()
    request = functools.partial(_ollama_request, prompt, model, reply_tokens=reply_tokens)
    try:
        with ollama.Client(host=host, timeout=timeout_seconds) as client:
            try:
                response = client.chat(**request())
            except Exception as exc:
                if not _schema_rejected(ollama, model, exc):
                    raise
                response = client.chat(**request(schema=False))
    except Exception as exc:
        raise _ollama_error(model, exc) from exc
    return _ollama_response(response, model)


async def _ainvoke_ollama(
    prompt: str,
    model: str,
    timeout_seconds: float | None = None,
    *,
    host: str | None = None,
    reply_tokens: int = _OLLAMA_MIN_RESPONSE_TOKENS,
) -> InvokerResponse:
    """:func:`_invoke_ollama` with ``ollama.AsyncClient``."""
    ollama = _import_ollama()
    request = functools.partial(_ollama_request, prompt, model, reply_tokens=reply_tokens)
    try:
        async with ollama.AsyncClient(host=host, timeout=timeout_seconds) as client:
            try:
                response = await client.chat(**request())
            except Exception as exc:
                if not _schema_rejected(ollama, model, exc):
                    raise
                response = await client.chat(**request(schema=False))
    except Exception as exc:
        raise _ollama_error(model, exc) from exc
    return _ollama_response(response, model)


# One retry for 429 (rate limit) and 503 (high demand); the fallback model
# handles persistent failures.
_TRANSIENT_STATUS_CODES = frozenset({429, 503})
_RETRY_DELAY_SECONDS = 1.0
_RETRY_JITTER = 0.2
# A Retry-After longer than this means a quota, not a blip: fall back instead.
_MAX_RETRY_AFTER_SECONDS = 10.0
# Logged once per retried cloud request; the eval harness counts these.
_RETRY_WARNING = "Cloud model '%s' answered %d; retrying once in %.1f s."
# The retried request needs at least this much of the model's budget left.
_MIN_RETRY_BUDGET_SECONDS = 1.0


def _transient_status(exc: Exception) -> int | None:
    """Return the HTTP status of a transient ``google-genai`` error, else ``None``."""
    code = getattr(exc, "code", None)
    return code if code in _TRANSIENT_STATUS_CODES else None


class _CloudCall:
    """One Gemini request, prepared before any network use, and its one retry."""

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
        self.retries = 0
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
            response_json_schema=response_schema(),
            # No tools: without this the SDK logs "AFC is enabled" on every call.
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
        # google-genai takes the request timeout in whole MILLISECONDS, at least 1.
        milliseconds = max(1, math.ceil(timeout_seconds * 1000))
        self._client_kwargs["http_options"] = self._types.HttpOptions(timeout=milliseconds)

    def client(self) -> GenaiClient:
        """Create the ``google-genai`` client for this call."""
        return self._genai.Client(**self._client_kwargs)

    def retry_delay(self, model: str, exc: Exception) -> float:
        """The seconds to wait before retrying a failed request, or raise its domain error.

        Only a first 429 or 503 is retried, and only with ``_MIN_RETRY_BUDGET_SECONDS``
        of budget left after the wait: the retry's timeout is cut to what remains.
        """
        delay = None if self.retries else self._delay(exc)
        if delay is not None and self._timeout_seconds is not None:
            remaining = self._timeout_seconds - (time.monotonic() - self._started) - delay
            if remaining < _MIN_RETRY_BUDGET_SECONDS:
                delay = None
            else:
                self._set_request_timeout(remaining)
        if delay is None:
            # "from None": the original exception could carry the API key.
            raise self._error(model, exc) from None
        logger.warning(_RETRY_WARNING, model, _transient_status(exc), delay)
        self.retries += 1
        return delay

    @staticmethod
    def _delay(exc: Exception) -> float | None:
        """The wait of a transient error: ``Retry-After`` up to a limit (else a quota), or ~1 s."""
        if _transient_status(exc) is None:
            return None
        headers: Any = getattr(getattr(exc, "response", None), "headers", None) or {}
        try:  # A numeric Retry-After header, if any.
            retry_after = float(headers.get("Retry-After"))
        except (TypeError, ValueError):
            retry_after = -1.0
        if not retry_after >= 0:  # also NaN
            return _RETRY_DELAY_SECONDS * random.uniform(1 - _RETRY_JITTER, 1 + _RETRY_JITTER)
        return retry_after if retry_after <= _MAX_RETRY_AFTER_SECONDS else None

    def _error(self, model: str, exc: Exception) -> ModelInvocationError:
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

    def response(self, response: GenerateContentResponse, model: str) -> InvokerResponse:
        """The non-empty text and the token counts of a Gemini response."""
        text = response.text
        if not text:
            raise ModelInvocationError(
                f"Cloud model '{model}' returned an empty response{_empty_reason(response)}."
            )
        usage = getattr(response, "usage_metadata", None)
        return InvokerResponse(
            text,
            prompt_tokens=_count(getattr(usage, "prompt_token_count", None)),
            completion_tokens=_count(getattr(usage, "candidates_token_count", None)),
            retries=self.retries,
        )


def _empty_reason(response: GenerateContentResponse) -> str:
    """Say why a Gemini response has no text: a blocked prompt or the finish reason."""
    block_reason = getattr(response.prompt_feedback, "block_reason", None)
    if block_reason:
        return f" (prompt blocked: {getattr(block_reason, 'value', block_reason)})"
    candidates = response.candidates or []
    finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    if finish_reason:
        return f" (finish reason: {getattr(finish_reason, 'value', finish_reason)})"
    return ""


def _invoke_cloud(
    prompt: str,
    model: str,
    timeout_seconds: float | None = None,
    *,
    settings: Settings | None = None,
) -> InvokerResponse:
    """Send ``prompt`` to Gemini, with the schema Ollama gets, at temperature 0.

    Authenticates with ``gemini_api_key``, else Vertex AI ADC, from ``settings`` or the env.

    Raises:
        BackendConfigurationError: If the ``[cloud]`` extra or a setting is wrong.
        CredentialsNotConfiguredError: If no usable credentials are set.
        ModelTimeoutError: If the request times out.
        ModelInvocationError: If the request fails or the response is empty.
    """
    call = _CloudCall(settings, timeout_seconds)
    logger.debug("Calling cloud model '%s' via %s.", model, call.credentials.describe())
    while True:
        try:
            with call.client() as client:
                response = client.models.generate_content(
                    model=model, contents=prompt, config=call.config
                )
        except Exception as exc:  # noqa: BLE001 - varied SDK/transport errors; see retry_delay.
            time.sleep(call.retry_delay(model, exc))
            continue
        return call.response(response, model)


async def _ainvoke_cloud(
    prompt: str,
    model: str,
    timeout_seconds: float | None = None,
    *,
    settings: Settings | None = None,
) -> InvokerResponse:
    """:func:`_invoke_cloud` with ``client.aio``."""
    call = _CloudCall(settings, timeout_seconds)
    logger.debug("Calling cloud model '%s' via %s (async).", model, call.credentials.describe())
    while True:
        try:
            with call.client() as client:
                async with client.aio as aio:
                    response = await aio.models.generate_content(
                        model=model, contents=prompt, config=call.config
                    )
        except Exception as exc:  # noqa: BLE001 - varied SDK/transport errors; see retry_delay.
            await asyncio.sleep(call.retry_delay(model, exc))
            continue
        return call.response(response, model)


def _bind(
    backend: LLMBackend,
    settings: Settings,
    fields: int | None,
    cloud: Callable[..., _R],
    local: Callable[..., _R],
) -> Callable[[str, str, float | None], _R]:
    """Bind the settings (and, for Ollama, the reply cap) of one inspection to an invoker."""
    if backend is LLMBackend.API:
        return functools.partial(cloud, settings=settings)
    reply_tokens = ollama_reply_tokens(fields)
    return functools.partial(local, host=settings.ollama_host, reply_tokens=reply_tokens)


def builtin_invoker(
    backend: LLMBackend, settings: Settings, *, fields: int | None = None
) -> Callable[[str, str, float | None], InvokerResponse]:
    """Return the built-in sync invoker as ``(prompt, model, timeout_seconds) -> response``.

    The seam the inspection calls: ``response.text`` is the raw text and the
    rest feeds ``Usage``, so a wrapper must return the :class:`InvokerResponse`
    unchanged, and a wrapper of this factory must pass ``fields`` (the head's
    field count, which sizes Ollama's reply cap) on.
    """
    return _bind(backend, settings, fields, _invoke_cloud, _invoke_ollama)


def builtin_async_invoker(
    backend: LLMBackend, settings: Settings, *, fields: int | None = None
) -> Callable[[str, str, float | None], Awaitable[InvokerResponse]]:
    """Return the built-in async invoker; the async twin of :func:`builtin_invoker`."""
    return _bind(backend, settings, fields, _ainvoke_cloud, _ainvoke_ollama)
