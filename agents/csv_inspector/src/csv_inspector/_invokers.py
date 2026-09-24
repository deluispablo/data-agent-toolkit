"""Model invokers (sync and async) for the local and cloud backends.

Every invoker builds its own client per call and closes it afterwards, so
there is no shared, mutable client state and calls are thread-safe. SDKs are
imported lazily: the local backend needs only ``ollama``, the cloud backend
only ``google-genai`` (``[cloud]`` extra).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Awaitable, Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol

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
        "options": {"temperature": 0.0, "num_ctx": _ollama_num_ctx(prompt)},
    }


def _ollama_error(model: str, exc: Exception) -> ModelInvocationError:
    """Map an Ollama client failure to a domain error."""
    if _is_timeout(exc):
        return ModelTimeoutError(f"Model '{model}' timed out: {exc}")
    return ModelInvocationError(f"Model '{model}' failed to respond: {exc}")


def _ollama_content(response: _OllamaChatResponse, model: str) -> str:
    """Extract the non-empty message content from an Ollama chat response."""
    content = response.message.content
    if not content:
        raise ModelInvocationError(f"Model '{model}' returned an empty response.")
    return content


def invoke_ollama_model(prompt: str, model: str, *, timeout_seconds: float | None = None) -> str:
    """Send a prompt to a local Ollama model and return its raw text response.

    Args:
        prompt: The fully-built prompt to send.
        model: Name of the Ollama model to invoke (e.g. ``"qwen2.5-coder:7b"``).
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
    ollama = _import_ollama()
    try:
        with ollama.Client(timeout=timeout_seconds) as client:
            response = client.chat(**_ollama_request(prompt, model))
    except Exception as exc:
        raise _ollama_error(model, exc) from exc
    return _ollama_content(response, model)


async def ainvoke_ollama_model(
    prompt: str, model: str, *, timeout_seconds: float | None = None
) -> str:
    """Async variant of :func:`invoke_ollama_model`, using ``ollama.AsyncClient``.

    Args:
        prompt: The fully-built prompt to send.
        model: Name of the Ollama model to invoke.
        timeout_seconds: Client-side timeout for the request, or ``None``.

    Returns:
        The raw text content of the model's response.

    Raises:
        BackendConfigurationError: If the ``ollama`` package is not installed.
        ModelTimeoutError: If the request times out.
        ModelInvocationError: If the request fails or the response is empty.
    """
    ollama = _import_ollama()
    try:
        async with ollama.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.chat(**_ollama_request(prompt, model))
    except Exception as exc:
        raise _ollama_error(model, exc) from exc
    return _ollama_content(response, model)


# ---------------------------------------------------------------------
# Cloud backend (Google Gemini via google-genai)
# ---------------------------------------------------------------------


def _to_milliseconds(timeout_seconds: float) -> int:
    """Convert seconds to the whole milliseconds ``google-genai`` expects (at least 1)."""
    return max(1, math.ceil(timeout_seconds * 1000))


class _CloudCall:
    """Everything needed for one Gemini request, prepared before any network use."""

    def __init__(self, settings: Settings | None, timeout_seconds: float | None) -> None:
        self.credentials: CloudCredentials = resolve_settings(
            settings, LLMBackend.API
        ).cloud_credentials()
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
        self._auth_errors = google_auth_exceptions
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
            response_json_schema=CSVInspectionResult.model_json_schema(),
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
            # google-genai takes the request timeout in MILLISECONDS.
            self._client_kwargs["http_options"] = types.HttpOptions(
                timeout=_to_milliseconds(timeout_seconds)
            )

    def client(self) -> GenaiClient:
        """Create the ``google-genai`` client for this call."""
        return self._genai.Client(**self._client_kwargs)

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


def _cloud_text(response: _GenaiResponse, model: str) -> str:
    """Extract the non-empty text of a Gemini response."""
    text = response.text
    if not text:
        raise ModelInvocationError(f"Cloud model '{model}' returned an empty response.")
    return text


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

    Note:
        Unit-tested against a mocked client only; not yet verified against
        the real service (see issue #5).

    Args:
        prompt: The fully-built prompt to send.
        model: Name of the Gemini model to invoke (e.g. ``"gemini-2.5-flash"``).
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
    call = _CloudCall(settings, timeout_seconds)
    logger.debug("Calling cloud model '%s' via %s.", model, call.credentials.describe())
    try:
        with call.client() as client:
            response = client.models.generate_content(
                model=model, contents=prompt, config=call.config
            )
    except Exception as exc:  # noqa: BLE001 - varied SDK/transport errors; see below.
        # Re-raised "from None" on purpose: the original exception (and its
        # traceback) could carry the API key, so only a redacted message is kept.
        raise call.error(model, exc) from None
    return _cloud_text(response, model)


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
    call = _CloudCall(settings, timeout_seconds)
    logger.debug("Calling cloud model '%s' via %s (async).", model, call.credentials.describe())
    try:
        with call.client() as client:
            async with client.aio as aio:
                response = await aio.models.generate_content(
                    model=model, contents=prompt, config=call.config
                )
    except Exception as exc:  # noqa: BLE001 - varied SDK/transport errors; see below.
        # Re-raised "from None" on purpose; see invoke_cloud_model.
        raise call.error(model, exc) from None
    return _cloud_text(response, model)


# ---------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------


def builtin_invoker(
    backend: LLMBackend, settings: Settings
) -> Callable[[str, str, float | None], str]:
    """Return the built-in sync invoker as ``(prompt, model, timeout_seconds) -> text``."""
    if backend is LLMBackend.API:
        return lambda prompt, model, timeout: invoke_cloud_model(
            prompt, model, settings=settings, timeout_seconds=timeout
        )
    return lambda prompt, model, timeout: invoke_ollama_model(
        prompt, model, timeout_seconds=timeout
    )


def builtin_async_invoker(
    backend: LLMBackend, settings: Settings
) -> Callable[[str, str, float | None], Awaitable[str]]:
    """Return the built-in async invoker as ``(prompt, model, timeout_seconds) -> text``."""
    if backend is LLMBackend.API:
        return lambda prompt, model, timeout: ainvoke_cloud_model(
            prompt, model, settings=settings, timeout_seconds=timeout
        )
    return lambda prompt, model, timeout: ainvoke_ollama_model(
        prompt, model, timeout_seconds=timeout
    )
