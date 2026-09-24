"""Inspection orchestration: sampling, prompting, model calls and grounding.

:func:`inspect_csv` is the synchronous entry point; :func:`ainspect_csv` is
its asyncio counterpart. Both share the same steps and guarantees:

* configuration is resolved and checked **before** the source is read, so a
  non-seekable stream is never consumed only to fail on a missing setting;
* ``timeout_seconds`` is one overall budget for the model phase, shared by
  the primary and fallback models, and enforced by the library itself, so
  custom invokers are bounded too;
* no state is shared between calls: every call builds its own clients.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ._backends import LLMBackend
from ._config import Settings
from ._exceptions import (
    BackendConfigurationError,
    InspectionFailedError,
    InspectionTimeoutError,
    ModelInvocationError,
    ModelTimeoutError,
    ResponseParsingError,
    SchemaValidationError,
)
from ._grounding import ground_in_samples
from ._invokers import (
    AsyncModelInvoker,
    ModelInvoker,
    builtin_async_invoker,
    builtin_invoker,
    ensure_backend_ready,
    resolve_settings,
)
from ._models import CSVInspectionResult
from ._prompt import build_prompt, parse_and_validate
from ._sampling import (
    DEFAULT_SAMPLE_BYTES,
    DEFAULT_TAIL_BYTES,
    CSVSource,
    Samples,
    sample_source,
)

logger = logging.getLogger(__name__)

_SyncCall = Callable[[str, str, float | None], str]
_AsyncCall = Callable[[str, str, float | None], Awaitable[str]]


@dataclass(frozen=True)
class _Plan:
    """Resolved models and settings for one inspection."""

    candidates: tuple[str, ...]
    settings: Settings


class _Deadline:
    """An optional overall time budget, measured on the monotonic clock."""

    def __init__(self, timeout_seconds: float | None) -> None:
        self._expires_at = None if timeout_seconds is None else time.monotonic() + timeout_seconds

    def remaining(self) -> float | None:
        """Seconds left (possibly <= 0), or ``None`` when there is no budget."""
        if self._expires_at is None:
            return None
        return self._expires_at - time.monotonic()

    @property
    def expired(self) -> bool:
        """Whether a budget exists and has run out."""
        remaining = self.remaining()
        return remaining is not None and remaining <= 0


def _plan(
    backend: LLMBackend,
    settings: Settings | None,
    model: str | None,
    fallback_model: str | None,
    timeout_seconds: float | None,
    *,
    uses_builtin_invoker: bool,
) -> _Plan:
    """Validate arguments and resolve models/settings, before touching the source.

    Settings are only resolved when something needs them (a default model
    name or a built-in invoker), so callers that inject an invoker and model
    names never cause the environment to be read.

    Raises:
        ValueError: If ``timeout_seconds`` is not positive.
        BackendConfigurationError: If the backend is unusable as configured.
    """
    if timeout_seconds is not None and not timeout_seconds > 0:
        raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}.")

    if model is not None and fallback_model is not None and not uses_builtin_invoker:
        # Everything was injected: the environment is never consulted.
        return _Plan(
            candidates=_candidates(model, fallback_model),
            settings=settings if settings is not None else Settings(),
        )

    resolved = resolve_settings(settings, backend)
    if uses_builtin_invoker:
        ensure_backend_ready(backend, resolved)
    primary = model if model is not None else resolved.model_for(backend)
    fallback = (
        fallback_model if fallback_model is not None else resolved.fallback_model_for(backend)
    )
    return _Plan(candidates=_candidates(primary, fallback), settings=resolved)


def _candidates(primary: str, fallback: str) -> tuple[str, ...]:
    """The models to try in order; the fallback is skipped when it equals the primary."""
    return (primary,) if primary == fallback else (primary, fallback)


def _prompt_for(samples: Samples) -> str:
    return build_prompt(
        samples.head_text,
        samples.encoding,
        tail_sample=samples.tail_text,
        covers_whole_file=samples.covers_whole_file,
    )


def _timed_out(model: str, remaining: float) -> ModelTimeoutError:
    return ModelTimeoutError(f"Model '{model}' did not answer within {max(remaining, 0):.2f}s.")


def _call_with_deadline(call: _SyncCall, prompt: str, model: str, remaining: float | None) -> str:
    """Run one sync model call, returning no later than ``remaining`` seconds.

    With a budget, the call runs in a worker thread and is abandoned when the
    budget runs out: Python cannot interrupt a blocking call, but the caller
    gets control back on time (built-in invokers also stop on their own, as
    the same timeout is passed to their HTTP client).

    Raises:
        ModelTimeoutError: If the budget runs out first.
    """
    if remaining is None:
        return call(prompt, model, None)
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="csv_inspector"
    )
    try:
        future = executor.submit(call, prompt, model, remaining)
        try:
            return future.result(timeout=remaining)
        except concurrent.futures.TimeoutError:
            raise _timed_out(model, remaining) from None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


class _Attempts:
    """Collects per-model failures and turns them into the final error."""

    def __init__(self, description: str) -> None:
        self._description = description
        self.errors: dict[str, Exception] = {}

    def record(self, model: str, exc: Exception) -> None:
        logger.warning("Model '%s' failed: %s", model, exc)
        self.errors[model] = exc

    def timeout_error(self, timeout_seconds: float | None) -> InspectionTimeoutError:
        return InspectionTimeoutError(
            f"Inspection of '{self._description}' ran out of its {timeout_seconds}s budget. "
            f"Tried: {list(self.errors)}.",
            attempts=self.errors,
        )

    def failure_error(self) -> InspectionFailedError:
        return InspectionFailedError(
            f"No configured model produced a valid inspection result for "
            f"'{self._description}'. Tried: {list(self.errors)}.",
            attempts=self.errors,
        )


def _succeeded(model: str, result: CSVInspectionResult, samples: Samples) -> CSVInspectionResult:
    logger.info(
        "Inspection of '%s' succeeded with model '%s' (confidence=%.2f).",
        samples.description,
        model,
        result.confidence,
    )
    return ground_in_samples(
        result,
        samples.head_text,
        samples.tail_text,
        covers_whole_file=samples.covers_whole_file,
    )


_RETRYABLE = (ModelInvocationError, ResponseParsingError, SchemaValidationError)


def inspect_csv(
    source: CSVSource,
    /,
    *,
    backend: LLMBackend = LLMBackend.LOCAL,
    settings: Settings | None = None,
    model: str | None = None,
    fallback_model: str | None = None,
    n_bytes: int = DEFAULT_SAMPLE_BYTES,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
    timeout_seconds: float | None = None,
    model_invoker: ModelInvoker | None = None,
) -> CSVInspectionResult:
    """Infer the dialect, header/footer layout and schema of a delimited source.

    Samples only the first ``n_bytes`` (head) and, when the source is larger,
    up to ``tail_bytes`` more from its end (tail), never loading it in full.
    Asks an LLM (local Ollama by default, or Google Gemini with
    ``backend=LLMBackend.API``) to infer the encoding, delimiter, quoting,
    header row, footer lines and a preliminary column schema, then grounds
    the answer in the sampled text and returns it validated.

    Blocking: in an asyncio application use :func:`ainspect_csv`, or run
    this function with ``asyncio.to_thread``; never call it on the event loop.

    Args:
        source: A path, in-memory bytes, or a binary file-like object (see
            :data:`CSVSource`). Streams are read from their current position,
            which is restored afterwards when the stream is seekable.
        backend: The LLM backend: local Ollama (default, no credentials) or
            the Gemini API (opt-in; needs the ``[cloud]`` extra).
        settings: Explicit :class:`Settings`. When given, the environment is
            never read. When ``None``, settings come from environment
            variables (never from a ``.env`` file); see :func:`load_settings`.
        model: Primary model name; defaults to the backend's configured model.
        fallback_model: Model to try if ``model`` fails; defaults to the
            backend's configured fallback. Skipped when equal to ``model``.
        n_bytes: Head sample size, in bytes. Must be at least 1.
        tail_bytes: Maximum tail sample size, in bytes. The tail never
            overlaps the head. ``0`` disables tail sampling.
        timeout_seconds: Overall time budget for the model phase, shared by
            the primary and fallback models and enforced even for custom
            invokers. ``None`` (default) means no limit.
        model_invoker: A custom ``(prompt, model) -> text`` callable. When
            given, it takes precedence over ``backend`` (which then only
            selects default model names).

    Returns:
        A validated :class:`CSVInspectionResult`.

    Raises:
        ValueError: If a byte budget or ``timeout_seconds`` is out of range.
        TypeError: If ``source`` is not a supported type, or is a text stream.
        FileSampleReadError: If the source cannot be read.
        EmptySampleError: If the source is empty.
        BackendConfigurationError: If the backend is unusable as configured
            (e.g. missing SDK or credentials); raised before the source is
            read, and never retried with the fallback model.
        InspectionTimeoutError: If ``timeout_seconds`` runs out.
        InspectionFailedError: If every model fails to produce a valid result.
    """
    plan = _plan(
        backend,
        settings,
        model,
        fallback_model,
        timeout_seconds,
        uses_builtin_invoker=model_invoker is None,
    )
    samples = sample_source(source, n_bytes, tail_bytes)
    prompt = _prompt_for(samples)

    call: _SyncCall
    if model_invoker is not None:
        custom = model_invoker
        call = lambda prompt, model, _timeout: custom(prompt, model)  # noqa: E731
    else:
        call = builtin_invoker(backend, plan.settings)

    deadline = _Deadline(timeout_seconds)
    attempts = _Attempts(samples.description)
    for candidate in plan.candidates:
        if deadline.expired:
            raise attempts.timeout_error(timeout_seconds)
        logger.info("Inspecting '%s' with model '%s'.", samples.description, candidate)
        try:
            raw = _call_with_deadline(call, prompt, candidate, deadline.remaining())
            result = parse_and_validate(raw, candidate)
        except BackendConfigurationError:
            raise
        except _RETRYABLE as exc:
            attempts.record(candidate, exc)
            if deadline.expired:
                raise attempts.timeout_error(timeout_seconds) from exc
            continue
        return _succeeded(candidate, result, samples)
    raise attempts.failure_error()


async def ainspect_csv(
    source: CSVSource,
    /,
    *,
    backend: LLMBackend = LLMBackend.LOCAL,
    settings: Settings | None = None,
    model: str | None = None,
    fallback_model: str | None = None,
    n_bytes: int = DEFAULT_SAMPLE_BYTES,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
    timeout_seconds: float | None = None,
    model_invoker: AsyncModelInvoker | None = None,
) -> CSVInspectionResult:
    """Asyncio counterpart of :func:`inspect_csv`; safe to await on the event loop.

    Sampling runs in a worker thread (``asyncio.to_thread``), since reading a
    file or a slow stream is blocking I/O. Model calls use the SDKs' native
    async clients (``ollama.AsyncClient`` / ``google-genai``'s ``client.aio``)
    and the time budget is enforced with ``asyncio.wait_for``, which cancels
    the pending call when it runs out.

    Args:
        source: See :func:`inspect_csv`.
        backend: See :func:`inspect_csv`.
        settings: See :func:`inspect_csv`.
        model: See :func:`inspect_csv`.
        fallback_model: See :func:`inspect_csv`.
        n_bytes: See :func:`inspect_csv`.
        tail_bytes: See :func:`inspect_csv`.
        timeout_seconds: See :func:`inspect_csv`.
        model_invoker: A custom async ``(prompt, model) -> text`` callable;
            takes precedence over ``backend``.

    Returns:
        A validated :class:`CSVInspectionResult`.

    Raises:
        Same as :func:`inspect_csv`.
    """
    plan = _plan(
        backend,
        settings,
        model,
        fallback_model,
        timeout_seconds,
        uses_builtin_invoker=model_invoker is None,
    )
    samples = await asyncio.to_thread(sample_source, source, n_bytes, tail_bytes)
    prompt = _prompt_for(samples)

    call: _AsyncCall
    if model_invoker is not None:
        custom = model_invoker
        call = lambda prompt, model, _timeout: custom(prompt, model)  # noqa: E731
    else:
        call = builtin_async_invoker(backend, plan.settings)

    deadline = _Deadline(timeout_seconds)
    attempts = _Attempts(samples.description)
    for candidate in plan.candidates:
        remaining = deadline.remaining()
        if remaining is not None and remaining <= 0:
            raise attempts.timeout_error(timeout_seconds)
        logger.info("Inspecting '%s' with model '%s'.", samples.description, candidate)
        try:
            try:
                raw = await asyncio.wait_for(call(prompt, candidate, remaining), remaining)
            except asyncio.TimeoutError:
                raise _timed_out(candidate, remaining or 0) from None
            result = parse_and_validate(raw, candidate)
        except BackendConfigurationError:
            raise
        except _RETRYABLE as exc:
            attempts.record(candidate, exc)
            if deadline.expired:
                raise attempts.timeout_error(timeout_seconds) from exc
            continue
        return _succeeded(candidate, result, samples)
    raise attempts.failure_error()
