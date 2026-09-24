"""Inspection orchestration: sampling, prompting, model calls and grounding.

:func:`inspect_csv` is the synchronous entry point; :func:`ainspect_csv` is
its asyncio counterpart. Both share the same steps and guarantees:

* configuration is resolved and checked **before** the source is read, so a
  non-seekable stream is never consumed only to fail on a missing setting;
* ``timeout_seconds`` is one overall budget for the model phase, shared by
  the primary and fallback models (the primary may use about 70 % of it,
  so a slowly loading primary rarely loses its answer while a hung one
  still leaves the fallback time), and enforced by the library itself, so
  custom invokers are bounded too;
* no state is shared between calls: every call builds its own clients.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import TypeVar

from ._backends import LLMBackend
from ._config import Settings, ensure_backend_ready, resolve_settings
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
    InvokerResponse,
    ModelInvoker,
    _redact,
    builtin_async_invoker,
    builtin_invoker,
)
from ._models import CSVInspectionResult, Usage
from ._prompt import PROMPT_VERSION, build_prompt, parse_and_validate
from ._sampling import (
    DEFAULT_SAMPLE_BYTES,
    DEFAULT_TAIL_BYTES,
    CSVSource,
    Samples,
    sample_source,
)

logger = logging.getLogger(__name__)

PRIMARY_SHARE = 0.7
"""Fraction of the remaining budget a model may use when another one follows it."""

_SyncCall = Callable[[str, str, float | None], InvokerResponse]
_AsyncCall = Callable[[str, str, float | None], Awaitable[InvokerResponse]]
_N = TypeVar("_N", int, float)


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

    def share(self, models_left: int) -> float | None:
        """This model's slice of the budget: most of what is left, or all of it.

        Giving the current model the whole remainder would let a hung primary
        spend it all, so the fallback, needed exactly then, would never run.
        An equal split has the opposite flaw: a cold 7B load on CPU often
        needs more than half, so the weaker fallback answered on every cold
        start. A model followed by another one gets :data:`PRIMARY_SHARE` of
        what is left; the last one gets everything. Time a model does not
        use carries over.
        """
        remaining = self.remaining()
        if remaining is None:
            return None
        return remaining * PRIMARY_SHARE if models_left > 1 else remaining

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

    resolved = resolve_settings(settings)
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


def _timed_out(model: str, remaining: float) -> ModelTimeoutError:
    return ModelTimeoutError(f"Model '{model}' did not answer within {max(remaining, 0):.2f}s.")


def _call_with_deadline(
    call: _SyncCall, prompt: str, model: str, remaining: float | None
) -> InvokerResponse:
    """Run one sync model call, returning no later than ``remaining`` seconds.

    With a budget, the call runs in a daemon worker thread and is abandoned
    when the budget runs out: Python cannot interrupt a blocking call, but
    the caller gets control back on time (built-in invokers also stop on
    their own, as the same timeout is passed to their HTTP client). The
    thread is a daemon so that a call that never returns cannot keep the
    interpreter from exiting, as a ``ThreadPoolExecutor`` worker would.

    Raises:
        ModelTimeoutError: If the budget runs out first.
    """
    if remaining is None:
        return call(prompt, model, None)
    future: concurrent.futures.Future[InvokerResponse] = concurrent.futures.Future()

    def run() -> None:
        try:
            future.set_result(call(prompt, model, remaining))
        except BaseException as exc:  # noqa: BLE001 - handed to the caller via the future
            future.set_exception(exc)

    threading.Thread(target=run, name="csv_inspector", daemon=True).start()
    try:
        return future.result(timeout=remaining)
    except concurrent.futures.TimeoutError:
        raise _timed_out(model, remaining) from None


async def _acall_with_deadline(
    call: _AsyncCall, prompt: str, model: str, remaining: float | None
) -> InvokerResponse:
    """Await one async model call, cancelling it after ``remaining`` seconds.

    Raises:
        ModelTimeoutError: If the budget runs out first.
    """
    try:
        return await asyncio.wait_for(call(prompt, model, remaining), remaining)
    except asyncio.TimeoutError:
        raise _timed_out(model, remaining or 0) from None


def _add(total: _N | None, value: _N | None) -> _N | None:
    """Sum two optional counters; ``None`` only when neither was reported."""
    if value is None:
        return total
    return value if total is None else total + value


_RETRYABLE: tuple[type[Exception], ...] = (
    ModelInvocationError,
    ResponseParsingError,
    SchemaValidationError,
)


def _retryable(custom_invoker: bool) -> tuple[type[Exception], ...]:
    """The errors that count as a failed attempt, moving on to the next model.

    Built-in invokers report every failure as a :class:`ModelInvocationError`.
    A custom invoker may raise anything (``RuntimeError``, raw ``httpx``
    errors...), so any exception from it is a failed attempt too, and ends
    up in :class:`InspectionFailedError` rather than escaping unwrapped.
    :class:`BackendConfigurationError` is re-raised before this applies.
    """
    return (Exception,) if custom_invoker else _RETRYABLE


class _Run:
    """The model phase of one inspection, shared by the sync and async entry points.

    Owns everything but the I/O mechanics of a model call: the prompt, the
    order of the candidate models, each one's share of the time budget, the
    errors that count as a failed attempt, and the final error when every
    model fails.
    """

    def __init__(
        self,
        plan: _Plan,
        samples: Samples,
        timeout_seconds: float | None,
        *,
        custom_invoker: bool,
    ) -> None:
        self.prompt = build_prompt(
            samples.head_text,
            samples.encoding,
            tail_sample=samples.tail_text,
            covers_whole_file=samples.covers_whole_file,
        )
        self.retryable = _retryable(custom_invoker)
        self._candidates = plan.candidates
        self._samples = samples
        self._timeout_seconds = timeout_seconds
        self._deadline = _Deadline(timeout_seconds)
        self._errors: dict[str, Exception] = {}
        # Usage, accumulated over every attempt that returned an answer.
        self._started = 0.0
        self._attempts = 0
        self._prompt_tokens: int | None = None
        self._completion_tokens: int | None = None
        self._retries = 0
        self._load_seconds: float | None = None
        # The built-in invokers redact their own errors; a custom invoker's
        # message may still carry the configured key, so the log line is redacted too.
        self._secret = plan.settings.gemini_api_key

    def attempts(self) -> Iterator[tuple[str, float | None]]:
        """Yield each model to try, in order, with its share of the time budget.

        Raises:
            InspectionTimeoutError: If the budget runs out before a model starts.
        """
        for index, model in enumerate(self._candidates):
            if self._deadline.expired:
                self._log_skipped(self._candidates[index:])
                raise self._timeout_error()
            logger.info("Inspecting '%s' with model '%s'.", self._samples.description, model)
            if not self._attempts:
                self._started = time.monotonic()
            self._attempts += 1
            yield model, self._deadline.share(len(self._candidates) - index)

    def responded(self, response: InvokerResponse) -> str:
        """Add one answer's usage to the inspection's, and return its raw text.

        Called before the answer is parsed, so an answer that then fails
        validation still counts. An attempt that raises (timeout, empty
        reply, transport error) reports nothing.
        """
        self._prompt_tokens = _add(self._prompt_tokens, response.prompt_tokens)
        self._completion_tokens = _add(self._completion_tokens, response.completion_tokens)
        self._load_seconds = _add(self._load_seconds, response.load_seconds)
        self._retries += response.retries
        return response.text

    def failed(self, model: str, exc: Exception) -> None:
        """Record a failed attempt; the next model is tried if time is left.

        Raises:
            InspectionTimeoutError: If the budget has run out.
        """
        logger.warning("Model '%s' failed: %s", model, _redact(str(exc), self._secret))
        self._errors[model] = exc
        # The last model's share is the whole remaining budget, so its timeout
        # means the budget ran out, even when a timed wait returned a hair
        # early and the clock still reads just before the deadline (seen on
        # Windows).
        last_timed_out = (
            model == self._candidates[-1]
            and self._timeout_seconds is not None
            and isinstance(exc, ModelTimeoutError)
        )
        if self._deadline.expired or last_timed_out:
            self._log_skipped(self._candidates[self._candidates.index(model) + 1 :])
            raise self._timeout_error() from exc

    def _log_skipped(self, models: tuple[str, ...]) -> None:
        """Name the models the time budget left out, to help tune ``timeout_seconds``."""
        for model in models:
            share = 1.0 if model == self._candidates[-1] else PRIMARY_SHARE
            logger.info(
                "Skipping model '%s': the %.2fs time budget ran out (it would have had "
                "%.0f%% of what the models before it left).",
                model,
                self._timeout_seconds,
                share * 100,
            )

    def succeeded(self, model: str, result: CSVInspectionResult) -> CSVInspectionResult:
        """Ground a validated answer in the samples and return it with its usage attached."""
        usage = Usage(
            model=model,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            latency_seconds=time.monotonic() - self._started,
            attempts=self._attempts,
            retries=self._retries,
            load_seconds=self._load_seconds,
            prompt_version=PROMPT_VERSION,
        )
        samples = self._samples
        logger.info(
            "Inspection of '%s' succeeded with model '%s' (confidence=%.2f).",
            samples.description,
            model,
            result.confidence,
        )
        logger.info(
            "Usage: model=%s prompt_tokens=%s completion_tokens=%s latency=%.2fs "
            "attempts=%d retries=%d prompt_version=%s",
            usage.model,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.latency_seconds,
            usage.attempts,
            usage.retries,
            usage.prompt_version,
        )
        grounded = ground_in_samples(
            result,
            samples.head_text,
            samples.tail_text,
            covers_whole_file=samples.covers_whole_file,
            detected_encoding=samples.encoding,
        )
        # The model is frozen: attach the usage to a copy.
        return grounded.model_copy(update={"usage": usage})

    def failure_error(self) -> InspectionFailedError:
        """The error for when every model has failed."""
        return InspectionFailedError(
            f"No configured model produced a valid inspection result for "
            f"'{self._samples.description}'. Tried: {list(self._errors)}.",
            attempts=self._errors,
        )

    def _timeout_error(self) -> InspectionTimeoutError:
        return InspectionTimeoutError(
            f"Inspection of '{self._samples.description}' ran out of its "
            f"{self._timeout_seconds}s budget. Tried: {list(self._errors)}.",
            attempts=self._errors,
        )


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
            invokers. With a fallback, the primary may use about 70 % of
            it (:data:`PRIMARY_SHARE`) and the fallback everything left;
            time the primary does not use carries over. ``None`` (default)
            means no limit.
        model_invoker: A custom ``(prompt, model) -> text`` callable. When
            given, it takes precedence over ``backend`` (which then only
            selects default model names). Any exception it raises, other
            than :class:`BackendConfigurationError`, counts as a failed
            attempt for that model.

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

    call: _SyncCall
    if model_invoker is not None:
        custom = model_invoker
        call = lambda prompt, model, _timeout: InvokerResponse(custom(prompt, model))  # noqa: E731
    else:
        call = builtin_invoker(backend, plan.settings)

    run = _Run(plan, samples, timeout_seconds, custom_invoker=model_invoker is not None)
    for candidate, budget in run.attempts():
        try:
            raw = run.responded(_call_with_deadline(call, run.prompt, candidate, budget))
            result = parse_and_validate(raw, candidate)
        except BackendConfigurationError:
            raise
        except run.retryable as exc:
            run.failed(candidate, exc)
            continue
        return run.succeeded(candidate, result)
    raise run.failure_error()


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
            takes precedence over ``backend``. Its exceptions are handled
            as in :func:`inspect_csv`.

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

    call: _AsyncCall
    if model_invoker is not None:
        custom = model_invoker

        async def call(prompt: str, model: str, _timeout: float | None) -> InvokerResponse:
            return InvokerResponse(await custom(prompt, model))

    else:
        call = builtin_async_invoker(backend, plan.settings)

    run = _Run(plan, samples, timeout_seconds, custom_invoker=model_invoker is not None)
    for candidate, budget in run.attempts():
        try:
            response = await _acall_with_deadline(call, run.prompt, candidate, budget)
            raw = run.responded(response)
            result = parse_and_validate(raw, candidate)
        except BackendConfigurationError:
            raise
        except run.retryable as exc:
            run.failed(candidate, exc)
            continue
        return run.succeeded(candidate, result)
    raise run.failure_error()
