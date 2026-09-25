"""Inspection orchestration: sampling, prompting, model calls and grounding.

:func:`inspect_csv` and its asyncio counterpart :func:`ainspect_csv` share
the same steps and guarantees:

* configuration is checked **before** the source is read, so a non-seekable
  stream is never consumed only to fail on a missing setting;
* ``timeout_seconds`` is one budget for the model phase, shared by the
  primary (about 70 % of it) and fallback models and enforced by the library,
  so custom invokers are bounded too;
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
from ._grounding import ground_in_samples, head_field_count
from ._invokers import (
    AsyncModelInvoker,
    InvokerResponse,
    ModelInvoker,
    _redact,
    builtin_async_invoker,
    builtin_invoker,
)
from ._models import CSVInspectionResult, Usage, _ModelAnswer
from ._prompt import PROMPT_VERSION, build_prompt, parse_and_validate
from ._sampling import DEFAULT_SAMPLE_BYTES, DEFAULT_TAIL_BYTES, CSVSource, Samples, sample_source

logger = logging.getLogger(__name__)

PRIMARY_SHARE = 0.7
"""Fraction of the remaining budget a model may use when another one follows it."""

_SyncCall = Callable[[str, str, float | None], InvokerResponse]
_AsyncCall = Callable[[str, str, float | None], Awaitable[InvokerResponse]]
_C = TypeVar("_C")

# Built-in invokers report every failure as one of these. A custom invoker may
# raise anything (RuntimeError, raw httpx errors...): any exception from it is
# a failed attempt too, never escaping unwrapped (BackendConfigurationError aside).
_RETRYABLE = (ModelInvocationError, ResponseParsingError, SchemaValidationError)
_USAGE_LOG = (
    "Usage: model=%s prompt_tokens=%s completion_tokens=%s latency=%.2fs "
    "attempts=%d retries=%d prompt_version=%s lines_omitted=%d"
)
_SKIPPED_LOG = (
    "Skipping model '%s': the %.2fs time budget ran out (it would have had "
    "%.0f%% of what the models before it left)."
)


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
        """:data:`PRIMARY_SHARE` of what is left, or all of it for the last model.

        All of it would let a hung primary starve the fallback; half made a cold 7B lose.
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


def _timed_out(model: str, remaining: float) -> ModelTimeoutError:
    return ModelTimeoutError(f"Model '{model}' did not answer within {max(remaining, 0):.2f}s.")


def _call_with_deadline(
    call: _SyncCall, prompt: str, model: str, remaining: float | None
) -> InvokerResponse:
    """Run one sync model call, returning no later than ``remaining`` seconds.

    With a budget the call runs in a daemon thread, abandoned (``ModelTimeoutError``)
    when the budget runs out: Python cannot interrupt a blocking call, and a call
    that never returns must not keep the interpreter from exiting.
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
    """Await one async model call, cancelling it (``ModelTimeoutError``) after ``remaining`` s."""
    try:
        return await asyncio.wait_for(call(prompt, model, remaining), remaining)
    except asyncio.TimeoutError:
        raise _timed_out(model, remaining or 0) from None


@dataclass
class _UsageTally:
    """The usage of every attempt of one inspection; answers count even if invalid."""

    attempts: int = 0
    started: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    retries: int = 0
    load_seconds: float | None = None

    def start(self) -> None:
        """Count one more attempt; the first one starts the clock."""
        if not self.attempts:
            self.started = time.monotonic()
        self.attempts += 1

    def add(self, response: InvokerResponse) -> str:
        """Add one answer's counters (a raising attempt reports none); return its text."""
        for name in ("prompt_tokens", "completion_tokens", "load_seconds"):
            value = getattr(response, name)  # None when not reported: the total stays
            if value is not None:
                setattr(self, name, value + (getattr(self, name) or 0))
        self.retries += response.retries
        return response.text

    def finish(self, model: str) -> Usage:
        """The :class:`Usage` of the inspection, answered by ``model``."""
        counters = {key: value for key, value in vars(self).items() if key != "started"}
        latency = time.monotonic() - self.started
        return Usage(
            model=model, latency_seconds=latency, prompt_version=PROMPT_VERSION, **counters
        )


class _Run:
    """One inspection: models and settings resolved up front, then the model phase.

    :meth:`run` and :meth:`arun` differ only in one ``await``.
    """

    def __init__(
        self,
        backend: LLMBackend,
        settings: Settings | None,
        model: str | None,
        fallback_model: str | None,
        timeout_seconds: float | None,
        *,
        custom_invoker: bool,
    ) -> None:
        """Validate and resolve everything, before the source is read.

        The environment is read only for a default model name or a built-in
        invoker; the fallback is tried only when it differs from the primary.

        Raises:
            ValueError: If ``timeout_seconds`` is not positive.
            BackendConfigurationError: If the backend is unusable as configured.
        """
        if timeout_seconds is not None and not timeout_seconds > 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}.")
        if model is None or fallback_model is None or not custom_invoker:
            settings = resolve_settings(settings)
            if not custom_invoker:
                ensure_backend_ready(backend, settings)
            model = model if model is not None else settings.model_for(backend)
            if fallback_model is None:
                fallback_model = settings.fallback_model_for(backend)
        # Both are set now: injected, or resolved above.
        assert model is not None
        assert fallback_model is not None
        self.settings = settings if settings is not None else Settings()
        self._candidates = (model,) if model == fallback_model else (model, fallback_model)
        self._backend = backend
        self._timeout_seconds = timeout_seconds
        self._retryable = (Exception,) if custom_invoker else _RETRYABLE
        self._errors: dict[str, Exception] = {}
        self._tally = _UsageTally()

    def run(self, samples: Samples, custom: ModelInvoker | None) -> CSVInspectionResult:
        """The model phase with ``custom`` or the built-in sync invoker."""
        call: _SyncCall
        if custom is None:
            call = self._builtin(samples, builtin_invoker)
        else:
            call = lambda prompt, model, _timeout: InvokerResponse(custom(prompt, model))  # noqa: E731
        for candidate, budget in self._attempts(samples):
            try:
                response = _call_with_deadline(call, self._prompt, candidate, budget)
                answer = parse_and_validate(self._tally.add(response), candidate)
            except BackendConfigurationError:
                raise
            except self._retryable as exc:
                self._failed(candidate, exc)
                continue
            return self._succeeded(candidate, answer)
        raise self._failure()

    async def arun(self, samples: Samples, custom: AsyncModelInvoker | None) -> CSVInspectionResult:
        """The model phase with ``custom`` or the built-in async invoker."""
        call: _AsyncCall
        if custom is None:
            call = self._builtin(samples, builtin_async_invoker)
        else:

            async def call(prompt: str, model: str, _timeout: float | None) -> InvokerResponse:
                return InvokerResponse(await custom(prompt, model))

        for candidate, budget in self._attempts(samples):
            try:
                response = await _acall_with_deadline(call, self._prompt, candidate, budget)
                answer = parse_and_validate(self._tally.add(response), candidate)
            except BackendConfigurationError:
                raise
            except self._retryable as exc:
                self._failed(candidate, exc)
                continue
            return self._succeeded(candidate, answer)
        raise self._failure()

    def _builtin(self, samples: Samples, factory: Callable[..., _C]) -> _C:
        """The built-in invoker, its reply sized from the head's field count."""
        return factory(self._backend, self.settings, fields=head_field_count(samples.head_text))

    def _attempts(self, samples: Samples) -> Iterator[tuple[str, float | None]]:
        """Build the prompt, then yield each model to try with its share of the budget."""
        self._samples = samples
        self._prompt = build_prompt(
            samples.head_text,
            samples.encoding,
            tail_sample=samples.tail_text,
            covers_whole_file=samples.covers_whole_file,
        )
        deadline = self._deadline = _Deadline(self._timeout_seconds)
        for index, model in enumerate(self._candidates):
            if deadline.expired:
                raise self._timeout(index)
            logger.info("Inspecting '%s' with model '%s'.", samples.description, model)
            self._tally.start()
            yield model, deadline.share(len(self._candidates) - index)

    def _failed(self, model: str, exc: Exception) -> None:
        """Record a failed attempt, or raise ``InspectionTimeoutError`` when the budget is out."""
        # A custom invoker's message may carry the configured key: redacted too.
        secret = self.settings.gemini_api_key
        logger.warning("Model '%s' failed: %s", model, _redact(str(exc), secret))
        self._errors[model] = exc
        # The last model's share is all that is left: its timeout means the budget ran
        # out, even when a timed wait returned a hair early (seen on Windows).
        last = model == self._candidates[-1]
        timed_out = self._timeout_seconds is not None and isinstance(exc, ModelTimeoutError)
        if self._deadline.expired or (last and timed_out):
            raise self._timeout(self._candidates.index(model) + 1) from exc

    def _succeeded(self, model: str, answer: _ModelAnswer) -> CSVInspectionResult:
        """Ground a validated answer in the samples: the result, with its usage attached."""
        usage, samples = self._tally.finish(model), self._samples
        msg = "Inspection of '%s' succeeded with model '%s' (confidence=%.2f)."
        logger.info(msg, samples.description, model, answer.confidence)
        counters = usage.model_dump(exclude={"load_seconds"}).values()
        logger.info(_USAGE_LOG, *counters, samples.lines_omitted)
        grounded = ground_in_samples(
            answer,
            samples.head_text,
            samples.tail_text,
            covers_whole_file=samples.covers_whole_file,
            detected_encoding=samples.encoding,
        )
        # The model is frozen: attach the usage to a copy.
        return grounded.model_copy(update={"usage": usage})

    def _timeout(self, skipped: int) -> InspectionTimeoutError:
        """The budget ran out: log the models from index ``skipped`` on (to tune it), and say so."""
        for model in self._candidates[skipped:]:
            share = 1.0 if model == self._candidates[-1] else PRIMARY_SHARE
            logger.info(_SKIPPED_LOG, model, self._timeout_seconds, share * 100)
        return InspectionTimeoutError(
            f"Inspection of '{self._samples.description}' ran out of its "
            f"{self._timeout_seconds}s budget. Tried: {list(self._errors)}.",
            attempts=self._errors,
        )

    def _failure(self) -> InspectionFailedError:
        return InspectionFailedError(
            f"No configured model produced a valid inspection result for "
            f"'{self._samples.description}'. Tried: {list(self._errors)}.",
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

    Samples a head of ``n_bytes`` and, when the source is larger, a tail of up
    to ``tail_bytes``, never loading it in full; asks an LLM (local Ollama, or
    Gemini with ``backend=LLMBackend.API``); grounds its answer in the samples.
    Blocking: in asyncio, use :func:`ainspect_csv` or ``asyncio.to_thread``.

    Args:
        source: A path, in-memory bytes, or a binary file-like object (see
            :data:`CSVSource`), read from its current position (restored if seekable).
        backend: Local Ollama (default, no credentials) or the Gemini API
            (opt-in; needs the ``[cloud]`` extra).
        settings: Explicit :class:`Settings`: the environment is never read.
            ``None`` reads environment variables (never ``.env``; see :func:`load_settings`).
        model: Primary model name; defaults to the backend's configured model.
        fallback_model: Model to try if ``model`` fails; defaults to the
            backend's configured fallback. Skipped when equal to ``model``.
        n_bytes: Head sample size, in bytes. Must be at least 1.
        tail_bytes: Maximum tail sample size, in bytes, never overlapping the
            head. ``0`` disables tail sampling.
        timeout_seconds: One budget for the model phase, enforced even for
            custom invokers: with a fallback, the primary may use about 70 %
            (:data:`PRIMARY_SHARE`) and the fallback the rest. ``None``: no limit.
        model_invoker: A custom ``(prompt, model) -> text`` callable, taking
            precedence over ``backend`` (which then only selects default model
            names). Any exception but :class:`BackendConfigurationError` is a
            failed attempt for that model.

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
    custom = model_invoker is not None
    run = _Run(backend, settings, model, fallback_model, timeout_seconds, custom_invoker=custom)
    return run.run(sample_source(source, n_bytes, tail_bytes), model_invoker)


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

    Sampling runs in a worker thread; model calls use the SDKs' async clients,
    and ``asyncio.wait_for`` cancels a call when the budget runs out.

    The arguments, result and exceptions are those of :func:`inspect_csv`,
    except that ``model_invoker`` is an async ``(prompt, model) -> text``
    callable.
    """
    custom = model_invoker is not None
    run = _Run(backend, settings, model, fallback_model, timeout_seconds, custom_invoker=custom)
    samples = await asyncio.to_thread(sample_source, source, n_bytes, tail_bytes)
    return await run.arun(samples, model_invoker)
