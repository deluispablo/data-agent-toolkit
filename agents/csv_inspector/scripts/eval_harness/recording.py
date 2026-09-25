"""The one recording seam: every built-in model call made inside a block, as it answered.

``inspect_csv`` looks up ``csv_inspector._inspect.builtin_invoker`` when no
``model_invoker`` is given. Wrapping that factory keeps the token counts in
``Usage`` and the errors that count as a failed attempt exactly as in a
normal run, which the public ``model_invoker`` (plain text in, plain text
out) would not. Used by ``--keep-raw`` and by ``capture_walkthrough.py``.
Sequential use only: it swaps a module attribute for the block.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from csv_inspector import LLMBackend, Settings, _inspect
from csv_inspector._invokers import InvokerResponse

OnCall = Callable[[str, str, InvokerResponse], None]
"""Called with ``(model, prompt, response)`` after each built-in call answers."""

_SyncCall = Callable[[str, str, float | None], InvokerResponse]
# The factory inspect_csv looks up in its module when no model_invoker is given.
_SEAM = "builtin_invoker"


@contextmanager
def recording_invoker(on_call: OnCall) -> Iterator[None]:
    """Hand every built-in model call made inside the block to ``on_call``, unchanged."""
    original: Callable[..., _SyncCall] = getattr(_inspect, _SEAM)

    def factory(backend: LLMBackend, settings: Settings, *, fields: int | None = None) -> _SyncCall:
        call = original(backend, settings, fields=fields)

        def recording(prompt: str, model: str, timeout: float | None) -> InvokerResponse:
            response = call(prompt, model, timeout)
            on_call(model, prompt, response)
            return response

        return recording

    setattr(_inspect, _SEAM, factory)
    try:
        yield
    finally:
        setattr(_inspect, _SEAM, original)
