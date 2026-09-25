"""Test doubles shared across the csv_inspector test suite (no network, ever)."""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from ollama import ResponseError

ChatHandler = Callable[..., Any]


def ollama_reply(
    content: str | None,
    *,
    prompt_eval_count: int | None = None,
    eval_count: int | None = None,
    load_duration: int | None = None,
) -> SimpleNamespace:
    """Build an object shaped like ``ollama``'s chat response.

    Args:
        content: The message text.
        prompt_eval_count: Prompt tokens, as Ollama reports them.
        eval_count: Completion tokens.
        load_duration: Model load time, in nanoseconds.
    """
    return SimpleNamespace(
        message=SimpleNamespace(content=content),
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
        load_duration=load_duration,
    )


class FakeOllama:
    """A stand-in ``ollama`` module whose sync and async clients call ``chat``.

    Records every client's constructor kwargs (e.g. ``timeout``) and every
    request, and can delay answers to exercise timeouts.

    Attributes:
        client_kwargs: Keyword arguments each client was constructed with.
        requests: Keyword arguments of every chat request, in call order.
        closed_clients: How many clients were closed (context exited).
    """

    def __init__(self, chat: ChatHandler, *, delay_seconds: float = 0.0) -> None:
        self._chat = chat
        self._delay = delay_seconds
        self.client_kwargs: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.closed_clients = 0
        fake = self

        class Client:
            def __init__(self, **kwargs: Any) -> None:
                fake.client_kwargs.append(kwargs)

            def __enter__(self) -> Client:
                return self

            def __exit__(self, *exc_info: object) -> None:
                fake.closed_clients += 1

            def chat(self, **kwargs: Any) -> Any:
                fake.requests.append(kwargs)
                if fake._delay:
                    time.sleep(fake._delay)
                return fake._chat(**kwargs)

        class AsyncClient:
            def __init__(self, **kwargs: Any) -> None:
                fake.client_kwargs.append(kwargs)

            async def __aenter__(self) -> AsyncClient:
                return self

            async def __aexit__(self, *exc_info: object) -> None:
                fake.closed_clients += 1

            async def chat(self, **kwargs: Any) -> Any:
                fake.requests.append(kwargs)
                if fake._delay:
                    await asyncio.sleep(fake._delay)
                return fake._chat(**kwargs)

        self.module = SimpleNamespace(
            Client=Client, AsyncClient=AsyncClient, ResponseError=ResponseError
        )


def install_fake_ollama(
    monkeypatch: pytest.MonkeyPatch, chat: ChatHandler, *, delay_seconds: float = 0.0
) -> FakeOllama:
    """Register a :class:`FakeOllama` as the ``ollama`` module for one test."""
    fake = FakeOllama(chat, delay_seconds=delay_seconds)
    monkeypatch.setitem(sys.modules, "ollama", fake.module)
    return fake
