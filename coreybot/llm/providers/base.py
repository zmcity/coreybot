"""Provider abstraction + a tiny registry that makes the layer extensible.

Design:
- ``LLMProvider`` is the contract every protocol adapter implements: given a
  list of unified ``Message`` objects, return a unified ``CompletionResult``.
- A registry maps a short name (e.g. ``"openai"``) to a provider class. To add
  a new protocol you write a subclass and decorate it with ``@register("name")``
  -- no other code needs to change. This is the Open/Closed Principle in action.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Callable, Dict, Iterator, List, Optional, Type

from coreybot.core.cancel import CancelToken
from coreybot.core.config import Config
from coreybot.core.message import CompletionResult, Message


class LLMProvider(ABC):
    """Common interface for all provider protocol adapters.

    The async methods (``acomplete`` / ``astream``) are the primary interface:
    they accept an optional :class:`CancelToken` so an in-flight request can be
    interrupted (like passing a ``context.Context`` in Go). The sync
    ``complete`` / ``stream`` methods remain as thin conveniences for scripts
    and tests that are not running an event loop.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    # --- async interface (primary) -------------------------------------
    @abstractmethod
    async def acomplete(
        self, messages: List[Message], cancel_token: Optional[CancelToken] = None
    ) -> CompletionResult:
        """Send ``messages`` and return a normalized result (cancellable)."""
        raise NotImplementedError

    async def astream(
        self, messages: List[Message], cancel_token: Optional[CancelToken] = None
    ) -> AsyncIterator[str]:
        """Yield text chunks as they arrive (cancellable).

        Default implementation awaits a single ``acomplete`` and yields the
        whole answer at once. Providers with real SSE override this.
        """
        result = await self.acomplete(messages, cancel_token)
        yield result.text

    # --- sync convenience ----------------------------------------------
    def complete(self, messages: List[Message]) -> CompletionResult:
        """Blocking convenience wrapper around :meth:`acomplete`."""
        import asyncio

        return asyncio.run(self.acomplete(messages))

    def stream(self, messages: List[Message]) -> Iterator[str]:
        """Blocking convenience wrapper around :meth:`astream`."""
        import asyncio

        async def _collect() -> List[str]:
            return [chunk async for chunk in self.astream(messages)]

        for chunk in asyncio.run(_collect()):
            yield chunk


# Internal registry: name -> provider class.
_REGISTRY: Dict[str, Type[LLMProvider]] = {}


def register(name: str) -> Callable[[Type[LLMProvider]], Type[LLMProvider]]:
    """Class decorator that registers a provider under ``name``."""

    def decorator(cls: Type[LLMProvider]) -> Type[LLMProvider]:
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"Provider '{key}' is already registered")
        _REGISTRY[key] = cls
        return cls

    return decorator


def create_provider(config: Config) -> LLMProvider:
    """Instantiate the provider selected by ``config.provider``."""

    key = config.provider.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown provider '{config.provider}'. Available: {available}"
        )
    return _REGISTRY[key](config)


def available_providers() -> List[str]:
    return sorted(_REGISTRY)
