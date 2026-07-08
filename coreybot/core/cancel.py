"""Cooperative cancellation, modeled loosely on Go\'s ``context.Context``.

A :class:`CancelToken` is a shared flag that any layer (HTTP, provider, agent)
can observe. Cancelling it does two things:

1. Sets a plain, thread-safe flag (``is_cancelled``) that blocking code running
   in worker threads can poll between chunks.
2. Sets an ``asyncio.Event`` so async code can *await* cancellation and race it
   against real work with :func:`run_cancellable`.

Why not just use ``asyncio.Task.cancel()``? We do use task cancellation at the
edges (the TUI/CLI cancel the running turn task). But our actual network I/O is
blocking ``urllib`` code executed in threads via ``asyncio.to_thread`` -- and a
thread cannot be force-killed. The token lets that blocking code *cooperate*:
it checks ``is_cancelled`` between streamed chunks and stops promptly. This is
the same cooperative model Go uses (a goroutine must select on ``ctx.Done()``).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Awaitable, Optional, TypeVar


class CancelledError(Exception):
    """Raised when work is abandoned because its token was cancelled.

    We deliberately subclass the builtin ``Exception`` (not
    ``asyncio.CancelledError``) so ordinary ``except Exception`` handlers in the
    loop can surface it as a normal, non-crashing "turn was interrupted" event.
    """


T = TypeVar("T")


class CancelToken:
    """A shared, thread-safe cancellation flag with an async waiter."""

    def __init__(self) -> None:
        self._flag = threading.Event()
        # The asyncio.Event is created lazily on first async use so a token can
        # be constructed outside a running loop (e.g. in a plain constructor).
        self._async_event: Optional[asyncio.Event] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # --- observing -----------------------------------------------------
    @property
    def is_cancelled(self) -> bool:
        """Thread-safe: pollable from any thread (blocking code uses this)."""
        return self._flag.is_set()

    def raise_if_cancelled(self) -> None:
        """Convenience guard: raise :class:`CancelledError` if cancelled."""
        if self._flag.is_set():
            raise CancelledError("operation cancelled")

    def _event(self) -> asyncio.Event:
        """Return (creating if needed) the asyncio.Event bound to this loop."""
        if self._async_event is None:
            self._async_event = asyncio.Event()
            self._loop = asyncio.get_running_loop()
            if self._flag.is_set():
                self._async_event.set()
        return self._async_event

    async def wait(self) -> None:
        """Await until this token is cancelled (for racing against work)."""
        await self._event().wait()

    # --- triggering ----------------------------------------------------
    def cancel(self) -> None:
        """Cancel from any thread; wakes both pollers and async waiters."""
        self._flag.set()
        event = self._async_event
        loop = self._loop
        if event is not None and loop is not None and not event.is_set():
            # The event must be set on its owning loop thread.
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                # Loop already closed; the thread-safe flag still did its job.
                pass


async def run_cancellable(coro: Awaitable[T], token: Optional[CancelToken]) -> T:
    """Await ``coro`` but abandon it if ``token`` is cancelled first.

    Returns the coroutine\'s result, or raises :class:`CancelledError` if the
    token fired first. The losing task is cancelled so we do not leak it.
    """
    if token is None:
        return await coro  # type: ignore[return-value]

    work = asyncio.ensure_future(coro)
    waiter = asyncio.ensure_future(token.wait())
    try:
        done, _pending = await asyncio.wait(
            {work, waiter}, return_when=asyncio.FIRST_COMPLETED
        )
        if work in done:
            return work.result()
        raise CancelledError("operation cancelled")
    finally:
        # Cancel whichever task is still pending and drain it so asyncio does
        # not warn about a never-retrieved task/exception.
        for task in (work, waiter):
            if not task.done():
                task.cancel()
                try:
                    await task
                except BaseException:
                    pass
