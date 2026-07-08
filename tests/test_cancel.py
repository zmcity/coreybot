"""Tests for cooperative cancellation (``coreybot.core.cancel``) and its use in the
agent loop and HTTP layer.

These are the heart of the "interruptible session" feature: a shared
:class:`CancelToken` (Go\'s ``context`` analogue) must stop in-flight async work
and any blocking work polled from a thread.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from coreybot.runtime.agent import Agent
from coreybot.core.cancel import CancelToken, CancelledError, run_cancellable
from coreybot.core.config import Config
from coreybot.core.message import CompletionResult


# --- CancelToken / run_cancellable ----------------------------------------
async def test_run_cancellable_returns_result_when_not_cancelled():
    async def work():
        await asyncio.sleep(0.01)
        return 42

    assert await run_cancellable(work(), CancelToken()) == 42


async def test_run_cancellable_raises_when_cancelled_midflight():
    token = CancelToken()

    async def canceller():
        await asyncio.sleep(0.02)
        token.cancel()

    async def slow():
        await asyncio.sleep(5)
        return "late"

    task = asyncio.ensure_future(canceller())
    t0 = time.time()
    with pytest.raises(CancelledError):
        await run_cancellable(slow(), token)
    assert time.time() - t0 < 1.0  # cancelled promptly, not after 5s
    await task


async def test_pre_cancelled_token_raises_immediately():
    token = CancelToken()
    token.cancel()

    async def slow():
        await asyncio.sleep(5)

    with pytest.raises(CancelledError):
        await run_cancellable(slow(), token)


async def test_token_cancel_is_visible_to_polling_thread():
    token = CancelToken()

    def blocking():
        for _ in range(200):
            if token.is_cancelled:
                return "stopped"
            time.sleep(0.01)
        return "ran-full"

    async def canceller():
        await asyncio.sleep(0.03)
        token.cancel()

    task = asyncio.ensure_future(canceller())
    result = await asyncio.to_thread(blocking)
    assert result == "stopped"
    await task


def test_run_turn_none_token_is_allowed():
    # A plain (non-cancellable) call path must still work.
    async def _amain():
        return await run_cancellable(asyncio.sleep(0, result="ok"), None)

    assert asyncio.run(_amain()) == "ok"


# --- Agent-level cancellation ---------------------------------------------
class _SlowProvider:
    """Async provider that sleeps ``delay`` seconds before replying."""

    def __init__(self, reply: str, delay: float) -> None:
        self.reply = reply
        self.delay = delay
        self.calls = 0

    async def acomplete(self, messages, cancel_token=None):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return CompletionResult(text=self.reply, model="fake")


async def test_agent_turn_can_be_cancelled():
    provider = _SlowProvider("<message>done</message>", delay=5)
    agent = Agent(Config(), provider=provider)
    token = CancelToken()

    async def canceller():
        await asyncio.sleep(0.05)
        token.cancel()

    task = asyncio.ensure_future(canceller())
    t0 = time.time()
    with pytest.raises(CancelledError):
        await agent.arun_turn("hello", cancel_token=token)
    assert time.time() - t0 < 1.0
    await task


async def test_agent_turn_completes_when_not_cancelled():
    provider = _SlowProvider("<message>fast</message>", delay=0.0)
    agent = Agent(Config(), provider=provider)
    resp = await agent.arun_turn("hi", cancel_token=CancelToken())
    assert resp.content == "fast"
