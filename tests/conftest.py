"""Shared pytest fixtures and helpers.

The most important helper here is :func:`fake_transport`, which lets a test
replace the network layer of a provider with an in-memory function. Providers
call ``post_json`` / ``post_sse`` from ``coreybot.llm.http_client``; by monking
those names *inside the provider module* we can assert exactly what request the
provider would send and control exactly what it receives -- all without a real
socket. This is the key technique that makes the provider tests fast and
deterministic.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pytest

from coreybot.core.message import Message, Role


@pytest.fixture
def sample_messages() -> List[Message]:
    """A tiny conversation with a system + user turn, reused across tests."""
    return [
        Message.system("You are helpful."),
        Message.user("Hello"),
    ]


class RecordingJSON:
    """Callable stand-in for ``post_json`` that records the call and replies.

    Usage:
        rec = RecordingJSON(response={...})
        provider_module.post_json = rec
        ...            # exercise the provider
        assert rec.url == "http://.../chat/completions"
        assert rec.payload["model"] == "..."
    """

    def __init__(self, response: Dict[str, Any]) -> None:
        self.response = response
        self.url: Optional[str] = None
        self.payload: Optional[Dict[str, Any]] = None
        self.headers: Optional[Dict[str, str]] = None
        self.timeout: Optional[float] = None
        self.calls = 0

    def __call__(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        self.calls += 1
        self.url = url
        self.payload = payload
        self.headers = headers
        self.timeout = timeout
        return self.response


class RecordingSSE:
    """Callable stand-in for ``post_sse`` that records and yields chunks."""

    def __init__(self, chunks: List[str]) -> None:
        self.chunks = chunks
        self.url: Optional[str] = None
        self.payload: Optional[Dict[str, Any]] = None
        self.headers: Optional[Dict[str, str]] = None
        self.timeout: Optional[float] = None
        self.calls = 0

    def __call__(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
    ) -> Iterator[str]:
        self.calls += 1
        self.url = url
        self.payload = payload
        self.headers = headers
        self.timeout = timeout
        yield from self.chunks


class RecordingAsyncJSON:
    """Awaitable stand-in for ``apost_json`` (records the call, returns reply)."""

    def __init__(self, response: Dict[str, Any]) -> None:
        self.response = response
        self.url: Optional[str] = None
        self.payload: Optional[Dict[str, Any]] = None
        self.headers: Optional[Dict[str, str]] = None
        self.timeout: Optional[float] = None
        self.cancel_token: Any = None
        self.calls = 0

    async def __call__(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
        cancel_token: Any = None,
    ) -> Dict[str, Any]:
        self.calls += 1
        self.url = url
        self.payload = payload
        self.headers = headers
        self.timeout = timeout
        self.cancel_token = cancel_token
        return self.response


class RecordingAsyncSSE:
    """Async-generator stand-in for ``apost_sse`` (records, yields chunks)."""

    def __init__(self, chunks: List[str]) -> None:
        self.chunks = chunks
        self.url: Optional[str] = None
        self.payload: Optional[Dict[str, Any]] = None
        self.headers: Optional[Dict[str, str]] = None
        self.timeout: Optional[float] = None
        self.cancel_token: Any = None
        self.calls = 0

    def __call__(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
        cancel_token: Any = None,
    ):
        self.calls += 1
        self.url = url
        self.payload = payload
        self.headers = headers
        self.timeout = timeout
        self.cancel_token = cancel_token

        async def _gen():
            for chunk in self.chunks:
                yield chunk

        return _gen()


@pytest.fixture
def recording_async_json() -> Callable[[Dict[str, Any]], RecordingAsyncJSON]:
    def _make(response: Dict[str, Any]) -> RecordingAsyncJSON:
        return RecordingAsyncJSON(response)
    return _make


@pytest.fixture
def recording_async_sse() -> Callable[[List[str]], RecordingAsyncSSE]:
    def _make(chunks: List[str]) -> RecordingAsyncSSE:
        return RecordingAsyncSSE(chunks)
    return _make


@pytest.fixture
def recording_json() -> Callable[[Dict[str, Any]], RecordingJSON]:
    """Factory fixture: ``recording_json({...})`` -> a RecordingJSON."""
    def _make(response: Dict[str, Any]) -> RecordingJSON:
        return RecordingJSON(response)
    return _make


@pytest.fixture
def recording_sse() -> Callable[[List[str]], RecordingSSE]:
    """Factory fixture: ``recording_sse([...])`` -> a RecordingSSE."""
    def _make(chunks: List[str]) -> RecordingSSE:
        return RecordingSSE(chunks)
    return _make


class FakeStreamProvider:
    """A minimal async provider used by TUI / chat-loop tests (no network).

    - ``acomplete`` returns a full reply (from ``replies`` queue or joined
      ``tokens``); ``astream`` yields tokens one by one.
    - If ``error`` is set it is raised, letting tests exercise error handling.
    - ``delay`` (seconds) makes ``acomplete`` await, so cancellation tests can
      interrupt an in-flight call.

    Sync ``complete`` / ``stream`` remain for any non-async caller.
    """

    def __init__(
        self,
        tokens: Optional[List[str]] = None,
        replies: Optional[List[str]] = None,
        error: Optional[Exception] = None,
        error_after: int = 0,
        delay: float = 0.0,
    ) -> None:
        self.tokens = tokens if tokens is not None else ["Hello", "!"]
        # ``replies``: a queue of full raw responses for multi-step (tool) turns.
        self.replies = list(replies) if replies is not None else None
        self.error = error
        self.error_after = error_after
        self.delay = delay
        self.seen_messages: List[List[Message]] = []

    def _next_text(self, messages: List[Message]) -> str:
        self.seen_messages.append(list(messages))
        if self.error is not None:
            raise self.error
        if self.replies is not None:
            return self.replies.pop(0) if self.replies else "<message>(done)</message>"
        return "".join(self.tokens)

    async def acomplete(self, messages: List[Message], cancel_token=None):
        import asyncio

        from coreybot.core.message import CompletionResult

        if self.delay:
            await asyncio.sleep(self.delay)
        return CompletionResult(text=self._next_text(messages), model="fake")

    async def astream(self, messages: List[Message], cancel_token=None):
        self.seen_messages.append(list(messages))
        for index, token in enumerate(self.tokens):
            if self.error is not None and index == self.error_after:
                raise self.error
            yield token
        if self.error is not None and self.error_after >= len(self.tokens):
            raise self.error

    def complete(self, messages: List[Message]):
        from coreybot.core.message import CompletionResult

        return CompletionResult(text=self._next_text(messages), model="fake")

    def stream(self, messages: List[Message]) -> Iterator[str]:
        self.seen_messages.append(list(messages))
        for index, token in enumerate(self.tokens):
            if self.error is not None and index == self.error_after:
                raise self.error
            yield token
        if self.error is not None and self.error_after >= len(self.tokens):
            raise self.error


@pytest.fixture
def fake_stream_provider() -> Callable[..., FakeStreamProvider]:
    def _make(**kwargs: Any) -> FakeStreamProvider:
        return FakeStreamProvider(**kwargs)
    return _make



@pytest.fixture
def local_tmp_path():
    """A per-test temp directory created INSIDE the repo.

    Why not the built-in ``tmp_path``? On this machine the system ``%TEMP%``
    directory is locked down by security software (creating files there raises
    ``PermissionError [WinError 5]``), which breaks pytest's default temp
    fixtures. Placing our scratch directory under ``tests/.artifacts`` sidesteps
    that entirely. Each test gets a unique subfolder that is cleaned up after.
    """
    import shutil
    import uuid
    from pathlib import Path

    base = Path(__file__).parent / ".artifacts"
    base.mkdir(parents=True, exist_ok=True)
    path = base / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)