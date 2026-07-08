"""A tiny JSON-over-HTTP client built only on the standard library.

Why not ``requests``? This is a learning project, so we implement the small
slice of HTTP we actually need (POST JSON, read JSON back) using ``urllib``.
This keeps the dependency footprint at zero for networking.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import urllib.error
import urllib.request
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from coreybot.core.cancel import CancelToken, CancelledError


class HTTPError(RuntimeError):
    """Raised when the server responds with a non-2xx status code."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


def _build_request(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]],
) -> urllib.request.Request:
    data = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    return urllib.request.Request(
        url, data=data, headers=request_headers, method="POST"
    )


def post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """POST ``payload`` as JSON and return the parsed JSON response.

    Raises ``HTTPError`` on non-2xx responses so callers can handle API errors
    uniformly.
    """

    request = _build_request(url, payload, headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:  # server returned 4xx/5xx
        error_body = exc.read().decode("utf-8", errors="replace")
        raise HTTPError(exc.code, error_body) from exc
    except urllib.error.URLError as exc:  # connection/DNS problems
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc


def post_sse(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 60.0,
) -> Iterator[str]:
    """POST ``payload`` and yield Server-Sent-Events ``data:`` lines.

    Streaming APIs (OpenAI/Anthropic/Gemini with ``stream=true``) return a text
    stream where each event looks like ``data: {json}\\n``. We parse just enough:
    strip the ``data: `` prefix and yield the raw payload string. The special
    ``[DONE]`` sentinel and blank keep-alive lines are skipped by the caller.
    """

    request = _build_request(url, payload, headers)
    request.add_header("Accept", "text/event-stream")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line or not line.startswith("data:"):
                    continue
                yield line[len("data:"):].strip()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise HTTPError(exc.code, error_body) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc


# ---------------------------------------------------------------------------
# Async wrappers with cooperative cancellation.
#
# The stdlib has no async HTTP client, and this is a learning project, so we
# reuse the blocking ``urllib`` code above and move it off the event loop with
# ``asyncio.to_thread``. Cancellation is cooperative:
# - ``apost_json`` races the blocking call against the token; if the token wins
#   we raise ``CancelledError`` and stop awaiting (the orphaned thread finishes
#   on its own and its result is dropped).
# - ``apost_sse`` streams: a producer thread reads the socket and pushes chunks
#   into a queue; the async side polls the queue and the token between chunks,
#   so cancelling stops delivery promptly (the socket read unwinds when the
#   response object is closed / the generator is dropped).
# ---------------------------------------------------------------------------


async def apost_json(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 60.0,
    cancel_token: Optional[CancelToken] = None,
) -> Dict[str, Any]:
    """Async, cancellable version of :func:`post_json`."""
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    work = asyncio.ensure_future(
        asyncio.to_thread(post_json, url, payload, headers, timeout)
    )
    if cancel_token is None:
        return await work

    waiter = asyncio.ensure_future(cancel_token.wait())
    try:
        done, _ = await asyncio.wait(
            {work, waiter}, return_when=asyncio.FIRST_COMPLETED
        )
        if work in done:
            return work.result()
        raise CancelledError("request cancelled")
    finally:
        for task in (work, waiter):
            if not task.done():
                task.cancel()
                try:
                    await task
                except BaseException:
                    pass


_SSE_DONE = object()  # sentinel pushed by the producer when the stream ends


def _sse_producer(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]],
    timeout: float,
    out: "queue.Queue",
    stop: threading.Event,
) -> None:
    """Run in a thread: read the SSE stream and push chunks into ``out``.

    Stops early if ``stop`` is set (cooperative cancellation). Any exception is
    forwarded to the consumer so it can re-raise on the event loop.
    """
    try:
        for chunk in post_sse(url, payload, headers=headers, timeout=timeout):
            if stop.is_set():
                break
            out.put(chunk)
    except BaseException as exc:  # forward transport errors to the consumer
        out.put(exc)
    finally:
        out.put(_SSE_DONE)


async def apost_sse(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 60.0,
    cancel_token: Optional[CancelToken] = None,
    poll_interval: float = 0.05,
) -> AsyncIterator[str]:
    """Async, cancellable version of :func:`post_sse`.

    Yields ``data:`` payload strings as they arrive. If ``cancel_token`` is
    cancelled, iteration stops promptly and :class:`CancelledError` is raised.
    """
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    out: "queue.Queue" = queue.Queue()
    stop = threading.Event()
    thread = threading.Thread(
        target=_sse_producer,
        args=(url, payload, headers, timeout, out, stop),
        daemon=True,
    )
    thread.start()
    try:
        while True:
            if cancel_token is not None and cancel_token.is_cancelled:
                raise CancelledError("stream cancelled")
            try:
                item = out.get_nowait()
            except queue.Empty:
                await asyncio.sleep(poll_interval)
                continue
            if item is _SSE_DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()  # ask the producer to stop reading on the next chunk
