"""Implementation of the ``webtool`` tool (bounded HTTP fetch).

Interface + safety profile live in ``spec.py``. This uses only the standard
library (``urllib``) to keep the project dependency-free. It bounds the request
with a timeout and caps how much body it reads back. It performs no safety
checks itself: :class:`~coreybot.security.policy.SafetyPolicy` classifies the
call before the agent invokes it. A future revision can add a headless-browser
backend behind this same signature without changing the contract.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from typing import Optional

from ...base import ToolResult, tool
from .spec import SPEC

__all__ = ["webtool"]

_DEFAULT_TIMEOUT = 20
_HARD_MAX_TIMEOUT = 120
# Cap the body pulled back into the conversation.
_MAX_BODY_CHARS = 20_000
_USER_AGENT = "coreybot-webtool/0.1"
_ALLOWED_SCHEMES = ("http", "https")


def _summarize_headers(headers) -> str:
    """A compact, readable subset of response headers."""
    interesting = ("Content-Type", "Content-Length", "Server", "Location")
    parts = []
    for key in interesting:
        value = headers.get(key)
        if value:
            parts.append(f"{key}: {value}")
    return "; ".join(parts)


@tool(spec=SPEC)
def webtool(
    url: str,
    method: Optional[str] = None,
    data: Optional[str] = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> ToolResult:
    """Perform an HTTP(S) request to ``url`` and return a text summary + body."""
    if not isinstance(url, str) or not url.strip():
        return ToolResult.failure("url must be a non-empty string")
    scheme = url.split("://", 1)[0].lower() if "://" in url else ""
    if scheme not in _ALLOWED_SCHEMES:
        return ToolResult.failure("url must start with http:// or https://")

    try:
        seconds = float(timeout)
    except (TypeError, ValueError):
        return ToolResult.failure(f"timeout must be a number, got {timeout!r}")
    if seconds <= 0:
        return ToolResult.failure("timeout must be positive")
    seconds = min(seconds, _HARD_MAX_TIMEOUT)

    body_bytes = data.encode("utf-8") if isinstance(data, str) else None
    resolved_method = (method or ("POST" if body_bytes is not None else "GET")).upper()

    request = urllib.request.Request(
        url, data=body_bytes, method=resolved_method, headers={"User-Agent": _USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=seconds) as response:
            raw = response.read(_MAX_BODY_CHARS * 4)
            status = getattr(response, "status", None) or response.getcode()
            header_summary = _summarize_headers(response.headers)
    except urllib.error.HTTPError as exc:
        # An HTTP error status is still a completed request; report it.
        detail = exc.read(_MAX_BODY_CHARS * 4) if hasattr(exc, "read") else b""
        text = detail.decode("utf-8", errors="replace")[:_MAX_BODY_CHARS]
        return ToolResult.failure(f"HTTP {exc.code} {exc.reason}\n{text}".rstrip())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return ToolResult.failure(f"request failed: {exc}")

    text = raw.decode("utf-8", errors="replace")
    if len(text) > _MAX_BODY_CHARS:
        text = text[:_MAX_BODY_CHARS] + "\n... (body truncated)"
    head = f"HTTP {status} {resolved_method} {url}"
    if header_summary:
        head += f"\n{header_summary}"
    return ToolResult.success(f"{head}\n\n{text}".rstrip())
