"""Implementation of the ``read_file`` tool (bounded file read).

Interface is declared in ``spec.py``. This is a reference for a tool with an
optional argument and basic guard rails: it caps how much it reads, validates
inputs, and turns every failure into a ``ToolResult`` instead of raising.

Note: this is intentionally NOT sandboxed (it can read any path the process
can) and is meant for a local learning project only.
"""

from __future__ import annotations

from pathlib import Path

from ...base import ToolResult, tool
from .spec import SPEC

__all__ = ["read_file"]

# Default and hard-ceiling byte caps. The hard ceiling bounds how much a single
# call can pull into the conversation even if the model asks for more.
_DEFAULT_MAX_BYTES = 20_000
_HARD_MAX_BYTES = 200_000


@tool(spec=SPEC)
def read_file(path: str, max_bytes: int = _DEFAULT_MAX_BYTES) -> ToolResult:
    """Read up to ``max_bytes`` bytes of ``path`` and return decoded text."""
    try:
        limit = int(max_bytes)
    except (TypeError, ValueError):
        return ToolResult.failure(f"max_bytes must be an integer, got {max_bytes!r}")
    if limit <= 0:
        return ToolResult.failure("max_bytes must be positive")
    limit = min(limit, _HARD_MAX_BYTES)

    file_path = Path(path)
    if not file_path.exists():
        return ToolResult.failure(f"file not found: {path}")
    if not file_path.is_file():
        return ToolResult.failure(f"not a file: {path}")

    try:
        data = file_path.read_bytes()[:limit]
        text = data.decode("utf-8", errors="replace")
    except Exception as exc:
        return ToolResult.failure(f"could not read: {exc}")

    return ToolResult.success(text)
