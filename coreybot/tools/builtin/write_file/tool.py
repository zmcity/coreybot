"""Implementation of the ``write_file`` tool (reversible file write).

Interface + safety profile are declared in ``spec.py``. This tool intentionally
performs a real, side-effecting write; it is safe to ship as a builtin because
the framework\'s :class:`~coreybot.security.policy.SafetyPolicy` governs it: an
in-workspace target is classified fully recoverable and snapshotted before the
write, while a target inside coreybot\'s own storage or outside the workspace is
classified unrecoverable and gated. The tool itself just writes and reports.
"""

from __future__ import annotations

from pathlib import Path

from ...base import ToolResult, tool
from .spec import SPEC

__all__ = ["write_file"]


@tool(spec=SPEC)
def write_file(path: str, content: str) -> ToolResult:
    """Write ``content`` to ``path`` (UTF-8), creating parent dirs as needed."""
    if not isinstance(content, str):
        return ToolResult.failure(f"content must be a string, got {type(content).__name__}")
    file_path = Path(path)
    try:
        if file_path.parent and not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
        existed = file_path.exists()
        with open(file_path, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
    except OSError as exc:
        return ToolResult.failure(f"could not write {path}: {exc}")
    verb = "overwrote" if existed else "created"
    return ToolResult.success(f"{verb} {path} ({len(content)} chars)")
