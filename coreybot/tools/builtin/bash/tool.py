"""Implementation of the ``bash`` tool (bounded shell execution).

Interface + safety profile live in ``spec.py``. This runs a command through the
platform shell, captures combined stdout/stderr, and enforces a timeout so a
runaway process cannot hang the agent. It performs NO safety checks itself --
that is the job of :class:`~coreybot.security.policy.SafetyPolicy`, which gates
the call *before* the agent ever invokes this function. Direct callers (e.g.
unit tests) bypass that gate, so only run trusted commands directly.
"""

from __future__ import annotations

import subprocess
from typing import Optional

from ...base import ToolResult, tool
from .spec import SPEC

__all__ = ["run_bash"]

# Defaults chosen to be safe for a local learning tool: a short timeout and a
# hard ceiling so a single call cannot block the loop indefinitely.
_DEFAULT_TIMEOUT = 30
_HARD_MAX_TIMEOUT = 600
# Cap how much output a single call can pull back into the conversation.
_MAX_OUTPUT_CHARS = 20_000


@tool(spec=SPEC)
def run_bash(
    command: str,
    timeout: float = _DEFAULT_TIMEOUT,
    workdir: Optional[str] = None,
) -> ToolResult:
    """Execute ``command`` via the shell and report output + exit code."""
    if not isinstance(command, str) or not command.strip():
        return ToolResult.failure("command must be a non-empty string")
    try:
        seconds = float(timeout)
    except (TypeError, ValueError):
        return ToolResult.failure(f"timeout must be a number, got {timeout!r}")
    if seconds <= 0:
        return ToolResult.failure("timeout must be positive")
    seconds = min(seconds, _HARD_MAX_TIMEOUT)

    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=workdir or None,
            capture_output=True,
            text=True,
            timeout=seconds,
        )
    except subprocess.TimeoutExpired:
        return ToolResult.failure(f"command timed out after {seconds:g}s")
    except OSError as exc:
        return ToolResult.failure(f"could not run command: {exc}")

    output = (completed.stdout or "") + (completed.stderr or "")
    if len(output) > _MAX_OUTPUT_CHARS:
        output = output[:_MAX_OUTPUT_CHARS] + "\n... (output truncated)"
    report = f"exit code: {completed.returncode}\n{output}".rstrip()
    if completed.returncode == 0:
        return ToolResult.success(report)
    return ToolResult.failure(report)
