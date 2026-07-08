"""Interface declaration for the ``read_file`` tool (see ``tool.py``)."""

from __future__ import annotations

from ...base import ToolSpec

SPEC = ToolSpec(
    name="read_file",
    description="Read a UTF-8 text file and return its contents (truncated if large).",
    parameters={
        "path": "string -- path to the file to read",
        "max_bytes": "number -- optional cap on bytes to read (default 20000)",
    },
)
