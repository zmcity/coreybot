"""Interface declaration for the ``current_time`` tool (see ``tool.py``)."""

from __future__ import annotations

from ...base import ToolSpec

SPEC = ToolSpec(
    name="current_time",
    description="Get the current local date and time as an ISO-8601 string.",
    parameters={},
)
