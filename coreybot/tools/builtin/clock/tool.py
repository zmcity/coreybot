"""Implementation of the ``current_time`` tool.

Interface is declared in ``spec.py``. This is the simplest possible tool: no
arguments, returns a string.
"""

from __future__ import annotations

from datetime import datetime

from ...base import tool
from .spec import SPEC

__all__ = ["current_time"]


@tool(spec=SPEC)
def current_time() -> str:
    """Return the current local time, e.g. ``2026-07-07T14:30:00``."""
    return datetime.now().isoformat(timespec="seconds")
