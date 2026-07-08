"""The ``current_time`` builtin tool.

    from coreybot.tools.builtin.clock import current_time, SPEC
"""

from __future__ import annotations

from .spec import SPEC
from .tool import current_time

__all__ = ["SPEC", "current_time"]
