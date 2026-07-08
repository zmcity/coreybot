"""The ``read_file`` builtin tool.

    from coreybot.tools.builtin.read_file import read_file, SPEC
"""

from __future__ import annotations

from .spec import SPEC
from .tool import read_file

__all__ = ["SPEC", "read_file"]
