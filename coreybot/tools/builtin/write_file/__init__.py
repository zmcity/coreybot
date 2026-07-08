"""The ``write_file`` builtin tool.

    from coreybot.tools.builtin.write_file import write_file, SPEC
"""

from __future__ import annotations

from .spec import SPEC
from .tool import write_file

__all__ = ["SPEC", "write_file"]
