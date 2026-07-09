"""The ``calc`` builtin tool (algebra/scientific calculator, backed by SymPy).

Importing this package registers the tool via ``tool.py``. The interface
(``SPEC``) and the function are re-exported for convenient inspection/testing:

    from coreybot.tools.builtin.calc import calc, SPEC
"""

from __future__ import annotations

from .spec import SPEC
from .tool import calc

__all__ = ["SPEC", "calc"]
