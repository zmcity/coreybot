"""Interface declaration for the ``calc`` tool.

This file holds only the *contract* the model sees (name, description,
parameters). The implementation lives in ``tool.py``. Keeping them apart makes
the tool\'s public surface easy to read and review in isolation.
"""

from __future__ import annotations

from ...base import ToolSpec

SPEC = ToolSpec(
    name="calc",
    description="Evaluate a basic arithmetic expression, e.g. '2 * (3 + 4)'.",
    parameters={"expression": "string -- the arithmetic expression to evaluate"},
)
