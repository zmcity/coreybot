"""Interface declaration for the ``calc`` tool.

This file holds only the *contract* the model sees (name, description,
parameters). The implementation lives in ``tool.py``. Keeping them apart makes
the tool\'s public surface easy to read and review in isolation.

``calc`` is a general algebra/scientific calculator (backed by SymPy): it
handles numeric arithmetic, symbolic simplification, calculus and equation
solving. It is pure compute with no side effect, so it declares an empty
:class:`SafetyProfile` (the risk engine classes it fully recoverable).
"""

from __future__ import annotations

from ...base import ToolSpec
from ....security.capabilities import make_profile

SPEC = ToolSpec(
    name="calc",
    description=(
        "Evaluate a math expression: arithmetic, algebra (factor/expand/"
        "simplify), calculus (diff/integrate/limit) and equation solving "
        "(solve). Supports ^ for power, implicit multiplication (2x), constants "
        "pi/e/i and functions like sin, cos, sqrt, log, factorial."
    ),
    parameters={
        "expression": "string -- the math expression, e.g. 'solve(x^2 - 1, x)' or 'sin(pi/2)'",
        "variables": "object -- optional name->value substitutions, e.g. {\"x\": 3}",
        "mode": "string -- optional: auto (default), symbolic, or numeric",
    },
    safety=make_profile(),
)
