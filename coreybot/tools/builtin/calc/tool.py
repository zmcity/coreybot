"""Implementation of the ``calc`` tool (general algebra/scientific calculator).

The model-facing interface is declared in ``spec.py``; here we implement and
register the behavior. ``calc`` is backed by SymPy so it can do far more than
arithmetic: symbolic simplification (``simplify``/``factor``/``expand``),
calculus (``diff``/``integrate``/``limit``) and equation solving (``solve``),
using a familiar math syntax (``^`` for power, implicit multiplication like
``2x``, constants ``pi``/``e``/``i`` and functions like ``sin``/``sqrt``/``log``).

Security note
-------------
SymPy\'s parser will *evaluate* function calls, so a naive ``parse_expr`` on
untrusted input is a remote-code-execution risk (e.g. ``__import__(\'os\')...``
actually runs). We defend in two independent layers:

1. A lexical pre-vet (:func:`_prevet`) rejects the two escape hatches that a math
   expression never needs: attribute access (a ``.`` followed by an identifier,
   which decimals like ``.5`` do not trigger) and any double-underscore name.
   This blocks ``os.system(...)`` and ``(1).__class__`` before parsing.
2. Parsing runs with a ``global_dict`` whose ``__builtins__`` is emptied, so even
   if something slipped through, ``__import__`` / ``open`` / ``eval`` are simply
   unreachable names.

Only names present in the curated SymPy namespace resolve; everything else
becomes a free symbol (a harmless algebraic variable), never a Python callable.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from ...base import ToolResult, tool
from .spec import SPEC

__all__ = ["calc"]

# --- lexical guard ---------------------------------------------------------
# Attribute access in a math expression would be a dot followed by an identifier
# start (``.foo``); a decimal literal is a dot followed by a digit (``.5``), so
# this pattern does not touch numbers.
_ATTR_ACCESS = re.compile(r"\.\s*[A-Za-z_]")
_DUNDER = re.compile(r"__")
_MAX_EXPRESSION_CHARS = 2000


def _prevet(expression: str) -> Optional[str]:
    """Return an error string if ``expression`` uses a forbidden construct."""
    if len(expression) > _MAX_EXPRESSION_CHARS:
        return f"expression too long (>{_MAX_EXPRESSION_CHARS} chars)"
    if _ATTR_ACCESS.search(expression):
        return "attribute access ('.') is not allowed"
    if _DUNDER.search(expression):
        return "double-underscore names are not allowed"
    return None


# --- SymPy engine (built once, lazily) -------------------------------------
# Populated on first use so importing the tool stays cheap and does not hard
# require SymPy until someone actually calls calc.
_ENGINE: Optional[Dict[str, Any]] = None


def _build_engine() -> Dict[str, Any]:
    """Assemble the SymPy parser namespace + transformations (memoized)."""
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    transformations = standard_transformations + (
        implicit_multiplication_application,  # 2x -> 2*x
        convert_xor,                          # ^ -> **
    )
    # Full SymPy namespace so its parser can build Integer/Function/etc., but
    # with Python builtins neutralized (no __import__/open/eval reachable).
    global_dict: Dict[str, Any] = {}
    exec("from sympy import *", global_dict)  # noqa: S102 - trusted, fixed input
    global_dict["__builtins__"] = {}
    # Make lowercase ``e`` the Euler number (SymPy only defines ``E``), so
    # ``log(e**2)`` simplifies to 2 as a user expects.
    global_dict["e"] = sp.E
    return {"sp": sp, "parse_expr": parse_expr, "transformations": transformations,
            "global_dict": global_dict}


def _engine() -> Dict[str, Any]:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _build_engine()
    return _ENGINE


def _format(value: Any, sp: Any) -> str:
    """Render a SymPy result as clean text (undefined for complex infinity)."""
    if value is sp.zoo or value is sp.nan:
        return "undefined"
    return str(value)


@tool(spec=SPEC)
def calc(
    expression: str,
    variables: Optional[Dict[str, Any]] = None,
    mode: str = "auto",
) -> ToolResult:
    """Evaluate ``expression`` with SymPy; return the result as text.

    ``variables`` substitutes name->value pairs (e.g. ``{"x": 3}``). ``mode`` is
    ``auto`` (numeric when fully determined, else symbolic), ``symbolic`` (keep
    the exact symbolic form) or ``numeric`` (force a floating-point value).
    """
    if not isinstance(expression, str) or not expression.strip():
        return ToolResult.failure("expression must be a non-empty string")
    if mode not in ("auto", "symbolic", "numeric"):
        return ToolResult.failure("mode must be auto, symbolic or numeric")

    forbidden = _prevet(expression)
    if forbidden is not None:
        return ToolResult.failure(forbidden)

    try:
        engine = _engine()
    except Exception as exc:  # pragma: no cover - only if SymPy is unavailable
        return ToolResult.failure(f"calculator engine unavailable: {exc}")

    sp = engine["sp"]
    try:
        expr = engine["parse_expr"](
            expression,
            transformations=engine["transformations"],
            global_dict=engine["global_dict"],
            evaluate=True,
        )
    except (SyntaxError, TypeError, ValueError, AttributeError) as exc:
        return ToolResult.failure(f"invalid expression: {exc}")
    except Exception as exc:  # defensive: SymPy can raise assorted errors
        return ToolResult.failure(f"could not parse: {type(exc).__name__}: {exc}")

    # Optional variable substitution.
    if variables:
        if not isinstance(variables, dict):
            return ToolResult.failure("variables must be an object of name->value")
        subs = {}
        for name, val in variables.items():
            bad = _prevet(str(name))
            if bad is not None:
                return ToolResult.failure(f"invalid variable name {name!r}: {bad}")
            try:
                subs[sp.Symbol(str(name))] = sp.sympify(
                    val, locals={}, evaluate=True
                ) if not isinstance(val, (int, float)) else val
            except Exception as exc:
                return ToolResult.failure(f"invalid value for {name!r}: {exc}")
        try:
            expr = expr.subs(subs) if hasattr(expr, "subs") else expr
        except Exception as exc:
            return ToolResult.failure(f"could not substitute variables: {exc}")

    try:
        if mode == "numeric":
            result: Any = sp.N(expr)
        else:
            # auto and symbolic both keep the EXACT form (14, 5/6, 2*I, sqrt(2),
            # x**2 + 1); the difference is only that "numeric" forces evalf.
            # SymPy already evaluated arithmetic during parsing, so this yields
            # clean exact results without noisy trailing decimals.
            result = expr
    except ZeroDivisionError:
        return ToolResult.failure("division by zero")
    except Exception as exc:
        return ToolResult.failure(f"could not evaluate: {type(exc).__name__}: {exc}")

    return ToolResult.success(_format(result, sp))
