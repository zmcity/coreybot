"""Implementation of the ``calc`` tool (safe arithmetic).

The model-facing interface is declared separately in ``spec.py``; here we only
implement and register the behavior.

Security note
-------------
We never use ``eval`` (which would allow arbitrary code execution). Instead the
expression is parsed into an AST and evaluated by walking an explicit *allowlist*
of node types. Anything outside the allowlist -- function calls, names,
attribute access, subscripts, comprehensions, etc. -- is rejected before any
evaluation happens, so payloads like ``__import__(\'os\').system(...)`` cannot run.

Supported: integer/float literals, the binary operators ``+ - * / // % **``,
unary ``+``/``-``, and parentheses.
"""

from __future__ import annotations

import ast
import operator
from typing import Callable, Dict, Type, Union

from ...base import ToolResult, tool
from .spec import SPEC

__all__ = ["calc"]

Number = Union[int, float]

# Allowlisted operators. Mapping AST node type -> the function that applies it.
_BINARY_OPERATORS: Dict[Type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS: Dict[Type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class UnsafeExpressionError(ValueError):
    """Raised when the expression contains a node outside the allowlist."""


def _evaluate(node: ast.AST) -> Number:
    """Recursively evaluate an allowlisted AST node, or raise on anything else."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant):
        # Reject bools (a subclass of int) and non-numeric constants.
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpressionError("only numeric literals are allowed")
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        return _BINARY_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))

    raise UnsafeExpressionError(f"unsupported syntax: {type(node).__name__}")


@tool(spec=SPEC)
def calc(expression: str) -> ToolResult:
    """Parse and safely evaluate ``expression``; return the result as text."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return ToolResult.failure(f"invalid expression: {exc.msg}")

    try:
        value = _evaluate(tree)
    except ZeroDivisionError:
        return ToolResult.failure("division by zero")
    except UnsafeExpressionError as exc:
        return ToolResult.failure(str(exc))
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return ToolResult.failure(f"could not evaluate: {exc}")

    return ToolResult.success(str(value))
