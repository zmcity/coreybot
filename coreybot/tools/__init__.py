"""Tool system package.

Public API:
    - ``tool``            : decorator to define + register a tool.
    - ``ToolSpec``        : a tool's interface declaration (name/desc/params).
    - ``Tool``            : the tool dataclass.
    - ``ToolResult``      : normalized execution outcome.
    - ``ToolRegistry``    : registry type.
    - ``get_registry``    : access the process-wide default registry.

Importing this package also imports the builtin tools so they self-register.
Add your own builtin by creating a new subpackage directory under
``coreybot/tools/builtin/<name>/`` with a ``spec.py`` (the ``ToolSpec``
interface declaration), a ``tool.py`` (the ``@tool(spec=SPEC)``-registered
implementation) and, ideally, a ``tests/`` folder next to it. Discovery is
automatic -- no central file needs editing.
"""

from __future__ import annotations

from .base import Tool, ToolRegistry, ToolResult, ToolSpec, get_registry, tool

# Import builtins for their registration side effects.
from . import builtin  # noqa: F401

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "get_registry",
    "tool",
]