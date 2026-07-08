"""Builtin tools package with automatic discovery.

Layout: each builtin tool is its own *subpackage* directory:

    coreybot/tools/builtin/
        calc/
            __init__.py        # re-exports SPEC + the tool function
            spec.py            # the ToolSpec interface declaration
            tool.py            # the @tool(spec=SPEC) implementation
            tests/
                test_calc.py   # colocated unit tests
        clock/
        read_file/

Importing this package imports every tool subpackage, and each subpackage
registers its tool via ``@tool(...)`` as an import side effect. Discovery is
dynamic: any immediate subpackage that contains a ``tool`` module is loaded, so
adding a new tool is just dropping in a new directory -- no edit here required
(Open/Closed Principle, the same idea used by the provider registry).

The mandatory layout for each tool subpackage (spec.py / tool.py / tests/)
is documented in ``README.md`` next to this file and enforced by
``tests/test_builtin_conventions.py`` (a pytest-collected structural check
that acts as a project-local lint rule).
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from types import ModuleType
from typing import List


def _discover_tool_modules() -> List[str]:
    """Return the import paths of every ``<subpackage>.tool`` under this package.

    We only treat a subpackage as a tool if it exposes a ``tool`` submodule,
    which keeps unrelated helpers (e.g. a ``tests`` package) from being loaded
    as tools.
    """
    found: List[str] = []
    for info in pkgutil.iter_modules(__path__):
        if not info.ispkg or info.name.startswith("_"):
            continue
        tool_module = f"{__name__}.{info.name}.tool"
        if importlib.util.find_spec(tool_module) is not None:
            found.append(tool_module)
    return sorted(found)


def load_builtins() -> List[ModuleType]:
    """Import all discovered tool modules (registering their tools). Idempotent.

    Returns the imported modules, mostly so tests can assert what was loaded.
    """
    modules: List[ModuleType] = []
    for module_path in _discover_tool_modules():
        modules.append(importlib.import_module(module_path))
    return modules


# Names of the discovered tool subpackages (e.g. ["calc", "clock", ...]).
BUILTIN_TOOL_PACKAGES = [
    path.rsplit(".", 2)[-2] for path in _discover_tool_modules()
]

# Load on import so tools self-register when the package is imported.
load_builtins()

__all__ = ["BUILTIN_TOOL_PACKAGES", "load_builtins"]
