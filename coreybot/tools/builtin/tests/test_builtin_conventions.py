"""Enforced structural conventions for builtin tools (a lint-style gate).

Each subpackage under ``coreybot/tools/builtin/`` must follow the rules in
``coreybot/tools/builtin/README.md``. These tests parse and import each tool
package and fail if any rule is violated, so ``pytest`` behaves like a project
-local linter for the builtin layout.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import List

import pytest

from coreybot.tools import ToolSpec, get_registry
from coreybot.tools.builtin import BUILTIN_TOOL_PACKAGES

BUILTIN_PACKAGE = "coreybot.tools.builtin"

# AST node types that count as logic and are therefore forbidden in spec.py
# (rule 2: spec.py holds only the interface declaration, no behavior).
_LOGIC_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Lambda,
)


def _package_dir(name: str) -> Path:
    """Filesystem directory of a builtin subpackage by short name."""
    module = importlib.import_module(f"{BUILTIN_PACKAGE}.{name}")
    assert module.__file__ is not None
    return Path(module.__file__).parent


def _module_level_assigns(tree: ast.Module) -> List[str]:
    """Names bound by top-level ``NAME = ...`` assignments in a module."""
    names: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return names


def test_there_is_at_least_one_builtin() -> None:
    """Sanity guard so an empty discovery does not vacuously pass."""
    assert BUILTIN_TOOL_PACKAGES, "expected at least one builtin tool package"


@pytest.mark.parametrize("name", BUILTIN_TOOL_PACKAGES)
def test_builtin_layout_files_exist(name: str) -> None:
    """Rule 1/3/5: spec.py, tool.py and a tests/ dir with a test_*.py exist."""
    pkg = _package_dir(name)
    assert (pkg / "spec.py").is_file(), f"{name}: missing spec.py"
    assert (pkg / "tool.py").is_file(), f"{name}: missing tool.py"
    assert (pkg / "__init__.py").is_file(), f"{name}: missing __init__.py"
    tests_dir = pkg / "tests"
    assert tests_dir.is_dir(), f"{name}: missing tests/ directory"
    test_files = list(tests_dir.glob("test_*.py"))
    assert test_files, f"{name}: tests/ has no test_*.py file"


@pytest.mark.parametrize("name", BUILTIN_TOOL_PACKAGES)
def test_spec_declares_module_level_SPEC(name: str) -> None:
    """Rule 1: spec.py binds a module-level ``SPEC`` name."""
    source = (_package_dir(name) / "spec.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "SPEC" in _module_level_assigns(tree), (
        f"{name}: spec.py must assign a module-level SPEC"
    )


@pytest.mark.parametrize("name", BUILTIN_TOOL_PACKAGES)
def test_spec_contains_no_logic(name: str) -> None:
    """Rule 2: spec.py is declaration-only (no def/class/branching/loops)."""
    source = (_package_dir(name) / "spec.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    offenders = sorted(
        {type(node).__name__ for node in ast.walk(tree) if isinstance(node, _LOGIC_NODES)}
    )
    assert not offenders, (
        f"{name}: spec.py must contain no logic, found: {offenders}"
    )


@pytest.mark.parametrize("name", BUILTIN_TOOL_PACKAGES)
def test_tool_registers_via_spec(name: str) -> None:
    """Rule 3: tool.py registers through ``@tool(spec=SPEC)`` (not inline)."""
    source = (_package_dir(name) / "tool.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    uses_spec = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_tool_call = (
            isinstance(func, ast.Name) and func.id == "tool"
        ) or (
            isinstance(func, ast.Attribute) and func.attr == "tool"
        )
        if is_tool_call and any(kw.arg == "spec" for kw in node.keywords):
            uses_spec = True
            break
    assert uses_spec, (
        f"{name}: tool.py must register via @tool(spec=SPEC)"
    )


@pytest.mark.parametrize("name", BUILTIN_TOOL_PACKAGES)
def test_package_reexports_spec(name: str) -> None:
    """Rule 4: importing the package exposes a ``SPEC`` of type ToolSpec."""
    module = importlib.import_module(f"{BUILTIN_PACKAGE}.{name}")
    assert hasattr(module, "SPEC"), f"{name}: __init__ must re-export SPEC"
    assert isinstance(module.SPEC, ToolSpec), (
        f"{name}: re-exported SPEC must be a ToolSpec"
    )


@pytest.mark.parametrize("name", BUILTIN_TOOL_PACKAGES)
def test_spec_name_matches_registry(name: str) -> None:
    """Rule 6: SPEC.name is registered in the default registry."""
    module = importlib.import_module(f"{BUILTIN_PACKAGE}.{name}")
    spec = module.SPEC
    assert spec.name in get_registry(), (
        f"{name}: SPEC.name {spec.name!r} is not in the registry"
    )


def test_spec_names_are_unique() -> None:
    """Rule 6: no two builtin tools share the same SPEC.name."""
    names: List[str] = []
    for name in BUILTIN_TOOL_PACKAGES:
        module = importlib.import_module(f"{BUILTIN_PACKAGE}.{name}")
        names.append(module.SPEC.name)
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert not duplicates, f"duplicate SPEC.name values: {duplicates}"


@pytest.mark.parametrize("name", BUILTIN_TOOL_PACKAGES)
def test_parameter_hint_format(name: str) -> None:
    """Rule 7: each parameter hint reads as '<type> -- <description>'."""
    module = importlib.import_module(f"{BUILTIN_PACKAGE}.{name}")
    for arg, hint in module.SPEC.parameters.items():
        assert isinstance(hint, str), f"{name}.{arg}: hint must be a string"
        assert " -- " in hint, (
            f"{name}.{arg}: hint {hint!r} must look like '<type> -- <description>'"
        )
