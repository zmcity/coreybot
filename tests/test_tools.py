"""Unit tests for the tool system (``coreybot.tools``)."""

from __future__ import annotations

import pytest

from coreybot.tools import Tool, ToolRegistry, ToolResult, ToolSpec, tool
from coreybot.tools.base import get_registry


# --- registry + decorator ------------------------------------------------
def test_tool_decorator_registers_into_custom_registry():
    reg = ToolRegistry()

    @tool(name="add", description="add", parameters={"a": "number", "b": "number"}, registry=reg)
    def add(a, b):
        return a + b

    assert "add" in reg
    assert reg.names() == ["add"]
    # The original function is returned unchanged and still directly callable.
    assert add(2, 3) == 5


def test_tool_spec_bind_produces_matching_tool():
    spec = ToolSpec(
        name="mul", description="multiply", parameters={"a": "number", "b": "number"}
    )

    def mul(a, b):
        return a * b

    built = spec.bind(mul)
    assert built.name == "mul"
    assert built.description == "multiply"
    assert built.parameters == {"a": "number", "b": "number"}
    assert built.call({"a": 3, "b": 4}).output == "12"


def test_tool_decorator_accepts_spec():
    reg = ToolRegistry()
    spec = ToolSpec(name="neg", description="negate", parameters={"x": "number"})

    @tool(spec=spec, registry=reg)
    def neg(x):
        return -x

    assert reg.get("neg").description == "negate"
    assert reg.get("neg").call({"x": 5}).output == "-5"


def test_tool_decorator_rejects_spec_and_inline_together():
    spec = ToolSpec(name="x", description="d")
    with pytest.raises(ValueError):

        @tool(name="x", description="d", spec=spec, registry=ToolRegistry())
        def x():
            return "x"


def test_tool_decorator_requires_name_or_spec():
    with pytest.raises(ValueError):

        @tool(registry=ToolRegistry())
        def y():
            return "y"


def test_duplicate_tool_name_raises():
    reg = ToolRegistry()

    @tool(name="x", description="d", registry=reg)
    def one():
        return "1"

    with pytest.raises(ValueError):

        @tool(name="x", description="d", registry=reg)
        def two():
            return "2"


def test_builtins_are_registered_by_default():
    reg = get_registry()
    assert {"calc", "current_time", "read_file"} <= set(reg.names())


# --- calling + argument validation --------------------------------------
def test_call_success_wraps_plain_string():
    reg = ToolRegistry()

    @tool(name="echo", description="d", parameters={"s": "string"}, registry=reg)
    def echo(s):
        return s.upper()

    result = reg.get("echo").call({"s": "hi"})
    assert isinstance(result, ToolResult)
    assert result.ok and result.output == "HI"


def test_call_missing_required_argument():
    reg = ToolRegistry()

    @tool(name="need", description="d", parameters={"x": "string"}, registry=reg)
    def need(x):
        return x

    result = reg.get("need").call({})
    assert not result.ok
    assert "missing required argument" in result.error


def test_call_unexpected_argument():
    reg = ToolRegistry()

    @tool(name="noargs", description="d", registry=reg)
    def noargs():
        return "ok"

    result = reg.get("noargs").call({"nope": 1})
    assert not result.ok
    assert "unexpected argument" in result.error


def test_call_captures_tool_exception():
    reg = ToolRegistry()

    @tool(name="boom", description="d", registry=reg)
    def boom():
        raise RuntimeError("kaboom")

    result = reg.get("boom").call({})
    assert not result.ok
    assert "kaboom" in result.output


def test_tool_may_return_toolresult_directly():
    reg = ToolRegistry()

    @tool(name="direct", description="d", registry=reg)
    def direct():
        return ToolResult.failure("nope")

    result = reg.get("direct").call({})
    assert not result.ok and result.output == "nope"


# --- prompt rendering ----------------------------------------------------
def test_render_for_prompt_lists_tools():
    reg = ToolRegistry()

    @tool(name="a", description="does a", parameters={"x": "string"}, registry=reg)
    def a(x):
        return x

    @tool(name="b", description="does b", parameters={}, registry=reg)
    def b():
        return "b"

    rendered = reg.render_for_prompt()
    assert "a(x: string) -- does a" in rendered
    assert "b((none)) -- does b" in rendered


def test_render_for_prompt_empty_registry():
    assert ToolRegistry().render_for_prompt() == ""

# --- builtins wiring (deep per-tool behavior lives beside each tool) --------
def test_builtin_tools_are_registered_and_callable():
    """Smoke test: discovery wired the builtins into the default registry.

    Detailed behavior for each tool is covered by the colocated tests under
    ``coreybot/tools/builtin/<name>/tests/``.
    """
    reg = get_registry()
    assert {"calc", "current_time", "read_file"} <= set(reg.names())
    # Each is invocable through the registry and returns a ToolResult.
    assert reg.get("calc").call({"expression": "1+1"}).output == "2"
    assert reg.get("current_time").call({}).ok
    assert not reg.get("read_file").call({"path": "nope.txt"}).ok
