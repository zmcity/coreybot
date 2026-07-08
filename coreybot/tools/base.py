"""The tool system core: define a tool, register it, render it for the model.

Design goals (mirrors the provider registry so the ideas feel familiar):
    - Defining a tool should be as easy as writing a function and adding a
      decorator. No base classes to subclass, no boilerplate.
    - Tools self-register into a global registry via ``@tool(...)`` as an import
      side effect -- adding a builtin never requires editing central code
      (Open/Closed Principle).
    - The registry can render a human/model-readable catalog that we inject into
      the system prompt, so the model knows what tools exist and how to call
      them.
    - Executing a tool never raises into the agent loop: errors are captured in
      a ``ToolResult`` so the model can see and recover from them.

This module has no third-party dependencies.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional

from coreybot.security.capabilities import SafetyProfile, UNKNOWN_PROFILE


@dataclass
class ToolResult:
    """Outcome of running a tool.

    ``output`` is the text shown back to the model. ``ok`` distinguishes success
    from failure so the loop/UI can label it, and ``error`` carries the reason.
    """

    output: str
    ok: bool = True
    error: Optional[str] = None

    @classmethod
    def success(cls, output: str) -> "ToolResult":
        return cls(output=output, ok=True)

    @classmethod
    def failure(cls, error: str) -> "ToolResult":
        return cls(output=error, ok=False, error=error)


@dataclass(frozen=True)
class ToolSpec:
    """A tool\'s *interface declaration* -- the model-facing contract.

    This is deliberately separate from the implementation so a tool\'s public
    surface (what the model sees) can live in its own ``spec.py`` file and be
    read at a glance, independent of the logic in ``tool.py``.

    - ``name``: unique identifier the model uses in ``<name>``.
    - ``description``: one line telling the model when to use it.
    - ``parameters``: mapping of ``arg_name -> hint string``. By convention the
      hint reads ``"<type> -- <what it is>"`` (e.g.
      ``{"expression": "string -- the expression to evaluate"}``). It only
      *describes* arguments to the model; values arrive as parsed JSON.
    """

    name: str
    description: str
    parameters: Dict[str, str] = field(default_factory=dict)
    # Optional safety profile (capabilities + how to make the call
    # recoverable). Defaults to opaque/unknown so tools that declare nothing
    # are treated conservatively by the safety policy.
    safety: SafetyProfile = UNKNOWN_PROFILE

    def bind(self, func: Callable[..., Any]) -> "Tool":
        """Attach an implementation ``func`` to this spec, producing a Tool."""
        return Tool(
            name=self.name,
            description=self.description,
            parameters=dict(self.parameters),
            func=func,
            safety=self.safety,
        )


@dataclass
class Tool:
    """A callable capability the model can invoke.

    - ``name``: unique identifier the model uses in ``<name>``.
    - ``description``: one line telling the model when to use it.
    - ``parameters``: mapping of ``arg_name -> hint string``. By convention the
      hint reads ``"<type> -- <what it is>"`` (e.g.
      ``{"expression": "string -- the expression to evaluate"}``). It is used
      only to *describe* the tool to the model; actual values arrive as parsed
      JSON and are validated by name in :meth:`call`.
    - ``func``: the Python callable implementing the tool. It receives keyword
      arguments and returns either a ``ToolResult`` or a plain string (which we
      wrap as a success).
    """

    name: str
    description: str
    parameters: Dict[str, str]
    func: Callable[..., Any]
    safety: SafetyProfile = UNKNOWN_PROFILE

    def call(self, arguments: Dict[str, Any]) -> ToolResult:
        """Invoke the tool with ``arguments`` (a dict), capturing failures.

        We validate that required parameters are present (any parameter the
        underlying function does not give a default), then call it. Any
        exception becomes a ``ToolResult.failure`` so the agent loop stays alive.
        """
        try:
            self._check_arguments(arguments)
            result = self.func(**arguments)
        except TypeError as exc:
            return ToolResult.failure(f"invalid arguments: {exc}")
        except Exception as exc:  # tool raised: surface it, do not crash
            return ToolResult.failure(f"{type(exc).__name__}: {exc}")

        if isinstance(result, ToolResult):
            return result
        return ToolResult.success(str(result))

    def _check_arguments(self, arguments: Dict[str, Any]) -> None:
        signature = inspect.signature(self.func)
        # Unexpected argument names (that the function cannot accept).
        accepts_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        if not accepts_kwargs:
            for key in arguments:
                if key not in signature.parameters:
                    raise TypeError(f"unexpected argument '{key}'")
        # Missing required arguments (no default provided).
        for pname, param in signature.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                continue
            if param.default is inspect.Parameter.empty and pname not in arguments:
                raise TypeError(f"missing required argument '{pname}'")


class ToolRegistry:
    """Holds registered tools and can render them for the model."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool_obj: Tool) -> None:
        key = tool_obj.name
        if key in self._tools:
            raise ValueError(f"Tool '{key}' is already registered")
        self._tools[key] = tool_obj

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def all(self) -> List[Tool]:
        return [self._tools[name] for name in sorted(self._tools)]

    def names(self) -> List[str]:
        return sorted(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def render_for_prompt(self) -> str:
        """Render the catalog injected into the system prompt.

        Kept compact and example-driven so the model reliably learns the call
        format. Returns an empty string when no tools are registered.
        """
        if not self._tools:
            return ""
        lines = ["You can call the following tools:"]
        for tool_obj in self.all():
            if tool_obj.parameters:
                args = ", ".join(
                    f"{name}: {hint}" for name, hint in tool_obj.parameters.items()
                )
            else:
                args = "(none)"
            lines.append(f"- {tool_obj.name}({args}) -- {tool_obj.description}")
        return "\n".join(lines)


# The process-wide default registry that ``@tool`` writes into.
_DEFAULT_REGISTRY = ToolRegistry()


def get_registry() -> ToolRegistry:
    return _DEFAULT_REGISTRY


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, str]] = None,
    registry: Optional[ToolRegistry] = None,
    *,
    spec: Optional[ToolSpec] = None,
    safety: Optional[SafetyProfile] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a function as a :class:`Tool`.

    Two equivalent styles:

    1) Declare the interface inline::

        @tool(name="add", description="Add two numbers",
              parameters={"a": "number", "b": "number"})
        def add(a, b): return str(a + b)

    2) Declare the interface in a separate ``ToolSpec`` (recommended for the
       builtins, so the model-facing contract lives in its own ``spec.py``)::

        # spec.py
        ADD = ToolSpec(name="add", description="Add two numbers",
                       parameters={"a": "number", "b": "number"})

        # tool.py
        @tool(spec=ADD)
        def add(a, b): return str(a + b)

    The original function is returned unchanged, so it stays directly callable
    and unit-testable; the registration is the side effect.
    """
    if spec is not None:
        if name is not None or description is not None or parameters is not None:
            raise ValueError("pass either spec=... or name/description/parameters, not both")
        resolved = spec if safety is None else replace(spec, safety=safety)
    else:
        if name is None or description is None:
            raise ValueError("tool() requires name and description (or spec=...)")
        resolved = ToolSpec(
            name=name, description=description, parameters=parameters or {},
            safety=safety if safety is not None else UNKNOWN_PROFILE,
        )

    target = registry if registry is not None else _DEFAULT_REGISTRY

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        target.register(resolved.bind(func))
        return func

    return decorator