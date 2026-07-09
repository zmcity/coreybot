"""Integration tests for the agent loop (``coreybot.runtime.agent``).

We drive the Agent with a scripted provider that returns pre-baked raw replies,
so we can exercise the model -> tool -> model cycle deterministically and
without any network.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from coreybot.runtime.agent import Agent, AgentEvent
from coreybot.runtime.session import Snapshot
from coreybot.core.config import Config
from coreybot.core.message import Message, CompletionResult, Role
from coreybot.tools import ToolRegistry, tool


class ScriptedProvider:
    """Returns queued raw replies in order, ignoring the request contents."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def _next(self):
        self.calls += 1
        if not self.replies:
            raise AssertionError("provider ran out of scripted replies")
        return CompletionResult(text=self.replies.pop(0), model="fake")

    async def acomplete(self, messages, cancel_token=None):
        return self._next()

    def complete(self, messages):
        return self._next()


def _registry_with_add():
    reg = ToolRegistry()

    @tool(name="add", description="add two numbers",
          parameters={"a": "number", "b": "number"}, registry=reg)
    def add(a, b):
        return str(a + b)

    return reg


def test_plain_message_turn_no_tools():
    provider = ScriptedProvider(["<message>hello there</message>"])
    agent = Agent(Config(), provider=provider, registry=ToolRegistry())
    resp = agent.run_turn("hi")
    assert resp.is_message
    assert resp.content == "hello there"
    assert provider.calls == 1


def test_tool_call_then_message():
    provider = ScriptedProvider([
        '<tool_call><name>add</name><arguments>{"a": 2, "b": 3}</arguments></tool_call>',
        "<message>the sum is 5</message>",
    ])
    events = []
    agent = Agent(Config(), provider=provider, registry=_registry_with_add())
    resp = agent.run_turn("add 2 and 3", on_event=events.append)

    assert resp.content == "the sum is 5"
    assert provider.calls == 2

    # The stream now includes lifecycle events (turn/llm); focus on tool ones.
    tool_events = [e for e in events if e.kind in ("tool_call", "tool_result")]
    kinds = [e.kind for e in tool_events]
    assert kinds == ["tool_call", "tool_result"]
    assert tool_events[0].name == "add" and tool_events[0].arguments == {"a": 2, "b": 3}
    assert tool_events[0].source == "tool"
    assert tool_events[1].ok and tool_events[1].output == "5"
    # Lifecycle events are present and tagged with sources.
    assert any(e.kind == "turn_start" for e in events)
    assert any(e.kind == "llm_call" and e.source == "llm" for e in events)
    assert any(e.kind == "turn_end" for e in events)

    # History carries the observation as a user turn between assistant turns.
    roles = [m.role for m in agent.history]
    assert roles == [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT]
    assert "<tool_result" in agent.history[3].content


def test_tool_result_carries_execution_log():
    """A tool_result event carries a structured execution log for the inspector.

    The log records resolution, outcome, output size and real elapsed time so
    the flow chart's LOG section shows what actually happened.
    """
    provider = ScriptedProvider([
        '<tool_call><name>add</name><arguments>{"a": 2, "b": 3}</arguments></tool_call>',
        "<message>done</message>",
    ])
    events = []
    agent = Agent(Config(), provider=provider, registry=_registry_with_add())
    agent.run_turn("add 2 and 3", on_event=events.append)

    result = next(e for e in events if e.kind == "tool_result")
    assert result.log, "tool_result should carry a log"
    log = result.log
    assert "tool: add" in log
    assert "resolve: found" in log
    assert "outcome: ok" in log
    assert "elapsed:" in log and "ms" in log


def test_blocked_tool_log_records_safety_decision():
    """A tool blocked by the safety policy logs the decision and skips execution."""
    from coreybot.security import SecurityContext
    from coreybot.security.capabilities import Capability, make_profile
    from coreybot.security.policy import SafetyPolicy

    reg = ToolRegistry()
    outside = str((Path.cwd().parent / "sibling_should_not_write.txt"))

    @tool(
        name="writer",
        description="write a file",
        parameters={"path": "string -- target"},
        registry=reg,
        safety=make_profile(
            Capability.FS_WRITE,
            affected_paths=lambda args: [args.get("path", "")],
        ),
    )
    def writer(path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("x")
        return "wrote"

    provider = ScriptedProvider([
        '<tool_call><name>writer</name><arguments>{"path": %r}</arguments></tool_call>' % outside,
        "<message>done</message>",
    ])
    events = []
    security = SecurityContext(safety=SafetyPolicy(workspace_root=str(Path.cwd())))
    agent = Agent(Config(), provider=provider, registry=reg, security=security)
    agent.run_turn("write outside", on_event=events.append)

    result = next(e for e in events if e.kind == "tool_result")
    assert result.ok is False
    assert "safety: none -> deny" in result.log
    assert "execution: skipped" in result.log
    assert not os.path.exists(outside)  # never executed


def test_unknown_tool_feeds_error_back():
    provider = ScriptedProvider([
        "<tool_call><name>ghost</name><arguments>{}</arguments></tool_call>",
        "<message>sorry, could not do that</message>",
    ])
    events = []
    agent = Agent(Config(), provider=provider, registry=ToolRegistry())
    resp = agent.run_turn("do magic", on_event=events.append)

    assert resp.content == "sorry, could not do that"
    result_events = [e for e in events if e.kind == "tool_result"]
    assert result_events and not result_events[0].ok
    assert "unknown tool" in result_events[0].output


def test_bad_tool_arguments_are_reported():
    provider = ScriptedProvider([
        '<tool_call><name>add</name><arguments>{"a": 1}</arguments></tool_call>',
        "<message>done</message>",
    ])
    events = []
    agent = Agent(Config(), provider=provider, registry=_registry_with_add())
    agent.run_turn("add", on_event=events.append)
    result_events = [e for e in events if e.kind == "tool_result"]
    assert not result_events[0].ok
    assert "missing required argument" in result_events[0].output


def test_max_steps_guard_stops_infinite_tool_loop():
    # Always returns a tool call -> the loop must stop at max_steps.
    always_tool = ["<tool_call><name>add</name><arguments>{\"a\":1,\"b\":1}</arguments></tool_call>"] * 10
    provider = ScriptedProvider(always_tool)
    agent = Agent(Config(), provider=provider, registry=_registry_with_add(), max_steps=3)
    resp = agent.run_turn("loop")
    assert provider.calls == 3
    assert resp.parse_error is not None
    assert "too many tool steps" in resp.content


def test_reset_clears_history_but_keeps_system():
    provider = ScriptedProvider(["<message>hi</message>"])
    agent = Agent(Config(), provider=provider, registry=ToolRegistry())
    agent.run_turn("hello")
    assert len(agent.history) > 1
    agent.reset()
    assert [m.role for m in agent.history] == [Role.SYSTEM]


def test_transport_error_propagates():
    class Boom:
        async def acomplete(self, messages, cancel_token=None):
            raise RuntimeError("network down")

    agent = Agent(Config(), provider=Boom(), registry=ToolRegistry())
    with pytest.raises(RuntimeError):
        agent.run_turn("hi")


def test_system_prompt_includes_tool_catalog():
    agent = Agent(Config(), provider=ScriptedProvider([]), registry=_registry_with_add())
    assert "add(" in agent.system_prompt
    assert "<tool_call>" in agent.system_prompt


def test_checkout_session_rewinds_history_only():
    """A session checkout restores just the conversation/telemetry."""
    agent = Agent(Config(), provider=ScriptedProvider([]), registry=ToolRegistry())
    base = list(agent.history)
    node = agent.sessions.commit(Snapshot(history=base), label="turn 1")
    agent.history = base + [Message.user("later")]
    agent.sessions.commit(Snapshot(history=list(agent.history)), label="turn 2")

    agent.checkout_session(node.id)
    assert agent.history == base


def test_restore_workspace_rewinds_session_and_applies_file_artifacts(monkeypatch):
    """The workspace scope rewinds the session AND re-applies captured files.

    ``restore_workspace`` writes back edited-file blobs and removes files that
    were tombstoned as deleted, on top of the plain session rewind, and
    returns the number of on-disk changes. The filesystem is faked (the
    sandbox forbids real writes) so the test pins the DISPATCH: which paths
    get written with which contents, and which get removed.
    """
    import coreybot.runtime.agent as agent_mod

    writes = {}
    removed = []

    class _FakeFile:
        def __init__(self, path):
            self._path = path
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
        def write(self, data):
            writes[self._path] = writes.get(self._path, "") + data

    def fake_open(path, mode="r", *args, **kwargs):
        assert "w" in mode
        return _FakeFile(path)

    monkeypatch.setattr(agent_mod, "open", fake_open, raising=False)
    monkeypatch.setattr(agent_mod.os, "remove", lambda p: removed.append(p))

    agent = Agent(Config(), provider=ScriptedProvider([]), registry=ToolRegistry())
    base = list(agent.history)
    artifacts = {
        "files": {"edited.txt": "restored contents"},
        "deleted": ["gone.txt"],
    }
    node = agent.sessions.commit(
        Snapshot(history=base, artifacts=artifacts), label="snap"
    )
    agent.history = base + [Message.user("drifted")]

    applied = agent.restore_workspace(node.id)

    assert agent.history == base                     # session rewound
    assert writes == {"edited.txt": "restored contents"}
    assert removed == ["gone.txt"]
    assert applied == 2


def test_restore_workspace_with_no_artifacts_just_rewinds_session():
    """With no captured files (the norm today) it degrades to a session
    restore and reports zero on-disk changes.
    """
    agent = Agent(Config(), provider=ScriptedProvider([]), registry=ToolRegistry())
    base = list(agent.history)
    node = agent.sessions.commit(Snapshot(history=base), label="snap")
    agent.history = base + [Message.user("drifted")]

    applied = agent.restore_workspace(node.id)
    assert agent.history == base
    assert applied == 0
def test_agent_persists_via_saver_after_turn_and_reset():
    """A saver-equipped Agent calls it after each tree mutation (commit / reset).

    We do not touch disk here -- a list-appending saver proves the wiring: one
    call per committed turn, plus one when ``reset`` forks a fresh root.
    """
    saved = []
    provider = ScriptedProvider(["<message>hi</message>"])
    agent = Agent(
        Config(), provider=provider, registry=ToolRegistry(),
        session_saver=lambda tree: saved.append(len(tree)),
    )
    agent.run_turn("hello")
    assert saved, "saver was not called after a turn"
    after_turn = saved[-1]

    agent.reset()
    assert len(saved) >= 2  # reset persisted too
    # reset forks a new child off the root, so the tree only grows.
    assert saved[-1] >= after_turn


def test_agent_without_saver_does_not_crash():
    """No saver injected (the headless/test default) -> turns still work."""
    provider = ScriptedProvider(["<message>ok</message>"])
    agent = Agent(Config(), provider=provider, registry=ToolRegistry())
    resp = agent.run_turn("hi")
    assert resp.content == "ok"


def test_agent_persists_after_checkout_session():
    """``checkout_session`` (restore) also triggers a save."""
    saved = []
    provider = ScriptedProvider(["<message>one</message>", "<message>two</message>"])
    agent = Agent(
        Config(), provider=provider, registry=ToolRegistry(),
        session_saver=lambda tree: saved.append(tree.head()),
    )
    agent.run_turn("first")
    first_head = agent.sessions.head()
    agent.run_turn("second")
    saved.clear()
    agent.checkout_session(first_head)
    assert saved, "checkout did not persist"
    assert saved[-1] == first_head
