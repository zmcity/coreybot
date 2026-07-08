"""Integration tests for the plain line-based loop (``coreybot.frontends.chat_loop``).

The loop now delegates to ``Agent``. We patch ``create_provider`` in the agent
module so the Agent picks up our fake provider, feed scripted user lines via
``input``, capture ``print`` output, and assert on behavior. No network.
"""

from __future__ import annotations

import builtins

import pytest

import coreybot.runtime.agent as agent_mod
import coreybot.frontends.chat_loop as chat_loop
from coreybot.core.config import Config
from coreybot.core.message import Role


class _ScriptedInput:
    """Returns queued lines, then raises EOFError to end the loop."""

    def __init__(self, lines):
        self._lines = list(lines)

    def __call__(self, prompt=""):
        if self._lines:
            return self._lines.pop(0)
        raise EOFError


def _run_loop_with(monkeypatch, lines, provider):
    monkeypatch.setattr(builtins, "input", _ScriptedInput(lines))
    # The Agent builds its provider via create_provider(); inject the fake.
    monkeypatch.setattr(agent_mod, "create_provider", lambda config: provider)
    printed = []
    monkeypatch.setattr(
        builtins, "print", lambda *a, **k: printed.append(" ".join(str(x) for x in a))
    )
    chat_loop.run_chat_loop(Config())
    return printed


def test_render_reply_plaintext_preserves_content(monkeypatch):
    # Non-TTY path: Markdown is rendered but literal tokens survive for reading.
    monkeypatch.setattr(chat_loop.sys.stdout, "isatty", lambda: False, raising=False)
    out = chat_loop._render_reply("use `nine` and:\n```python\nprint(1)\n```")
    assert "nine" in out
    assert "print(1)" in out
    # No ANSI escapes when not a terminal.
    assert "\x1b[" not in out


def test_render_reply_falls_back_when_rich_missing(monkeypatch):
    # Simulate Rich being unavailable -> return the raw text unchanged.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("rich"):
            raise ImportError("no rich")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    raw = "**bold** and `code`"
    assert chat_loop._render_reply(raw) == raw


def test_loop_echoes_reply_and_keeps_memory(monkeypatch, fake_stream_provider):
    provider = fake_stream_provider(replies=["<message>Hello!</message>"])
    printed = _run_loop_with(monkeypatch, ["hi", "/exit"], provider)

    joined = "\n".join(printed)
    assert "Hello!" in joined
    # provider saw a history that included the user message
    assert provider.seen_messages
    last_seen = provider.seen_messages[-1]
    assert last_seen[-1].role is Role.USER
    assert last_seen[-1].content == "hi"


def test_loop_runs_a_tool_then_answers(monkeypatch, fake_stream_provider):
    provider = fake_stream_provider(
        replies=[
            '<tool_call><name>calc</name><arguments>{"expression": "3*3"}</arguments></tool_call>',
            "<message>nine</message>",
        ]
    )
    printed = _run_loop_with(monkeypatch, ["3*3?", "/exit"], provider)
    joined = "\n".join(printed)
    assert "calling calc" in joined
    assert "nine" in joined


def test_loop_reset_clears_memory(monkeypatch, fake_stream_provider):
    provider = fake_stream_provider(replies=["<message>ok</message>", "<message>again-ok</message>"])
    printed = _run_loop_with(monkeypatch, ["hi", "/reset", "again", "/exit"], provider)
    joined = "\n".join(printed)
    assert "memory cleared" in joined
    # After reset, the next turn should only carry system + the new user msg.
    second_call = provider.seen_messages[-1]
    assert [m.role for m in second_call] == [Role.SYSTEM, Role.USER]


class _InterruptingProvider:
    """Async provider whose first turn raises CancelledError (as if Ctrl+C)."""

    def __init__(self):
        self.seen_messages = []

    async def acomplete(self, messages, cancel_token=None):
        from coreybot.core.cancel import CancelledError

        self.seen_messages.append(list(messages))
        raise CancelledError("interrupted")


def test_loop_reports_interrupted_turn_without_crashing(monkeypatch):
    provider = _InterruptingProvider()
    printed = _run_loop_with(monkeypatch, ["hi", "/exit"], provider)
    joined = "\n".join(printed)
    assert "interrupted" in joined
    # After an interrupt the loop keeps running and exits cleanly on /exit.
    assert "bye!" in joined


def test_loop_recovers_from_provider_error(monkeypatch, fake_stream_provider):
    provider = fake_stream_provider(error=RuntimeError("api down"))
    printed = _run_loop_with(monkeypatch, ["hi", "/exit"], provider)
    joined = "\n".join(printed)
    assert "error" in joined.lower()
    assert "api down" in joined