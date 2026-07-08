"""Unit tests for the unified message model (``coreybot.core.message``).

These verify the tiny data structures every provider relies on:
- ``Role`` is a string enum (so it serializes cleanly to the wire format).
- ``Message`` factory helpers and ``to_dict`` produce the OpenAI-style shape.
"""

from __future__ import annotations

from coreybot.core.message import (
    CompletionResult,
    Message,
    Role,
    messages_to_dicts,
)


def test_role_is_str_enum():
    # Subclassing str means a Role IS a string equal to its wire value.
    assert Role.USER == "user"
    assert Role.SYSTEM.value == "system"
    assert isinstance(Role.ASSISTANT, str)


def test_message_factories_set_role():
    assert Message.system("s").role is Role.SYSTEM
    assert Message.user("u").role is Role.USER
    assert Message.assistant("a").role is Role.ASSISTANT


def test_message_to_dict_shape():
    msg = Message.user("hello")
    assert msg.to_dict() == {"role": "user", "content": "hello"}


def test_message_metadata_defaults_to_empty_dict():
    a = Message.user("x")
    b = Message.user("y")
    # Each message must get its own dict (no shared mutable default).
    a.metadata["k"] = 1
    assert b.metadata == {}


def test_messages_to_dicts_roundtrip():
    msgs = [Message.system("s"), Message.user("u")]
    assert messages_to_dicts(msgs) == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]


def test_completion_result_defaults():
    result = CompletionResult(text="hi", model="m")
    assert result.text == "hi"
    assert result.model == "m"
    assert result.raw == {}
