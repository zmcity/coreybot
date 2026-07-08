"""Unit tests for the structured XML-tag protocol (``coreybot.llm.protocol``).

These pin down two things:
1. ``build_system_prompt`` injects the protocol instructions.
2. ``extract_tag`` / ``parse_agent_response`` are robust: they pull the
   ``<message>`` content out of clean, multi-line, attribute-tagged, and
   prose-wrapped output, and degrade gracefully (never raise) when the tag is
   missing.
"""

from __future__ import annotations

from coreybot.llm.protocol import (
    AgentResponse,
    PROTOCOL_INSTRUCTIONS,
    ResponseType,
    build_system_prompt,
    extract_tag,
    parse_agent_response,
)


def test_build_system_prompt_appends_instructions():
    out = build_system_prompt("You are nice.")
    assert "You are nice." in out
    assert PROTOCOL_INSTRUCTIONS.strip() in out


def test_build_system_prompt_handles_empty_base():
    out = build_system_prompt("   ")
    assert out == PROTOCOL_INSTRUCTIONS.strip()


def test_extract_tag_basic():
    assert extract_tag("<message>hi</message>", "message") == "hi"


def test_extract_tag_is_case_insensitive():
    assert extract_tag("<MESSAGE>hey</MESSAGE>", "message") == "hey"


def test_extract_tag_multiline():
    text = "<message>line1\nline2\n  line3</message>"
    assert extract_tag(text, "message") == "line1\nline2\n  line3"


def test_extract_tag_ignores_attributes():
    assert extract_tag('<message id="1">yo</message>', "message") == "yo"


def test_extract_tag_returns_none_when_absent():
    assert extract_tag("no tags here", "message") is None


def test_extract_tag_first_match_only():
    text = "<message>a</message><message>b</message>"
    assert extract_tag(text, "message") == "a"


def test_parse_message_clean():
    r = parse_agent_response("<message>hello</message>")
    assert isinstance(r, AgentResponse)
    assert r.is_message
    assert r.content == "hello"
    assert r.parse_error is None
    assert r.raw == {"message": "hello"}


def test_parse_message_with_surrounding_prose():
    text = "Sure!\n<message>the answer</message>\nHope that helps."
    r = parse_agent_response(text)
    assert r.content == "the answer"
    assert r.parse_error is None


def test_parse_message_preserves_special_characters():
    # No escaping needed inside tags: quotes, braces, ampersands, code.
    body = 'use {curly}, "quotes", & <=> operators\nprint("hi")'
    r = parse_agent_response(f"<message>{body}</message>")
    assert r.content == body


def test_parse_missing_tag_falls_back_to_plain_text():
    r = parse_agent_response("just plain text, no tags")
    assert r.is_message
    assert r.content == "just plain text, no tags"
    assert r.parse_error == "no <message> or <tool_call> tag found"


def test_parse_response_type_constant():
    r = parse_agent_response("<message>x</message>")
    assert r.type == ResponseType.MESSAGE

# --- tool_call parsing ---------------------------------------------------
def test_parse_tool_call_basic():
    text = '<tool_call><name>calc</name><arguments>{"expression": "2+2"}</arguments></tool_call>'
    r = parse_agent_response(text)
    assert r.is_tool_call
    assert r.tool_name == "calc"
    assert r.tool_arguments == {"expression": "2+2"}
    assert r.parse_error is None


def test_parse_tool_call_multiline_and_spacing():
    text = """
    <tool_call>
      <name>read_file</name>
      <arguments>{"path": "a.txt"}</arguments>
    </tool_call>
    """
    r = parse_agent_response(text)
    assert r.is_tool_call
    assert r.tool_name == "read_file"
    assert r.tool_arguments == {"path": "a.txt"}


def test_parse_tool_call_no_arguments_defaults_empty():
    text = "<tool_call><name>current_time</name></tool_call>"
    r = parse_agent_response(text)
    assert r.is_tool_call
    assert r.tool_name == "current_time"
    assert r.tool_arguments == {}


def test_parse_tool_call_invalid_json_records_error():
    text = "<tool_call><name>calc</name><arguments>{bad json}</arguments></tool_call>"
    r = parse_agent_response(text)
    assert r.is_tool_call
    assert r.tool_name == "calc"
    assert r.parse_error is not None
    assert "invalid JSON" in r.parse_error


def test_parse_tool_call_missing_name():
    text = "<tool_call><arguments>{}</arguments></tool_call>"
    r = parse_agent_response(text)
    assert r.is_tool_call
    assert r.parse_error == "tool_call missing <name>"


def test_first_tag_by_position_wins():
    # tool_call opens before message -> treated as a tool call.
    text = "<tool_call><name>x</name></tool_call> then <message>hi</message>"
    r = parse_agent_response(text)
    assert r.is_tool_call


def test_build_system_prompt_appends_tool_catalog():
    out = build_system_prompt("Base.", "You can call the following tools:\n- calc(...)")
    assert "Base." in out
    assert "<tool_call>" in out
    assert "calc(" in out