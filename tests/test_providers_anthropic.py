"""Protocol tests for the Anthropic-compatible provider.

These assert the *differences* from OpenAI that the compat layer smooths over:
- endpoint is ``{base_url}/messages``
- the system prompt is lifted to a top-level ``system`` field (not a message)
- auth uses ``x-api-key`` + ``anthropic-version`` headers
- ``max_tokens`` is always present
- reply text is joined from ``content[]`` blocks of type ``text``
"""

from __future__ import annotations

import coreybot.llm.providers.anthropic_provider as anthropic_mod
from coreybot.core.config import Config
from coreybot.core.message import Message
from coreybot.llm.providers.anthropic_provider import AnthropicProvider


async def test_anthropic_lifts_system_and_sets_headers(monkeypatch, recording_async_json):
    rec = recording_async_json(
        {"model": "claude", "content": [{"type": "text", "text": "hello"}]}
    )
    monkeypatch.setattr(anthropic_mod, "apost_json", rec)

    messages = [
        Message.system("Be nice."),
        Message.user("Hi"),
        Message.assistant("Yo"),
    ]
    provider = AnthropicProvider(Config(api_key="k"))
    result = await provider.acomplete(messages)

    assert rec.url == "http://127.0.0.1:23333/api/openai/v1/messages"
    # system is extracted, NOT in the messages array
    assert rec.payload["system"] == "Be nice."
    assert rec.payload["messages"] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Yo"},
    ]
    assert "max_tokens" in rec.payload
    assert rec.headers["x-api-key"] == "k"
    assert "anthropic-version" in rec.headers

    assert result.text == "hello"


async def test_anthropic_joins_multiple_text_blocks(monkeypatch, sample_messages, recording_async_json):
    rec = recording_async_json(
        {
            "content": [
                {"type": "text", "text": "foo "},
                {"type": "tool_use", "id": "x"},   # non-text -> ignored
                {"type": "text", "text": "bar"},
            ]
        }
    )
    monkeypatch.setattr(anthropic_mod, "apost_json", rec)
    provider = AnthropicProvider(Config())
    result = await provider.acomplete(sample_messages)
    assert result.text == "foo bar"


async def test_anthropic_no_system_when_absent(monkeypatch, recording_async_json):
    rec = recording_async_json({"content": [{"type": "text", "text": "ok"}]})
    monkeypatch.setattr(anthropic_mod, "apost_json", rec)
    provider = AnthropicProvider(Config())
    await provider.acomplete([Message.user("hi")])
    assert "system" not in rec.payload
