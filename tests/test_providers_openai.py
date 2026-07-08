"""Protocol tests for the OpenAI-compatible provider.

We replace ``apost_json`` / ``apost_sse`` inside the provider module so no socket
is used. The assertions document the OpenAI wire contract:
- endpoint is ``{base_url}/chat/completions``
- messages are passed through as-is (system stays a message)
- reply text is at ``choices[0].message.content``
- streaming reads ``choices[0].delta.content`` and stops on ``[DONE]``
"""

from __future__ import annotations

import coreybot.llm.providers.openai_provider as openai_mod
from coreybot.core.config import Config
from coreybot.llm.providers.openai_provider import OpenAIProvider


async def test_openai_complete_request_and_parse(monkeypatch, sample_messages, recording_async_json):
    rec = recording_async_json(
        {"model": "srv-model", "choices": [{"message": {"content": "hi there"}}]}
    )
    monkeypatch.setattr(openai_mod, "apost_json", rec)

    provider = OpenAIProvider(Config(api_key="tok"))
    result = await provider.acomplete(sample_messages)

    # URL + payload contract
    assert rec.url == "http://127.0.0.1:23333/api/openai/v1/chat/completions"
    assert rec.payload["model"] == "claude-opus-4.8"
    assert rec.payload["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert rec.headers == {"Authorization": "Bearer tok"}

    # Response parsing
    assert result.text == "hi there"
    assert result.model == "srv-model"
    assert result.raw["choices"][0]["message"]["content"] == "hi there"


async def test_openai_complete_raises_without_choices(monkeypatch, sample_messages, recording_async_json):
    rec = recording_async_json({"choices": []})
    monkeypatch.setattr(openai_mod, "apost_json", rec)
    provider = OpenAIProvider(Config())
    try:
        await provider.acomplete(sample_messages)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


async def test_openai_stream_sets_flag_and_parses_deltas(monkeypatch, sample_messages, recording_async_sse):
    rec = recording_async_sse(
        [
            '{"choices":[{"delta":{"content":"Hel"}}]}',
            '{"choices":[{"delta":{"content":"lo"}}]}',
            '{"choices":[{"delta":{}}]}',   # no content -> skipped
            "not-json",                       # malformed -> skipped
            "[DONE]",
        ]
    )
    monkeypatch.setattr(openai_mod, "apost_sse", rec)

    provider = OpenAIProvider(Config())
    out = "".join([chunk async for chunk in provider.astream(sample_messages)])

    assert out == "Hello"
    assert rec.payload["stream"] is True  # streaming flag set on the request
