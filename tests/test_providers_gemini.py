"""Protocol tests for the Gemini-compatible provider.

Gemini differs the most from OpenAI, so these tests are especially useful as
documentation:
- endpoint embeds model + API key: ``.../models/{model}:generateContent?key=``
- roles are remapped: assistant -> ``model``; there is no system role
- a system prompt goes to top-level ``systemInstruction``
- message text lives under ``parts: [{text: ...}]``
- reply text is at ``candidates[0].content.parts[*].text``
"""

from __future__ import annotations

import coreybot.llm.providers.gemini_provider as gemini_mod
from coreybot.core.config import Config
from coreybot.core.message import Message
from coreybot.llm.providers.gemini_provider import GeminiProvider


async def test_gemini_url_embeds_model_and_key(monkeypatch, sample_messages, recording_async_json):
    rec = recording_async_json(
        {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
    )
    monkeypatch.setattr(gemini_mod, "apost_json", rec)

    provider = GeminiProvider(Config(api_key="KEY", model="gemini-x"))
    result = await provider.acomplete(sample_messages)

    assert rec.url == (
        "http://127.0.0.1:23333/api/openai/v1/models/gemini-x:generateContent?key=KEY"
    )
    assert result.text == "hi"


async def test_gemini_role_mapping_and_system_instruction(monkeypatch, recording_async_json):
    rec = recording_async_json(
        {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
    )
    monkeypatch.setattr(gemini_mod, "apost_json", rec)

    messages = [
        Message.system("sys"),
        Message.user("u1"),
        Message.assistant("a1"),
    ]
    provider = GeminiProvider(Config())
    await provider.acomplete(messages)

    # system pulled to systemInstruction
    assert rec.payload["systemInstruction"]["parts"][0]["text"] == "sys"
    # assistant remapped to "model", text under parts
    assert rec.payload["contents"] == [
        {"role": "user", "parts": [{"text": "u1"}]},
        {"role": "model", "parts": [{"text": "a1"}]},
    ]


async def test_gemini_joins_parts(monkeypatch, sample_messages, recording_async_json):
    rec = recording_async_json(
        {"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]}
    )
    monkeypatch.setattr(gemini_mod, "apost_json", rec)
    provider = GeminiProvider(Config())
    assert (await provider.acomplete(sample_messages)).text == "ab"


async def test_gemini_raises_without_candidates(monkeypatch, sample_messages, recording_async_json):
    rec = recording_async_json({"candidates": []})
    monkeypatch.setattr(gemini_mod, "apost_json", rec)
    provider = GeminiProvider(Config())
    try:
        await provider.acomplete(sample_messages)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
