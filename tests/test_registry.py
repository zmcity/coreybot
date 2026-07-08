"""Unit tests for the provider registry (``coreybot.llm.providers.base``).

The registry is what makes the compat layer extensible: a name maps to a
provider class, and ``create_provider`` instantiates the one named by config.
"""

from __future__ import annotations

import pytest

from coreybot.core.config import Config
from coreybot.core.message import CompletionResult, Message
from coreybot.llm.providers import available_providers, create_provider
from coreybot.llm.providers.base import LLMProvider, register


def test_builtin_providers_registered():
    names = available_providers()
    assert {"openai", "anthropic", "gemini"} <= set(names)


def test_create_provider_returns_correct_type():
    provider = create_provider(Config(provider="openai"))
    from coreybot.llm.providers.openai_provider import OpenAIProvider

    assert isinstance(provider, OpenAIProvider)


def test_create_provider_is_case_insensitive():
    provider = create_provider(Config(provider="OpenAI"))
    assert isinstance(provider, LLMProvider)


def test_unknown_provider_raises_keyerror():
    with pytest.raises(KeyError) as excinfo:
        create_provider(Config(provider="does-not-exist"))
    # Error message should list available providers to aid debugging.
    assert "Available" in str(excinfo.value)


def test_duplicate_registration_raises():
    with pytest.raises(ValueError):

        @register("openai")  # already taken
        class _Dupe(LLMProvider):
            def complete(self, messages):
                return CompletionResult(text="", model="")


def test_default_stream_falls_back_to_complete():
    # A provider that only implements acomplete() should still stream() as a
    # single chunk, thanks to the base-class default astream().
    @register("unit-fallback")
    class OneShot(LLMProvider):
        async def acomplete(self, messages, cancel_token=None):
            return CompletionResult(text="one shot", model="m")

    provider = OneShot(Config(provider="unit-fallback"))
    chunks = list(provider.stream([Message.user("hi")]))
    assert chunks == ["one shot"]
