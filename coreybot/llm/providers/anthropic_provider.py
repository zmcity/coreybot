"""Anthropic Messages API compatible provider.

Key protocol differences vs OpenAI (this is *why* a compat layer earns its keep):
- Endpoint is ``{base_url}/messages``.
- The system prompt is a *top-level* ``system`` field, NOT a message with
  ``role: "system"``. So we split it out of the message list.
- Auth uses ``x-api-key`` + ``anthropic-version`` headers, not ``Authorization``.
- ``max_tokens`` is required.
- Response content is a list of typed blocks:
    { "content": [ { "type": "text", "text": "..." }, ... ] }
"""

from __future__ import annotations

from typing import Dict, List, Optional

from coreybot.core.cancel import CancelToken
from coreybot.core.config import Config
from coreybot.llm.http_client import apost_json
from coreybot.core.message import CompletionResult, Message, Role
from coreybot.llm.providers.base import LLMProvider, register


@register("anthropic")
class AnthropicProvider(LLMProvider):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.max_tokens = 1024
        self.anthropic_version = "2023-06-01"

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.config.api_key,
            "anthropic-version": self.anthropic_version,
        }

    async def acomplete(
        self, messages: List[Message], cancel_token: Optional[CancelToken] = None
    ) -> CompletionResult:
        url = f"{self.config.base_url.rstrip('/')}/messages"

        system_parts: List[str] = []
        chat: List[Dict[str, str]] = []
        for message in messages:
            if message.role == Role.SYSTEM:
                system_parts.append(message.content)
            else:
                chat.append({"role": message.role.value, "content": message.content})

        payload: Dict[str, object] = {
            "model": self.config.model,
            "max_tokens": self.max_tokens,
            "messages": chat,
        }
        if system_parts:
            payload["system"] = "\\n\\n".join(system_parts)

        data = await apost_json(
            url,
            payload,
            headers=self._headers(),
            timeout=self.config.timeout,
            cancel_token=cancel_token,
        )

        blocks = data.get("content") or []
        text = "".join(
            block.get("text", "")
            for block in blocks
            if block.get("type") == "text"
        )
        model = data.get("model", self.config.model)
        return CompletionResult(text=text, model=model, raw=data)
