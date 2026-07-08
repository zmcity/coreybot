"""OpenAI Chat Completions compatible provider.

Wire format (request):
    POST {base_url}/chat/completions
    { "model": ..., "messages": [{"role": ..., "content": ...}, ...] }

Wire format (response):
    { "choices": [ { "message": { "role": "assistant", "content": "..." } } ] }

Streaming (``stream: true``) instead returns SSE events shaped like:
    data: { "choices": [ { "delta": { "content": "par" } } ] }
    data: [DONE]

Your local endpoint speaks this protocol, so this adapter is the default. Both
requests are cancellable via a :class:`CancelToken`.
"""

from __future__ import annotations

import json
from typing import AsyncIterator, List, Optional

from coreybot.core.cancel import CancelToken
from coreybot.llm.http_client import apost_json, apost_sse
from coreybot.core.message import CompletionResult, Message, messages_to_dicts
from coreybot.llm.providers.base import LLMProvider, register


@register("openai")
class OpenAIProvider(LLMProvider):
    def _url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    async def acomplete(
        self, messages: List[Message], cancel_token: Optional[CancelToken] = None
    ) -> CompletionResult:
        payload = {
            "model": self.config.model,
            "messages": messages_to_dicts(messages),
        }
        data = await apost_json(
            self._url(),
            payload,
            headers=self.config.auth_headers(),
            timeout=self.config.timeout,
            cancel_token=cancel_token,
        )

        # Defensive parsing: shapes vary slightly across compatible servers.
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices in response: {data}")
        text = (choices[0].get("message") or {}).get("content", "")
        model = data.get("model", self.config.model)
        return CompletionResult(text=text, model=model, raw=data)

    async def astream(
        self, messages: List[Message], cancel_token: Optional[CancelToken] = None
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.config.model,
            "messages": messages_to_dicts(messages),
            "stream": True,
        }
        async for chunk in apost_sse(
            self._url(),
            payload,
            headers=self.config.auth_headers(),
            timeout=self.config.timeout,
            cancel_token=cancel_token,
        ):
            if chunk == "[DONE]":
                break
            try:
                event = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}).get("content")
            if delta:
                yield delta
