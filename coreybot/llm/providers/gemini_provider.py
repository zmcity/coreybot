"""Google Gemini (generateContent) compatible provider.

Key protocol differences vs OpenAI:
- Endpoint embeds the model + API key:
    POST {base_url}/models/{model}:generateContent?key={api_key}
- Roles are ``user`` / ``model`` (assistant -> ``model``); there is no ``system``
  role in the ``contents`` array. A system instruction goes into a top-level
  ``systemInstruction`` field instead.
- Message text lives under ``parts``:
    { "contents": [ { "role": "user", "parts": [ { "text": "..." } ] } ] }
- Response:
    { "candidates": [ { "content": { "parts": [ { "text": "..." } ] } } ] }
"""

from __future__ import annotations

from typing import Dict, List, Optional

from coreybot.core.cancel import CancelToken
from coreybot.llm.http_client import apost_json
from coreybot.core.message import CompletionResult, Message, Role
from coreybot.llm.providers.base import LLMProvider, register


# Map our unified roles to Gemini's role vocabulary.
_ROLE_MAP = {
    Role.USER: "user",
    Role.ASSISTANT: "model",
    Role.TOOL: "user",
}


@register("gemini")
class GeminiProvider(LLMProvider):
    async def acomplete(
        self, messages: List[Message], cancel_token: Optional[CancelToken] = None
    ) -> CompletionResult:
        base = self.config.base_url.rstrip("/")
        url = (
            f"{base}/models/{self.config.model}:generateContent"
            f"?key={self.config.api_key}"
        )

        system_parts: List[str] = []
        contents: List[Dict[str, object]] = []
        for message in messages:
            if message.role == Role.SYSTEM:
                system_parts.append(message.content)
                continue
            contents.append(
                {
                    "role": _ROLE_MAP.get(message.role, "user"),
                    "parts": [{"text": message.content}],
                }
            )

        payload: Dict[str, object] = {"contents": contents}
        if system_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\\n\\n".join(system_parts)}]
            }

        data = await apost_json(
            url, payload, timeout=self.config.timeout, cancel_token=cancel_token
        )

        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"No candidates in response: {data}")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        return CompletionResult(text=text, model=self.config.model, raw=data)
