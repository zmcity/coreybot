"""Unified message representation shared across all provider protocols.

We deliberately define our own tiny data structures instead of relying on any
vendor SDK. Every provider adapter converts *to* and *from* these types, so the
rest of the framework never has to care whether we are talking to OpenAI,
Anthropic, or Gemini.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class Role(str, Enum):
    """Who produced a message.

    Subclassing ``str`` means a ``Role`` is also a plain string, which makes it
    trivial to serialize to JSON (``role.value`` == the wire format string used
    by OpenAI-compatible APIs).
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single chat message in our provider-neutral format."""

    role: Role
    content: str
    # Free-form bag for provider-specific extras (tool calls, names, etc.).
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role.value, "content": self.content}

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(Role.SYSTEM, content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(Role.USER, content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(Role.ASSISTANT, content)


@dataclass
class CompletionResult:
    """Normalized result returned by every provider.

    ``raw`` keeps the untouched provider payload so you can inspect protocol
    differences while learning.
    """

    text: str
    model: str
    raw: Dict[str, Any] = field(default_factory=dict)


def messages_to_dicts(messages: List[Message]) -> List[Dict[str, Any]]:
    return [m.to_dict() for m in messages]
