"""Provider package.

Importing this package imports every adapter module, and each adapter registers
itself via the ``@register(...)`` decorator as a side effect. That is what makes
``create_provider`` aware of all protocols without a hand-maintained list.
"""

from __future__ import annotations

from coreybot.llm.providers.base import (
    LLMProvider,
    available_providers,
    create_provider,
    register,
)

# Import adapters for their registration side effects.
from coreybot.llm.providers import anthropic_provider  # noqa: F401
from coreybot.llm.providers import gemini_provider  # noqa: F401
from coreybot.llm.providers import openai_provider  # noqa: F401

__all__ = [
    "LLMProvider",
    "available_providers",
    "create_provider",
    "register",
]
