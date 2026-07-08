"""Configuration for the agent.

Loads settings from environment variables (optionally via a ``.env`` file that
we parse ourselves, again to avoid extra dependencies). Defaults point at the
local OpenAI-compatible endpoint you described.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


def _load_dotenv(path: Path) -> None:
    """Very small ``.env`` parser: ``KEY=VALUE`` lines, ``#`` comments.

    Only sets variables that are not already present in the environment, so
    real environment variables always win.
    """

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass
class Config:
    """Runtime configuration for talking to an LLM provider."""

    provider: str = "openai"
    base_url: str = "http://127.0.0.1:23333/api/openai/v1"
    api_key: str = "not-needed"
    model: str = "claude-opus-4.8"
    system_prompt: str = "You are a helpful assistant."
    timeout: float = 60.0

    @classmethod
    def from_env(cls, dotenv_path: str = ".env") -> "Config":
        _load_dotenv(Path(dotenv_path))
        return cls(
            provider=os.environ.get("LLM_PROVIDER", cls.provider),
            base_url=os.environ.get("LLM_BASE_URL", cls.base_url),
            api_key=os.environ.get("LLM_API_KEY", cls.api_key),
            model=os.environ.get("LLM_MODEL", cls.model),
            system_prompt=os.environ.get("LLM_SYSTEM_PROMPT", cls.system_prompt),
            timeout=float(os.environ.get("LLM_TIMEOUT", cls.timeout)),
        )

    def auth_headers(self) -> Dict[str, str]:
        """Default bearer-token style auth. Providers may override this."""

        return {"Authorization": f"Bearer {self.api_key}"}
