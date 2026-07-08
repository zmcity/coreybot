"""Unit tests for configuration loading (``coreybot.core.config``).

Covers the custom ``.env`` parser and precedence rules: real environment
variables must win over ``.env`` file values, and explicit ``.env`` values win
over dataclass defaults.

Note: these use the project-local ``local_tmp_path`` fixture instead of pytest's
built-in ``tmp_path`` because this machine's system ``%TEMP%`` is locked down by
security software (see conftest for details).
"""

from __future__ import annotations

from coreybot.core.config import Config


def test_defaults_point_at_local_endpoint():
    cfg = Config()
    assert cfg.provider == "openai"
    assert cfg.base_url == "http://127.0.0.1:23333/api/openai/v1"
    assert cfg.model == "claude-opus-4.8"


def test_auth_headers_use_bearer():
    cfg = Config(api_key="secret")
    assert cfg.auth_headers() == {"Authorization": "Bearer secret"}


def test_from_env_reads_dotenv(local_tmp_path, monkeypatch):
    # Ensure no real env vars interfere with this test.
    for key in ["LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY"]:
        monkeypatch.delenv(key, raising=False)

    env_file = local_tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# a comment",
                'LLM_PROVIDER=anthropic',
                'LLM_MODEL="claude-x"',
                "LLM_BASE_URL=http://example/v1",
                "not_a_pair_line",
            ]
        ),
        encoding="utf-8",
    )

    cfg = Config.from_env(dotenv_path=str(env_file))
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-x"  # surrounding quotes stripped
    assert cfg.base_url == "http://example/v1"


def test_real_env_overrides_dotenv(local_tmp_path, monkeypatch):
    env_file = local_tmp_path / ".env"
    env_file.write_text("LLM_MODEL=from-file", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "from-env")

    cfg = Config.from_env(dotenv_path=str(env_file))
    assert cfg.model == "from-env"


def test_from_env_missing_file_uses_defaults(local_tmp_path, monkeypatch):
    for key in ["LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    missing = local_tmp_path / "does-not-exist.env"
    cfg = Config.from_env(dotenv_path=str(missing))
    assert cfg.provider == "openai"