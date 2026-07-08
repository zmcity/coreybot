"""Unit tests for the on-disk path resolver (``coreybot.core.paths``).

These pin the override/env/default precedence, the Codex-shaped child paths,
the date-bucketed session file name, and that ``ensure`` materializes the tree
plus a ``version.json``. All filesystem work happens under a repo-local temp
dir (``local_tmp_path``) because the system %TEMP% is locked on this machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coreybot.core import paths as paths_mod
from coreybot.core.paths import AgentPaths, DEFAULT_HOME_DIRNAME, HOME_ENV, default_home, resolve_home


def test_default_home_is_dirname_under_user_home():
    assert default_home() == Path.home() / DEFAULT_HOME_DIRNAME
    # Built at call time, not frozen at import: still tracks Path.home().
    assert default_home().name == DEFAULT_HOME_DIRNAME


def test_resolve_precedence_override_beats_env(monkeypatch, local_tmp_path):
    monkeypatch.setenv(HOME_ENV, str(local_tmp_path / "from_env"))
    override = local_tmp_path / "from_override"
    assert resolve_home(override) == override.absolute()


def test_resolve_uses_env_when_no_override(monkeypatch, local_tmp_path):
    target = local_tmp_path / "envhome"
    monkeypatch.setenv(HOME_ENV, str(target))
    assert resolve_home() == target.absolute()


def test_resolve_falls_back_to_default(monkeypatch):
    monkeypatch.delenv(HOME_ENV, raising=False)
    assert resolve_home() == default_home().absolute()


def test_resolve_ignores_empty_env(monkeypatch):
    monkeypatch.setenv(HOME_ENV, "")
    assert resolve_home() == default_home().absolute()


def test_resolve_expands_user_and_vars(monkeypatch):
    monkeypatch.delenv(HOME_ENV, raising=False)
    # ``~`` expands to the real home; the result is absolute.
    resolved = resolve_home("~/somewhere")
    assert resolved.is_absolute()
    assert resolved == (Path.home() / "somewhere").absolute()


def test_child_paths_are_codex_shaped(local_tmp_path):
    ap = AgentPaths.resolve(local_tmp_path)
    assert ap.config_file == ap.home / "config.toml"
    assert ap.version_file == ap.home / "version.json"
    assert ap.history_file == ap.home / "history.jsonl"
    assert ap.sessions_dir == ap.home / "sessions"
    assert ap.logs_dir == ap.home / "logs"


def test_session_file_is_date_bucketed(local_tmp_path):
    from datetime import datetime

    ap = AgentPaths.resolve(local_tmp_path)
    when = datetime(2024, 3, 7, 13, 5, 9)
    path = ap.session_file("abc123", when)
    assert path == ap.sessions_dir / "2024" / "03" / "07" / "rollout-20240307T130509-abc123.jsonl"


def test_exists_and_ensure_creates_tree(local_tmp_path):
    ap = AgentPaths.resolve(local_tmp_path / "home")
    assert not ap.exists()
    ap.ensure(version="1.2.3")
    assert ap.exists()
    assert ap.sessions_dir.is_dir()
    assert ap.logs_dir.is_dir()
    payload = json.loads(ap.version_file.read_text(encoding="utf-8"))
    assert payload["app"] == "coreybot"
    assert payload["version"] == "1.2.3"
    assert payload["home"] == str(ap.home)
    assert "updated_at" in payload


def test_ensure_is_idempotent(local_tmp_path):
    ap = AgentPaths.resolve(local_tmp_path / "home")
    ap.ensure()
    # Drop a user file, re-ensure, confirm it survives (only version.json is rewritten).
    marker = ap.home / "config.toml"
    marker.write_text("keep me", encoding="utf-8")
    ap.ensure()
    assert marker.read_text(encoding="utf-8") == "keep me"
