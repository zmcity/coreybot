"""Resolve the on-disk home directory for the agent (Codex-style layout).

The agent keeps all of its durable state under a single home directory,
mirroring how Codex uses ``~/.codex`` (selectable via ``CODEX_HOME``). Here the
default is ``~/.coreybot`` -- built at *runtime* from a static dirname joined
onto the user's home -- and it can be overridden two ways, in priority order:

1. an explicit path passed in code (``resolve_home(override=...)``), then
2. the ``COREYBOT_HOME`` environment variable, then
3. the default ``Path.home() / ".coreybot"``.

The layout under the home directory (also Codex-shaped)::

    <home>/
      config.toml                              # user-editable settings
      version.json                             # what wrote this dir + when
      history.jsonl                            # cross-session prompt history
      sessions/YYYY/MM/DD/rollout-<ts>-<id>.jsonl   # one file per session
      logs/

Nothing here creates directories on import: paths are pure values. Call
:meth:`AgentPaths.ensure` (after confirming with the user) to materialize the
tree. This module has no third-party dependencies.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Environment variable that overrides the home directory (cf. ``CODEX_HOME``).
HOME_ENV = "COREYBOT_HOME"
# The default home is this dirname under the user's home directory. Kept as a
# static string and joined onto ``Path.home()`` at call time so the resolved
# path always tracks the *current* user rather than being frozen at import.
DEFAULT_HOME_DIRNAME = ".coreybot"


def default_home() -> Path:
    """The built-in default home: ``~/.coreybot`` (computed at call time)."""
    return Path.home() / DEFAULT_HOME_DIRNAME


def resolve_home(override=None) -> Path:
    """Return the home directory, honoring override > env var > default.

    ``override`` (an explicit path from code / a CLI flag) wins; otherwise the
    ``COREYBOT_HOME`` environment variable is used if set and non-empty;
    otherwise the default ``~/.coreybot``. The path is expanded (``~`` and
    env vars) and made absolute, but NOT created -- see ``AgentPaths.ensure``.
    """
    raw = override
    if raw is None:
        env = os.environ.get(HOME_ENV)
        raw = env if env else None
    if raw is None:
        path = default_home()
    else:
        path = Path(os.path.expandvars(os.fspath(raw))).expanduser()
    return path.absolute()


@dataclass(frozen=True)
class AgentPaths:
    """The set of durable paths under one agent home directory.

    Construct via ``resolve`` (which applies the override/env/default
    precedence). All attributes are pure path values; ``ensure`` is the only
    method that touches the filesystem.
    """

    home: Path

    @classmethod
    def resolve(cls, override=None) -> "AgentPaths":
        return cls(home=resolve_home(override))

    # --- child paths (Codex-shaped) -------------------------------------
    @property
    def config_file(self) -> Path:
        return self.home / "config.toml"

    @property
    def version_file(self) -> Path:
        return self.home / "version.json"

    @property
    def history_file(self) -> Path:
        return self.home / "history.jsonl"

    @property
    def sessions_dir(self) -> Path:
        return self.home / "sessions"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    def session_file(self, session_id: str, created=None) -> Path:
        """Path for one session's rollout file, bucketed by date like Codex.

        ``sessions/YYYY/MM/DD/rollout-<YYYYMMDDTHHMMSS>-<session_id>.jsonl``.
        ``created`` defaults to now (local time). The date buckets keep a busy
        directory from ballooning into thousands of sibling files.
        """
        when = created or datetime.now()
        bucket = self.sessions_dir / ("%04d" % when.year) / ("%02d" % when.month) / ("%02d" % when.day)
        stamp = when.strftime("%Y%m%dT%H%M%S")
        return bucket / ("rollout-%s-%s.jsonl" % (stamp, session_id))

    # --- filesystem ------------------------------------------------------
    def exists(self) -> bool:
        """True if the home directory already exists on disk."""
        return self.home.is_dir()

    def ensure(self, version=None) -> "AgentPaths":
        """Create the home directory tree (idempotent) and stamp version.json.

        Creates ``home``, ``sessions/`` and ``logs/`` if missing, then writes
        ``version.json`` recording the writer + timestamp. Safe to call every
        launch; existing files are left untouched except that ``version.json``
        is refreshed with the last-run time.
        """
        self.home.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "app": "coreybot",
            "version": version or "0",
            "home": str(self.home),
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        self.version_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return self
