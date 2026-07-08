"""Tie the on-disk home directory to a live session tree.

This is the small glue layer between :mod:`coreybot.core.paths` (where things
live) and :mod:`coreybot.runtime.session_store` (how a tree is written), plus
the FIRST-LAUNCH confirmation. The entry point calls :func:`open_session` once
at startup to:

1. resolve the home directory (override > ``COREYBOT_HOME`` > ``~/.coreybot``);
2. if it does not exist yet, ASK the user before creating it (a plain y/N on a
   TTY; auto-create when stdin is not interactive, e.g. tests / pipes, so a
   non-interactive run never hangs);
3. ensure the directory tree + ``version.json``;
4. pick a fresh session (rollout) file for this run;
5. hand back a :class:`SessionHandle` whose ``saver`` the :class:`Agent`
   calls after every commit to keep the rollout current.

Nothing here imports the TUI, so it works for both the CLI and the TUI before
the full-screen app takes over the terminal.
"""

from __future__ import annotations

import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from coreybot.core.message import Message, Role
from coreybot.core.paths import AgentPaths
from coreybot.runtime.session import SessionTree
from coreybot.runtime.session_store import load_tree, save_tree


def _is_interactive() -> bool:
    """True only when we can actually prompt the user (both ends are a TTY)."""
    try:
        return bool(sys.stdin and sys.stdin.isatty() and sys.stdout and sys.stdout.isatty())
    except Exception:
        return False


def confirm_create_home(
    paths: AgentPaths,
    *,
    assume_yes: bool = False,
    input_fn: "Optional[Callable[[str], str]]" = None,
    output_fn: "Optional[Callable[[str], None]]" = None,
) -> bool:
    """Ask whether to create the (missing) home directory; return the decision.

    Returns ``True`` to proceed with creation. When ``assume_yes`` is set, or
    when the session is non-interactive, it returns ``True`` without asking (so
    tests / pipes never block). On a TTY it prints a one-line y/N prompt
    defaulting to YES. ``input_fn``/``output_fn`` are injectable for tests.
    """
    # Auto-create (no prompt) only when we truly cannot ask: not forced-yes and
    # not interactive AND no explicit input_fn was injected. An injected
    # input_fn means the caller wants the prompt (e.g. tests), so honor it even
    # off a TTY.
    if assume_yes:
        return True
    if input_fn is None and not _is_interactive():
        return True
    ask = input_fn or input
    say = output_fn or (lambda text: print(text))
    say(f"First run: create the coreybot home directory at {paths.home}?")
    try:
        answer = ask("Create it now? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("", "y", "yes")


@dataclass
class SessionHandle:
    """A resolved home + the live tree + a saver bound to this run's file.

    ``saver`` is what the agent calls after each mutation; ``session_file`` is
    where it writes. ``created`` is ``False`` when the home already existed.
    """

    paths: AgentPaths
    tree: Optional[SessionTree]
    session_file: Optional[Path]
    saver: Optional[Callable[[SessionTree], None]]
    created: bool

    @property
    def home(self) -> Path:
        return self.paths.home


def _make_saver(session_file: Path) -> Callable[[SessionTree], None]:
    def _save(tree: SessionTree) -> None:
        save_tree(tree, session_file)

    return _save


def open_session(
    override: "str | Path | None" = None,
    *,
    version: str = "0",
    assume_yes: bool = False,
    resume: "str | Path | None" = None,
    input_fn: "Optional[Callable[[str], str]]" = None,
    output_fn: "Optional[Callable[[str], None]]" = None,
) -> SessionHandle:
    """Resolve the home, confirm/ensure it, and return a persistence handle.

    If the home directory is missing the user is asked first (see
    :func:`confirm_create_home`); declining returns a handle with NO saver, so
    the app still runs but keeps nothing on disk. ``resume`` (a rollout path)
    loads an existing tree to continue it; otherwise a brand-new session file
    is allocated and the agent seeds its own root.
    """
    paths = AgentPaths.resolve(override)
    if not paths.exists():
        if not confirm_create_home(
            paths, assume_yes=assume_yes, input_fn=input_fn, output_fn=output_fn
        ):
            return SessionHandle(
                paths=paths, tree=None, session_file=None, saver=None, created=False
            )
    paths.ensure(version=version)

    if resume is not None:
        session_file = Path(resume)
        tree = load_tree(session_file) if session_file.exists() else None
    else:
        session_id = uuid.uuid4().hex[:12]
        session_file = paths.session_file(session_id, datetime.now())
        tree = None

    return SessionHandle(
        paths=paths,
        tree=tree,
        session_file=session_file,
        saver=_make_saver(session_file),
        created=True,
    )
# --- cross-session discovery (the "all sessions" browser) --------------
# A rollout file is named ``rollout-<YYYYMMDDTHHMMSS>-<id>.jsonl`` (see
# ``AgentPaths.session_file``); this regex pulls the timestamp + id back out
# so the browser can label a session without opening it.
_ROLLOUT_RE = re.compile(
    r"^rollout-(?P<stamp>\d{8}T\d{6})-(?P<id>.+)\.jsonl$"
)


@dataclass
class SessionInfo:
    """One row in the global session list -- cheap metadata about a rollout.

    Built from the file name (``created``/``session_id``) plus a light peek at
    the file for a human ``title`` (the first user line) and ``message_count``.
    ``is_current`` marks the rollout this run is writing to, so the browser can
    flag "this is the session you are in".
    """

    path: Path
    session_id: str
    created: Optional[datetime]
    title: str
    message_count: int
    is_current: bool = False

    @property
    def created_text(self) -> str:
        return self.created.strftime("%Y-%m-%d %H:%M:%S") if self.created else "?"


def _parse_rollout_name(path: Path):
    """Return ``(created, session_id)`` parsed from a rollout file name.

    Falls back to the file's mtime + stem when the name does not match the
    expected pattern, so hand-dropped or renamed files still list.
    """
    match = _ROLLOUT_RE.match(path.name)
    if match:
        try:
            created = datetime.strptime(match.group("stamp"), "%Y%m%dT%H%M%S")
        except ValueError:
            created = None
        return created, match.group("id")
    try:
        created = datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        created = None
    return created, path.stem


def _first_user_line(messages: "List[Message]") -> str:
    """The first user message's first non-empty line -- a session's title."""
    for message in messages:
        if message.role is Role.USER:
            for line in message.content.splitlines():
                text = line.strip()
                if text:
                    return text
    return "(no messages)"


def flatten_session(path: "str | Path") -> "List[Message]":
    """Load a rollout and return its main-line conversation, oldest-first.

    The preview does NOT show branches: it walks the root -> head path of the
    ``main`` actor and concatenates each node's history *delta* (the messages
    that node added over its parent), yielding a simple time-ordered
    transcript. Corrupt/older files degrade to the head snapshot's history.
    """
    tree = load_tree(path)
    head = tree.head()
    chain = tree.path(head)
    if not chain:
        return []
    flat: "List[Message]" = []
    seen = 0
    for node in chain:
        history = list(node.snapshot.history)
        # Each node's snapshot is the FULL history at that commit, so only the
        # tail past what we already have is new (append-only conversation).
        if len(history) >= seen:
            flat.extend(history[seen:])
            seen = len(history)
        else:
            # History shrank (a reset/fork) -- restart from this snapshot.
            flat = list(history)
            seen = len(history)
    return flat


def session_summary(path: "str | Path"):
    """Return ``(title, message_count)`` for a rollout, best-effort.

    Opens the file once; any error yields a safe placeholder so a single bad
    file never breaks the whole listing.
    """
    try:
        messages = flatten_session(path)
    except Exception:
        return "(unreadable)", 0
    return _first_user_line(messages), len(messages)


def list_sessions(
    paths: AgentPaths,
    *,
    current: "str | Path | None" = None,
    limit: "int | None" = None,
) -> "List[SessionInfo]":
    """List every saved session under ``paths.sessions_dir``, newest first.

    Scans ``sessions/**/*.jsonl`` (the date-bucketed rollout tree), reads a
    little metadata from each, and sorts by creation time descending so the
    most recent work is on top. ``current`` (the rollout this run writes) is
    flagged via ``is_current``. ``limit`` caps how many are returned.
    """
    sessions_dir = paths.sessions_dir
    if not sessions_dir.is_dir():
        return []
    current_path = Path(current).absolute() if current is not None else None
    infos: "List[SessionInfo]" = []
    for file in sessions_dir.rglob("*.jsonl"):
        if not file.is_file():
            continue
        created, session_id = _parse_rollout_name(file)
        title, count = session_summary(file)
        infos.append(
            SessionInfo(
                path=file.absolute(),
                session_id=session_id,
                created=created,
                title=title,
                message_count=count,
                is_current=(current_path is not None and file.absolute() == current_path),
            )
        )
    infos.sort(key=lambda info: (info.created or datetime.min), reverse=True)
    if limit is not None:
        infos = infos[:limit]
    return infos
