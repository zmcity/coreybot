"""Unit tests for the session service glue (``coreybot.runtime.session_service``).

Covers the first-launch confirmation logic (create / decline / assume-yes /
non-interactive auto-create) and ``open_session`` returning a coherent handle
in each case: a fresh session file + working saver when created, no saver when
declined, and a loaded tree when resuming. Filesystem work stays under a
repo-local temp dir since the system %TEMP% is locked on this machine.
"""

from __future__ import annotations

import pytest

from coreybot.core.message import Message
from coreybot.core.paths import AgentPaths
from coreybot.runtime.session import SessionTree, Snapshot
from coreybot.runtime.session_service import (
    SessionHandle,
    SessionInfo,
    confirm_create_home,
    flatten_session,
    list_sessions,
    open_session,
    session_summary,
)
from coreybot.runtime.session_store import save_tree


def _paths(local_tmp_path):
    return AgentPaths.resolve(local_tmp_path / "home")


def test_confirm_assume_yes_skips_prompt(local_tmp_path):
    asked = []
    ok = confirm_create_home(
        _paths(local_tmp_path), assume_yes=True,
        input_fn=lambda prompt: asked.append(prompt) or "n",
    )
    assert ok is True
    assert asked == []  # never prompted


def test_confirm_yes_answers(local_tmp_path):
    for answer in ("", "y", "yes", "Y", " Yes "):
        ok = confirm_create_home(
            _paths(local_tmp_path), input_fn=lambda prompt, a=answer: a,
            output_fn=lambda text: None,
        )
        assert ok is True, answer


def test_confirm_decline_answers(local_tmp_path):
    for answer in ("n", "no", "N", "nope"):
        ok = confirm_create_home(
            _paths(local_tmp_path), input_fn=lambda prompt, a=answer: a,
            output_fn=lambda text: None,
        )
        assert ok is False, answer


def test_confirm_eof_declines(local_tmp_path):
    def _raise(prompt):
        raise EOFError

    ok = confirm_create_home(
        _paths(local_tmp_path), input_fn=_raise, output_fn=lambda text: None
    )
    assert ok is False


def test_open_session_creates_and_persists(local_tmp_path):
    home = local_tmp_path / "home"
    handle = open_session(home, input_fn=lambda prompt: "y", output_fn=lambda t: None)
    assert isinstance(handle, SessionHandle)
    assert handle.created is True
    assert handle.home == home.absolute()
    assert handle.paths.exists()
    assert handle.saver is not None
    # The saver writes the tree to the allocated (date-bucketed) session file.
    tree = SessionTree(Snapshot(history=[]))
    handle.saver(tree)
    assert handle.session_file.exists()
    assert handle.session_file.suffix == ".jsonl"


def test_open_session_decline_makes_no_dir(local_tmp_path):
    home = local_tmp_path / "home"
    handle = open_session(home, input_fn=lambda prompt: "n", output_fn=lambda t: None)
    assert handle.created is False
    assert handle.saver is None
    assert handle.session_file is None
    assert not home.exists()  # declining must NOT create anything


def test_open_session_existing_home_no_prompt(local_tmp_path):
    home = local_tmp_path / "home"
    AgentPaths.resolve(home).ensure()
    asked = []
    handle = open_session(home, input_fn=lambda p: asked.append(p) or "n")
    # Home already exists -> no confirmation, saver present.
    assert asked == []
    assert handle.created is True
    assert handle.saver is not None


def test_open_session_resume_loads_tree(local_tmp_path):
    home = local_tmp_path / "home"
    ap = AgentPaths.resolve(home)
    ap.ensure()
    # Seed a rollout file with a small tree.
    seed = SessionTree(Snapshot(history=[]))
    seed.commit(Snapshot(history=[]), label="q1")
    rollout = ap.session_file("seed12345678")
    save_tree(seed, rollout)

    handle = open_session(home, resume=rollout, input_fn=lambda p: "n")
    assert handle.tree is not None
    assert len(handle.tree) == len(seed)
    assert handle.session_file == rollout


def test_open_session_resume_missing_file_returns_none_tree(local_tmp_path):
    home = local_tmp_path / "home"
    AgentPaths.resolve(home).ensure()
    missing = home / "sessions" / "nope.jsonl"
    handle = open_session(home, resume=missing, input_fn=lambda p: "n")
    assert handle.tree is None
    assert handle.saver is not None  # can still save a new tree there
# --- cross-session discovery (the "all sessions" browser) --------------
def _seed_session(paths, session_id, when, users):
    from datetime import datetime as _dt

    tree = SessionTree(Snapshot(history=[Message.system("sys")]))
    hist = [Message.system("sys")]
    for text in users:
        hist = hist + [Message.user(text), Message.assistant("ok:" + text)]
        tree.commit(Snapshot(history=list(hist)), label=text)
    target = paths.session_file(session_id, when)
    save_tree(tree, target)
    return target


def test_list_sessions_empty_when_no_dir(local_tmp_path):
    paths = AgentPaths.resolve(local_tmp_path / "home")
    # sessions dir does not exist yet -> empty list, no crash.
    assert list_sessions(paths) == []


def test_list_sessions_newest_first_and_current_flag(local_tmp_path):
    from datetime import datetime

    paths = AgentPaths.resolve(local_tmp_path / "home")
    paths.ensure()
    old = _seed_session(paths, "old00001", datetime(2024, 1, 1, 8, 0, 0), ["oldest"])
    mid = _seed_session(paths, "mid00002", datetime(2024, 3, 3, 9, 0, 0), ["middle"])
    new = _seed_session(paths, "new00003", datetime(2024, 5, 5, 10, 0, 0), ["newest"])

    infos = list_sessions(paths, current=mid)
    assert [i.session_id for i in infos] == ["new00003", "mid00002", "old00001"]
    assert [i.is_current for i in infos] == [False, True, False]
    # Titles come from the first user line.
    assert infos[0].title == "newest"


def test_list_sessions_limit(local_tmp_path):
    from datetime import datetime

    paths = AgentPaths.resolve(local_tmp_path / "home")
    paths.ensure()
    for n in range(5):
        _seed_session(paths, "s%08d" % n, datetime(2024, 1, 1 + n, 8, 0, 0), ["m%d" % n])
    infos = list_sessions(paths, limit=2)
    assert len(infos) == 2


def test_session_info_created_text(local_tmp_path):
    from datetime import datetime

    paths = AgentPaths.resolve(local_tmp_path / "home")
    paths.ensure()
    f = _seed_session(paths, "aaa11111", datetime(2024, 7, 8, 13, 5, 9), ["hi"])
    info = list_sessions(paths)[0]
    assert info.created_text == "2024-07-08 13:05:09"
    assert info.message_count == 3  # sys + user + assistant


def test_flatten_session_is_time_ordered(local_tmp_path):
    from datetime import datetime

    paths = AgentPaths.resolve(local_tmp_path / "home")
    paths.ensure()
    f = _seed_session(paths, "aaa11111", datetime(2024, 1, 1, 8, 0, 0),
                      ["first question", "second question"])
    flat = flatten_session(f)
    contents = [m.content for m in flat]
    # Root sys, then the two Q/A pairs in order (no tree, just the main line).
    assert contents == [
        "sys",
        "first question", "ok:first question",
        "second question", "ok:second question",
    ]


def test_session_summary_reads_title_and_count(local_tmp_path):
    from datetime import datetime

    paths = AgentPaths.resolve(local_tmp_path / "home")
    paths.ensure()
    f = _seed_session(paths, "aaa11111", datetime(2024, 1, 1, 8, 0, 0), ["hello there"])
    title, count = session_summary(f)
    assert title == "hello there"
    assert count == 3


def test_session_summary_handles_unreadable(local_tmp_path):
    bad = local_tmp_path / "broken.jsonl"
    bad.write_text("not json at all\n", encoding="utf-8")
    title, count = session_summary(bad)
    assert count == 0  # degrades safely
