"""Unit tests for JSONL session persistence (``coreybot.runtime.session_store``).

The store must be LOSSLESS: saving a tree and loading it back reproduces the
exact node set, parent/child links, per-actor heads, roots and -- crucially --
the id counter, so a post-load ``commit`` mints a fresh, non-colliding id
(this is what keeps "restore to node" stable). History, telemetry and the
artifacts bag ride along on each snapshot. Files are written under a repo-local
temp dir because the system %TEMP% is locked on this machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coreybot.core.message import Message, Role
from coreybot.runtime.agent import AgentEvent
from coreybot.runtime.session import SessionTree, Snapshot
from coreybot.runtime.session_store import (
    FORMAT_VERSION,
    load_tree,
    save_tree,
    tree_from_records,
    tree_to_records,
)


def _snap(*contents, telemetry=None, artifacts=None):
    return Snapshot(
        history=[Message.user(c) for c in contents],
        telemetry=list(telemetry or []),
        artifacts=dict(artifacts or {}),
    )


def _sample_tree():
    tree = SessionTree(_snap("sys"))
    a = tree.commit(_snap("sys", "q1"), label="q1")
    tree.commit(_snap("sys", "q1", "q2"), label="q2")
    # Branch off ``a`` to create a fork (two children under one node).
    tree.checkout(a.id)
    tree.commit(_snap("sys", "q1", "q2b"), label="q2b")
    # A second actor commits on its own head off the root.
    tree.checkout(tree.root_id, actor="worker")
    tree.commit(_snap("sys", "w1"), actor="worker", label="w1")
    return tree


def _structure(tree):
    return {
        "ids": sorted(n.id for n in (tree.get(i) for i in _all_ids(tree)) if n),
        "parents": {i: tree.get(i).parent for i in _all_ids(tree)},
        "children": {i: sorted(tree.get(i).children) for i in _all_ids(tree)},
        "heads": tree.branch_heads(),
        "roots": tree.roots,
        "root_id": tree.root_id,
    }


def _all_ids(tree):
    return [r.node.id for r in tree.rows()]


def test_records_meta_first_then_nodes():
    tree = _sample_tree()
    records = tree_to_records(tree)
    assert records[0]["type"] == "meta"
    assert records[0]["format"] == FORMAT_VERSION
    assert all(r["type"] == "node" for r in records[1:])
    assert {r["id"] for r in records[1:]} == set(_all_ids(tree))


def test_round_trip_is_lossless():
    tree = _sample_tree()
    restored = tree_from_records(tree_to_records(tree))
    assert _structure(restored) == _structure(tree)


def test_round_trip_preserves_snapshot_payloads():
    events = [
        AgentEvent(kind="llm_call", source="llm", text="in"),
        AgentEvent(kind="tool_result", source="tool", name="calc",
                   arguments={"expression": "1+1"}, output="2", ok=True),
    ]
    tree = SessionTree()
    tree.commit(_snap("q1", telemetry=events, artifacts={"files": {"a.txt": "x"}}), label="q1")
    restored = tree_from_records(tree_to_records(tree))

    head = restored.get(restored.head())
    assert [m.content for m in head.snapshot.history] == ["q1"]
    assert head.snapshot.artifacts == {"files": {"a.txt": "x"}}
    kinds = [e.kind for e in head.snapshot.telemetry]
    assert kinds == ["llm_call", "tool_result"]
    tool_evt = head.snapshot.telemetry[1]
    assert tool_evt.name == "calc"
    assert tool_evt.arguments == {"expression": "1+1"}
    assert tool_evt.output == "2"


def test_post_load_commit_mints_fresh_id():
    tree = _sample_tree()
    restored = tree_from_records(tree_to_records(tree))
    existing = set(_all_ids(restored))
    node = restored.commit(_snap("new"), label="new")
    assert node.id not in existing


def test_save_and_load_file_round_trip(local_tmp_path):
    tree = _sample_tree()
    target = local_tmp_path / "sub" / "rollout.jsonl"
    save_tree(tree, target)
    assert target.exists()
    reloaded = load_tree(target)
    assert _structure(reloaded) == _structure(tree)


def test_saved_file_is_utf8_jsonl(local_tmp_path):
    tree = SessionTree(_snap("hi"))
    target = local_tmp_path / "r.jsonl"
    save_tree(tree, target)
    raw = target.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")  # no BOM
    lines = [l for l in raw.decode("utf-8").splitlines() if l.strip()]
    first = json.loads(lines[0])
    assert first["type"] == "meta"


def test_unknown_telemetry_degrades_to_notice():
    # A record whose telemetry payload lacks a known kind must not crash the
    # loader; it degrades to a ``notice`` with default fields.
    records = [
        {"type": "meta", "format": FORMAT_VERSION, "root_id": "n0",
         "roots": ["n0"], "heads": {"main": "n0"}, "counter": 1},
        {"type": "node", "id": "n0", "parent": None, "actor": "main",
         "label": "root", "seq": 0,
         "snapshot": {"history": [], "telemetry": [{"weird": True}], "artifacts": {}}},
    ]
    tree = tree_from_records(records)
    evt = tree.get("n0").snapshot.telemetry[0]
    assert evt.kind == "notice"
