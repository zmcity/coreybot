"""Persist a :class:`SessionTree` to disk as a JSONL rollout (Codex-style).

Codex stores each session as a line-delimited JSON "rollout" file. We do the
same: one file per :class:`SessionTree`, whose FIRST line is a ``meta`` record
(format version, tree bookkeeping) and whose remaining lines are one ``node``
record per commit. Loading rebuilds the tree's internal state EXACTLY (same
ids, heads, roots and id counter) rather than replaying ``commit`` -- replaying
would re-mint ids and could change structure, breaking "restore to node".

Why a separate module
---------------------
:class:`SessionTree` stays a pure in-memory data structure (no I/O, no JSON).
All serialization lives here so the tree has zero persistence coupling. This
module knows how to turn the concrete payloads a snapshot carries -- provider
neutral :class:`Message` history and :class:`AgentEvent` telemetry -- into
plain dicts and back. Unknown telemetry payloads degrade to a ``notice`` event
so an older/newer file never crashes the loader.

The on-disk shape is intentionally simple and greppable::

    {"type": "meta", "format": 1, "root_id": "n0", "roots": ["n0"], ...}
    {"type": "node", "id": "n0", "parent": null, "actor": "main", ...}
    {"type": "node", "id": "n1", "parent": "n0", ...}
"""

from __future__ import annotations

import io
import itertools
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from coreybot.core.message import Message, Role
from coreybot.runtime.agent import AgentEvent
from coreybot.runtime.session import SessionNode, SessionTree, Snapshot

# Bump when the on-disk record shape changes incompatibly.
FORMAT_VERSION = 1


# --- payload (de)serialization -----------------------------------------
def _message_to_dict(message: Message) -> dict:
    return {
        "role": message.role.value,
        "content": message.content,
        "metadata": message.metadata or {},
    }


def _message_from_dict(data: dict) -> Message:
    role = Role(data.get("role", "user"))
    return Message(
        role=role,
        content=data.get("content", ""),
        metadata=dict(data.get("metadata") or {}),
    )


def _event_to_dict(event: AgentEvent) -> dict:
    return {
        "kind": event.kind,
        "source": event.source,
        "name": event.name,
        "arguments": event.arguments,
        "output": event.output,
        "ok": event.ok,
        "text": event.text,
    }


def _event_from_dict(data: dict) -> AgentEvent:
    return AgentEvent(
        kind=data.get("kind", "notice"),
        source=data.get("source", ""),
        name=data.get("name", ""),
        arguments=data.get("arguments"),
        output=data.get("output", ""),
        ok=bool(data.get("ok", True)),
        text=data.get("text", ""),
    )


def _snapshot_to_dict(snapshot: Snapshot) -> dict:
    return {
        "history": [_message_to_dict(m) for m in snapshot.history],
        "telemetry": [_event_to_dict(e) for e in snapshot.telemetry],
        "artifacts": snapshot.artifacts or {},
    }


def _snapshot_from_dict(data: dict) -> Snapshot:
    return Snapshot(
        history=[_message_from_dict(m) for m in data.get("history", [])],
        telemetry=[_event_from_dict(e) for e in data.get("telemetry", [])],
        artifacts=dict(data.get("artifacts") or {}),
    )


# --- tree <-> records ---------------------------------------------------
def tree_to_records(tree: SessionTree) -> List[dict]:
    """Flatten a tree into a list of JSON-able records (meta first, then nodes).

    Reads the tree's internal state under its lock so the snapshot is
    consistent. The ``counter`` is the next value the id/seq generator would
    produce, so a load can resume minting collision-free ids.
    """
    with tree._lock:  # noqa: SLF001 -- store is the tree's persistence sibling
        nodes = list(tree._nodes.values())
        # Next counter value: one past the highest number embedded in an id
        # (ids are ``n<counter>``) or a seq, whichever is larger.
        used = [0]
        for node in nodes:
            if node.id.startswith("n") and node.id[1:].isdigit():
                used.append(int(node.id[1:]) + 1)
            used.append(int(node.seq) + 1)
        meta = {
            "type": "meta",
            "format": FORMAT_VERSION,
            "app": "coreybot",
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "root_id": tree._root_id,
            "roots": list(tree._roots),
            "heads": dict(tree._heads),
            "counter": max(used),
        }
        records: List[dict] = [meta]
        for node in nodes:
            records.append({
                "type": "node",
                "id": node.id,
                "parent": node.parent,
                "actor": node.actor,
                "label": node.label,
                "seq": node.seq,
                "children": list(node.children),
                "snapshot": _snapshot_to_dict(node.snapshot),
            })
        return records


def tree_from_records(records: List[dict]) -> SessionTree:
    """Rebuild a :class:`SessionTree` from :func:`tree_to_records` output.

    Reconstructs the exact node set, parent/child links, per-actor heads,
    roots and id counter. Node ``children`` are rebuilt from parents so the
    stored order is authoritative and any inconsistency self-heals.
    """
    meta = {}
    node_records: List[dict] = []
    for record in records:
        if record.get("type") == "meta":
            meta = record
        elif record.get("type") == "node":
            node_records.append(record)

    tree = SessionTree.__new__(SessionTree)  # bypass __init__ (which seeds a root)
    import threading

    tree._lock = threading.RLock()
    tree._nodes = {}
    tree._heads = {}

    for record in node_records:
        node = SessionNode(
            id=record["id"],
            parent=record.get("parent"),
            actor=record.get("actor", "main"),
            label=record.get("label", ""),
            seq=int(record.get("seq", 0)),
            snapshot=_snapshot_from_dict(record.get("snapshot") or {}),
            children=[],
        )
        tree._nodes[node.id] = node

    # Rebuild children from parent links (stored order preserved when present).
    stored_children = {r["id"]: list(r.get("children") or []) for r in node_records}
    for node_id, node in tree._nodes.items():
        kids = [c for c in stored_children.get(node_id, []) if c in tree._nodes]
        seen = set(kids)
        for other in tree._nodes.values():
            if other.parent == node_id and other.id not in seen:
                kids.append(other.id)
                seen.add(other.id)
        node.children = kids

    roots = [r for r in (meta.get("roots") or []) if r in tree._nodes]
    if not roots:
        roots = [n.id for n in tree._nodes.values() if n.parent is None]
    tree._roots = roots
    tree._root_id = meta.get("root_id") or (roots[0] if roots else None)

    heads = {a: h for a, h in (meta.get("heads") or {}).items() if h in tree._nodes}
    if not heads:
        heads = {"main": tree._root_id}
    tree._heads = heads

    start = int(meta.get("counter", len(tree._nodes)))
    tree._counter = itertools.count(start)
    return tree


# --- file I/O -----------------------------------------------------------
def save_tree(tree: SessionTree, path) -> Path:
    """Write ``tree`` to ``path`` as JSONL, creating parent dirs as needed."""
    target = Path(os.fspath(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(rec, ensure_ascii=False) for rec in tree_to_records(tree)]
    with io.open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return target


def load_tree(path) -> SessionTree:
    """Read a JSONL rollout written by :func:`save_tree` back into a tree."""
    target = Path(os.fspath(path))
    records: List[dict] = []
    with io.open(target, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return tree_from_records(records)
