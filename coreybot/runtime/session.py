"""A git-like session tree: branching, restorable snapshots of a run.

Why this exists
---------------
"Restore to here" started life as a destructive truncation (drop history +
telemetry after a point). That loses the branch you were on. Real work -- and
especially multi-agent work -- needs the *git* model instead: every restorable
point is a **commit** (a :class:`SessionNode`) in a tree; "restoring" is a
**checkout** that moves a branch head, and the branch you came from survives so
you can jump back to it (or to any other node) later.

Model
-----
- :class:`Snapshot` -- an immutable capture of the agent's state at a commit:
  the ``history`` (conversation) and ``telemetry`` (flow events) at that point,
  plus an extensible ``artifacts`` bag reserved for future file snapshots
  (edited-file contents, deletion tombstones, ...). Nothing in the tree assumes
  what an artifact *is*, so file recovery can be layered on without touching the
  tree.
- :class:`SessionNode` -- one commit: a stable ``id``, its ``parent`` id, the
  ``actor`` that made it (an agent id -- the multi-agent hook), a human
  ``label``, a monotonic ``seq``, and its ``snapshot``.
- :class:`SessionTree` -- the append-only store. Nodes are never mutated or
  removed; branching just adds a child. Each ``actor`` has its own **head**
  ref (like a git branch), so several agents can commit onto their own heads
  concurrently and share one tree without clobbering each other.

Concurrency / multi-agent
-------------------------
The tree is append-only and every write goes through :meth:`commit` /
:meth:`checkout`, which take a short lock. An ``actor`` only ever moves *its
own* head, so two agents committing at the same time produce two children of
their respective heads -- divergent branches, never a lost update. Reads
(:meth:`rows`, :meth:`path`, :meth:`branch_heads`) are snapshots of the current
node set.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from coreybot.core.message import Message

# The default actor when a single agent drives the session (multi-agent runs
# pass their own ids). Kept as a plain string so third parties can invent their
# own without touching this module.
DEFAULT_ACTOR = "main"
ROOT_LABEL = "root"
# Label for the node a ``clear`` forks off the original root (see
# ``SessionTree.new_root``). A distinct, plain string so the history-tree
# map can mark where the context was reset.
CLEAR_LABEL = "cleared"


@dataclass
class Snapshot:
    """An immutable capture of the agent's state at one commit.

    ``history``/``telemetry`` are copied on capture so later mutation of the
    live agent never rewrites history. ``artifacts`` is a free-form,
    forward-compatible bag: file-recovery will store edited-file blobs and
    deletion tombstones here (e.g. ``{"files": {...}, "deleted": [...]}``)
    without any change to the tree itself.
    """

    history: List[Message] = field(default_factory=list)
    telemetry: List[object] = field(default_factory=list)
    artifacts: Dict[str, object] = field(default_factory=dict)

    def clone(self) -> "Snapshot":
        # Shallow-copy the sequences (Messages/events are treated as immutable
        # once committed); deep-copy the artifacts bag one level so callers can
        # keep mutating their own dict safely.
        return Snapshot(
            history=list(self.history),
            telemetry=list(self.telemetry),
            artifacts=dict(self.artifacts),
        )


@dataclass
class SessionNode:
    """One commit in the session tree."""

    id: str
    parent: Optional[str]
    actor: str
    label: str
    seq: int
    snapshot: Snapshot
    children: List[str] = field(default_factory=list)

    @property
    def is_root(self) -> bool:
        return self.parent is None


class SessionTree:
    """An append-only, branching store of restorable session commits."""

    def __init__(self, snapshot: Optional[Snapshot] = None) -> None:
        self._lock = threading.RLock()
        self._counter = itertools.count()
        self._nodes: Dict[str, SessionNode] = {}
        # Per-actor branch heads (git refs). ``checkout``/``commit`` move only
        # the acting actor's head, which is what makes multi-agent safe.
        self._heads: Dict[str, str] = {}
        root = SessionNode(
            id=self._new_id(),
            parent=None,
            actor=DEFAULT_ACTOR,
            label=ROOT_LABEL,
            seq=0,
            snapshot=(snapshot.clone() if snapshot else Snapshot()),
        )
        self._nodes[root.id] = root
        self._root_id = root.id
        # Detached roots, oldest first. Normally there is just the one first
        # root: ``clear`` no longer starts a separate root but FORKS a new
        # child off this first root (see ``new_root``), so the pre-clear line
        # stays reachable/restorable as a sibling branch. The list (and the
        # multi-root walks in ``rows``/``graph_layout``) are kept so a future
        # multi-agent flow could seed additional independent roots.
        # ``root_id`` always means this ORIGINAL (first) root.
        self._roots: List[str] = [root.id]
        self._heads[DEFAULT_ACTOR] = root.id

    # --- ids ------------------------------------------------------------
    def _new_id(self) -> str:
        # Short, stable, human-scannable ids (n0, n1, ...). Not cryptographic;
        # the tree is per-session so collisions are impossible.
        return f"n{next(self._counter)}"

    # --- refs -----------------------------------------------------------
    @property
    def root_id(self) -> str:
        return self._root_id

    def head(self, actor: str = DEFAULT_ACTOR) -> str:
        """The node id this ``actor`` currently points at (its branch head)."""
        with self._lock:
            return self._heads.get(actor, self._root_id)

    def branch_heads(self) -> Dict[str, str]:
        """A copy of every actor -> head-node-id mapping (the branch refs)."""
        with self._lock:
            return dict(self._heads)

    @property
    def roots(self) -> List[str]:
        """Every root id, oldest first (normally just the original root)."""
        with self._lock:
            return list(self._roots)

    def new_root(
        self,
        snapshot: Optional[Snapshot] = None,
        actor: str = DEFAULT_ACTOR,
        label: str = CLEAR_LABEL,
    ) -> SessionNode:
        """Fork a fresh line off the original root and move the head to it.

        This backs ``clear``: it drops the conversation context WITHOUT
        discarding the tree. Rather than a detached, parentless root, the
        new node is attached as ANOTHER CHILD of the very first root, so the
        original root becomes a visible fork: one child is the pre-clear
        conversation, the other is the fresh post-clear line. The old branch
        stays fully in the store (still restorable from the history-tree
        map); the next commit simply grows beneath this new child.
        """
        with self._lock:
            parent_id = self._root_id
            node = SessionNode(
                id=self._new_id(),
                parent=parent_id,
                actor=actor,
                label=label or CLEAR_LABEL,
                seq=next(self._counter),
                snapshot=(snapshot.clone() if snapshot else Snapshot()),
            )
            self._nodes[node.id] = node
            self._nodes[parent_id].children.append(node.id)
            self._heads[actor] = node.id
            return node

    def get(self, node_id: str) -> Optional[SessionNode]:
        with self._lock:
            return self._nodes.get(node_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._nodes)

    # --- writes ---------------------------------------------------------
    def commit(
        self,
        snapshot: Snapshot,
        actor: str = DEFAULT_ACTOR,
        label: str = "",
        parent: Optional[str] = None,
    ) -> SessionNode:
        """Record a new commit as a child of ``parent`` (default: actor head).

        Advances ``actor``'s head to the new node. Because each actor only
        moves its own head, concurrent commits from different actors create
        divergent branches instead of overwriting one another.
        """
        with self._lock:
            parent_id = parent if parent is not None else self.head(actor)
            if parent_id not in self._nodes:
                raise KeyError(f"unknown parent node: {parent_id}")
            node = SessionNode(
                id=self._new_id(),
                parent=parent_id,
                actor=actor,
                label=label or "commit",
                seq=next(self._counter),
                snapshot=snapshot.clone(),
            )
            self._nodes[node.id] = node
            self._nodes[parent_id].children.append(node.id)
            self._heads[actor] = node.id
            return node

    def checkout(self, node_id: str, actor: str = DEFAULT_ACTOR) -> SessionNode:
        """Point ``actor``'s head at an existing node (git checkout).

        Non-destructive: no node is removed, so every previously reachable
        branch stays available. New commits after a checkout branch off from
        here, which is exactly the git 'detached-then-branch' behaviour.
        """
        with self._lock:
            if node_id not in self._nodes:
                raise KeyError(f"unknown node: {node_id}")
            self._heads[actor] = node_id
            return self._nodes[node_id]

    # --- reads ----------------------------------------------------------
    def path(self, node_id: str) -> List[SessionNode]:
        """Root -> ``node_id`` ancestry (the 'current line' for that node)."""
        with self._lock:
            chain: List[SessionNode] = []
            current = self._nodes.get(node_id)
            while current is not None:
                chain.append(current)
                current = self._nodes.get(current.parent) if current.parent else None
            chain.reverse()
            return chain

    def children(self, node_id: str) -> List[SessionNode]:
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return []
            return [self._nodes[c] for c in node.children if c in self._nodes]

    def rows(self) -> List["SessionRow"]:
        """A flattened, depth-annotated view for rendering an ASCII tree.

        Pre-order DFS from EACH root (usually one; a ``clear`` adds another);
        ``depth`` is the indent level and
        ``is_branch_point`` marks nodes with more than one child (a fork). The
        result is stable and cheap for a modal to redraw on every open.
        """
        with self._lock:
            rows: List[SessionRow] = []

            def walk(node_id: str, depth: int) -> None:
                node = self._nodes.get(node_id)
                if node is None:
                    return
                rows.append(
                    SessionRow(
                        node=node,
                        depth=depth,
                        is_branch_point=len(node.children) > 1,
                    )
                )
                for child_id in node.children:
                    walk(child_id, depth + 1)

            for root_id in self._roots:
                walk(root_id, 0)
            return rows

    def graph_rows(self) -> List["SessionGraphRow"]:
        """A pre-order view carrying git-graph connectors for DRAWING the tree.

        Unlike :meth:`rows` (which only knows a node's ``depth``), each row
        here also carries the ASCII connector strings needed to render the
        branching structure like ``git log --graph`` / ``tree``:

        - ``prefix`` -- the vertical guides for the ancestor levels: a
          ``\u2502`` (with trailing spaces) where an ancestor still has a
          following sibling, or blank spaces where it does not.
        - ``connector`` -- this node's own tee: ``\u251c\u2500`` if it has a
          following sibling, ``\u2514\u2500`` if it is the last child (the
          root gets an empty connector).

        Concatenating ``prefix + connector`` gives the full left gutter for
        the row, so forks and abandoned branches are visually explicit.
        The traversal order matches :meth:`rows` (pre-order DFS over every
        root -- a ``clear`` forks a new child off the original root).
        """
        with self._lock:
            out: List[SessionGraphRow] = []

            def walk(node_id: str, prefix: str, is_last: bool, depth: int) -> None:
                node = self._nodes.get(node_id)
                if node is None:
                    return
                if depth == 0:
                    connector = ""
                else:
                    connector = ("\u2514\u2500 " if is_last else "\u251c\u2500 ")
                out.append(
                    SessionGraphRow(
                        node=node,
                        depth=depth,
                        prefix=prefix,
                        connector=connector,
                        is_last=is_last,
                        is_branch_point=len(node.children) > 1,
                    )
                )
                # Children indent under this node; the guide for THIS level is
                # a vertical bar only if this node itself has a later sibling.
                child_prefix = prefix + ("   " if is_last else "\u2502  ")
                child_ids = [c for c in node.children if c in self._nodes]
                for i, child_id in enumerate(child_ids):
                    walk(
                        child_id,
                        child_prefix,
                        i == len(child_ids) - 1,
                        depth + 1,
                    )

            for root_id in self._roots:
                walk(root_id, "", True, 0)
            return out

    def graph_layout(self, hide_labels=None) -> List["SessionLayoutNode"]:
        """A 2D (column, row) placement for drawing the tree as stacked boxes.

        This is the seam for rendering the session tree the SAME way the
        telemetry dashboard draws its flow: a straight vertical spine for a
        linear history (so multi-turn sessions do NOT creep rightward), with
        a fork stepping out to a new column only where a node truly has more
        than one child.

        ``hide_labels`` (a set/collection of label strings) makes matching
        nodes TRANSPARENT: they are not placed, and their children re-attach
        to the nearest visible ancestor. This backs the dashboard's choice
        to hide ``clear`` markers (a clear restores no context, so it is not
        a useful restore target) WITHOUT detaching the post-clear line --
        that line simply hangs off the original root as a fork instead of
        off the (now hidden) clear node. The underlying tree is untouched;
        only this drawing view skips them.

        Placement (pre-order DFS over every root; normally one root, whose
        children include the pre-clear line AND the post-clear fork a
        ``clear`` grows via ``new_root``):

        - ``row`` -- a monotonic counter, so every node stacks on its own row
          top-to-bottom (never overlapping), exactly like the flow canvas.
        - ``col`` -- the root is column 0; a node's FIRST child inherits its
          column (the mainline stays put), and each LATER child of a fork
          opens a BRAND-NEW column to the right (``max column used so far``
          + 1). Columns are NEVER reused: every branch keeps its own lane for
          life. This is what makes the drawn tree PLANAR (zero crossings) --
          a later child's lane is always further right than every lane opened
          before it, so the connector into it only ever traverses empty
          columns and never cuts across another branch's rail. The trade is
          width (deep/serial forks creep rightward), which is fine on an
          unbounded, pannable canvas; a linear chain still stays in column 0.

        Returns :class:`SessionLayoutNode` records (node + col + row + parent
        + fork flag) that a UI turns into box positions + connector edges.
        """
        hidden = set(hide_labels or ())
        with self._lock:
            def is_hidden(node_id: str) -> bool:
                node = self._nodes.get(node_id)
                return node is not None and node.label in hidden

            def visible_children(node_id: str) -> List[str]:
                # Children of ``node_id`` in the VISIBLE tree: descend through
                # hidden nodes so their visible descendants re-attach here.
                out: List[str] = []
                node = self._nodes.get(node_id)
                if node is None:
                    return out
                for child_id in node.children:
                    if child_id not in self._nodes:
                        continue
                    if is_hidden(child_id):
                        out.extend(visible_children(child_id))
                    else:
                        out.append(child_id)
                return out

            visible_parent: Dict[str, Optional[str]] = {}

            # First pass: monotonic row per VISIBLE node (pre-order DFS),
            # skipping hidden nodes but continuing into their children so the
            # post-clear line still appears (attached to a visible ancestor).
            order: List[str] = []
            row_of: Dict[str, int] = {}

            def measure(node_id: str, parent: Optional[str]) -> None:
                if node_id not in self._nodes:
                    return
                visible_parent[node_id] = parent
                row_of[node_id] = len(order)
                order.append(node_id)
                for child_id in visible_children(node_id):
                    measure(child_id, node_id)

            # Second pass: allocate columns over the VISIBLE tree, NEVER
            # reusing one -- first child keeps the lane, later children open a
            # fresh lane to the right (keeps the drawing planar).
            col_of: Dict[str, int] = {}
            next_col = [0]

            def assign(node_id: str, col: int) -> None:
                col_of[node_id] = col
                for i, child_id in enumerate(visible_children(node_id)):
                    if i == 0:
                        child_col = col            # mainline stays in column
                    else:
                        next_col[0] += 1           # a fresh lane, never reused
                        child_col = next_col[0]
                    assign(child_id, child_col)

            # Visible roots: a root that is itself hidden contributes its
            # visible descendants as roots (so hiding never orphans a line).
            visible_roots: List[str] = []
            for root_id in self._roots:
                if root_id not in self._nodes:
                    continue
                if is_hidden(root_id):
                    visible_roots.extend(visible_children(root_id))
                else:
                    visible_roots.append(root_id)

            for root_id in visible_roots:
                measure(root_id, None)
            for root_id in visible_roots:
                assign(root_id, next_col[0])
                next_col[0] += 1

            placed: List[SessionLayoutNode] = []
            for node_id in order:
                node = self._nodes[node_id]
                placed.append(
                    SessionLayoutNode(
                        node=node,
                        col=col_of[node_id],
                        row=row_of[node_id],
                        parent=visible_parent.get(node_id),
                        is_branch_point=len(visible_children(node_id)) > 1,
                    )
                )
            return placed


@dataclass
class SessionRow:
    """One line of a rendered session tree (see :meth:`SessionTree.rows`)."""

    node: SessionNode
    depth: int
    is_branch_point: bool


@dataclass
class SessionGraphRow:
    """One drawable line of the tree (see :meth:`SessionTree.graph_rows`).

    Carries the git-graph connector strings so a UI can render the branching
    structure directly: ``prefix`` (ancestor vertical guides) + ``connector``
    (this node's tee) form the left gutter drawn before the node's label.
    """

    node: SessionNode
    depth: int
    prefix: str
    connector: str
    is_last: bool
    is_branch_point: bool


@dataclass
class SessionLayoutNode:
    """A node placed on a 2D grid for drawing (see :meth:`SessionTree.graph_layout`).

    ``col``/``row`` are grid coordinates: a linear chain shares one column
    (stacking straight down), forks step to a new column. A UI multiplies
    these by its box width/height + gaps to get pixel positions and draws a
    connector from each node to its ``parent``.
    """

    node: SessionNode
    col: int
    row: int
    parent: Optional[str]
    is_branch_point: bool
