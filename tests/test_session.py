"""Unit tests for the git-like session tree (coreybot/runtime/session.py).

These exercise the pure data model in isolation: commits build a tree,
checkout is non-destructive (branches survive), each actor moves only its own
head (the multi-agent guarantee), and snapshots are immutable captures.
"""

from __future__ import annotations

from coreybot.core.message import Message
from coreybot.runtime.session import (
    CLEAR_LABEL,
    DEFAULT_ACTOR,
    SessionGraphRow,
    SessionLayoutNode,
    SessionTree,
    Snapshot,
)


def _hist(*contents):
    return [Message.user(c) for c in contents]


def test_root_exists_and_head_starts_at_root():
    tree = SessionTree(Snapshot(history=_hist("sys")))
    assert len(tree) == 1
    assert tree.head() == tree.root_id
    assert tree.get(tree.root_id).is_root


def test_commit_appends_child_and_moves_head():
    tree = SessionTree()
    a = tree.commit(Snapshot(history=_hist("q1")), label="q1")
    b = tree.commit(Snapshot(history=_hist("q1", "q2")), label="q2")
    assert a.parent == tree.root_id
    assert b.parent == a.id
    assert tree.head() == b.id
    assert [n.id for n in tree.path(b.id)] == [tree.root_id, a.id, b.id]


def test_checkout_is_non_destructive_and_branches():
    tree = SessionTree()
    a = tree.commit(Snapshot(history=_hist("q1")), label="q1")
    b = tree.commit(Snapshot(history=_hist("q1", "q2")), label="q2")
    # Go back to a and commit a divergent line.
    tree.checkout(a.id)
    assert tree.head() == a.id
    c = tree.commit(Snapshot(history=_hist("q1", "q2b")), label="q2b")
    # b still exists (non-destructive); a now forks into b and c.
    assert tree.get(b.id) is not None
    child_ids = {n.id for n in tree.children(a.id)}
    assert child_ids == {b.id, c.id}
    branch_rows = [r for r in tree.rows() if r.is_branch_point]
    assert [r.node.id for r in branch_rows] == [a.id]


def test_each_actor_moves_only_its_own_head():
    tree = SessionTree()
    main_a = tree.commit(Snapshot(history=_hist("q1")), label="q1")
    # A second agent branches off the root on its own head.
    tree.checkout(tree.root_id, actor="agent2")
    other = tree.commit(Snapshot(history=_hist("x")), actor="agent2", label="agent2")
    heads = tree.branch_heads()
    assert heads[DEFAULT_ACTOR] == main_a.id
    assert heads["agent2"] == other.id
    # Neither actor disturbed the other's head.
    assert tree.head() == main_a.id
    assert tree.head("agent2") == other.id
    # Both commits are children of the root: divergent branches, no conflict.
    assert {n.id for n in tree.children(tree.root_id)} == {main_a.id, other.id}


def test_snapshot_is_an_immutable_capture():
    history = _hist("q1")
    tree = SessionTree()
    node = tree.commit(Snapshot(history=history, telemetry=[1, 2]), label="q1")
    # Mutating the caller's lists must not change the committed snapshot.
    history.append(Message.user("late"))
    assert len(node.snapshot.history) == 1
    node.snapshot.telemetry.append(99)
    # A fresh read of the same node is unaffected by that external mutation
    # of a different list object.
    assert tree.get(node.id).snapshot.history[0].content == "q1"


def test_rows_are_preorder_with_depth():
    tree = SessionTree()
    a = tree.commit(Snapshot(), label="a")
    tree.commit(Snapshot(), label="b")
    tree.checkout(a.id)
    tree.commit(Snapshot(), label="c")
    rows = tree.rows()
    labels = [r.node.label for r in rows]
    depths = [r.depth for r in rows]
    assert labels[0] == "root" and depths[0] == 0
    # a is depth 1; its two children b and c are depth 2.
    assert depths[labels.index("a")] == 1
    assert depths[labels.index("b")] == 2
    assert depths[labels.index("c")] == 2


def test_graph_rows_carry_git_graph_connectors_for_drawing():
    """graph_rows() yields the ASCII connectors needed to DRAW the branch tree.

    Tree built:  root -> a -> b   and a fork off a: a -> c   (so ``a`` is a
    branch point with children [b, c]). The connectors must reflect that: ``b``
    (not last child of a) gets a tee, ``c`` (last child) gets an elbow, and the
    guide under a keeps a vertical bar for the row between b and c.
    """
    V = "\u2502"          # vertical guide
    TEE = "\u251c\u2500"    # branch (has a following sibling)
    ELBOW = "\u2514\u2500"  # last child

    tree = SessionTree()
    a = tree.commit(Snapshot(), label="a")
    tree.commit(Snapshot(), label="b")
    tree.checkout(a.id)
    tree.commit(Snapshot(), label="c")

    rows = tree.graph_rows()
    assert all(isinstance(r, SessionGraphRow) for r in rows)
    by_label = {r.node.label: r for r in rows}
    # Same pre-order traversal as rows(), starting at the root.
    assert [r.node.label for r in rows] == ["root", "a", "b", "c"]

    # Root has no connector; a is the (only) child of root -> last -> elbow.
    assert by_label["root"].connector == ""
    assert by_label["a"].connector == ELBOW + " "
    # a has two children: b is NOT last (tee), c IS last (elbow).
    assert by_label["b"].connector == TEE + " "
    assert by_label["c"].connector == ELBOW + " "
    # a is a fork; b and c are leaves.
    assert by_label["a"].is_branch_point is True
    assert by_label["b"].is_branch_point is False
    # ``a`` is root's LAST (only) child, so the guide under it is blank;
    # ``b``'s prefix is therefore all spaces (root spaces + a's spaces).
    assert V not in by_label["b"].prefix
    assert by_label["b"].prefix.strip() == ""
    # The full drawn gutter is prefix + connector; c ends in an elbow.
    c_gutter = by_label["c"].prefix + by_label["c"].connector
    assert c_gutter.endswith(ELBOW + " ")

    # When a node has a FOLLOWING sibling, its descendants keep a vertical
    # guide. Fork off the root (root now has children [a, d]) so a is no
    # longer last -> b/c under a must show a's bar in their prefix.
    tree.checkout(tree.root_id)
    tree.commit(Snapshot(), label="d")
    rows2 = {r.node.label: r for r in tree.graph_rows()}
    assert V in rows2["b"].prefix
    assert V in rows2["c"].prefix
    # d is now root's last child -> elbow; a is not last -> tee.
    assert rows2["a"].connector == TEE + " "
    assert rows2["d"].connector == ELBOW + " "


def test_graph_layout_keeps_a_linear_history_in_one_column():
    """A linear multi-turn session stacks straight down (no rightward creep).

    The restore map draws boxes at (col, row); a linear chain must keep every
    node in column 0 (only ``row`` advances) so the tree does not indent with
    each turn -- the whole point of the telemetry-style layout.
    """
    tree = SessionTree()
    tree.commit(Snapshot(history=_hist("q1")), label="q1")
    tree.commit(Snapshot(history=_hist("q1", "q2")), label="q2")
    tree.commit(Snapshot(history=_hist("q1", "q2", "q3")), label="q3")

    layout = tree.graph_layout()
    assert all(isinstance(node, SessionLayoutNode) for node in layout)
    # Every node sits in column 0; rows are the monotonic 0..N stack order.
    assert {node.col for node in layout} == {0}
    assert [node.row for node in layout] == list(range(len(layout)))
    # A straight line -> no branch points.
    assert not any(node.is_branch_point for node in layout)


def test_graph_layout_steps_forks_into_new_columns():
    """A fork keeps the mainline in place and steps the later branch right.

    The first child of a fork inherits the parent's column (mainline stays put);
    each additional child gets a fresh column to the right, and the forking node
    is flagged so the UI can draw the branch.
    """
    tree = SessionTree()
    a = tree.commit(Snapshot(history=_hist("q1")), label="q1")
    tree.commit(Snapshot(history=_hist("q1", "q2")), label="q2")
    # Diverge from a into a second line.
    tree.checkout(a.id)
    tree.commit(Snapshot(history=_hist("q1", "q2b")), label="q2b")

    by_label = {node.node.label: node for node in tree.graph_layout()}
    # Root and the mainline (q1 -> q2) stay in column 0.
    assert by_label["q1"].col == 0
    assert by_label["q2"].col == 0
    # The forking node is flagged and the later branch steps to a new column.
    assert by_label["q1"].is_branch_point is True
    assert by_label["q2b"].col >= 1
    # Rows never collide (each node stacks on its own row).
    rows = [node.row for node in tree.graph_layout()]
    assert len(rows) == len(set(rows))


def test_graph_layout_hide_labels_skips_clear_and_reparents_children():
    """``graph_layout(hide_labels=...)`` drops matching nodes, transparently.

    A hidden node is not placed, and its children re-attach to the nearest
    VISIBLE ancestor -- so hiding the ``clear`` marker keeps the post-clear
    line, just hanging off the original root as a fork instead of off the
    (now hidden) clear node. Rows stay dense and columns stay planar.
    """
    tree = SessionTree(Snapshot(history=_hist("sys")))
    tree.commit(Snapshot(history=_hist("sys", "q1")), label="q1")
    tree.commit(Snapshot(history=_hist("sys", "q1", "q2")), label="q2")
    # Clear: a fresh line forks off the original root.
    tree.new_root(Snapshot(history=_hist("sys")), label=CLEAR_LABEL)
    tree.commit(Snapshot(history=_hist("c1")), label="c1")
    tree.commit(Snapshot(history=_hist("c1", "c2")), label="c2")
    root_id = tree.root_id

    layout = tree.graph_layout(hide_labels={CLEAR_LABEL})
    labels = [node.node.label for node in layout]
    # The clear marker is gone; every other commit survives.
    assert CLEAR_LABEL not in labels
    assert set(labels) == {"root", "q1", "q2", "c1", "c2"}

    by_label = {node.node.label: node for node in layout}
    # The post-clear line (c1) re-parents to the ORIGINAL root...
    assert by_label["c1"].parent == root_id
    # ...and the root is a fork again (pre-clear q1 line + post-clear c1 line).
    assert by_label["root"].is_branch_point is True
    # Mainline stays in column 0; the post-clear branch steps right.
    assert by_label["q1"].col == 0
    assert by_label["c1"].col >= 1
    # Rows are a dense 0..N-1 range (no gap where the clear node was).
    rows = sorted(node.row for node in layout)
    assert rows == list(range(len(layout)))


def test_graph_layout_without_hide_labels_is_unchanged():
    """The default (no ``hide_labels``) still places every node, clear included."""
    tree = SessionTree(Snapshot(history=_hist("sys")))
    tree.commit(Snapshot(history=_hist("sys", "q1")), label="q1")
    tree.new_root(Snapshot(history=_hist("sys")), label=CLEAR_LABEL)
    tree.commit(Snapshot(history=_hist("c1")), label="c1")
    labels = [node.node.label for node in tree.graph_layout()]
    assert CLEAR_LABEL in labels
    assert len(tree.graph_layout()) == len(tree.rows())


def test_new_root_forks_off_root_and_keeps_old_branch():
    """clear == FORK a fresh line off the original root, keeping the tree.

    ``new_root`` is what backs the UI 'clear': the previous line stays in the
    store (still restorable) as a sibling branch, and the head moves to a new
    child of the ORIGINAL root -- so that root becomes a visible split.
    """
    tree = SessionTree()
    a = tree.commit(Snapshot(history=_hist("q1")), label="q1")
    b = tree.commit(Snapshot(history=_hist("q1", "q2")), label="q2")
    before = len(tree)
    original_root = tree.root_id

    fresh = tree.new_root(Snapshot(history=_hist("sys")), label="cleared")

    # The old nodes all survive (nothing removed) and the tree GREW.
    assert len(tree) == before + 1
    assert tree.get(a.id) is not None
    assert tree.get(b.id) is not None
    # The new node is a CHILD of the original root (not a detached root) and
    # the head now points at it, so the root forks into two lines.
    assert not fresh.is_root
    assert fresh.parent == original_root
    assert tree.head() == fresh.id
    # ``root_id`` still means the ORIGINAL root, which is now a branch point.
    assert tree.root_id == original_root
    assert tree.roots == [original_root]
    assert set(tree.get(original_root).children) == {a.id, fresh.id}
    # A commit after the clear grows the NEW line, not the abandoned one.
    c = tree.commit(Snapshot(history=_hist("sys", "q3")), label="q3")
    assert c.parent == fresh.id
    assert [n.id for n in tree.path(c.id)] == [original_root, fresh.id, c.id]


def test_rows_and_layout_fork_at_root_after_clear():
    """After a clear, the tree views show the root FORKED into two lines.

    ``rows``/``graph_layout`` must enumerate both children of the original
    root: the abandoned pre-clear branch AND the fresh line -- so the
    history-tree map draws the split instead of looking empty.
    """
    tree = SessionTree()
    tree.commit(Snapshot(history=_hist("q1")), label="q1")
    tree.new_root(Snapshot(history=_hist("sys")), label="cleared")
    tree.commit(Snapshot(history=_hist("sys", "q2")), label="q2")

    labels = [row.node.label for row in tree.rows()]
    # Old line (root -> q1) AND the new line (root -> cleared -> q2) present.
    assert labels == ["root", "q1", "cleared", "q2"]
    # The layout places every node once, on its own row (no collisions).
    layout = tree.graph_layout()
    assert isinstance(layout[0], SessionLayoutNode)
    assert [n.node.label for n in layout] == ["root", "q1", "cleared", "q2"]
    layout_rows = [n.row for n in layout]
    assert len(layout_rows) == len(set(layout_rows))
    # The root is a branch point; the pre-clear line stays in column 0 and
    # the fresh post-clear fork steps out to column 1, both linked to root.
    by_label = {n.node.label: n for n in layout}
    assert by_label["root"].is_branch_point
    assert by_label["root"].col == 0
    assert by_label["q1"].col == 0
    cleared = by_label["cleared"]
    assert cleared.parent == tree.root_id
    assert cleared.col == 1
    assert by_label["q2"].col == 1


def test_graph_layout_never_reuses_a_column_so_the_tree_stays_planar():
    """Every fork opens a BRAND-NEW column; columns are never reused.

    Reusing a freed lane would let a later fork's connector cut back across a
    live branch. Giving each branch its own permanent lane (monotone right)
    is what makes the drawn tree planar (zero crossings). The trade is width,
    which is fine on an unbounded, pannable canvas.
    """
    tree = SessionTree()
    root = tree.root_id
    # mainline in column 0
    tree.commit(Snapshot(history=_hist("a1")), label="A1")
    tree.commit(Snapshot(history=_hist("a2")), label="A2")
    # a first fork off root -> a new column (1), with its own two-node subtree
    tree._heads["main"] = root
    tree.commit(Snapshot(history=_hist("b1")), label="B1")
    tree.commit(Snapshot(history=_hist("b2")), label="B2")
    # a second fork off root -> even though the B branch already ended, its
    # column is NOT reused: C1 opens a fresh column (2).
    tree._heads["main"] = root
    tree.commit(Snapshot(history=_hist("c1")), label="C1")

    by_label = {n.node.label: n for n in tree.graph_layout()}
    assert by_label["A1"].col == 0 and by_label["A2"].col == 0
    assert by_label["B1"].col == 1 and by_label["B2"].col == 1
    # No reuse: C1 takes a new lane to the right, not B's freed column 1.
    assert by_label["C1"].col == 2
    # Columns are the contiguous set 0..2 (one lane per branch, none reused).
    assert sorted({n.col for n in by_label.values()}) == [0, 1, 2]
