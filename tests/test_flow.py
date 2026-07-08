"""Tests for the interactive flow graph (coreybot.frontends.tui.flow).

The flow graph is a *projection* of the agent's append-only telemetry log, not
a widget that owns turn state. These tests therefore drive it the same way the
TUI does: build a list of AgentEvents (a mini telemetry log) and feed it via
``set_history`` (full rebuild) or ``append`` (incremental fast path).

Two layers are covered:
  * the visual *model* (nodes/edges/status + multi-turn context), built without
    a running app;
  * the *widget* behavior (one FlowNode per node; pan the background to move the
    whole canvas; auto-follow the newest node), driven with a headless Pilot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.geometry import Offset

from coreybot.core.config import Config
from coreybot.runtime.agent import Agent, AgentEvent, Source
from coreybot.frontends.tui.flow import (
    FlowPanel,
    FlowNode,
    FlowCanvas,
    _EdgeLayer,
    SOURCE_STYLE,
    _style_for,
    STATUS_OK,
    STATUS_FAIL,
    STATUS_RUNNING,
    STATUS_INFO,
    _format_duration,
)

# Reuse the shared async test double (FakeStreamProvider) for the roundtrip.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import FakeStreamProvider  # noqa: E402

pytestmark = pytest.mark.integration


# --- telemetry builders ---------------------------------------------------
# A telemetry log always opens with a ``turn_start`` (which creates the user
# node) and closes with ``turn_end`` (the answer node). Helpers below return
# such lists so tests can project them exactly like the live loop does.
def _full_turn_events(question: str = "2+2?", answer: str = "it is 4") -> list:
    """One tool-using turn: user -> llm -> tool -> llm -> answer (5 nodes)."""
    return [
        AgentEvent(kind="turn_start", name="user", text=question),
        AgentEvent(kind="llm_call", name="m"),
        AgentEvent(kind="llm_result", name="m", ok=True, text="tool call"),
        AgentEvent(kind="tool_call", name="calc", arguments={"expression": "2+2"}),
        AgentEvent(kind="tool_result", name="calc", ok=True, output="4"),
        AgentEvent(kind="llm_call", name="m"),
        AgentEvent(kind="llm_result", name="m", ok=True, text="message"),
        AgentEvent(kind="turn_end", ok=True, text=answer),
    ]


def _busy_turn_events(steps: int = 8) -> list:
    """A turn with many llm steps so the graph overflows a small viewport."""
    events = [AgentEvent(kind="turn_start", name="user", text="hello")]
    for i in range(steps):
        events.append(AgentEvent(kind="llm_call", name=f"m{i}"))
        events.append(AgentEvent(kind="llm_result", name=f"m{i}", ok=True, text="ok"))
    return events


def _project(panel: FlowPanel, events: list) -> None:
    """Feed events incrementally, exactly like ChatApp._on_event does."""
    for event in events:
        panel.append(event)


# --- model-only tests (no app) -------------------------------------------
def test_style_lookup_has_all_known_sources():
    for src in (Source.LLM, Source.TOOL, Source.MCP, Source.SKILL, Source.AGENT):
        assert src in SOURCE_STYLE
        assert "icon" in SOURCE_STYLE[src]


def test_unknown_source_falls_back():
    assert _style_for("something-new")["label"] == "other"


def test_flow_builds_full_turn_as_node_chain():
    panel = FlowPanel()
    _project(panel, _full_turn_events())
    nodes = panel.nodes()
    # user + llm + tool + llm + answer = 5 nodes, chained by 4 edges.
    assert len(nodes) == 5
    assert nodes[0].kind == "user"
    assert nodes[-1].kind == "answer"
    titles = " ".join(n.title for n in nodes)
    assert "you" in titles and "calc" in titles
    calc = next(n for n in nodes if "calc" in n.title)
    assert calc.status == STATUS_OK and calc.status_text == "4"
    assert len(panel.edges()) == len(nodes) - 1
    # edges connect consecutive nodes.
    keys = [n.key for n in nodes]
    assert [(e.src, e.dst) for e in panel.edges()] == list(zip(keys, keys[1:]))


def test_flow_marks_tool_failure():
    panel = FlowPanel()
    _project(panel, [
        AgentEvent(kind="turn_start", name="user", text="do x"),
        AgentEvent(kind="tool_call", name="calc", arguments={}),
        AgentEvent(kind="tool_result", name="calc", ok=False, output="boom"),
    ])
    tool_node = panel.nodes()[1]
    assert tool_node.status == STATUS_FAIL and tool_node.status_text == "boom"


def test_flow_supports_new_source_without_code_change():
    panel = FlowPanel()
    _project(panel, [
        AgentEvent(kind="turn_start", name="user", text="call mcp"),
        AgentEvent(kind="tool_call", source=Source.MCP, name="fetch", arguments={}),
    ])
    node = panel.nodes()[1]
    assert node.source == Source.MCP and node.key.startswith("mcp-")
    assert node.title.endswith("fetch")


def test_clear_resets_model():
    panel = FlowPanel()
    _project(panel, _full_turn_events())
    assert panel.nodes()
    panel.clear()
    assert panel.nodes() == [] and panel.edges() == []


# --- expand / collapse + dynamic-height layout ---------------------------
def test_steps_start_collapsed_and_carry_full_text():
    """Every step keeps its full message but shows a short preview collapsed."""
    panel = FlowPanel()
    _project(panel, [
        AgentEvent(kind="turn_start", name="user", text="hi"),
        AgentEvent(kind="tool_call", name="calc", arguments={"expression": "2+2"}),
        AgentEvent(kind="tool_result", name="calc", ok=True,
                   output="a long tool result that will need to wrap across rows"),
    ])
    tool = panel.nodes()[1]
    assert tool.expanded is False              # steps default to collapsed
    assert tool.collapsible is True
    # The full message is retained even though only a preview is displayed.
    assert "long tool result" in tool.message
    assert len(tool.body_rows()) == 1          # collapsed -> single preview row


def test_notice_auto_expands():
    """Notices reveal their message by default (no click needed)."""
    panel = FlowPanel()
    _project(panel, [
        AgentEvent(kind="turn_start", name="user", text="hi"),
        AgentEvent(kind="notice", text="memory cleared and here is a longer explanation"),
    ])
    notice = panel.nodes()[-1]
    assert notice.kind == "notice"
    assert notice.expanded is True
    assert len(notice.body_rows()) >= 1


def test_toggle_expands_node_and_reflows_neighbors():
    """Expanding a node grows it and pushes later nodes further down."""
    panel = FlowPanel()
    _project(panel, [
        AgentEvent(kind="turn_start", name="user", text="hi"),
        AgentEvent(kind="tool_call", name="calc", arguments={"expression": "2+2"}),
        AgentEvent(kind="tool_result", name="calc", ok=True,
                   output="a deliberately long tool result that wraps onto "
                          "several rows once the node is expanded fully"),
        AgentEvent(kind="turn_end", ok=True, text="done"),
    ])
    tool = panel.nodes()[1]
    answer = panel.nodes()[-1]
    y_before = [n.y for n in panel.nodes()]
    h_tool_before = tool.height
    # Expanding the tool node (long body) must grow it and shove the answer down.
    panel.toggle(tool.key)
    assert panel.nodes()[1].expanded is True
    assert tool.height > h_tool_before
    assert answer.y > y_before[-1]             # everything below reflowed down
    # Toggling back restores the original layout (pure function of the model).
    panel.toggle(tool.key)
    assert [n.y for n in panel.nodes()] == y_before
    assert tool.height == h_tool_before


def test_set_expanded_and_bulk_helpers():
    panel = FlowPanel()
    _project(panel, _full_turn_events())
    keys = [n.key for n in panel.nodes()]
    panel.set_expanded(keys[0], True)
    assert panel.nodes()[0].expanded is True
    panel.set_expanded(keys[0], False)
    assert panel.nodes()[0].expanded is False
    panel.expand_all()
    assert all(n.expanded for n in panel.nodes() if n.collapsible)
    panel.collapse_all()
    assert all(not n.expanded for n in panel.nodes())


def test_relayout_is_pure_and_non_overlapping():
    """Positions are a pure function of the model, and boxes never overlap."""
    panel = FlowPanel()
    _project(panel, _full_turn_events())
    _project(panel, _full_turn_events("again?", "sure"))
    # Recomputing layout yields identical coordinates (deterministic).
    before = [(n.key, n.x, n.y) for n in panel.nodes()]
    panel._relayout()
    assert [(n.key, n.x, n.y) for n in panel.nodes()] == before
    # Vertically stacked boxes do not overlap: each starts below the previous.
    nodes = panel.nodes()
    for prev, node in zip(nodes, nodes[1:]):
        assert node.y >= prev.y + prev.height


# --- projection semantics: multi-turn context, idempotency, equivalence ---
def test_multiple_turns_preserve_context():
    """Two turns must both stay on the chart (context is not wiped)."""
    panel = FlowPanel()
    _project(panel, _full_turn_events("2+2?", "it is 4"))
    _project(panel, _full_turn_events("3+3?", "it is 6"))
    nodes = panel.nodes()
    # 5 nodes per turn -> 10 total, with BOTH user nodes and BOTH answers.
    assert len(nodes) == 10
    assert [n.kind for n in nodes].count("user") == 2
    assert [n.kind for n in nodes].count("answer") == 2
    # The second turn branches from its own user node, not the first answer.
    second_user = nodes[5]
    assert second_user.kind == "user" and second_user.parent is None


def test_new_turn_breaks_edge_chain():
    """turn_start starts a fresh branch: no edge crosses the turn boundary."""
    panel = FlowPanel()
    _project(panel, _full_turn_events())
    _project(panel, _full_turn_events("next?", "done"))
    nodes = panel.nodes()
    second_user_key = nodes[5].key
    # No edge points *into* the second user node.
    assert all(e.dst != second_user_key for e in panel.edges())


def test_set_history_is_idempotent():
    """Rebuilding from the same log yields an identical graph."""
    panel = FlowPanel()
    log = _full_turn_events() + _full_turn_events("again?", "again 4")
    panel.set_history(log)
    first = [(n.key, n.kind, n.x, n.y) for n in panel.nodes()]
    first_edges = [(e.src, e.dst) for e in panel.edges()]
    panel.set_history(log)
    second = [(n.key, n.kind, n.x, n.y) for n in panel.nodes()]
    second_edges = [(e.src, e.dst) for e in panel.edges()]
    assert first == second and first_edges == second_edges


def test_incremental_append_matches_full_set_history():
    """append() step-by-step == set_history() of the whole log."""
    log = _full_turn_events() + _busy_turn_events(3)

    incremental = FlowPanel()
    _project(incremental, log)

    whole = FlowPanel()
    whole.set_history(log)

    inc_nodes = [(n.key, n.kind, n.status, n.x, n.y) for n in incremental.nodes()]
    whole_nodes = [(n.key, n.kind, n.status, n.x, n.y) for n in whole.nodes()]
    assert inc_nodes == whole_nodes
    assert [(e.src, e.dst) for e in incremental.edges()] == \
           [(e.src, e.dst) for e in whole.edges()]


async def test_projects_real_agent_telemetry_roundtrip():
    """End-to-end: run turns on a real Agent, then project its telemetry.

    This is the property the whole task is about -- the chart is a faithful
    projection of ``Agent.telemetry`` and survives across turns.
    """
    provider = FakeStreamProvider(replies=["<message>hi there</message>",
                                            "<message>bye now</message>"])
    agent = Agent(Config(), provider=provider)
    await agent.arun_turn("hello")
    await agent.arun_turn("goodbye")

    panel = FlowPanel()
    panel.set_history(agent.telemetry)
    nodes = panel.nodes()
    # Two no-tool turns: each is user -> llm -> answer (3 nodes) = 6 total.
    assert [n.kind for n in nodes].count("user") == 2
    assert [n.kind for n in nodes].count("answer") == 2
    answers = [n.detail for n in nodes if n.kind == "answer"]
    assert answers == ["hi there", "bye now"]


# --- widget/interaction tests (headless app) -----------------------------
class _Harness(App):
    def compose(self) -> ComposeResult:
        self.panel = FlowPanel()
        yield self.panel


class _MouseStub:
    """Minimal stand-in for a Textual mouse event (only screen_offset needed)."""
    def __init__(self, x: int, y: int) -> None:
        self.screen_offset = Offset(x, y)

    def stop(self) -> None:
        pass


async def test_each_node_materializes_a_widget():
    app = _Harness()
    async with app.run_test(size=(70, 30)) as pilot:
        _project(app.panel, _full_turn_events())
        await pilot.pause()
        widgets = list(app.query(FlowNode))
        assert len(widgets) == len(app.panel.nodes()) == 5


async def test_nodes_are_not_individually_draggable():
    """FlowNode no longer captures the mouse; it has no drag handlers."""
    app = _Harness()
    async with app.run_test(size=(70, 30)) as pilot:
        _project(app.panel, _full_turn_events())
        await pilot.pause()
        widget = list(app.query(FlowNode))[0]
        assert not hasattr(widget, "on_mouse_down")


async def test_autofollow_scrolls_to_newest_node():
    app = _Harness()
    # Small viewport so a busy turn overflows and must scroll.
    async with app.run_test(size=(40, 12)) as pilot:
        _project(app.panel, _busy_turn_events())
        await pilot.pause()
        await pilot.pause()
        canvas = app.query_one(FlowCanvas)
        assert canvas.max_scroll_y > 0        # content exceeds the viewport
        assert canvas.scroll_offset.y > 0     # followed the tail downward
        assert canvas.follow_tail is True


async def test_panning_background_moves_viewport_and_stops_follow():
    app = _Harness()
    async with app.run_test(size=(40, 12)) as pilot:
        _project(app.panel, _busy_turn_events())
        await pilot.pause()
        await pilot.pause()
        canvas = app.query_one(FlowCanvas)
        before = canvas.scroll_offset.y
        # Grab the background and drag it down by 4 rows.
        canvas.on_mouse_down(_MouseStub(5, 5))
        canvas.on_mouse_move(_MouseStub(5, 9))
        canvas.on_mouse_up(_MouseStub(5, 9))
        await pilot.pause()
        assert canvas.follow_tail is False        # user took control
        assert canvas.scroll_offset.y != before   # viewport actually moved


async def test_small_graph_can_pan_freely_in_both_axes():
    """Even a tiny graph has slack to pan in X and Y (map-like free drag).

    Regression: without canvas padding the virtual area equalled the content, so
    max_scroll was (0, 0) and dragging did nothing. Padding on every side gives
    real pan travel horizontally and vertically regardless of graph size.
    """
    app = _Harness()
    async with app.run_test(size=(80, 24)) as pilot:
        # A short one-turn graph that easily fits the viewport.
        _project(app.panel, [
            AgentEvent(kind="turn_start", text="hi"),
            AgentEvent(kind="llm_call", name="m"),
            AgentEvent(kind="llm_result", name="m", ok=True, text="ok"),
            AgentEvent(kind="turn_end", text="done"),
        ])
        await pilot.pause()
        await pilot.pause()
        canvas = app.query_one(FlowCanvas)
        # There is room to scroll in *both* directions (not glued to content).
        assert canvas.max_scroll_x > 0
        assert canvas.max_scroll_y > 0

        # Drag the canvas up-left: the scroll offset increases on both axes.
        start = canvas.scroll_offset
        canvas.on_mouse_down(_MouseStub(40, 12))
        canvas.on_mouse_move(_MouseStub(34, 8))   # move by (-6, -4) -> pan
        canvas.on_mouse_up(_MouseStub(34, 8))
        await pilot.pause()
        after = canvas.scroll_offset
        assert after.x != start.x        # horizontal pan actually moved
        assert after.y != start.y        # vertical pan actually moved
        assert canvas.follow_tail is False


async def test_over_dragging_past_edge_clamps_without_drift():
    """Dragging past a boundary stops crisply and does not self-drift.

    Regression: panning used a deferred, eased ``scroll_to`` that kept gliding to
    the clamped max after the cursor stopped ("drift at the boundary"). The pan
    now clamps the target itself and applies it immediately, so the offset tracks
    the cursor 1:1 and holds still once the drag ends.
    """
    app = _Harness()
    async with app.run_test(size=(60, 20)) as pilot:
        _project(app.panel, _busy_turn_events())
        await pilot.pause()
        await pilot.pause()
        canvas = app.query_one(FlowCanvas)
        # This graph must be pannable on both axes for the test to be meaningful.
        assert canvas.max_scroll_x > 0
        assert canvas.max_scroll_y > 0

        # Start from the top-left corner, then drag the cursor down-right so the
        # target (origin - delta) goes negative. It must clamp to the min corner.
        canvas.scroll_to(0, 0, animate=False, immediate=True)
        await pilot.pause()
        canvas.on_mouse_down(_MouseStub(5, 5))
        canvas.on_mouse_move(_MouseStub(58, 19))
        canvas.on_mouse_up(_MouseStub(58, 19))
        settled = (canvas.scroll_x, canvas.scroll_y)
        for _ in range(6):
            await pilot.pause()
        # Pinned to the min corner with no post-release motion.
        assert settled == (0, 0)
        assert (canvas.scroll_x, canvas.scroll_y) == settled

        # Start from the max corner, then drag the cursor up-left so the target
        # exceeds the max. It must clamp to the max corner (no glide past / back).
        canvas.scroll_to(canvas.max_scroll_x, canvas.max_scroll_y, animate=False, immediate=True)
        await pilot.pause()
        canvas.on_mouse_down(_MouseStub(55, 18))
        canvas.on_mouse_move(_MouseStub(1, 1))
        canvas.on_mouse_up(_MouseStub(1, 1))
        settled2 = (canvas.scroll_x, canvas.scroll_y)
        for _ in range(6):
            await pilot.pause()
        assert settled2 == (canvas.max_scroll_x, canvas.max_scroll_y)
        assert (canvas.scroll_x, canvas.scroll_y) == settled2


async def test_drag_starting_on_a_node_pans_instead_of_toggling():
    """Pressing on a node and dragging pans the canvas (and does not toggle)."""
    app = _Harness()
    async with app.run_test(size=(80, 24)) as pilot:
        _project(app.panel, _full_turn_events())
        await pilot.pause()
        await pilot.pause()
        canvas = app.query_one(FlowCanvas)
        node = app.panel.nodes()[0]
        assert node.expanded is False
        widget = canvas._widgets[node.key]
        sx = widget.region.x + 3
        sy = widget.region.y + 1
        start = canvas.scroll_offset
        # Press on the node, then move well past the click slop -> this is a pan.
        canvas.on_mouse_down(_MouseStub(sx, sy))
        canvas.on_mouse_move(_MouseStub(sx + 8, sy + 5))
        canvas.on_mouse_up(_MouseStub(sx + 8, sy + 5))
        await pilot.pause()
        # The canvas moved and the node was NOT toggled (drag != click).
        assert canvas.scroll_offset != start
        assert app.panel.nodes()[0].expanded is False


async def test_new_turn_resumes_follow():
    app = _Harness()
    async with app.run_test(size=(40, 12)) as pilot:
        _project(app.panel, _busy_turn_events())
        await pilot.pause()
        canvas = app.query_one(FlowCanvas)
        canvas.on_mouse_down(_MouseStub(5, 5))
        canvas.on_mouse_move(_MouseStub(5, 9))
        canvas.on_mouse_up(_MouseStub(5, 9))
        await pilot.pause()
        assert canvas.follow_tail is False
        # Projecting a new turn_start re-enables auto-follow.
        app.panel.append(AgentEvent(kind="turn_start", name="user", text="again"))
        await pilot.pause()
        assert canvas.follow_tail is True
        # resume_follow() is also exposed on the panel.
        canvas.follow_tail = False
        app.panel.resume_follow()
        assert canvas.follow_tail is True


async def test_click_on_node_toggles_expansion():
    """A press+release on a node (no drag) expands/collapses it.

    The node's screen region is re-read before each click because expanding a
    node reflows the canvas; a toggle itself does not auto-scroll (only appends
    do), so once the view has settled the clicked node stays put.
    """
    app = _Harness()
    async with app.run_test(size=(60, 30)) as pilot:
        _project(app.panel, _full_turn_events(
            answer="a long answer that has enough text to wrap when expanded fully"))
        # Let the append's auto-follow scroll settle before measuring.
        await pilot.pause()
        await pilot.pause()
        canvas = app.query_one(FlowCanvas)
        node = app.panel.nodes()[0]
        assert node.expanded is False

        def click_node0():
            widget = canvas._widgets[node.key]
            cx = widget.region.x + 2
            cy = widget.region.y + 1
            canvas.on_mouse_down(_MouseStub(cx, cy))
            canvas.on_mouse_up(_MouseStub(cx, cy))

        click_node0()
        await pilot.pause()
        assert app.panel.nodes()[0].expanded is True
        # A toggle does not chase the tail, so node 0 stays under the cursor.
        click_node0()
        await pilot.pause()
        assert app.panel.nodes()[0].expanded is False


async def test_click_on_background_does_not_toggle():
    app = _Harness()
    async with app.run_test(size=(60, 30)) as pilot:
        _project(app.panel, _full_turn_events())
        await pilot.pause()
        canvas = app.query_one(FlowCanvas)
        before = [n.expanded for n in app.panel.nodes()]
        # A click far to the right of the single node column hits no node.
        canvas.on_mouse_down(_MouseStub(55, 25))
        canvas.on_mouse_up(_MouseStub(55, 25))
        await pilot.pause()
        assert [n.expanded for n in app.panel.nodes()] == before


def _llm_turn_with_payload(prompt="[user]\nexplain this", reply="<message>a long reply</message>"):
    """One llm round-trip carrying full input (prompt) and response (raw)."""
    return [
        AgentEvent(kind="turn_start", name="user", text="explain this"),
        AgentEvent(kind="llm_call", name="m", text=prompt),
        AgentEvent(kind="llm_result", name="m", ok=True, text="message", output=reply),
        AgentEvent(kind="turn_end", ok=True, text="a long reply"),
    ]


def test_llm_node_captures_input_and_response():
    """The llm node stores the prompt and raw reply and is inspectable."""
    panel = FlowPanel()
    _project(panel, _llm_turn_with_payload())
    node = next(n for n in panel.nodes() if n.source == Source.LLM)
    assert "explain this" in node.full_input
    assert node.full_response == "<message>a long reply</message>"
    assert node.inspectable is True
    labels = [label for label, _ in node.inspect_sections()]
    assert labels == ["INPUT", "RESPONSE"]


def test_non_llm_nodes_are_not_inspectable():
    """User/answer/tool nodes without long input+response stay non-inspectable."""
    panel = FlowPanel()
    _project(panel, _full_turn_events())
    for node in panel.nodes():
        if node.source != Source.LLM:
            assert node.inspectable is False


async def test_open_inspector_posts_message():
    """open_inspector emits an InspectRequested carrying the sections."""
    messages = []

    class _Catch(App):
        def compose(self) -> ComposeResult:
            self.panel = FlowPanel()
            yield self.panel

        def on_flow_panel_inspect_requested(self, message) -> None:
            messages.append(message)

    app = _Catch()
    async with app.run_test(size=(80, 30)) as pilot:
        _project(app.panel, _llm_turn_with_payload())
        await pilot.pause()
        key = next(k for k, v in app.panel._model.items() if v.source == Source.LLM)
        app.panel.open_inspector(key)
        await pilot.pause()
        await pilot.pause()
    assert len(messages) == 1
    labels = [label for label, _ in messages[0].sections]
    assert labels == ["INPUT", "RESPONSE"]


async def test_inspectable_node_shows_button_glyph():
    """An inspectable node renders the inspect button glyph in its header."""
    panel = FlowPanel()
    _project(panel, _llm_turn_with_payload())
    node = next(n for n in panel.nodes() if n.source == Source.LLM)
    content = panel.node_content(node)
    assert "⤢" in content.plain


def test_inspectable_node_header_has_no_inline_caret():
    """A popup-only (inspectable) node shows the inspect glyph but NO caret.

    The inline expand/collapse caret would imply a second, in-place expand
    gesture; inspectable nodes are popup-only, so only the inspect glyph is
    drawn (one gesture, one place).
    """
    panel = FlowPanel()
    _project(panel, _llm_turn_with_payload())
    node = next(n for n in panel.nodes() if n.source == Source.LLM)
    assert node.inspectable is True and node.expandable is False
    plain = panel.node_content(node).plain
    assert "⤢" in plain                 # popup affordance present
    assert "▸" not in plain and "▾" not in plain  # no inline caret



def test_non_inspectable_node_with_body_still_shows_caret():
    """A non-inspectable node with a body keeps its inline expand caret."""
    panel = FlowPanel()
    _project(panel, [
        AgentEvent(kind="turn_start", name="user", text="hi"),
        AgentEvent(kind="tool_call", name="calc", arguments={"expression": "2+2"}),
        AgentEvent(kind="tool_result", name="calc", ok=True,
                   output="a long tool result that wraps over several rows once expanded"),
        AgentEvent(kind="turn_end", ok=True, text="done"),
    ])
    tool = next(n for n in panel.nodes() if n.source == Source.TOOL)
    assert tool.inspectable is False and tool.expandable is True
    plain = panel.node_content(tool).plain
    assert "▸" in plain                 # collapsed caret shown


async def test_clicking_inspectable_node_always_opens_inspector_never_toggles():
    """An inspectable node is popup-only: ANY click opens the inspector.

    Two different expand gestures in two different spots (inline caret vs the
    popup button) was confusing, so an inspectable node (e.g. a model call)
    no longer inline-expands at all -- a click on the header OR the body row
    both open the full-screen inspector, and the node never toggles.
    """
    opened = []

    class _Catch(App):
        def compose(self) -> ComposeResult:
            self.panel = FlowPanel()
            yield self.panel

        def on_flow_panel_inspect_requested(self, message) -> None:
            opened.append(message)

    app = _Catch()
    async with app.run_test(size=(80, 30)) as pilot:
        _project(app.panel, _llm_turn_with_payload())
        await pilot.pause()
        canvas = app.query_one(FlowCanvas)
        key = next(k for k, v in app.panel._model.items() if v.source == Source.LLM)
        node = app.panel._model[key]
        assert node.inspectable is True
        assert node.expandable is False   # popup-only: no inline expand
        widget = next(w for w in app.query(FlowNode) if w._key == key)
        r = widget.region
        # Header row (region.y + 1) -> inspector.
        canvas.on_mouse_down(_MouseStub(r.x + 2, r.y + 1))
        canvas.on_mouse_up(_MouseStub(r.x + 2, r.y + 1))
        await pilot.pause()
        await pilot.pause()
        assert len(opened) == 1
        assert app.panel._model[key].expanded is False
        # Body row (region.y + 2) -> ALSO the inspector, and still no toggle.
        canvas.on_mouse_down(_MouseStub(r.x + 2, r.y + 2))
        canvas.on_mouse_up(_MouseStub(r.x + 2, r.y + 2))
        await pilot.pause()
        await pilot.pause()
        assert len(opened) == 2            # a second inspector open, not a toggle
        assert app.panel._model[key].expanded is False


async def test_edge_layer_renders_all_rows():
    app = _Harness()
    async with app.run_test(size=(70, 30)) as pilot:
        _project(app.panel, _full_turn_events())
        await pilot.pause()
        edge_layer = app.query_one(_EdgeLayer)
        height = edge_layer.size.height
        for y in range(height):
            strip = edge_layer.render_line(y)
            assert strip is not None


async def test_edge_connector_runs_down_the_box_center_not_the_left_edge():
    """The vertical connector exits/enters each box at its horizontal CENTER.

    Nodes stack in one column at a fixed width, so every edge is a straight
    vertical. It must sit on the box's rounded-center column
    (``x + (width - 1) // 2``), NOT hugging the left border (``x + 1``) --
    the off-center exit was a reported bug.
    """
    V = "\u2502"
    app = _Harness()
    async with app.run_test(size=(70, 30)) as pilot:
        _project(app.panel, _full_turn_events())
        await pilot.pause()
        edge_layer = app.query_one(_EdgeLayer)
        grid = [
            "".join(seg.text for seg in edge_layer.render_line(y))
            for y in range(edge_layer.size.height)
        ]
        node = app.panel.nodes()[0]
        center = node.x + (node.width - 1) // 2
        left_edge = node.x + 1
        assert center != left_edge  # otherwise the test proves nothing
        # Rows that contain a connector glyph have it ONLY at the center
        # column (the box layer hides anything over a box, so what remains
        # is the gap-row vertical).
        vbar_rows = [row for row in grid if V in row]
        assert vbar_rows  # at least one edge was drawn
        for row in vbar_rows:
            cols = [i for i, ch in enumerate(row) if ch == V]
            assert cols == [center], (cols, center)
            assert left_edge not in cols


# --- running-node blink + live timer --------------------------------------
def test_format_duration_ms_under_2s_seconds_beyond():
    """<2s shows milliseconds; >=2s shows seconds (per the timing spec)."""
    assert _format_duration(None) == ""
    assert _format_duration(0.0) == "0ms"
    assert _format_duration(0.012) == "12ms"
    assert _format_duration(0.84) == "840ms"
    assert _format_duration(1.999) == "1999ms"
    # 2s is the boundary -> switch to seconds.
    assert _format_duration(2.0) == "2.0s"
    assert _format_duration(3.44) == "3.4s"
    assert _format_duration(9.95) == "9.9s"
    # >=10s drops the decimal for compactness.
    assert _format_duration(12.7) == "12s"
    assert _format_duration(60.0) == "60s"


class _Clock:
    """A controllable monotonic clock for deterministic duration tests."""
    def __init__(self, start: float = 100.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


def test_running_node_stamps_started_and_duration_is_live():
    """A node entering RUNNING is stamped, and its duration ticks with time."""
    panel = FlowPanel()
    clock = _Clock(100.0)
    panel._clock = clock
    panel._project_event(AgentEvent(kind="turn_start", name="user", text="hi"))
    panel._project_event(AgentEvent(kind="llm_call", name="m", text="prompt"))
    node = next(n for n in panel.nodes() if n.source == Source.LLM)
    assert node.is_running and node.started_at == 100.0 and node.finished_at is None
    # Live: measured against 'now'.
    assert node.duration(100.84) == pytest.approx(0.84)
    assert _format_duration(node.duration(100.84)) == "840ms"


def test_finished_node_freezes_its_duration():
    """Once a node settles, its duration stops changing (frozen elapsed)."""
    panel = FlowPanel()
    clock = _Clock(100.0)
    panel._clock = clock
    panel._project_event(AgentEvent(kind="turn_start", name="user", text="hi"))
    panel._project_event(AgentEvent(kind="llm_call", name="m"))
    clock.t = 103.4  # 3.4s later
    panel._project_event(
        AgentEvent(kind="llm_result", name="m", ok=True, text="message", output="x")
    )
    node = next(n for n in panel.nodes() if n.source == Source.LLM)
    assert not node.is_running and node.finished_at == 103.4
    # Frozen: a much later 'now' does not change it.
    assert node.duration(9999.0) == pytest.approx(3.4)
    assert _format_duration(node.duration(9999.0)) == "3.4s"


def test_interruption_freezes_a_running_node():
    """A notice (e.g. 'interrupted') freezes any still-running node's timer."""
    panel = FlowPanel()
    clock = _Clock(200.0)
    panel._clock = clock
    panel._project_event(AgentEvent(kind="turn_start", name="user", text="hi"))
    panel._project_event(AgentEvent(kind="llm_call", name="m"))
    clock.t = 200.5
    panel._project_event(AgentEvent(kind="notice", text="interrupted"))
    node = next(n for n in panel.nodes() if n.source == Source.LLM)
    assert node.finished_at == 200.5
    assert _format_duration(node.duration(9999.0)) == "500ms"


def test_set_history_freezes_historical_running_nodes():
    """Rebuilding from a log that ends mid-run must not tick forever."""
    panel = FlowPanel()
    panel._clock = _Clock(100.0)
    # A log whose last event leaves the llm node running.
    panel.set_history([
        AgentEvent(kind="turn_start", name="user", text="hi"),
        AgentEvent(kind="llm_call", name="m"),
    ])
    node = next(n for n in panel.nodes() if n.source == Source.LLM)
    # Frozen on rebuild -> not counting up from the rebuild timestamp.
    assert node.finished_at is not None


async def test_running_node_blinks_and_timer_ticks_then_stops():
    """Mounted: a running node starts the ~10fps timer; it stops when done.

    The blink pulse alternates between frames, and the animation timer runs
    only while something is running -- it stops itself once nothing is.
    """
    app = _Harness()
    async with app.run_test(size=(80, 30)) as pilot:
        clock = _Clock(100.0)
        app.panel._clock = clock
        app.panel.append(AgentEvent(kind="turn_start", name="user", text="hi"))
        app.panel.append(AgentEvent(kind="llm_call", name="m"))
        await pilot.pause()
        # Timer is running while a node runs.
        assert app.panel._tick_timer is not None
        # The pulse alternates across frames. It is a slow, gentle pulse now
        # (~1.1Hz), so pump enough frames to span a full period rather than a
        # handful (which could all fall in one half).
        seen = set()
        for _ in range(40):
            app.panel._tick()
            seen.add(app.panel.pulse_on)
        assert seen == {True, False}
        # Finish the node -> the timer stops on the next tick.
        clock.t = 100.84
        app.panel.append(
            AgentEvent(kind="llm_result", name="m", ok=True, text="message", output="x")
        )
        app.panel._tick()
        assert app.panel._tick_timer is None


async def test_running_node_border_blinks_and_header_shows_timer():
    """The RUNNING state blinks the node's *border* (a CSS class), not a dot.

    The header shows a static running glyph plus the live timer; the border
    pulse is the `.blink` class toggled on the FlowNode widget by the
    animation timer. When the node finishes, the header shows the OK check,
    the timer freezes, and no blink/running classes remain.
    """
    app = _Harness()
    async with app.run_test(size=(80, 30)) as pilot:
        clock = _Clock(100.0)
        app.panel._clock = clock
        app.panel.append(AgentEvent(kind="turn_start", name="user", text="hi"))
        app.panel.append(AgentEvent(kind="llm_call", name="m"))
        await pilot.pause()
        clock.t = 100.5
        key = next(k for k, n in app.panel._model.items() if n.is_running)
        widget = app.panel._canvas._widgets[key]
        node = app.panel._model[key]
        header = app.panel.node_content(node).plain
        # No blink DOT in the header anymore; the running glyph + timer are.
        assert "\u25cf" not in header and "\u25cb" not in header
        assert "\u23f3" in header  # running hourglass glyph
        assert "500ms" in header
        # The widget carries the running class, and the .blink class toggles
        # across animation frames (the border pulse).
        assert widget.has_class("running")
        seen = set()
        for _ in range(40):
            app.panel._tick()
            seen.add(widget.has_class("blink"))
        assert seen == {True, False}
        # Finish -> OK glyph, frozen timer, and no running/blink classes.
        clock.t = 100.9
        app.panel.append(
            AgentEvent(kind="llm_result", name="m", ok=True, text="message", output="x")
        )
        app.panel._tick()
        node = app.panel._model[key]
        header2 = app.panel.node_content(node).plain
        assert "\u2713" in header2  # OK glyph
        assert "900ms" in header2
        assert not widget.has_class("running")
        assert not widget.has_class("blink")


async def test_node_border_colour_encodes_type_and_pulse_uses_own_accent():
    """Border colour encodes the node TYPE (source); the pulse stays same-hue.

    Each step node is tagged with a `src-<label>` class (llm/tool/...), so its
    border colour differs by type. A running node adds only `.running`/`.blink`
    (the `.blink` CSS eases the border to a lighter accent of that SAME type
    colour, never a different colour like amber). A failure adds `.fail` (red)
    regardless of type. user/answer nodes keep their kind classes.
    """
    app = _Harness()
    async with app.run_test(size=(90, 40)) as pilot:
        panel = app.panel
        panel.append(AgentEvent(kind="turn_start", name="user", text="hi"))
        panel.append(AgentEvent(kind="llm_call", name="gpt", source=Source.LLM))
        await pilot.pause()

        # Running LLM node: type class src-llm + running + (bright half) blink.
        key_llm = next(k for k, n in panel._model.items() if n.is_running)
        w_llm = panel._canvas._widgets[key_llm]
        panel._pulse = 0  # pulse_on -> True (bright half)
        panel._canvas.repaint_running(panel._model)
        await pilot.pause()
        assert w_llm.has_class("src-llm")
        assert w_llm.has_class("running") and w_llm.has_class("blink")
        # It is NOT coloured by a generic amber/running-specific hue class.
        assert not w_llm.has_class("src-tool")

        # A tool node is a DIFFERENT type colour (src-tool, not src-llm).
        panel.append(AgentEvent(kind="llm_result", name="gpt", ok=True, text="m", output="x"))
        panel.append(
            AgentEvent(kind="tool_call", name="calc", source=Source.TOOL,
                       arguments={"expression": "1+1"})
        )
        await pilot.pause()
        key_tool = next(k for k, n in panel._model.items() if n.is_running)
        w_tool = panel._canvas._widgets[key_tool]
        assert w_tool.has_class("src-tool") and not w_tool.has_class("src-llm")

        # A failure paints .fail (red) on top of the type colour.
        panel.append(AgentEvent(kind="tool_result", name="calc", ok=False, output="boom"))
        await pilot.pause()
        assert w_tool.has_class("fail")
        assert not w_tool.has_class("running") and not w_tool.has_class("blink")

        # user / answer nodes keep their kind classes (own hues).
        panel.append(AgentEvent(kind="turn_end", ok=True, text="done"))
        await pilot.pause()
        user_w = next(
            panel._canvas._widgets[k] for k, n in panel._model.items() if n.kind == "user"
        )
        answer_w = next(
            panel._canvas._widgets[k] for k, n in panel._model.items() if n.kind == "answer"
        )
        assert user_w.has_class("user")
        assert answer_w.has_class("answer")
