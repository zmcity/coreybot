"""FlowPanel: a live, mouse-driven flow *canvas* of agent activity.

Each telemetry event becomes a node placed on an absolutely-positioned, pannable
2D canvas, with connector edges drawn from a node to its parent. The canvas is a
*projection* of the agent's append-only telemetry, and layout is a pure function
of the model (:meth:`FlowPanel._relayout`) -- which is the seam where a future
DAG layout (multi-agent / parallel tool calls needing horizontal breadth) can be
swapped in without touching rendering or interaction.

Interaction:
  * Drag the *background* to pan the whole canvas (grab-to-pan); no scrollbars.
  * *Click* a node to expand/collapse its message. ``notice`` nodes auto-expand;
    other steps start collapsed (a one-line preview) and reveal the full text on
    demand. The view auto-follows the newest node while a turn runs.

Widget tree::

    FlowPanel (Container, id="flow")
      FlowCanvas (ScrollableContainer)         # the pannable drawing area
        _EdgeLayer (Static, layer 'edges')     # paints connectors behind nodes
        FlowNode * N (Static, layer 'nodes')   # one per step, position: absolute

Extensibility: each node carries a *source* (llm / tool / mcp / skill / agent).
Icons and colors come from SOURCE_STYLE keyed by source string, so new injectors
render correctly just by tagging events with a new source -- no widget change.

The visual model (nodes + edges + positions) lives in plain attributes on
FlowPanel, so it can be built and asserted on headlessly, while the widgets
render it when mounted.
"""

from __future__ import annotations

import io
import time

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rich.console import Console
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.containers import Container, ScrollableContainer
from textual.geometry import Offset
from textual.strip import Strip
from textual import events
from textual.message import Message
from textual.widgets import Static

from coreybot.runtime.agent import AgentEvent, Source


# icon + Rich color per source. Unknown sources fall back to a generic style.
SOURCE_STYLE: Dict[str, Dict[str, str]] = {
    Source.LLM: {"icon": "🧠", "color": "cyan", "label": "llm"},
    Source.TOOL: {"icon": "🔧", "color": "green", "label": "tool"},
    Source.MCP: {"icon": "🔌", "color": "magenta", "label": "mcp"},
    Source.SKILL: {"icon": "✨", "color": "yellow", "label": "skill"},
    Source.AGENT: {"icon": "🤖", "color": "blue", "label": "agent"},
    Source.SYSTEM: {"icon": "•", "color": "grey62", "label": "system"},
}
_FALLBACK_STYLE = {"icon": "▫", "color": "white", "label": "other"}


def _style_for(source: str) -> Dict[str, str]:
    return SOURCE_STYLE.get(source, _FALLBACK_STYLE)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]`` (``high`` wins if the range is empty)."""
    if high < low:
        return low
    return max(low, min(value, high))


# Node box geometry (cells) and canvas layout spacing.
NODE_WIDTH = 30
_NODE_INNER_WIDTH = NODE_WIDTH - 4     # usable text width inside border+padding
HEADER_ROWS = 1                        # the title row
_COLLAPSED_PREVIEW_ROWS = 1            # body rows shown while collapsed
_MAX_BODY_ROWS = 10                    # cap expanded body so one node stays sane
_BORDER_ROWS = 2                       # round border adds a row top and bottom
_V_GAP = 1                             # blank rows between vertically stacked nodes
_H_GAP = 4                             # column gap (reserved for future DAG breadth)
_TURN_GAP = 1                          # extra blank rows between successive turns
_MARGIN_X = 2
_MARGIN_Y = 1
# Blank breathing room added on *every* side of the content, so the virtual
# canvas is always bigger than the viewport and can be dragged freely in any
# direction (like a map) -- even when the graph itself is small. Content is
# offset by this much from the origin (room to pan up/left), and the same pad
# is appended past the content (room to pan down/right).
_CANVAS_PAD = 12

# collapse/expand carets shown when a node has a body to reveal.
_CARET_COLLAPSED = "▸"
_CARET_EXPANDED = "▾"
# Shown on a node header when it has long input/response to open full-screen.
_INSPECT_GLYPH = "⤢"

# status codes kept as plain data so the model is easy to assert on.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_OK = "ok"
STATUS_FAIL = "fail"
STATUS_INFO = "info"

_STATUS_GLYPH = {
    STATUS_RUNNING: ("⏳", "yellow"),
    STATUS_OK: ("✓", "green"),
    STATUS_FAIL: ("✗", "red bold"),
}


def _wrap_rows(text: str, width: int, limit: int) -> List[str]:
    """Soft-wrap ``text`` to ``width`` cells, returning up to ``limit`` rows."""
    if not text:
        return []
    console = Console(width=max(1, width), file=io.StringIO(), highlight=False)
    console.print(Text(text, no_wrap=False, overflow="fold"))
    rows = [ln.rstrip() for ln in console.file.getvalue().splitlines()]
    rows = [ln for ln in rows if ln != ""] or [""]
    if len(rows) > limit:
        rows = rows[: limit - 1] + [rows[limit - 1][: width - 1] + "…"]
    return rows


@dataclass
class GraphNode:
    """One box in the flow canvas (pure data; a FlowNode widget renders it).

    The node stores the *full* message (``full_detail``); how much of it is
    shown depends on ``expanded``. ``height`` is therefore dynamic, which the
    layout pass consumes so an expanded node pushes its neighbors down instead
    of overlapping them.
    """
    key: str                       # stable id, e.g. 'llm-1' or 'tool-2'
    source: str                    # Source.* value -> icon/color lookup
    title: str                     # primary label (model name, tool name, ...)
    detail: str = ""               # short collapsed preview (one line)
    full_detail: str = ""          # complete message revealed when expanded
    status: str = STATUS_PENDING
    status_text: str = ""
    x: int = 0                     # canvas position (cells), top-left corner
    y: int = 0
    parent: Optional[str] = None   # key of the node this one flows from
    kind: str = "step"             # step | user | answer | notice
    expanded: bool = False         # whether the full body is shown
    full_input: str = ""           # long model INPUT (prompt) for the inspector
    full_response: str = ""        # long model RESPONSE (raw) for the inspector
    # Wall-clock timing (monotonic seconds). ``started_at`` is stamped when
    # the node enters RUNNING; ``finished_at`` when it settles (ok/fail).
    # A running node has ``finished_at is None`` -> its timer ticks live.
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def width(self) -> int:
        return NODE_WIDTH

    @property
    def message(self) -> str:
        """The complete text this node can reveal (body when expanded)."""
        return self.full_detail or self.detail or self.status_text

    @property
    def collapsible(self) -> bool:
        """True when there is a body worth expanding/collapsing."""
        return bool(self.message.strip())

    @property
    def inspectable(self) -> bool:
        """True when this node has long input/response worth a full-screen view."""
        return bool(self.full_input.strip() or self.full_response.strip())

    @property
    def expandable(self) -> bool:
        """True when this node offers the INLINE expand/collapse affordance.

        A node that is :attr:`inspectable` (e.g. a model call) opens a
        full-screen popup instead, so it deliberately has NO inline
        expand/collapse -- two different expand gestures in two different
        spots is confusing. Only non-inspectable nodes with a body toggle
        inline.
        """
        return self.collapsible and not self.inspectable

    def inspect_sections(self) -> List[tuple]:
        """Return ``(label, body)`` sections for the inspector modal."""
        sections: List[tuple] = []
        if self.full_input.strip():
            sections.append(("INPUT", self.full_input.rstrip()))
        if self.full_response.strip():
            sections.append(("RESPONSE", self.full_response.rstrip()))
        return sections

    def body_rows(self) -> List[str]:
        """The body text rows to render given the current expanded state."""
        if not self.collapsible:
            return []
        # Inspectable nodes are popup-only: they never inline-expand, so they
        # always show just the one-line preview (detail lives in the modal).
        if self.expanded and self.expandable:
            return _wrap_rows(self.message, _NODE_INNER_WIDTH, _MAX_BODY_ROWS)
        preview = self.detail or self.status_text or self.message
        preview = preview.strip().replace(chr(10), " ")
        return _wrap_rows(preview, _NODE_INNER_WIDTH, _COLLAPSED_PREVIEW_ROWS)

    @property
    def height(self) -> int:
        """Box height in cells: border + header + however many body rows."""
        return _BORDER_ROWS + HEADER_ROWS + len(self.body_rows())

    @property
    def is_running(self) -> bool:
        return self.status == STATUS_RUNNING

    def duration(self, now: float) -> Optional[float]:
        """Elapsed seconds for this node, or ``None`` if it never started.

        Frozen once finished; live (measured against ``now``) while running.
        """
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else now
        return max(0.0, end - self.started_at)


@dataclass
class GraphEdge:
    """A directed connector from one node (parent) to another (child)."""
    src: str
    dst: str


class FlowPanel(Container):
    """Interactive, pannable flow canvas that *projects* session telemetry.

    The panel does not own turn logic; it is a pure view over an append-only
    list of :class:`AgentEvent` (the agent's ``telemetry``). Every turn stays on
    the canvas (turns are delimited by ``turn_start``/``turn_end``), so context
    is preserved.

    Public API:
        - set_history(events): rebuild the whole canvas from a full telemetry log.
        - append(event): project one newly-emitted event (incremental fast path).
        - clear(): empty the canvas.
        - toggle(key) / set_expanded(key, value): reveal or hide a node's message.
        - collapse_all() / expand_all(): bulk expansion helpers.

    Layout (positions) is recomputed from the model by :meth:`_relayout` after
    every change, so it stays a pure function of the data -- the seam a future
    DAG layout would replace.
    """

    DEFAULT_CSS = """
    FlowPanel {
        layout: vertical;
    }
    FlowPanel > FlowCanvas {
        width: 1fr;
        height: 1fr;
        /* Scrollable so scroll_to() works for grab-to-pan and auto-follow, but
           the scrollbar chrome is hidden (size 0): the canvas only moves by
           naturally dragging it. */
        overflow: auto auto;
        scrollbar-size: 0 0;
        layers: edges nodes;
    }
    FlowCanvas > _EdgeLayer {
        layer: edges;
    }
    FlowCanvas > FlowNode {
        layer: nodes;
        position: absolute;
        width: 30;
        height: auto;
        border: round $primary;
        background: $panel;
        padding: 0 1;
    }
    /* Border colour encodes the node TYPE (its source): each kind of step --
       llm / tool / mcp / skill / agent -- gets its own hue, and the user /
       answer / notice nodes keep theirs. See FlowNode.apply, which adds a
       `src-<label>` (or kind) class. */
    FlowCanvas > FlowNode.src-llm    { border: round $primary; }
    FlowCanvas > FlowNode.src-tool   { border: round $success; }
    FlowCanvas > FlowNode.src-mcp    { border: round $secondary; }
    FlowCanvas > FlowNode.src-skill  { border: round $warning; }
    FlowCanvas > FlowNode.src-agent  { border: round $accent; }
    FlowCanvas > FlowNode.src-system { border: round $primary-darken-2; }
    FlowCanvas > FlowNode.notice  { border: round $accent; }
    FlowCanvas > FlowNode.user    { border: round $secondary; }
    FlowCanvas > FlowNode.answer  { border: round $success; }
    /* A failure always shows a red border regardless of type (glyph aside). */
    FlowCanvas > FlowNode.fail    { border: round $error; }
    /* Running nodes gently PULSE their border toward an APPROXIMATE ACCENT of
       their own type colour (a lighter shade of the same hue -- never a
       different colour like amber). The border STYLE stays ``round`` in both
       phases so the edge never thickens/jumps; the .blink class (toggled
       ~1.1Hz) is the brighter half, so it reads as a soft breath. */
    FlowCanvas > FlowNode.src-llm.blink    { border: round $primary-lighten-3; }
    FlowCanvas > FlowNode.src-tool.blink   { border: round $success-lighten-3; }
    FlowCanvas > FlowNode.src-mcp.blink    { border: round $secondary-lighten-3; }
    FlowCanvas > FlowNode.src-skill.blink  { border: round $warning-lighten-3; }
    FlowCanvas > FlowNode.src-agent.blink  { border: round $accent-lighten-3; }
    FlowCanvas > FlowNode.src-system.blink { border: round $primary-darken-1; }
    FlowCanvas > FlowNode.expanded { background: $boost; }
    FlowCanvas.panning { background: $boost; }
    """

    def __init__(self) -> None:
        super().__init__(id="flow")
        self._model: Dict[str, GraphNode] = {}
        self._order: List[str] = []
        self._edges: List[GraphEdge] = []
        self._counter = 0
        # keys of the most recent llm/tool nodes so a *_result updates them.
        self._pending_llm: Optional[str] = None
        self._pending_tool: Optional[str] = None
        self._last_key: Optional[str] = None   # for chaining edges
        self._turns = 0                        # how many turns are on the canvas
        self._canvas: Optional[FlowCanvas] = None
        # Live animation for RUNNING nodes: a ~10fps timer advances a pulse
        # phase (for the gentle border pulse) and repaints running nodes (for
        # the live timer). Injectable clock keeps duration tests deterministic.
        self._clock = time.monotonic
        self._pulse = 0
        self._tick_timer = None

    # --- Textual lifecycle ---------------------------------------------
    def compose(self):
        self._canvas = FlowCanvas(self)
        yield self._canvas

    def on_mount(self) -> None:
        self._sync_widgets()

    # --- public API (projection over telemetry) ------------------------
    def set_history(self, events: List[AgentEvent]) -> None:
        """Rebuild the entire canvas from a full telemetry log (idempotent)."""
        self._stop_ticking()
        self._reset_model()
        for event in events:
            self._project_event(event)
        # A rebuild replays history: nothing is genuinely live, so freeze any
        # node the log left running (e.g. an interrupted turn) instead of
        # letting its timer count up from a rebuild timestamp.
        self._freeze_running()
        self._relayout()
        if self._canvas is not None:
            self._canvas.follow_tail = True
        self._sync_widgets(follow=True)

    def append(self, event: AgentEvent) -> None:
        """Project a single newly-emitted event (incremental fast path)."""
        self._project_event(event)
        self._relayout()
        self._sync_widgets(follow=True)
        self._ensure_ticking()

    def clear(self) -> None:
        """Empty the canvas (e.g. after the conversation is reset)."""
        self._reset_model()
        self._sync_widgets()

    def toggle(self, key: str) -> None:
        """Flip one INLINE-expandable node between collapsed/expanded.

        Inspectable nodes are popup-only (see ``GraphNode.expandable``), so
        they are ignored here -- their click opens the inspector instead.
        """
        node = self._model.get(key)
        if node is None or not node.expandable:
            return
        node.expanded = not node.expanded
        self._relayout()
        self._sync_widgets()

    class InspectRequested(Message):
        """Posted when a node's inspect button is clicked.

        Carries the node title and its ``(label, body)`` sections so the app
        can open a full-screen modal without reaching back into the model.
        """
        def __init__(self, title: str, sections: List[tuple]) -> None:
            super().__init__()
            self.title = title
            self.sections = sections

    def open_inspector(self, key: str) -> None:
        """Request the app to open the full-screen inspector for ``key``."""
        node = self._model.get(key)
        if node is None or not node.inspectable:
            return
        self.post_message(self.InspectRequested(node.title, node.inspect_sections()))

    def set_expanded(self, key: str, value: bool) -> None:
        node = self._model.get(key)
        if node is None:
            return
        # Only inline-expandable nodes carry an expanded body (inspectable
        # nodes are popup-only), so never mark those expanded.
        node.expanded = bool(value) and node.expandable
        self._relayout()
        self._sync_widgets()

    def expand_all(self) -> None:
        for node in self._model.values():
            node.expanded = node.expandable
        self._relayout()
        self._sync_widgets()

    def collapse_all(self) -> None:
        for node in self._model.values():
            node.expanded = False
        self._relayout()
        self._sync_widgets()

    # --- projection internals ------------------------------------------
    def _reset_model(self) -> None:
        self._model.clear()
        self._order.clear()
        self._edges.clear()
        self._counter = 0
        self._pending_llm = None
        self._pending_tool = None
        self._last_key = None
        self._turns = 0

    def _project_event(self, event: AgentEvent) -> None:
        """Mutate the model from one telemetry event (no clearing)."""
        handler = {
            "turn_start": self._on_turn_start,
            "llm_call": self._on_llm_call,
            "llm_result": self._on_llm_result,
            "tool_call": self._on_tool_call,
            "tool_result": self._on_tool_result,
            "notice": self._on_notice,
            "turn_end": self._on_turn_end,
        }.get(event.kind)
        if handler is not None:
            handler(event)

    def resume_follow(self) -> None:
        """Re-enable auto-follow and jump to the newest node."""
        if self._canvas is not None:
            self._canvas.resume_follow()

    # accessors used by tests
    def nodes(self) -> List[GraphNode]:
        return [self._model[k] for k in self._order]

    def edges(self) -> List[GraphEdge]:
        return list(self._edges)

    # --- per-event handlers --------------------------------------------
    def _on_turn_start(self, event: AgentEvent) -> None:
        """Open a new turn: add a user node, keeping prior turns on the canvas."""
        self._last_key = None
        self._turns += 1
        text = (event.text or "").strip()
        self._add_node(
            source=Source.SYSTEM, title="🧑 you",
            detail=_preview(text), full_detail=text,
            status=STATUS_INFO, kind="user",
        )
        if self._canvas is not None:
            self._canvas.follow_tail = True

    def _on_llm_call(self, event: AgentEvent) -> None:
        key = self._add_node(
            source=event.source, title=event.name or "model",
            status=STATUS_RUNNING, kind="step",
        )
        # Keep the exact prompt around so the inspector modal can show it.
        self._model[key].full_input = (event.text or "").strip()
        self._model[key].started_at = self._clock()
        self._pending_llm = key
        self._ensure_ticking()

    def _on_llm_result(self, event: AgentEvent) -> None:
        if self._pending_llm is None:
            return
        node = self._model[self._pending_llm]
        node.status = STATUS_OK if event.ok else STATUS_FAIL
        node.finished_at = self._clock()
        text = (event.text or "").strip()
        node.status_text = text
        # The raw model reply is the long RESPONSE for the inspector, and also
        # the node's expandable inline body (fall back to the kind label).
        raw = (event.output or "").strip()
        node.full_response = raw
        node.full_detail = raw or node.full_detail or text
        self._pending_llm = None

    def _on_tool_call(self, event: AgentEvent) -> None:
        full = ""
        if event.arguments:
            full = ", ".join(f"{k}={v}" for k, v in event.arguments.items())
        key = self._add_node(
            source=event.source, title=event.name or "?",
            detail=_preview(full), full_detail=full,
            status=STATUS_RUNNING, kind="step",
        )
        self._model[key].started_at = self._clock()
        self._pending_tool = key
        self._ensure_ticking()

    def _on_tool_result(self, event: AgentEvent) -> None:
        if self._pending_tool is None:
            return
        node = self._model[self._pending_tool]
        node.status = STATUS_OK if event.ok else STATUS_FAIL
        node.finished_at = self._clock()
        out = (event.output or "").strip()
        node.status_text = _preview(out)
        # Prefer showing the full result as the expandable body.
        node.full_detail = out or node.full_detail
        self._pending_tool = None

    def _on_notice(self, event: AgentEvent) -> None:
        self._freeze_running()
        text = (event.text or "").strip()
        self._add_node(
            source=Source.SYSTEM, title="⚠ notice",
            detail=_preview(text), full_detail=text,
            status=STATUS_INFO, kind="notice",
            expanded=True,   # notices auto-expand so the message is visible
        )

    def _on_turn_end(self, event: AgentEvent) -> None:
        self._freeze_running()
        text = (event.text or "").strip()
        self._add_node(
            source=Source.SYSTEM, title="✅ answer",
            detail=_preview(text), full_detail=text,
            status=STATUS_OK, kind="answer",
        )

    # --- model mutation helpers ----------------------------------------
    def _new_key(self, source: str) -> str:
        self._counter += 1
        label = _style_for(source).get("label", "node")
        return f"{label}-{self._counter}"

    def _add_node(self, source, title, detail="", full_detail="",
                  status=STATUS_PENDING, kind="step", expanded=False) -> str:
        """Append a node, chaining an edge from the previous node."""
        key = self._new_key(source)
        parent = self._last_key
        node = GraphNode(
            key=key, source=source, title=title, detail=detail,
            full_detail=full_detail, status=status, parent=parent,
            kind=kind, expanded=expanded,
        )
        self._model[key] = node
        self._order.append(key)
        if parent is not None:
            self._edges.append(GraphEdge(src=parent, dst=key))
        self._last_key = key
        return key

    # --- layout (pure function of the model) ---------------------------
    def _relayout(self) -> None:
        """Recompute every node's (x, y) from the model.

        Default layout is a single vertical column that stacks nodes using their
        *actual* heights (so expanding a node reflows the ones below it). Turns
        are separated by an extra gap. This method is intentionally the only
        place positions are decided: a future DAG layout (assigning columns by
        graph depth, using ``_H_GAP`` for horizontal breadth) drops in here
        without changing rendering, hit-testing or panning.
        """
        y = _CANVAS_PAD
        prev_turn_root: Optional[str] = None
        for key in self._order:
            node = self._model[key]
            # A turn root (parent is None) after the first adds a turn gap.
            if node.parent is None and prev_turn_root is not None:
                y += _TURN_GAP
            node.x = _CANVAS_PAD
            node.y = y
            y += node.height + _V_GAP
            if node.parent is None:
                prev_turn_root = key

    # --- widget materialization ----------------------------------------
    def _sync_widgets(self, follow: bool = False) -> None:
        """Create/update/remove FlowNode widgets to match the model.

        ``follow`` auto-scrolls to the newest node -- only appends/rebuilds
        do that. Toggling/expanding a node reflows *without* chasing the tail,
        so clicking an old node keeps it under the cursor instead of yanking
        the view to the bottom.
        """
        if self._canvas is None or not self.is_mounted:
            return
        self._canvas.sync(self._model, self._order, follow=follow)

    # --- live timing / blink -------------------------------------------
    def _any_running(self) -> bool:
        return any(n.is_running for n in self._model.values())

    def _freeze_running(self) -> None:
        """Stamp ``finished_at`` on any node still marked RUNNING.

        Used when a turn ends / is interrupted and after a history rebuild,
        so an abandoned running node stops ticking (its timer freezes at the
        moment it was frozen instead of counting up forever).
        """
        now = self._clock()
        for node in self._model.values():
            if node.is_running and node.finished_at is None:
                node.finished_at = now

    def _ensure_ticking(self) -> None:
        """Start the ~10fps animation timer if a node is running.

        100ms is the fastest the millisecond timer updates (per the spec) --
        fast enough to feel live, slow enough to stay cheap. The timer only
        repaints running nodes, and stops itself once none remain.
        """
        if self._tick_timer is not None or not self.is_mounted:
            return
        if not self._any_running():
            return
        self._tick_timer = self.set_interval(0.1, self._tick)

    def _stop_ticking(self) -> None:
        if self._tick_timer is not None:
            self._tick_timer.stop()
            self._tick_timer = None

    def _tick(self) -> None:
        """One animation frame: advance the pulse and repaint running nodes.

        Cheap by design: it does NOT reflow the layout (positions/heights are
        unchanged) -- it only re-renders the content of the nodes that are
        currently running, then stops the timer when none are left.
        """
        self._pulse = (self._pulse + 1) % 1000
        if self._canvas is not None:
            self._canvas.repaint_running(self._model)
        if not self._any_running():
            self._stop_ticking()

    @property
    def pulse_on(self) -> bool:
        """Whether the pulse is in its (slightly) brighter half this frame.

        Softer than a strobe: with a 0.1s tick, a 9-frame half-period makes the
        border ease between its two amber shades roughly every ~0.45s (~1.1Hz),
        which reads as a calm breath rather than a hard blink. Decoupled from
        the live ms timer, which repaints every tick regardless.
        """
        return (self._pulse // 9) % 2 == 0

    def node_content(self, node: GraphNode) -> Text:
        """Build the Rich text shown inside a node box.

        Row 1 is the header (source icon + title + status glyph + a caret when
        the node has a body). Remaining rows are the body: a one-line preview
        while collapsed, or the full wrapped message while expanded.
        """
        style = _style_for(node.source)
        color = style["color"]
        text = Text(no_wrap=True, overflow="ellipsis")
        text.append(f"{style['icon']} ", style=color)
        text.append(node.title, style=f"{color} bold")
        # The RUNNING state is signalled by a gently pulsing *border* (see the
        # .running/.blink CSS + FlowNode.apply), not a glyph in the header,
        # so the header just shows the status glyph and the live timer.
        glyph = _STATUS_GLYPH.get(node.status)
        if glyph:
            text.append(f" {glyph[0]}", style=glyph[1])
        # Live/elapsed timer: ms under 2s, seconds beyond (see _format_duration).
        label = _format_duration(node.duration(self._clock()))
        if label:
            text.append(f" {label}", style="grey58")
        if node.expandable:
            # Inline expand/collapse caret (only for popup-less nodes).
            caret = "▾" if node.expanded else "▸"
            text.append(f" {caret}", style="grey58")
        if node.inspectable:
            # Popup affordance: clicking ANYWHERE on this node opens the
            # full-screen view (no inline caret -- one gesture, one place).
            text.append(f" {_INSPECT_GLYPH}", style="bright_cyan bold")
        for row in node.body_rows():
            text.append(chr(10) + row, style="grey70")
        return text


# Below this many seconds a node's timer is shown in milliseconds; at/above
# it, in seconds. (User spec: <2s -> ms, >=2s -> s.)
_MS_THRESHOLD_S = 2.0


def _format_duration(seconds: Optional[float]) -> str:
    """Human timer label: ms under 2s, seconds at/after 2s.

    Examples: ``12ms``, ``840ms``, ``1999ms``, ``2.0s``, ``3.4s``, ``12s``.
    Returns '' when the node never started (nothing to show).
    """
    if seconds is None:
        return ""
    if seconds < _MS_THRESHOLD_S:
        return f"{int(seconds * 1000)}ms"
    if seconds < 10.0:
        return f"{seconds:.1f}s"
    return f"{int(seconds)}s"


def _preview(text: str) -> str:
    """One-line, length-capped preview of a possibly multi-line message."""
    line = (text or "").strip().replace(chr(10), " ")
    if len(line) > 24:
        line = line[:23] + "…"
    return line


class FlowCanvas(ScrollableContainer):
    """Pannable drawing area holding the edge layer plus node widgets.

    Two mouse gestures share the background:
      * a *drag* pans the whole canvas (grab-to-pan) by moving the scroll offset
        opposite to the cursor;
      * a *click* (press+release with no meaningful movement) on a node toggles
        that node's expand/collapse state.

    Auto-follow: while ``follow_tail`` is on, the canvas scrolls to keep the most
    recent node visible; panning turns it off, a new turn turns it back on.
    """

    # A press that moves more than this many cells counts as a pan, not a click.
    _CLICK_SLOP = 1

    def __init__(self, panel: "FlowPanel") -> None:
        super().__init__()
        self._panel = panel
        self._edge_layer = _EdgeLayer(panel)
        self._widgets: Dict[str, FlowNode] = {}
        # panning state
        self._pan_from: Optional[Offset] = None       # cursor screen pos at grab
        self._pan_origin: Offset = Offset(0, 0)        # scroll offset at grab
        self._moved: bool = False                      # did this drag move enough?
        self.follow_tail: bool = True                  # auto-scroll to newest node
        self._last_key: Optional[str] = None           # newest node to follow

    def compose(self):
        yield self._edge_layer

    def on_mount(self) -> None:
        self.sync(self._panel._model, self._panel._order)

    def on_resize(self, event: events.Resize) -> None:
        # Re-flow so the virtual area (and thus pan slack) tracks the new
        # pane size -- keeps free dragging available after a terminal resize.
        self.reflow(self._panel._model, self._panel._order)

    # --- widget reconciliation -----------------------------------------
    def sync(self, model: Dict[str, GraphNode], order: List[str],
             follow: bool = False) -> None:
        """Reconcile FlowNode widgets with the model (add/remove), then reflow.

        ``follow`` is forwarded to :meth:`reflow`; only appends/rebuilds set it,
        so a plain expand/collapse reflow does not auto-scroll.
        """
        if not self.is_mounted:
            return
        wanted = set(order)
        for key in list(self._widgets):
            if key not in wanted:
                self._widgets.pop(key).remove()
        for key in order:
            if key not in self._widgets:
                node = model[key]
                widget = FlowNode(self._panel, node.key)
                self._widgets[key] = widget
                self.mount(widget)
        self._last_key = order[-1] if order else None
        self.reflow(model, order, follow=follow)

    def reflow(self, model: Dict[str, GraphNode], order: List[str],
               follow: bool = False) -> None:
        """Push positions/content/height into widgets and resize the canvas.

        Auto-scrolls to the newest node only when ``follow`` is set (an append
        or a rebuild) and auto-follow is still enabled -- never on a toggle.
        """
        if not self.is_mounted:
            return
        max_x = 0
        max_y = 0
        for key in order:
            node = model[key]
            widget = self._widgets.get(key)
            if widget is None:
                continue
            widget.apply(node, self._panel.node_content(node))
            max_x = max(max_x, node.x + node.width)
            max_y = max(max_y, node.y + node.height)
        # The edge layer (a normal child) defines the scrollable virtual area.
        # Make it bigger than the viewport in BOTH axes so the canvas can
        # always be dragged freely like a map -- even when the graph is small
        # or narrower than the pane. We floor each dimension at
        # ``viewport + _CANVAS_PAD`` (giving pan slack when content is small)
        # and also honor ``content + _CANVAS_PAD`` (slack past a large graph).
        viewport = self.size
        virtual_w = max(max_x + _CANVAS_PAD, viewport.width + _CANVAS_PAD)
        virtual_h = max(max_y + _CANVAS_PAD, viewport.height + _CANVAS_PAD)
        self._edge_layer.styles.width = virtual_w
        self._edge_layer.styles.height = virtual_h
        self._edge_layer.refresh()
        if follow and self.follow_tail:
            self.call_after_refresh(self._scroll_to_tail)

    def repaint_running(self, model: Dict[str, GraphNode]) -> None:
        """Re-render only the RUNNING node widgets (animation fast path).

        Called every animation frame. Deliberately avoids :meth:`reflow` --
        running nodes don't change size, so we just push fresh content into
        the few running widgets. That keeps the ~10fps blink/timer cheap even
        with a large graph.
        """
        if not self.is_mounted:
            return
        for key, widget in self._widgets.items():
            node = model.get(key)
            if node is not None and node.is_running:
                widget.apply(node, self._panel.node_content(node))

    def _scroll_to_tail(self) -> None:
        """Scroll so the most recently added node is visible."""
        if self._last_key is None:
            return
        node = self._panel._model.get(self._last_key)
        if node is None:
            return
        viewport_h = self.size.height
        viewport_w = self.size.width
        target_y = max(0, node.y + node.height + _MARGIN_Y - viewport_h)
        target_x = max(0, node.x + node.width + _MARGIN_X - viewport_w)
        self.scroll_to(target_x, target_y, animate=False, immediate=True)

    # --- background panning + click-to-toggle --------------------------
    def on_mouse_down(self, event: events.MouseDown) -> None:
        self._pan_from = event.screen_offset
        self._pan_origin = self.scroll_offset
        self._moved = False
        self.capture_mouse()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._pan_from is None:
            return
        delta = event.screen_offset - self._pan_from
        if abs(delta.x) > self._CLICK_SLOP or abs(delta.y) > self._CLICK_SLOP:
            if not self._moved:
                # First real movement: this gesture is a pan, not a click.
                self._moved = True
                self.follow_tail = False
                self.add_class("panning")
            # Clamp the target to the valid scroll range *ourselves*, then
            # apply it immediately. Without this, dragging past an edge asked
            # Textual to scroll out of range; the deferred/eased scroll then
            # kept gliding to the clamped max after the mouse stopped -- the
            # "drift at the boundary" bug. immediate=True + no animation makes
            # panning track the cursor 1:1 and stop crisply at the edge.
            target_x = _clamp(self._pan_origin.x - delta.x, 0, self.max_scroll_x)
            target_y = _clamp(self._pan_origin.y - delta.y, 0, self.max_scroll_y)
            self.scroll_to(target_x, target_y, animate=False, immediate=True)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._pan_from is None:
            return
        was_click = not self._moved
        self._pan_from = None
        self.remove_class("panning")
        self.release_mouse()
        if was_click:
            key = self._node_key_at(event.screen_offset)
            if key is not None:
                self._handle_node_click(key, event.screen_offset)

    def _node_key_at(self, screen_offset: Offset) -> Optional[str]:
        """Return the key of the node whose box contains ``screen_offset``."""
        for key, widget in self._widgets.items():
            region = widget.region
            if region.contains(screen_offset.x, screen_offset.y):
                return key
        return None

    def _handle_node_click(self, key: str, screen_offset: Offset) -> None:
        """Route a node click to the RIGHT single gesture.

        Inspectable nodes (e.g. a model call) are popup-only: a click
        ANYWHERE on the box opens the full-screen inspector -- there is no
        inline expand for them, so the two-gestures-in-two-places confusion
        is gone. Every other node with a body toggles inline as before.
        """
        node = self._panel._model.get(key)
        if node is not None and node.inspectable:
            self._panel.open_inspector(key)
            return
        self._panel.toggle(key)

    def resume_follow(self) -> None:
        """Re-enable auto-follow and jump to the newest node."""
        self.follow_tail = True
        self._scroll_to_tail()


class FlowNode(Static):
    """A positioned box representing one graph node.

    Nodes are not individually draggable; the whole canvas pans when you drag
    the background. Clicking a node toggles its expansion (handled by the
    canvas). A node renders its content at the model position, sizes itself to
    the model height, and reflects status/expansion via CSS classes.
    """

    def __init__(self, panel: "FlowPanel", key: str) -> None:
        super().__init__()
        self._panel = panel
        self._key = key

    # Border-colour classes managed here: the per-source type hue, the
    # user/answer/notice kinds, the fail override, plus running/blink/expanded.
    _MANAGED_CLASSES = (
        "running", "fail", "blink", "expanded",
        "user", "answer", "notice",
        "src-llm", "src-tool", "src-mcp", "src-skill", "src-agent", "src-system",
    )

    def apply(self, node: GraphNode, content: Text) -> None:
        """Update position, height, type/status/expansion classes and content."""
        self.styles.offset = (node.x, node.y)
        self.styles.height = node.height
        for cls in self._MANAGED_CLASSES:
            self.remove_class(cls)
        # Border colour = node TYPE. user/answer/notice keep their kind hue;
        # every other (step) node is coloured by its source (llm/tool/mcp/...).
        if node.kind in ("user", "answer", "notice"):
            self.add_class(node.kind)
        else:
            label = _style_for(node.source).get("label", "system")
            self.add_class(f"src-{label}")
        # A failure paints the border red regardless of type (the header also
        # shows the fail glyph); running keeps the type hue and pulses it.
        if node.status == STATUS_FAIL:
            self.add_class("fail")
        if node.is_running:
            self.add_class("running")
            # Border pulse: on the brighter half of the pulse, a running node
            # gets .blink, which eases the border to a lighter accent of its
            # OWN type colour (same round style, so no thickness jump). The
            # animation timer repaints running nodes each frame.
            if self._panel.pulse_on:
                self.add_class("blink")
        if node.expanded:
            self.add_class("expanded")
        self.update(content)


class _EdgeLayer(Static):
    """Fills the canvas and paints connector edges behind the node boxes.

    Each edge is an orthogonal connector from the parent box's bottom-CENTER
    straight down to the child box's top-CENTER (plain line, no arrow head --
    the flow reads top-to-bottom as time). In the current single-column layout
    the two centers share a column, so the line is a clean vertical exiting/
    entering each box at its horizontal middle; if a future DAG layout puts the
    child in a different column, the line drops from the parent center then
    elbows across to the child center. Positions and heights are read live from
    the model, so edges stay correct as nodes expand, collapse, or pan.
    """

    def __init__(self, panel: "FlowPanel") -> None:
        super().__init__()
        self._panel = panel

    def _edge_style(self, src_key: str) -> Style:
        node = self._panel._model.get(src_key)
        color = _style_for(node.source)["color"] if node is not None else "grey50"
        return Style.parse(color)

    def render_line(self, y: int) -> Strip:
        """Render one horizontal slice (row ``y``) of the edge layer."""
        width = self.size.width
        if width <= 0:
            return Strip.blank(0)
        cells: List[Optional[tuple]] = [None] * width

        V = chr(0x2502); H = chr(0x2500); CORNER = chr(0x2514)

        for edge in self._panel._edges:
            src = self._panel._model.get(edge.src)
            dst = self._panel._model.get(edge.dst)
            if src is None or dst is None:
                continue
            style = self._edge_style(edge.src)
            # Use (width - 1) // 2 (the rounded true center) so an even-width
            # box's line sits on the LEFT of the two middle cells rather than
            # biased a half-cell right (which read as an off-center, corner-ish
            # exit). For width 30 (border cols 12..41, center 26.5) this is
            # col 26, matching round(center).
            sx = src.x + (src.width - 1) // 2    # parent bottom-CENTER (exit)
            sy = src.y + src.height - 1          # bottom border row of the parent
            dx = dst.x + (dst.width - 1) // 2    # child top-CENTER (entry)
            dy = dst.y                           # top border row of the child
            # Drop down the parent's center column to the child's top row; if
            # the child is in the same column (today's single-column stack)
            # this is a straight vertical, exiting/entering both boxes at their
            # horizontal middle. Elbow across only when the columns differ.
            elbow_y = dy

            def put(px: int, ch: str) -> None:
                if 0 <= px < width and 0 <= y < self.size.height:
                    cells[px] = (ch, style)

            top = min(sy, elbow_y)
            bottom = max(sy, elbow_y)
            if top <= y < bottom and y >= 0:
                put(sx, V)
            if y == elbow_y and sx != dx:
                # Different columns: turn at the child's top row and run across
                # to the child center. No arrow head (top-to-bottom == time).
                put(sx, CORNER)
                x0 = min(sx, dx)
                x1 = max(sx, dx)
                for px in range(x0 + 1, x1 + 1):
                    put(px, H)

        segments: List[Segment] = []
        run_text = ""
        run_style: Optional[Style] = None
        for cell in cells:
            if cell is None:
                ch, st = " ", None
            else:
                ch, st = cell
            if st is run_style:
                run_text += ch
            else:
                if run_text:
                    segments.append(Segment(run_text, run_style))
                run_text = ch
                run_style = st
        if run_text:
            segments.append(Segment(run_text, run_style))
        return Strip(segments, width)
