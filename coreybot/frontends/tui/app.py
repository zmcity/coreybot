"""A Codex-style terminal UI: compact chat on the left, a live flow chart on
the right.

Layout (no top bar -- the top of the terminal is free for overflowing chat)
    ┌───────────────────────────┬──────────────────────┐
    │ chat (compact bubbles)    │ FlowPanel (live tree) │
    └───────────────────────────┴──────────────────────┘
    Input
    StatusBar   (one row: animated info on the left + clickable key hints on
                 the right; replaces both Header and Footer)

Compactness: chat messages are borderless, zero-margin lines with a colored
role prefix, so the transcript reads densely like a real chat client.

Live flow: every ``AgentEvent`` is fed to both the transcript and the
``FlowPanel`` (right), which draws the current turn as a growing tree
(user -> llm -> tool -> llm -> answer) with per-source icons and live status.

Async + cancellation: a turn runs as a native async worker on Textual's own
event loop (no background thread), so UI updates need no thread marshaling and
pressing ``Esc`` cancels the in-flight turn via a Go-style ``CancelToken``.
"""

from __future__ import annotations

import os
from pathlib import Path

from rich.console import RenderableType
from rich.markdown import Markdown
from rich.segment import Segment
from rich.style import Style
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.geometry import Offset
from textual.message import Message as TextualMessage
from textual.containers import (
    Horizontal,
    ScrollableContainer,
    Vertical,
    VerticalScroll,
)
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.widgets import (
    Button,
    Input,
    ListItem,
    ListView,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from coreybot.runtime.agent import Agent, AgentEvent
from coreybot.runtime.session import CLEAR_LABEL
from coreybot.core.cancel import CancelToken, CancelledError
from coreybot.core.config import Config
from coreybot.core.message import Role
from coreybot.llm.protocol import parse_agent_response
from coreybot.llm.providers import available_providers
from coreybot.frontends.tui.flow import FlowPanel


# Compact, single-line role prefixes (no heavy borders).
_ROLE_PREFIX = {
    Role.USER: ("you", "bold cyan"),
    Role.ASSISTANT: ("bot", "bold green"),
    Role.SYSTEM: ("sys", "dim yellow"),
    Role.TOOL: ("tool", "bold magenta"),
}

# Standby 'breathing LED' for the status bar -- like the notification light on
# an early smartphone that pulses while the screen is off. A single dot whose
# brightness eases up and down (a genuine fade via truecolor, not a hard
# on/off), so it is calm enough to ignore yet clearly says 'the agent is
# alive'. State is conveyed by BOTH colour and blink speed.
_LED_GLYPH = "●"                 # the LED dot
_LED_OFF_RGB = (30, 30, 30)         # darkest point of the breath (nearly off)
# Per-state look: (r, g, b) fully-lit colour + breathing period in ticks
# (the timer ticks at ~0.12s, so period 16 ~= 1.9s per full breath). Add new
# entries here to signal more states (e.g. an error red fast-blink) later.
_LED_STATES = {
    "ready":   {"rgb": (80, 220, 120), "period": 16},   # calm green, slow
    "working": {"rgb": (240, 190, 60), "period": 8},    # amber, quicker pulse
}


def _display_answer(raw_reply: str) -> str:
    """Turn a stored assistant raw reply into the text a bubble shows.

    History keeps the model's raw ``<message>...</message>`` turn; the chat
    bubble shows the parsed message body (same as the live turn path). Used
    when rebuilding the transcript after a session checkout.
    """
    parsed = parse_agent_response(raw_reply)
    return parsed.content or raw_reply


class MessageBubble(Static):
    """A compact chat line: a colored ``prefix`` label followed by content.

    Assistant replies are rendered as Markdown (so inline code, fenced code
    blocks and blockquotes display richly); user/system/tool lines stay as
    dense single-column plain text. Either way ``text`` returns the raw string.
    """

    def __init__(
        self, role: Role, text: str = "", history_index: int = -1
    ) -> None:
        self.role = role
        self._text = text
        # Index of the message this bubble represents in ``agent.history``.
        # -1 means 'not tied to a history entry' (e.g. a system/help line).
        # Used by the per-bubble session modal (Edit rewinds to this index).
        self.history_index = history_index
        # Two-step open: the first click only *selects* the bubble; a second
        # click on the already-selected bubble opens the session modal. This
        # avoids opening the full-screen modal by accident.
        self._selected = False
        super().__init__(self.build_renderable(text))
        self.add_class(f"role-{role.value}")

    # Width of the fixed label column: "  you" (4) + " " + separator "│".
    _LABEL_WIDTH = 6

    def _label(self) -> "Text":
        label, style = _ROLE_PREFIX.get(self.role, (self.role.value, "white"))
        prefix = Text()
        prefix.append(f"{label:>4}", style=style)
        prefix.append(" │", style="grey37")
        return prefix

    def build_renderable(self, body: str) -> RenderableType:
        """Build the Rich renderable for ``body`` based on this bubble's role.

        A two-column ``Table.grid`` keeps the role label in a fixed-width left
        column and the content in the right column, so long/soft-wrapped or
        multi-line content stays under the content column and never bleeds into
        the ``you │`` / ``bot │`` label. Assistant content is Markdown; other
        roles are plain text.
        """
        shown = body if body else "…"
        content: RenderableType
        if self.role is Role.ASSISTANT:
            # Rich Markdown: fenced code blocks, inline code, quotes, lists.
            content = Markdown(shown, code_theme="ansi_dark")
        else:
            content = Text(shown)

        grid = Table.grid(padding=(0, 1, 0, 0), expand=True)
        grid.add_column(width=self._LABEL_WIDTH, no_wrap=True, vertical="top")
        grid.add_column(ratio=1, overflow="fold")
        grid.add_row(self._label(), content)
        return grid

    def set_text(self, text: str) -> None:
        self._text = text
        self.update(self.build_renderable(text))

    @property
    def text(self) -> str:
        return self._text

    def get_selection(self, selection):
        """Return this bubble's body text for drag-to-select + copy.

        Textual's default extraction only works when a widget renders a plain
        ``Text``/``Content``. Our bubble renders a two-column ``Table.grid``
        (label + content, with Markdown for answers), so the default returns
        ``None`` and nothing in the chat is copyable. We override it to yield
        the raw message body (never the ``you |`` prefix), so selecting a
        bubble copies clean text. When the selection covers only part of the
        body we honour it; otherwise the whole body is returned.
        """
        body = self._text
        if not body:
            return None
        extracted = None
        try:
            extracted = selection.extract(body)
        except Exception:
            extracted = None
        return (extracted if extracted else body, "\n")

    # --- selection + two-step open ------------------------------------
    class SelectRequested(TextualMessage):
        """Posted on the first click: ask the app to select this bubble."""

        def __init__(self, bubble: "MessageBubble") -> None:
            super().__init__()
            self.bubble = bubble

    class OpenRequested(TextualMessage):
        """Posted on the second click of a selected bubble: open its modal."""

        def __init__(self, bubble: "MessageBubble") -> None:
            super().__init__()
            self.bubble = bubble

    def set_selected(self, value: bool) -> None:
        """Toggle the visual 'selected' highlight (see the ``-selected`` CSS)."""
        self._selected = bool(value)
        self.set_class(self._selected, "-selected")

    @property
    def selected(self) -> bool:
        return self._selected

    def on_click(self, event: events.Click) -> None:
        """First click selects; a click while already selected opens the modal.

        Clicking never moves keyboard focus (Textual clicks don't focus a
        non-focusable widget, and the app pins focus to ``#prompt`` anyway),
        so typing still lands in the input even with a bubble selected.
        """
        event.stop()
        if self._selected:
            self.post_message(self.OpenRequested(self))
        else:
            self.post_message(self.SelectRequested(self))


class _KeyHint(Static):
    """A clickable key hint in the status bar (e.g. 'Esc interrupt').

    Restores the click-to-run behavior the old ``Footer`` gave: clicking the
    hint runs the same bound action as pressing the key. It renders the key in
    bold plus a dim label, and highlights on hover so it reads as clickable.
    """

    def __init__(self, key_label: str, action: str, description: str) -> None:
        super().__init__()
        self._key_label = key_label
        self._action = action
        self._description = description
        self.render_hint()

    def render_hint(self) -> None:
        text = Text(no_wrap=True)
        text.append(self._key_label, style="bold")
        text.append(f" {self._description}", style="grey62")
        self.update(text)

    async def on_click(self, event: events.Click) -> None:
        # Run the same action the key binding would (interrupt/clear/quit).
        # ``run_action`` is a coroutine, so it must be awaited; target the app
        # where these actions (action_interrupt/clear/quit) are defined.
        event.stop()
        await self.app.run_action(self._action)


class InspectModal(ModalScreen):
    """Full-screen viewer for a node's long INPUT / RESPONSE text.

    Opened from a flow-chart node's inspect button. Covers ~90% of the screen,
    grabs focus (a deliberate exception to the app's focus-pinned-to-input
    policy -- see ``ChatApp.on_descendant_focus``), scrolls, and can copy the
    whole body to the clipboard. ``Esc``/``q`` closes it; ``c`` copies.
    """

    DEFAULT_CSS = """
    InspectModal {
        align: center middle;
    }
    InspectModal > #inspect-box {
        width: 90%;
        height: 90%;
        border: round $primary;
        background: $surface;
        padding: 0 1;
    }
    InspectModal #inspect-title {
        height: 1;
        color: $text;
        text-style: bold;
        content-align: left middle;
    }
    InspectModal #inspect-body {
        height: 1fr;
        border: none;
    }
    InspectModal #inspect-actions {
        height: 1;
        align: left middle;
    }
    InspectModal #inspect-actions Button {
        height: 1;
        min-width: 10;
        margin: 0 1 0 0;
        border: none;
    }
    /* The hint is a flexible 1fr gap: it shows 'copied' AND shoves the Close
       button to the bottom-right corner, so every modal's exit is in the same
       place (mirrors the app status bar's right-aligned hints). */
    InspectModal #inspect-hint {
        width: 1fr;
        height: 1;
        color: $text 60%;
        content-align: left middle;
        padding: 0 1;
    }
    InspectModal #inspect-close { margin: 0; }
    """

    BINDINGS = [
        ("escape", "close", "Close"),
        ("q", "close", "Close"),
        ("c", "copy", "Copy"),
    ]

    def __init__(self, title: str, sections: list) -> None:
        super().__init__()
        self._title = title
        # Compose one labeled document from the (label, body) sections.
        parts = []
        for label, body in sections:
            parts.append(f"===== {label} =====\n{body}")
        self._body_text = "\n\n".join(parts) if parts else "(empty)"

    def compose(self) -> ComposeResult:
        with Vertical(id="inspect-box"):
            yield Static(f"\U0001f50d {self._title}", id="inspect-title")
            # A read-only TextArea gives scrolling + selection for free and is
            # focusable, so opening the modal moves focus here.
            area = TextArea(self._body_text, read_only=True, id="inspect-body")
            area.show_line_numbers = False
            yield area
            with Horizontal(id="inspect-actions"):
                # The shortcut is shown inside each button label. The hint is a
                # flexible 1fr gap that pushes Close to the bottom-right corner
                # (every modal keeps its exit in the same place).
                yield Button("Copy (c)", id="inspect-copy", variant="primary")
                yield Static("", id="inspect-hint")
                yield Button("Close (Esc)", id="inspect-close")

    def on_mount(self) -> None:
        self.query_one("#inspect-body", TextArea).focus()

    def action_close(self) -> None:
        self.dismiss(None)

    def action_copy(self) -> None:
        self.app.copy_to_clipboard(self._body_text)
        self.query_one("#inspect-hint", Static).update(" copied ✓")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "inspect-copy":
            self.action_copy()
        elif event.button.id == "inspect-close":
            self.action_close()


class SessionModal(ModalScreen):
    """Full-screen viewer for ONE chat message: read-only, selectable, copyable.

    Terminal-native drag-to-select is flaky in some terminals, so double-clicking
    a chat bubble opens this modal to reliably read / copy the message text. For
    USER messages it also offers ``Edit & resend`` (rewind the conversation to
    just before this message and prefill the input so a modified copy can be
    re-sent, which branches). The modal grabs focus (a deliberate exception to
    the focus-pinned-to-input policy) so the text is selectable; it stays
    decoupled from app state (Edit is posted as a message).
    """

    DEFAULT_CSS = """
    SessionModal {
        align: center middle;
    }
    SessionModal > #session-box {
        width: 90%;
        height: 90%;
        border: round $primary;
        background: $surface;
        padding: 0 1;
    }
    SessionModal #session-title {
        height: 1;
        color: $text;
        text-style: bold;
        content-align: left middle;
    }
    SessionModal #session-body {
        height: 1fr;
        border: none;
    }
    SessionModal #session-actions {
        height: 1;
        align: left middle;
    }
    SessionModal #session-actions Button {
        height: 1;
        min-width: 8;
        margin: 0 1 0 0;
        border: none;
    }
    /* Hint = flexible 1fr gap that pushes Close to the bottom-right corner. */
    SessionModal #session-hint {
        width: 1fr;
        height: 1;
        color: $text 60%;
        content-align: left middle;
        padding: 0 1;
    }
    SessionModal #session-close { margin: 0; }
    """

    BINDINGS = [
        ("escape", "close", "Close"),
        ("q", "close", "Close"),
        ("c", "copy", "Copy"),
        ("e", "edit", "Edit"),
    ]

    class EditRequested(TextualMessage):
        """Posted when the user picks Edit on a user message.

        Edit == rewind the conversation to JUST BEFORE this message, then
        prefill the input with its text so a modified version can be re-sent
        (which branches). Carries ``history_index`` so the app knows the
        rewind point.
        """

        def __init__(self, text: str, history_index: int) -> None:
            super().__init__()
            self.text = text
            self.history_index = history_index

    def __init__(
        self,
        text: str,
        role: Role,
        history_index: int = -1,
    ) -> None:
        super().__init__()
        self._body_text = text if text else "(empty)"
        self._role = role
        self._history_index = history_index
        # Edit only makes sense for a USER message with a real history entry:
        # it rewinds to just before that message and re-sends a modified copy.
        # (Editing an assistant answer to 're-send' it is meaningless.)
        self._can_edit = role is Role.USER and history_index >= 0

    def compose(self) -> ComposeResult:
        label = _ROLE_PREFIX.get(self._role, (self._role.value, "white"))[0]
        with Vertical(id="session-box"):
            yield Static(f"💬 {label} message", id="session-title")
            area = TextArea(self._body_text, read_only=True, id="session-body")
            area.show_line_numbers = False
            yield area
            with Horizontal(id="session-actions"):
                yield Button("Copy (c)", id="session-copy", variant="primary")
                # Edit = rewind-to-before + prefill; only for user messages.
                if self._can_edit:
                    yield Button("Edit & resend (e)", id="session-edit")
                # Hint = flexible 1fr gap -> Close in the bottom-right corner.
                yield Static("", id="session-hint")
                yield Button("Close (Esc)", id="session-close")

    def on_mount(self) -> None:
        # Focus the text so it can be selected/copied immediately.
        self.query_one("#session-body", TextArea).focus()

    def action_close(self) -> None:
        self.dismiss(None)

    def action_copy(self) -> None:
        self.app.copy_to_clipboard(self._body_text)
        self.query_one("#session-hint", Static).update(" copied ✓")

    def action_edit(self) -> None:
        if not self._can_edit:
            return
        self.post_message(
            self.EditRequested(self._body_text, self._history_index)
        )
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "session-copy": self.action_copy,
            "session-edit": self.action_edit,
            "session-close": self.action_close,
        }
        handler = actions.get(event.button.id)
        if handler is not None:
            handler()


# --- session-tree mini-canvas (same visual style as the telemetry flow) --
# A compact, pannable canvas that draws the session tree as stacked boxes
# with connector lines -- exactly like the right-hand flow dashboard -- so a
# linear multi-turn history stacks straight down (no rightward creep) and a
# fork steps out to a new column.
_SNODE_W = 26                 # node box width (cells)
_SNODE_H = 3                  # node box height: border + 1 label row + border
_SCOL_GAP = 4                 # gutter between columns (holds fork rails)
_SROW_GAP = 0                 # boxes touch (parent bottom border abuts child top border)
_SPAD = 2                     # canvas padding around the graph
_SPENDING_ID = "__session_pending__"   # synthetic in-flight-turn marker


class _SessionNodeBox(Static):
    """One session commit rendered as a bordered box on the mini-canvas.

    Mirrors the telemetry ``FlowNode`` look: a round border whose colour marks
    state (green = current head, primary = a fork, grey = a plain commit) and
    a `-selected` accent for the keyboard/mouse cursor. Absolutely positioned
    by :class:`_SessionCanvas` from the node's (col, row).
    """

    DEFAULT_CSS = """
    _SessionNodeBox {
        width: %d;
        height: %d;
        border: round $primary-darken-2;
        background: $panel;
        padding: 0 1;
        layer: nodes;
        position: absolute;
    }
    _SessionNodeBox.current { border: round $success; }
    _SessionNodeBox.fork    { border: round $primary; }
    _SessionNodeBox.pending {
        border: dashed $warning;
        color: $text-muted;
        background: $panel-darken-1;
    }
    _SessionNodeBox.-selected { background: $boost; border: round $accent; }
    """ % (_SNODE_W, _SNODE_H)

    def __init__(self, node_id: str) -> None:
        super().__init__()
        self._node_id = node_id


class _SessionEdgeLayer(Static):
    """Draws PLANAR (zero-crossing) git-style connectors between boxes.

    Because ``graph_layout`` never reuses a column, a fork's child always
    sits in a lane to the right of every lane opened before it. Each fork is
    a STRAIGHT orthogonal L drawn with box-drawing lines only (no diagonal):
    a horizontal leaves the parent's right-border MIDPOINT, runs across the
    empty lanes to the child's lane, a ``\u2510`` corner turns DOWN, a
    straight vertical drops the child's OWN (never-reused) lane, and a
    ``\u2514`` corner + short horizontal lands on the child's left-border
    MIDPOINT. Both endpoints are vertically CENTERED on their bubbles. The
    top run only ever crosses lanes that open LOWER down (still empty), so no
    connector crosses another or cuts a box interior. A linear (same-column)
    child draws nothing -- the touching boxes show the link. Fills the whole
    virtual canvas and sits on a lower layer than the boxes.
    """

    def __init__(self, canvas: "_SessionCanvas") -> None:
        super().__init__()
        self._canvas = canvas

    # Box-drawing glyphs for the STRAIGHT (orthogonal) connectors.
    _V = chr(0x2502)     # vertical  │
    _H = chr(0x2500)     # horizontal  ─
    _DL = chr(0x2510)    # down + left corner  ┐ (top run turns DOWN into lane)
    _UR = chr(0x2514)    # up + right corner  └ (lane turns RIGHT into child)
    _TD = chr(0x252C)    # tee-down  ┬ (comb bar drops into an inner lane)

    def render_line(self, y: int) -> Strip:
        width = self.size.width
        height = self.size.height
        if width <= 0:
            return Strip.blank(0)
        cells = [None] * width
        V, H, DL, UR, TD = self._V, self._H, self._DL, self._UR, self._TD

        def put(x_: int, ch: str, style: Style, over=False) -> None:
            if 0 <= x_ < width and 0 <= y < height:
                if over or cells[x_] is None:
                    cells[x_] = (ch, style)

        # Forks are drawn git-graph style as a per-parent COMB: one horizontal
        # bar on the parent's mid row leaves the parent's right border and
        # runs to the farthest child's lane, dropping DOWN into each child's
        # OWN (never-reused) lane -- a tee-down ``\u252c`` at an inner lane, a
        # ``\u2510`` down+LEFT corner at the last (farthest) lane. Each dropped
        # lane is a straight vertical ending on the child's left-border
        # midpoint with a ``\u2514`` elbow. So a parent with N children fans
        # out from ONE bar (a clean T), every endpoint stays centered on its
        # bubble, and no vertical is ever shared -- the picture stays planar.
        for g in self._canvas._fork_groups():
            pr = g["parent_right"]; bar = g["bar_row"]; style = g["style"]
            children = g["children"]
            last_lane = children[-1][0]
            # (1) The shared comb BAR on the parent mid row: from just past the
            #     parent border to the farthest lane, teeing DOWN at each lane.
            if y == bar:
                lanes = {lane for lane, _cmy, _cl in children}
                for x_ in range(pr + 1, last_lane + 1):
                    if x_ == last_lane:
                        put(x_, DL, style, over=True)
                    elif x_ in lanes:
                        put(x_, TD, style, over=True)
                    else:
                        put(x_, H, style, over=True)
            # (2) Each child's straight vertical down its OWN lane, then an
            #     up+right elbow + short run to the child's left-border mid.
            for lane, cmy, child_left in children:
                if bar < y < cmy:
                    put(lane, V, style)
                elif y == cmy:
                    put(lane, UR, style, over=True)
                    for x_ in range(lane + 1, child_left):
                        put(x_, H, style, over=True)
        segments = []
        run_text = []
        run_style = None
        for cell in cells:
            if cell is None:
                ch, st = " ", None
            else:
                ch, st = cell
            if st is run_style:
                run_text.append(ch)
            else:
                if run_text:
                    segments.append(Segment("".join(run_text), run_style))
                run_text = [ch]; run_style = st
        if run_text:
            segments.append(Segment("".join(run_text), run_style))
        return Strip(segments, width)


class _SessionCanvas(ScrollableContainer):
    """Pannable area that lays out session boxes + draws their connectors.

    Layout comes from :meth:`SessionTree.graph_layout` -- a linear chain stays
    in one column (stacking straight down), forks step right -- so this reads
    like the telemetry flow rather than an ever-indenting outline. Arrow keys
    move a selection cursor (row order); a click selects the box under it. A
    background drag pans the canvas (grab-to-pan), like the flow chart.
    """

    DEFAULT_CSS = """
    _SessionCanvas { }
    _SessionCanvas > _SessionEdgeLayer { layer: edges; }
    """

    class Selected(TextualMessage, namespace="session_canvas"):
        """Posted on a click / activation of a node box.

        The ``namespace="session_canvas"`` pins the Textual handler name to
        ``on_session_canvas_selected``. Without it, the leading underscore in
        ``_SessionCanvas`` makes Textual derive ``on__session_canvas_selected``
        (double underscore) from the class qualname, so the modal handler
        would never fire and activation (Enter / second click) would silently
        do nothing.
        """

        def __init__(self, node_id: str, activate: bool) -> None:
            super().__init__()
            self.node_id = node_id
            self.activate = activate

    _CLICK_SLOP = 1

    def __init__(
        self,
        layout,
        current_head: str,
        selected_id: str,
        pending_text: "str | None" = None,
    ) -> None:
        super().__init__()
        self._layout = list(layout)
        self._current_head = current_head
        self._selected_id = selected_id
        self._edge_layer = _SessionEdgeLayer(self)
        self._boxes = {}                 # node_id -> _SessionNodeBox
        self._pos = {}                   # node_id -> (x, y) top-left
        self._node_w = {}                # node_id -> box width (cells)
        self._node_col = {}              # node_id -> column index
        self._col_w = {}                 # col -> column width (widest box)
        self._col_x = {}                 # col -> left x of that column
        self._pan_from = None
        self._pan_origin = Offset(0, 0)
        self._moved = False
        # A turn is committed to the SessionTree only when it FINISHES, so an
        # in-flight send has no real node yet. When one is running we draw a
        # synthetic, dashed "pending" box hanging off the current head, so the
        # map shows the turn being waited on. It is NOT part of the tree and is
        # not selectable/restorable (arrow-nav and clicks skip it).
        self._pending = self._make_pending(pending_text)

    def _make_pending(self, pending_text):
        if not pending_text:
            return None
        head_lay = next(
            (l for l in self._layout if l.node.id == self._current_head), None
        )
        if head_lay is None:
            return None
        rows = [l.row for l in self._layout] or [0]
        cols = [l.col for l in self._layout] or [0]
        head_has_child = any(l.parent == self._current_head for l in self._layout)
        # Continue the mainline column when the head is a leaf; otherwise the
        # pending turn is a new branch and steps to a fresh column.
        col = head_lay.col if not head_has_child else max(cols) + 1
        row = max(rows) + 1
        return {
            "id": _SPENDING_ID,
            "label": pending_text,
            "col": col,
            "row": row,
            "parent": self._current_head,
        }

    def compose(self) -> ComposeResult:
        yield self._edge_layer

    def on_mount(self) -> None:
        for lay in self._layout:
            box = _SessionNodeBox(lay.node.id)
            self._boxes[lay.node.id] = box
            self.mount(box)
        if self._pending is not None:
            box = _SessionNodeBox(self._pending["id"])
            self._boxes[self._pending["id"]] = box
            self.mount(box)
        self._reflow()

    def on_resize(self, event: events.Resize) -> None:
        # Re-flow so the virtual (edge-layer) area tracks the modal size: this
        # both FILLS the pane and keeps free drag-pan slack, like FlowCanvas.
        self._reflow()

    def _cells(self):
        """(node_id, col, row, parent, is_branch, is_pending) for every box."""
        for lay in self._layout:
            yield (lay.node.id, lay.col, lay.row, lay.parent,
                   lay.is_branch_point, False)
        if self._pending is not None:
            p = self._pending
            yield (p["id"], p["col"], p["row"], p["parent"], False, True)

    def _max_box_w(self) -> int:
        """Cap a bubble at the visible width minus a margin (modal - 10).

        The canvas fills the modal, so its width is the modal inner width;
        keeping a 10-cell margin means the widest utterance still fits in
        view (no horizontal pan needed just to read one bubble).
        """
        return max(_SNODE_W, self.size.width - 10)

    def _box_width_for(self, node_id: str, is_pending: bool) -> int:
        """Content-sized box width: label cells + chrome, clamped to caps."""
        label = self._label(node_id, is_pending)
        chrome = 4  # round border (2) + horizontal padding (2)
        want = label.cell_len + chrome
        return max(_SNODE_W, min(self._max_box_w(), want))

    def _fork_edges(self):
        """Yield one drawable record per FORK edge (child in a right column).

        Linear edges (child in the SAME column as its parent) are skipped --
        the boxes touch, so the stack itself shows the link. Because
        ``graph_layout`` never reuses a column, a fork's child always sits in
        a lane to the right of every lane opened before it, so the connector
        can run straight DOWN the child's own lane and reach it with a single
        horizontal over empty columns -- i.e. the drawing is planar (no line
        ever crosses another). Each record is a dict with the parent's bottom
        row + lane x, the child's lane x + mid row + left edge, and a style.
        """
        real = Style.parse("grey42")
        pending = Style.parse("yellow")
        edges = []
        for node_id, col, _row, parent, _b, is_pending in self._cells():
            if (parent is None or parent not in self._pos
                    or node_id not in self._pos):
                continue
            pcol = self._node_col.get(parent, 0)
            ccol = self._node_col.get(node_id, 0)
            if ccol <= pcol:
                continue
            px, py = self._pos[parent]
            cx, cy = self._pos[node_id]
            pw = self._node_w.get(parent, _SNODE_W)
            edges.append({
                "parent": parent,
                "parent_col": pcol,
                "parent_right": px + pw - 1,
                "parent_mid": py + _SNODE_H // 2,
                "child_lane": self._lane_x(ccol),
                "child_mid": cy + _SNODE_H // 2,
                "child_left": cx,
                "style": pending if is_pending else real,
            })
        return edges

    def _fork_groups(self):
        """Group fork edges by PARENT into a git-style comb (shared bar).

        A parent with one or more fork children draws a SINGLE horizontal bar
        along its own mid row (the OUT endpoint, vertically centered on the
        parent box): the bar leaves the parent's right border and runs to the
        farthest child's lane, dropping DOWN into each child's OWN
        (never-reused) lane with a tee. So a parent with N children fans out
        from ONE bar (a clean comb / T) instead of stacking N separate
        horizontals. Each dropped lane is a straight vertical that ends on the
        child's left-border midpoint with an up+right elbow, so every endpoint
        stays centered and no vertical is ever shared (planarity holds).
        Yields one dict per parent: ``parent_right``, ``bar_row`` (parent mid),
        ``children`` (a list of ``(lane, child_mid, child_left)`` sorted by
        lane), and ``style``.
        """
        by_parent = {}
        order = []
        for e in self._fork_edges():
            if e["child_mid"] <= e["parent_mid"] or e["child_lane"] <= e["parent_right"]:
                continue
            pid = e["parent"]
            if pid not in by_parent:
                by_parent[pid] = []
                order.append(pid)
            by_parent[pid].append(e)
        groups = []
        for pid in order:
            edges = sorted(by_parent[pid], key=lambda e: e["child_lane"])
            children = [
                (e["child_lane"], e["child_mid"], e["child_left"]) for e in edges
            ]
            groups.append({
                "parent_right": edges[0]["parent_right"],
                "bar_row": edges[0]["parent_mid"],
                "children": children,
                "style": edges[0]["style"],
            })
        return groups

    def _lane_x(self, col: int) -> int:
        """X of a column's connector lane: one cell LEFT of its box border.

        The branch drops down the gutter just left of the child column, so
        the elbow lands exactly on the child box's left border (never inside
        it) and the vertical never overlaps a box. Column 0's lane sits in
        the left pad.
        """
        return max(0, self._col_x.get(col, _SPAD) - 1)

    def _compute_widths(self) -> None:
        """Size every box to its content, then pack columns by their widest.

        Each column is only as wide as its widest box, and the next column
        starts after it (plus a gap) -- so a fork does not reserve a fixed
        slot and wide utterances stretch rightward instead of being clipped.
        """
        self._node_w = {}
        self._node_col = {}
        col_w = {}
        for node_id, col, _row, _parent, _b, is_pending in self._cells():
            w = self._box_width_for(node_id, is_pending)
            self._node_w[node_id] = w
            self._node_col[node_id] = col
            col_w[col] = max(col_w.get(col, _SNODE_W), w)
        self._col_w = col_w
        # Columns are packed left-to-right, each as wide as its widest box
        # plus a fixed gutter. A fork's connector runs down the CHILD's own
        # lane (see ``_lane_x``), so the gutter no longer has to host stacked
        # rails -- a small, constant gap is enough.
        self._col_x = {}
        x = _SPAD
        for col in sorted(col_w):
            self._col_x[col] = x
            x += col_w[col] + _SCOL_GAP

    def _reflow(self) -> None:
        self._compute_widths()
        max_x = max_y = 0
        for node_id, col, row, _parent, is_branch, is_pending in self._cells():
            width = self._node_w.get(node_id, _SNODE_W)
            x = self._col_x.get(col, _SPAD)
            y = _SPAD + row * (_SNODE_H + _SROW_GAP)
            self._pos[node_id] = (x, y)
            box = self._boxes.get(node_id)
            if box is None:
                continue
            box.styles.width = width
            box.styles.offset = (x, y)
            box.update(self._label(node_id, is_pending))
            self._paint_classes(box, node_id, is_branch, is_pending)
            max_x = max(max_x, x + width)
            max_y = max(max_y, y + _SNODE_H)
        viewport = self.size
        self._edge_layer.styles.width = max(max_x + _SPAD, viewport.width + _SPAD)
        self._edge_layer.styles.height = max(max_y + _SPAD, viewport.height + _SPAD)
        self._edge_layer.refresh()

    def _label(self, node_id: str, is_pending: bool = False) -> Text:
        text = Text(no_wrap=True, overflow="ellipsis")
        if is_pending:
            text.append("… ", style="yellow")
            text.append(self._pending["label"], style="italic yellow")
            return text
        node = self._node_for(node_id)
        is_head = node_id == self._current_head
        text.append(
            ("● " if is_head else "○ "),
            style=("bold green" if is_head else "grey62"),
        )
        label = (node.label if node is not None else "") or node_id
        text.append(label, style=("bold" if is_head else ""))
        return text

    def _node_for(self, node_id: str):
        for lay in self._layout:
            if lay.node.id == node_id:
                return lay.node
        return None

    def _paint_classes(self, box, node_id, is_branch, is_pending) -> None:
        box.set_class(is_pending, "pending")
        box.set_class((not is_pending) and node_id == self._current_head, "current")
        box.set_class((not is_pending) and is_branch, "fork")
        box.set_class(node_id == self._selected_id, "-selected")

    # --- selection ------------------------------------------------------
    def _order(self):
        return [lay.node.id for lay in self._layout]

    def select(self, node_id: str, activate: bool = False) -> None:
        # The pending marker is not a commit -> never selectable/restorable.
        if node_id not in self._boxes or node_id == _SPENDING_ID:
            return
        self._selected_id = node_id
        for nid, _c, _r, _p, is_branch, is_pending in self._cells():
            self._paint_classes(self._boxes[nid], nid, is_branch, is_pending)
        self._scroll_to_node(node_id)
        self.post_message(self.Selected(node_id, activate))

    def move(self, delta: int) -> None:
        order = self._order()
        if not order:
            return
        try:
            idx = order.index(self._selected_id)
        except ValueError:
            idx = 0
        idx = max(0, min(len(order) - 1, idx + delta))
        self.select(order[idx])

    def _scroll_to_node(self, node_id: str) -> None:
        box = self._boxes.get(node_id)
        if box is not None:
            self.scroll_to_widget(box, animate=False)

    def on_key(self, event: events.Key) -> None:
        if event.key in ("up", "left"):
            self.move(-1); event.stop()
        elif event.key in ("down", "right"):
            self.move(1); event.stop()
        elif event.key == "enter":
            if self._selected_id:
                self.post_message(self.Selected(self._selected_id, True))
            event.stop()

    # --- mouse: click selects, drag pans -------------------------------
    def _node_at(self, off: Offset):
        for node_id, box in self._boxes.items():
            if node_id == _SPENDING_ID:
                continue
            if box.region.contains(off.x, off.y):
                return node_id
        return None

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self._pan_from = event.screen_offset
        self._pan_origin = self.scroll_offset
        self._moved = False
        self.capture_mouse()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._pan_from is None:
            return
        dx = event.screen_offset.x - self._pan_from.x
        dy = event.screen_offset.y - self._pan_from.y
        if abs(dx) > self._CLICK_SLOP or abs(dy) > self._CLICK_SLOP:
            self._moved = True
        self.scroll_to(
            self._pan_origin.x - dx, self._pan_origin.y - dy, animate=False
        )

    def on_mouse_up(self, event: events.MouseUp) -> None:
        try:
            self.release_mouse()
        except Exception:
            pass
        was_pan = self._moved
        self._pan_from = None
        self._moved = False
        if was_pan:
            return
        node_id = self._node_at(event.screen_offset)
        if node_id is not None:
            # A click only SELECTS a node -- it never restores. Restore is an
            # explicit, scoped choice (session vs workspace) made from the
            # modal's buttons, so a stray click can never rewind anything.
            self.select(node_id, activate=False)


class RestoreModal(ModalScreen):
    """A two-tab session manager: the current session tree + a global browser.

    Opened from the status bar (``sessions`` hint / ``Ctrl+R``). Tabs sit in
    the TOP-RIGHT corner:

    - **Session tree** (tab 1, unchanged): draws THIS run's session the same
      way the telemetry dashboard draws its flow -- stacked boxes with
      connector lines (see :class:`_SessionCanvas`). A linear history stacks
      straight down; forks step right. Selecting a node + a scope (``s`` /
      ``w`` or the buttons) posts :class:`RestoreRequested`; the app cancels
      any in-flight turn, awaits it, then checks the node out
      (non-destructively).
    - **All sessions** (tab 2): a global browser over every rollout under the
      home directory. Left = a searchable list (newest first); right = a flat,
      time-ordered preview of the highlighted session (no tree -- just the
      conversation in order). ``Enter`` / ``Load`` opens the highlighted
      session into the app (posts :class:`OpenRequested`).

    The modal owns focus (a deliberate exception to focus-pinned-to-input, the
    same one the inspector/session modals use) so the canvas / list are
    keyboard- and mouse-navigable.
    """

    DEFAULT_CSS = """
    RestoreModal {
        align: center middle;
    }
    RestoreModal > #restore-box {
        width: 80%;
        height: 80%;
        border: round $primary;
        background: $surface;
        padding: 0 1;
    }
    /* Title + the two tab buttons share ONE compact row. */
    RestoreModal #restore-titlebar {
        height: 1;
    }
    RestoreModal #restore-title {
        width: 1fr;
        height: 1;
        color: $text;
        text-style: bold;
        content-align: left middle;
    }
    /* Tab switchers styled exactly like the modal's other buttons. Scope
       by the titlebar id (like #restore-actions Button) so this beats the
       Button widget's own -style-default border rules; a class-only
       selector loses to them and the label ends up with zero rows. */
    RestoreModal #restore-titlebar Button {
        height: 1;
        min-width: 8;
        margin: 0 0 0 1;
        border: none;
    }
    /* The built-in TabbedContent tab bar is hidden -- our buttons drive it
       -- so the panes take the whole height with no extra tab row. */
    RestoreModal TabbedContent {
        height: 1fr;
    }
    RestoreModal TabbedContent > ContentTabs {
        display: none;
        height: 0;
    }
    RestoreModal TabPane {
        padding: 0;
    }
    RestoreModal #restore-canvas {
        height: 1fr;
        border: none;
        background: $surface;
        /* Scrollable so grab-to-pan works, but the scrollbar chrome is
           hidden (size 0) like the flow canvas: it would otherwise sit on
           top of the tree and cover boxes/connectors. Pan by dragging. */
        overflow: auto auto;
        scrollbar-size: 0 0;
    }
    /* Global browser: search on top, then a left list + right preview. */
    RestoreModal #global-search {
        height: 3;
        border: round $primary 50%;
    }
    RestoreModal #global-split {
        height: 1fr;
    }
    RestoreModal #global-list {
        width: 2fr;
        border: none;
        background: $surface;
    }
    RestoreModal #global-preview {
        width: 3fr;
        border: none;
        background: $surface;
    }
    RestoreModal #global-empty {
        width: 3fr;
        height: 1fr;
        color: $text 60%;
        content-align: center middle;
    }
    RestoreModal .global-row-current {
        color: $success;
        text-style: bold;
    }
    RestoreModal #restore-actions {
        height: 1;
        align: left middle;
    }
    RestoreModal #restore-actions Button {
        height: 1;
        min-width: 8;
        margin: 0 1 0 0;
        border: none;
    }
    /* Hint = flexible 1fr gap that pushes Close to the bottom-right corner. */
    RestoreModal #restore-hint {
        width: 1fr;
        height: 1;
        color: $text 60%;
        content-align: left middle;
        padding: 0 1;
    }
    RestoreModal #restore-close { margin: 0; }
    """

    BINDINGS = [
        ("escape", "close", "Close"),
        # Restore is an explicit, SCOPED action -- never a bare Enter/click,
        # so the two scopes can never be confused or triggered by accident.
        ("s", "restore_session", "Restore session"),
        ("w", "restore_workspace", "Restore workspace"),
    ]

    class RestoreRequested(TextualMessage):
        """Posted when the user picks a node + scope to restore to.

        ``scope`` is ``"session"`` (rewind the conversation only) or
        ``"workspace"`` (also re-apply the captured on-disk files). It
        defaults to ``"session"`` so the safer scope is the implicit one.
        """

        def __init__(self, node_id: str, scope: str = "session") -> None:
            super().__init__()
            self.node_id = node_id
            self.scope = scope

    class OpenRequested(TextualMessage):
        """Posted when the user opens a session from the GLOBAL browser.

        Carries the rollout file ``path`` to load. Unlike ``RestoreRequested``
        (which moves a head WITHIN the live tree), this swaps the whole active
        session over to another saved rollout.
        """

        def __init__(self, path: str) -> None:
            super().__init__()
            self.path = path

    def __init__(self, sessions, current_head: str,
                 pending_text: "str | None" = None,
                 sessions_dir=None, current_file=None) -> None:
        super().__init__()
        # ``sessions`` is the SessionTree; its ``graph_layout`` gives the 2D
        # (col,row) placement so linear history stacks and forks step right.
        # ``pending_text`` (set while a turn is in flight) draws a synthetic
        # dashed marker for the not-yet-committed turn.
        # Hide ``clear`` markers on the dashboard: a clear restores no
        # context, so it is not a useful restore target. Its post-clear line
        # re-attaches to the original root as a fork (see graph_layout).
        self._layout = sessions.graph_layout(hide_labels={CLEAR_LABEL})
        self._current_head = current_head
        # The cursor must land on a VISIBLE node. When the head is itself a
        # hidden ``clear`` marker (e.g. right after Ctrl+L, before any send),
        # fall back to its nearest visible ancestor so 's'/Enter never
        # restores to the hidden clear node.
        visible_ids = {lay.node.id for lay in self._layout}
        selected = current_head
        if selected not in visible_ids:
            selected = None
            for node in reversed(sessions.path(current_head)):
                if node.id in visible_ids:
                    selected = node.id
                    break
            if selected is None and self._layout:
                selected = self._layout[-1].node.id
        self._canvas = _SessionCanvas(
            self._layout, current_head, selected, pending_text=pending_text
        )
        # Global-browser state (tab 2): where rollouts live + which one is the
        # live session, plus the currently listed/loaded infos.
        self._sessions_dir = sessions_dir
        self._current_file = current_file
        self._all_sessions: list = []
        self._filtered: list = []

    def compose(self) -> ComposeResult:
        with Vertical(id="restore-box"):
            # Title and the two tab switchers live on ONE compact row; the
            # switchers are Buttons styled like the modal's other buttons
            # (the built-in TabbedContent tab bar is hidden -- see CSS).
            with Horizontal(id="restore-titlebar"):
                yield Static("sessions", id="restore-title")
                yield Button(
                    "Session tree",
                    id="tabbtn-tree",
                    classes="tabbtn",
                    variant="primary",
                    compact=True,
                )
                yield Button(
                    "All sessions",
                    id="tabbtn-global",
                    classes="tabbtn",
                    variant="default",
                    compact=True,
                )
            with TabbedContent(id="restore-tabs"):
                with TabPane("Session tree", id="tab-tree"):
                    self._canvas.id = "restore-canvas"
                    yield self._canvas
                with TabPane("All sessions", id="tab-global"):
                    yield Input(
                        placeholder="\U0001f50d  search sessions\u2026",
                        id="global-search",
                    )
                    with Horizontal(id="global-split"):
                        yield ListView(id="global-list")
                        yield TextArea(
                            "", read_only=True, id="global-preview"
                        )
            with Horizontal(id="restore-actions"):
                yield Button(
                    "Restore session (s)",
                    id="restore-session",
                    variant="primary",
                )
                yield Button(
                    "Restore workspace (w)",
                    id="restore-workspace",
                    variant="warning",
                )
                # Hint = flexible 1fr gap -> Close in the bottom-right corner.
                yield Static("", id="restore-hint")
                yield Button("Close (Esc)", id="restore-close")

    def on_mount(self) -> None:
        # Focus the canvas so arrow keys and the scope hotkeys (s/w) work
        # immediately on the (default) tree tab.
        self._canvas.focus()
        self.query_one("#global-preview", TextArea).show_line_numbers = False
        self._load_global_sessions()

    # --- global browser (tab 2) ----------------------------------------
    def _load_global_sessions(self) -> None:
        """Scan the home dir for rollouts once and populate the list.

        Best-effort: any failure just leaves an empty list, so the tree tab is
        never affected by a disk problem.
        """
        from coreybot.core.paths import AgentPaths
        from coreybot.runtime.session_service import list_sessions

        self._all_sessions = []
        if self._sessions_dir is not None:
            try:
                paths = AgentPaths(home=Path(self._sessions_dir).parent)
                self._all_sessions = list_sessions(
                    paths, current=self._current_file
                )
            except Exception:
                self._all_sessions = []
        self._apply_filter("")

    def _apply_filter(self, query: str) -> None:
        """Filter the session list by a case-insensitive substring match.

        Matches the title, the session id, or the timestamp text, so typing a
        date or a word from the first message both narrow the list.
        """
        needle = query.strip().lower()
        if needle:
            self._filtered = [
                info for info in self._all_sessions
                if needle in info.title.lower()
                or needle in info.session_id.lower()
                or needle in info.created_text.lower()
            ]
        else:
            self._filtered = list(self._all_sessions)
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        listview = self.query_one("#global-list", ListView)
        listview.clear()
        for index, info in enumerate(self._filtered):
            marker = "\u25cf " if info.is_current else "  "
            head = Text()
            head.append(marker, style="bold green" if info.is_current else "dim")
            head.append(info.title or "(no messages)")
            meta = Text(
                "    %s \u00b7 %d msg%s"
                % (info.created_text, info.message_count,
                   "" if info.message_count == 1 else "s"),
                style="dim",
            )
            body = Vertical(Static(head), Static(meta))
            item = ListItem(body)
            if info.is_current:
                item.add_class("global-row-current")
            listview.append(item)
        if self._filtered:
            listview.index = 0
            self._preview_index(0)
        else:
            self._show_preview_placeholder()

    def _show_preview_placeholder(self) -> None:
        area = self.query_one("#global-preview", TextArea)
        area.load_text("(no sessions match)" if self._all_sessions
                       else "(no saved sessions yet)")

    def _preview_index(self, index: int) -> None:
        if not (0 <= index < len(self._filtered)):
            return
        info = self._filtered[index]
        from coreybot.runtime.session_service import flatten_session

        try:
            messages = flatten_session(info.path)
        except Exception:
            messages = []
        area = self.query_one("#global-preview", TextArea)
        area.load_text(self._format_preview(info, messages))

    @staticmethod
    def _format_preview(info, messages) -> str:
        """Render a session as a flat, time-ordered transcript (no tree)."""
        header = "%s\n%s \u00b7 %d message%s\n%s\n" % (
            info.title or "(no messages)",
            info.created_text,
            len(messages),
            "" if len(messages) == 1 else "s",
            "\u2500" * 40,
        )
        prefix = {
            Role.USER: "you",
            Role.ASSISTANT: "bot",
            Role.SYSTEM: "sys",
            Role.TOOL: "tool",
        }
        blocks = [header]
        for message in messages:
            who = prefix.get(message.role, message.role.value)
            blocks.append("%s: %s" % (who, message.content))
        if not messages:
            blocks.append("(this session has no messages)")
        return "\n".join(blocks)

    def on_input_changed(self, event: "Input.Changed") -> None:
        if event.input.id == "global-search":
            self._apply_filter(event.value)

    def on_list_view_highlighted(self, event: "ListView.Highlighted") -> None:
        listview = self.query_one("#global-list", ListView)
        index = listview.index
        if index is not None:
            self._preview_index(index)

    def on_list_view_selected(self, event: "ListView.Selected") -> None:
        # Enter / click a row: open that session into the app.
        self._open_selected_global()

    def _open_selected_global(self) -> None:
        listview = self.query_one("#global-list", ListView)
        index = listview.index
        if index is None or not (0 <= index < len(self._filtered)):
            return
        info = self._filtered[index]
        self.post_message(self.OpenRequested(str(info.path)))
        self.dismiss(None)

    # --- tree tab (unchanged behavior) ---------------------------------
    def _selected_node_id(self) -> str:
        # The canvas owns the selection; read it straight from there so the
        # modal and canvas can never drift out of sync.
        return self._canvas._selected_id

    def on_session_canvas_selected(self, message: "_SessionCanvas.Selected") -> None:
        # Selecting a node NEVER restores -- it only moves the cursor. Restore
        # is always an explicit, scoped choice (the s/w keys or the buttons),
        # so a click or Enter can never silently rewind the session or files.
        return

    def action_close(self) -> None:
        self.dismiss(None)

    def _on_global_tab(self) -> bool:
        try:
            return self.query_one("#restore-tabs", TabbedContent).active == "tab-global"
        except Exception:
            return False

    def _switch_tab(self, tab_id: str) -> None:
        """Activate a pane from a title-bar tab button and reflect state.

        The visible TabbedContent tab bar is hidden, so these buttons ARE
        the tabs: set ``active`` and mark the chosen button ``primary`` (the
        other ``default``) so the active tab reads like a pressed button.
        Focus goes to the natural first widget of the pane so keys work.
        """
        tabs = self.query_one("#restore-tabs", TabbedContent)
        tabs.active = tab_id
        is_tree = tab_id == "tab-tree"
        self.query_one("#tabbtn-tree", Button).variant = (
            "primary" if is_tree else "default"
        )
        self.query_one("#tabbtn-global", Button).variant = (
            "default" if is_tree else "primary"
        )
        if is_tree:
            self._canvas.focus()
        else:
            try:
                self.query_one("#global-search", Input).focus()
            except Exception:
                pass

    def _restore(self, scope: str) -> None:
        # The scope hotkeys only mean "restore" on the tree tab; on the global
        # tab 's'/'w' are ordinary typing into the search box.
        if self._on_global_tab():
            return
        node_id = self._canvas._selected_id
        if not node_id:
            return
        self.post_message(self.RestoreRequested(node_id, scope))
        self.dismiss(None)

    def action_restore_session(self) -> None:
        """Rewind only the conversation to the selected node."""
        self._restore("session")

    def action_restore_workspace(self) -> None:
        """Rewind the conversation AND the on-disk files to the node."""
        self._restore("workspace")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "tabbtn-tree":
            self._switch_tab("tab-tree")
        elif event.button.id == "tabbtn-global":
            self._switch_tab("tab-global")
        elif event.button.id == "restore-session":
            self.action_restore_session()
        elif event.button.id == "restore-workspace":
            self.action_restore_workspace()
        elif event.button.id == "restore-close":
            self.action_close()


class ChatApp(App):
    """The main Textual application."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #body {
        height: 1fr;
    }
    #chat {
        width: 2fr;
        padding: 0 1;
        background: $surface;
    }
    #flow {
        width: 1fr;
        min-width: 28;
        padding: 0 1;
        background: $panel;
        border-left: solid $primary 30%;
    }
    MessageBubble {
        margin: 0;
        padding: 0;
        height: auto;
    }
    MessageBubble.role-user {
        background: $boost;
    }
    /* A selected bubble (first click) is outlined so the two-step open is
       obvious; a second click on it opens the full-screen session modal. */
    MessageBubble.-selected {
        background: $primary 20%;
    }
    /* A slim separator row above the input keeps the layout tidy without
       stealing the input's own text row. */
    #inputbar {
        height: 1;
        color: $primary 40%;
        background: $surface;
    }
    /* Compact, single-line input (Claude-Code style): NO border (a border
       would consume the single row and make text spill downward). One real
       text row, tinted background, stacked above the status bar (no dock). */
    #prompt {
        height: 1;
        margin: 0;
        padding: 0 1;
        border: none;
        background: $boost;
    }
    #prompt:focus {
        background: $panel;
    }
    /* The single bottom line (replaces both the old top Header and the
       Footer): a horizontal row with the animated info on the left and
       clickable key hints on the right. */
    #statusbar {
        height: 1;
        background: $panel;
        color: $text 70%;
    }
    #statusinfo {
        width: 1fr;
        height: 1;
        padding: 0 1;
        content-align: left middle;
    }
    _KeyHint {
        width: auto;
        height: 1;
        padding: 0 1;
    }
    _KeyHint:hover {
        background: $boost;
        color: $text;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear", "Clear"),
        ("ctrl+r", "restore", "Restore"),
        ("escape", "interrupt", "Interrupt"),
    ]

    def __init__(self, config: Config, session_saver=None, sessions_dir=None, current_file=None) -> None:
        super().__init__()
        self.config = config
        self.agent = Agent(config, session_saver=session_saver)
        # Where saved rollouts live (for the "all sessions" browser) and
        # which rollout THIS run writes (flagged as current in the list).
        self._sessions_dir = sessions_dir
        self._current_session_file = current_file
        self._busy = False
        self._flow = FlowPanel()
        # Per-turn cancellation handle (a Go-style context). ``Esc`` cancels it.
        self._cancel: CancelToken | None = None
        self._worker = None
        # Heartbeat animation: a frame index advanced by a repeating timer so
        # the status bar keeps moving, proving the UI/event loop is alive.
        self._beat = 0
        # The chat bubble currently selected (first click); a second click
        # on it opens the session modal. None means nothing is selected.
        self._selected_bubble: MessageBubble | None = None
        # Last status line rendered (exposed for tests/inspection).
        self._status_text = Text()
        # Text of the turn currently in flight (None when idle). Surfaced in
        # the history-tree map as a synthetic "pending" node -- a committed
        # SessionTree node only appears once the turn finishes.
        self._pending_user_text: str | None = None

    # --- layout ---------------------------------------------------------
    def compose(self) -> ComposeResult:
        # No Header: the top of the terminal is left free so overflowing chat
        # scrolls up instead of being hidden behind a bar.
        with Horizontal(id="body"):
            # can_focus=False: the chat must NOT take keyboard focus. Otherwise
            # pressing to start a text selection focuses the scroll, which fires
            # on_descendant_focus and yanks focus back to the input mid-drag --
            # cancelling the selection. It still scrolls by mouse wheel.
            yield VerticalScroll(id="chat", can_focus=False)
            yield self._flow
        yield Static("─" * 200, id="inputbar")
        yield Input(placeholder="›  Message  (/help for commands)", id="prompt")
        # The status bar is the single bottom line: animated info on the left
        # (#statusinfo) plus clickable key hints on the right (_KeyHint), all
        # on one row. There is no separate Footer.
        with Horizontal(id="statusbar"):
            yield Static(id="statusinfo")
            yield _KeyHint("Esc", "interrupt", "interrupt")
            yield _KeyHint("Ctrl+R", "restore", "sessions")
            yield _KeyHint("Ctrl+L", "clear", "clear")
            yield _KeyHint("Ctrl+C", "quit", "quit")

    def on_mount(self) -> None:
        tools = ", ".join(self.agent.registry.names()) or "(none)"
        # Terminal window title (harmless); the visible chrome now lives in
        # the bottom status bar instead of a top Header.
        self.title = "coreybot"
        self.sub_title = f"{self.config.provider} · {self.config.model}"
        self._notice(
            f"provider {self.config.provider} · model {self.config.model} · "
            f"tools: {tools}"
        )
        # Drive the heartbeat ~8fps: always animating so a live loop is obvious.
        self.set_interval(0.12, self._tick_status)
        self._render_status()
        self._focus_prompt()

    # --- focus policy ---------------------------------------------------
    def _focus_prompt(self) -> None:
        """Put keyboard focus on the message input.

        The input is the only place text is ever typed, so the app keeps focus
        pinned there (see :meth:`on_descendant_focus`).
        """
        try:
            self.query_one("#prompt", Input).focus()
        except Exception:
            pass  # not mounted yet

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        """Keep focus on the message input, always.

        Nothing else in the UI takes typed input, so focus never needs to move.
        If anything else grabs it -- clicking the flow chart / a node, or Tab --
        bounce focus straight back to ``#prompt``. Clicking a node to expand it
        and dragging the canvas are mouse events, so they keep working. The
        redirect runs after this event settles to avoid fighting Textual's own
        focus handling.
        """
        # A modal (e.g. the inspector) is allowed to own focus: only pin focus
        # to the input while the main screen is on top of the stack.
        if len(self.screen_stack) > 1:
            return
        try:
            prompt = self.query_one("#prompt", Input)
        except Exception:
            return
        if event.widget is not prompt:
            self.call_after_refresh(self._focus_prompt)

    # --- helpers --------------------------------------------------------
    def _chat(self) -> VerticalScroll:
        return self.query_one("#chat", VerticalScroll)

    def _add_bubble(
        self, role: Role, text: str = "", history_index: int = -1
    ) -> MessageBubble:
        bubble = MessageBubble(role, text, history_index=history_index)
        self._chat().mount(bubble)
        self.call_after_refresh(self._scroll_to_end)
        return bubble

    def _add_system_line(self, text: str) -> None:
        self._add_bubble(Role.SYSTEM, text)

    def _notice(self, text: str) -> None:
        """Surface a lifecycle/system line in the flow graph, not the chat.

        The left transcript is kept clean (human conversation only): every
        system/debug/activity message becomes a ``notice`` telemetry event so
        it shows up as a node on the right and survives a set_history rebuild.
        """
        event = AgentEvent(kind="notice", text=text)
        self.agent.telemetry.append(event)
        if self._flow is not None:
            self._flow.append(event)

    def _scroll_to_end(self) -> None:
        self._chat().scroll_end(animate=False)

    # --- bottom status bar (heartbeat) ---------------------------------
    def _tick_status(self) -> None:
        """Advance the breathing LED one tick and repaint the status bar."""
        # A plain, ever-growing counter; the LED brightness is derived from it
        # in :meth:`_led_indicator`. Kept unbounded-modulo so it always moves
        # (that is what proves the event loop is alive).
        self._beat = (self._beat + 1) % 1000
        self._render_status()

    def _status_state(self) -> str:
        """Return the current LED state key (looked up in ``_LED_STATES``).

        Split out so more states are trivial to add later (e.g. return
        ``\"error\"`` on a failed turn and give it a red fast-blink entry).
        """
        return "working" if self._busy else "ready"

    def _led_indicator(self) -> Text:
        """Build the standby 'breathing LED' dot reflecting the run state.

        The dot's brightness eases 0 -> 1 -> 0 on a triangle wave whose period
        comes from the current state (slow when idle, quicker when busy), and
        its hue comes from that same state. Brightness is applied by blending
        from a near-off colour toward the state colour, i.e. a real fade
        (truecolor), not a hard on/off blink. Everything derives from
        :attr:`_beat`, so the animation is a pure function of one counter.
        """
        spec = _LED_STATES[self._status_state()]
        period = spec["period"]
        # Triangle wave in [0, 1]: rises for half the period, falls for half.
        phase = self._beat % period
        half = period / 2
        bright = phase / half if phase < half else (period - phase) / half
        # Blend OFF_RGB -> state RGB by ``bright`` and format as #rrggbb.
        r0, g0, b0 = _LED_OFF_RGB
        r1, g1, b1 = spec["rgb"]
        r = round(r0 + (r1 - r0) * bright)
        g = round(g0 + (g1 - g0) * bright)
        b = round(b0 + (b1 - b0) * bright)
        return Text(_LED_GLYPH, style=f"#{r:02x}{g:02x}{b:02x}")

    def _render_status(self) -> None:
        """Paint the single bottom line: breathing LED + info + key hints.

        This one row replaces both the old top Header and the Footer. A
        breathing LED dot pulses every tick (see :meth:`_tick_status`), so the
        bar is visibly 'alive'; its colour + speed alone convey the run state
        (calm green when idle, amber and quicker while a turn runs) -- there is
        no redundant 'ready'/'working' text. No clock is shown. Key hints live
        on the same line (right side) instead of a separate Footer.
        """
        try:
            info = self.query_one("#statusinfo", Static)
        except Exception:
            return  # not mounted yet
        sep = "  │  "
        line = Text(no_wrap=True, overflow="ellipsis")
        # Standby breathing-LED dot stands in for the old spinner glyph; its
        # hue/speed is the sole run-state indicator (see _led_indicator).
        line.append_text(self._led_indicator())
        line.append(" ")
        # Left brand slot shows the current working directory (where the agent
        # is running), not a fixed product name -- handy when hopping repos.
        line.append(os.getcwd(), style="bold")
        line.append(sep, style="grey37")
        line.append(f"{self.config.provider} · {self.config.model}", style="cyan")
        # Key hints live in their own clickable _KeyHint widgets (right side).
        self._status_text = line
        info.update(line)

    # --- input handling -------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if self._handle_command(text):
            return
        if self._busy:
            self._add_system_line("(please wait for the current reply…)")
            return
        self._send(text)

    def _handle_command(self, text: str) -> bool:
        if not text.startswith("/"):
            return False
        command = text.lower()
        if command in ("/exit", "/quit"):
            self.exit()
        elif command == "/reset":
            self.agent.reset()
            self._selected_bubble = None
            self._chat().remove_children()
            self._flow.set_history(self.agent.telemetry)
            self._notice("memory cleared")
        elif command == "/tools":
            self._add_system_line(self.agent.registry.render_for_prompt() or "(no tools)")
        elif command == "/history":
            lines = "\n".join(f"[{m.role.value}] {m.content}" for m in self.agent.history)
            self._add_system_line(lines or "(empty)")
        elif command == "/help":
            self._add_system_line(
                "commands:\n"
                "  /help     show this help\n"
                "  /reset    clear conversation memory\n"
                "  /history  dump raw message history\n"
                "  /tools    list available tools\n"
                "  /exit     quit\n"
                "shortcuts: Esc interrupt · Ctrl+L clear · Ctrl+C quit\n"
                "flow graph: drag the background to pan; auto-scrolls to the newest node (resumes each turn)"
            )
        else:
            self._add_system_line(f"(unknown command: {text})")
        return True

    def action_clear(self) -> None:
        self.agent.reset()
        self._selected_bubble = None
        self._chat().remove_children()
        self._flow.set_history(self.agent.telemetry)
        self._notice("cleared")

    # --- talking to the model ------------------------------------------
    def _send(self, text: str) -> None:
        # ``arun_turn`` appends the user Message next, so it will land at the
        # current length (system prompt occupies index 0). Tag the bubble
        # with that index so 'Restore to here' can rewind to just before it.
        user_index = len(self.agent.history)
        self._add_bubble(Role.USER, text, history_index=user_index)
        self._pending_user_text = text
        # No explicit start_turn: the agent emits a 'turn_start' telemetry event,
        # which the flow panel projects (keeping prior turns on the chart).
        self._busy = True
        self._render_status()
        self.query_one("#prompt", Input).disabled = True
        self._cancel = CancelToken()
        # A native async worker: it runs on Textual's own event loop, so the
        # agent's awaits interleave with the UI and cancellation is immediate.
        self._worker = self.run_worker(
            self._turn_worker(text), exclusive=True, name="turn"
        )

    async def _turn_worker(self, text: str) -> None:
        """Async worker: drives one agent turn, cancellable via ``Esc``."""
        try:
            response = await self.agent.arun_turn(
                text, on_event=self._on_event, cancel_token=self._cancel
            )
        except CancelledError:
            self._on_interrupted()
            return
        except Exception as exc:  # transport error
            self._on_error(str(exc))
            return
        shown = response.content or "(empty response)"
        if response.parse_error:
            shown = f"{shown}\n[protocol warning] {response.parse_error}"
        self._on_done(shown)

    # --- UI callbacks (run on Textual's event loop) --------------------
    def _on_event(self, event: AgentEvent) -> None:
        # The flow graph is the single home for agent activity (llm/tool/
        # notice steps). The left transcript stays clean -- human turns and
        # final answers only -- so we just project onto the graph here.
        self._flow.append(event)

    def _on_error(self, message: str) -> None:
        self._add_bubble(Role.ASSISTANT, f"[error] {message}")
        if self.agent.history and self.agent.history[-1].role == Role.USER:
            self.agent.history.pop()
        self._finish()

    def _on_done(self, shown: str) -> None:
        # ``_drive`` already appended the assistant's raw reply as the last
        # history entry; point the bubble at it for Restore.
        answer_index = len(self.agent.history) - 1
        self._add_bubble(Role.ASSISTANT, shown, history_index=answer_index)
        self._finish()

    def _on_interrupted(self) -> None:
        # Record the interruption in telemetry so the graph stays a faithful
        # projection of the session (survives a later set_history rebuild).
        interrupted = AgentEvent(kind="notice", text="interrupted")
        self.agent.telemetry.append(interrupted)
        if self._flow is not None:
            self._flow.append(interrupted)
        # Drop the user turn we abandoned so history stays consistent.
        if self.agent.history and self.agent.history[-1].role == Role.USER:
            self.agent.history.pop()
        self._finish()

    def action_interrupt(self) -> None:
        """Cancel the in-flight turn (bound to ``Esc``)."""
        if self._busy and self._cancel is not None:
            self._cancel.cancel()

    def on_flow_panel_inspect_requested(
        self, message: FlowPanel.InspectRequested
    ) -> None:
        """Open the full-screen inspector for a node's long input/response."""
        self.push_screen(InspectModal(message.title, message.sections))

    # --- restore-to-node (session tree map, opened from the status bar) -
    def action_restore(self) -> None:
        """Open the session-tree map so any commit can be restored.

        Bound to ``Ctrl+R`` and to the clickable status-bar hint. Unlike a
        bubble's ``Edit`` (which rewinds to *before* a message and prefills
        the input), this restores the FULL context (history + telemetry) to
        whichever node the user picks -- a parent, a sibling branch, or an
        abandoned line -- non-destructively via the git-like session tree.
        """
        head = self.agent.sessions.head()
        pending = self._pending_user_text if self._busy else None
        self.push_screen(
            RestoreModal(
                self.agent.sessions,
                head,
                pending_text=pending,
                sessions_dir=self._sessions_dir,
                current_file=self._current_session_file,
            )
        )

    async def on_restore_modal_restore_requested(
        self, message: RestoreModal.RestoreRequested
    ) -> None:
        """Restore the whole session to the chosen node.

        A restore rewrites ``history``/``telemetry``; if a turn were still
        running its late callback would land an orphan reply in the restored
        transcript (and commit a stray node). So -- exactly like Edit -- we
        first fire the ``CancelToken`` and AWAIT the worker until the turn
        has fully unwound (see ``_cancel_active_turn``); only then is it
        safe to checkout. The checkout is non-destructive, so the line we
        came from stays reachable and re-sending branches off here.
        """
        await self._cancel_active_turn()
        if getattr(message, "scope", "session") == "workspace":
            applied = self.agent.restore_workspace(message.node_id)
            note = "restored session + workspace"
            if applied:
                note += " (%d file%s)" % (applied, "" if applied == 1 else "s")
        else:
            self.agent.checkout_session(message.node_id)
            note = "restored session"
        self._rebuild_transcript()
        self._flow.set_history(self.agent.telemetry)
        self._notice(note)
        self._focus_prompt()
        self._scroll_to_end()

    async def on_restore_modal_open_requested(
        self, message: RestoreModal.OpenRequested
    ) -> None:
        """Load a session picked in the global browser into the app.

        This swaps the whole active session over to another saved rollout.
        Like a restore it first cancels + AWAITS any in-flight turn so a
        late callback cannot land in the newly loaded transcript, then it
        replaces the agent's live tree/history/telemetry from the rollout
        and rebinds the saver so subsequent turns persist to THAT file.
        """
        from coreybot.runtime.session_store import load_tree, save_tree

        await self._cancel_active_turn()
        path = Path(message.path)
        try:
            tree = load_tree(path)
        except Exception as exc:
            self._notice(f"could not open session: {exc}")
            return
        self.agent.sessions = tree
        head = tree.get(tree.head())
        if head is not None:
            self.agent.history = list(head.snapshot.history)
            self.agent.telemetry = list(head.snapshot.telemetry)
        # Continue writing to the opened rollout from now on.
        self._current_session_file = str(path)
        self.agent._session_saver = lambda t, _p=path: save_tree(t, _p)
        self._rebuild_transcript()
        self._flow.set_history(self.agent.telemetry)
        self._notice("opened session")
        self._focus_prompt()
        self._scroll_to_end()

    # --- per-bubble session modal (double-click a chat bubble) ---------
    def on_message_bubble_select_requested(
        self, message: MessageBubble.SelectRequested
    ) -> None:
        """First click: select the bubble (a second click opens the modal)."""
        self._select_bubble(message.bubble)

    def on_message_bubble_open_requested(
        self, message: MessageBubble.OpenRequested
    ) -> None:
        """Second click on a selected bubble: open its read/copy/edit modal."""
        bubble = message.bubble
        self.push_screen(
            SessionModal(bubble.text, bubble.role, bubble.history_index)
        )

    def _node_before_history_index(self, history_index: int) -> str:
        """Find the session node whose snapshot ends JUST BEFORE this message.

        Used by Edit: rewinding to this node drops the target user message and
        everything after it, so re-sending a modified copy branches cleanly.
        The node is the one whose snapshot history length equals
        ``history_index`` (its last entry is the message right before the
        target). Returns '' if none (e.g. the very first user turn maps to the
        root, whose snapshot is the system-only prompt).
        """
        if history_index < 1:
            return ""
        for row in self.agent.sessions.rows():
            if len(row.node.snapshot.history) == history_index:
                return row.node.id
        return ""

    def _select_bubble(self, bubble: "MessageBubble") -> None:
        if self._selected_bubble is bubble:
            return
        if self._selected_bubble is not None:
            self._selected_bubble.set_selected(False)
        bubble.set_selected(True)
        self._selected_bubble = bubble

    async def _cancel_active_turn(self) -> None:
        """Cancel the in-flight turn (if any) and AWAIT its worker's completion.

        Editing rewinds ``history``; if a turn were still running, its late
        ``_on_done``/``_on_interrupted`` would land an orphan response in the
        freshly-rewound transcript (a reply with no matching request) and even
        commit a stray session node. So we reuse the Go-style ``CancelToken``:
        fire it, then block on the worker until the turn has actually unwound
        (its ``CancelledError`` handler pops the abandoned user turn and calls
        ``_finish``). Only after this is it safe to rewind + prefill.
        """
        if not self._busy or self._cancel is None:
            return
        worker = self._worker
        self._cancel.cancel()
        if worker is not None:
            try:
                await worker.wait()
            except Exception:
                # WorkerCancelled/WorkerFailed just mean it has stopped; the
                # turn's own except-handlers already reconciled state. We only
                # needed to be sure it is no longer running.
                pass

    async def _edit_message(self, history_index: int, text: str) -> None:
        """Edit a user message == rewind to just before it, then prefill input.

        Shared by the session modal's ``Edit & resend`` and the inline "edit"
        link on a user bubble. If a turn is still in flight it is cancelled and
        awaited FIRST (see ``_cancel_active_turn``), so no orphan response can
        arrive after the rewind. Then it checks out the session node ending
        right before the target user message (dropping it and anything after
        from the *current* line, non-destructively -- the old branch survives
        via the underlying session tree), rebuilds the view, and drops the text
        into the input so a modified version can be re-sent. Re-sending then
        branches off that point, exactly like editing a message in Claude/ChatGPT.
        """
        await self._cancel_active_turn()
        node_id = self._node_before_history_index(history_index)
        if node_id:
            self.agent.checkout_session(node_id)
        else:
            # First user turn: rewind to the root (system-only) state.
            self.agent.checkout_session(self.agent.sessions.root_id)
        self._rebuild_transcript()
        self._flow.set_history(self.agent.telemetry)
        self._notice("editing")
        prompt = self.query_one("#prompt", Input)
        prompt.value = text
        prompt.cursor_position = len(text)
        self._focus_prompt()

    async def on_session_modal_edit_requested(
        self, message: SessionModal.EditRequested
    ) -> None:
        """Modal 'Edit & resend' -> shared rewind-to-before + prefill flow."""
        await self._edit_message(message.history_index, message.text)

    def _rebuild_transcript(self) -> None:
        """Redraw the left chat from ``agent.history`` (checkout can branch).

        Only human turns and final answers become bubbles (tool-result
        observations and the system prompt are skipped, matching the live
        transcript). Each bubble keeps its real ``history`` index so it can be
        opened/edited again.
        """
        self._selected_bubble = None
        self._chat().remove_children()
        for index, msg in enumerate(self.agent.history):
            if msg.role == Role.SYSTEM:
                continue
            if msg.role == Role.USER and msg.content.startswith("<tool_result"):
                continue
            if msg.role == Role.ASSISTANT:
                # An assistant history entry is the model's RAW turn, which is
                # either a <tool_call> (an intermediate step) or a <message>
                # (the final answer). The live transcript only ever shows the
                # final answer -- tool-call steps live on the flow graph -- so
                # skip tool-call turns here instead of rendering their XML.
                parsed = parse_agent_response(msg.content)
                if parsed.is_tool_call:
                    continue
                shown = _display_answer(msg.content)
            else:
                shown = msg.content
            self._add_bubble(msg.role, shown, history_index=index)

    def _finish(self) -> None:
        self._busy = False
        self._cancel = None
        self._worker = None
        self._pending_user_text = None
        self._render_status()
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        self._focus_prompt()
        self._scroll_to_end()


def run_tui(config: Config, session_saver=None, sessions_dir=None,
            current_file=None) -> None:
    ChatApp(
        config,
        session_saver=session_saver,
        sessions_dir=sessions_dir,
        current_file=current_file,
    ).run()