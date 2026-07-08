"""Integration tests for the Textual TUI (``coreybot.frontends.tui.app``).

Textual's headless ``run_test()`` gives us a ``Pilot`` to mount the real app,
type into widgets, press keys, and inspect state -- no terminal, no network.
We inject a fake provider into the app's Agent so turns run deterministically.
These are marked ``integration`` (widgets + workers + agent + tools).
"""

from __future__ import annotations

import pytest

from coreybot.core.config import Config
from coreybot.core.message import Role
from coreybot.frontends.tui import app as tui_app
from coreybot.frontends.tui.app import ChatApp, MessageBubble

pytestmark = pytest.mark.integration


async def _drain_worker(pilot, app, limit=200):
    """Pump the event loop until the turn worker reports it is done."""
    for _ in range(limit):
        await pilot.pause()
        if not app._busy:
            return
    raise AssertionError("worker did not finish in time")


async def test_banner_on_mount_goes_to_flow_not_chat():
    """The startup banner is a flow-graph notice; the chat starts empty."""
    from coreybot.frontends.tui.flow import FlowPanel

    app = ChatApp(Config())
    async with app.run_test(size=(80, 24)):
        # Left transcript is clean at startup (no system/debug bubbles).
        assert len(app.query(MessageBubble)) == 0
        # The banner is projected as a notice node on the right instead.
        panel = app.query_one(FlowPanel)
        assert any(n.kind == "notice" for n in panel.nodes())
        assert app.agent.telemetry and app.agent.telemetry[-1].kind == "notice"


async def test_help_and_reset_commands():
    app = ChatApp(Config())
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#prompt").value = "/help"
        await pilot.press("enter")
        await pilot.pause()
        # /help is an explicit user request for text -> one chat bubble.
        assert len(app.query(MessageBubble)) == 1

        app.query_one("#prompt").value = "/reset"
        await pilot.press("enter")
        await pilot.pause()
        # /reset clears the transcript; its confirmation goes to the flow graph.
        assert len(app.query(MessageBubble)) == 0
        assert [m.role for m in app.agent.history] == [Role.SYSTEM]


async def test_reply_parses_message_tag_into_bubble(fake_stream_provider):
    app = ChatApp(Config())
    # complete() returns "".join(tokens) -> a full <message> block.
    app.agent.provider = fake_stream_provider(tokens=["<message>Hello!</message>"])
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#prompt").value = "hi"
        await pilot.press("enter")
        await _drain_worker(pilot, app)

        assert [m.role for m in app.agent.history] == [Role.SYSTEM, Role.USER, Role.ASSISTANT]
        assert app.agent.history[-1].content == "<message>Hello!</message>"

        bubbles = list(app.query(MessageBubble))
        assert bubbles[-1].text == "Hello!"
        assert app.query_one("#prompt").disabled is False


async def test_tool_call_shows_activity_then_answer(fake_stream_provider):
    # First reply asks to call calc; second reply is the final message.
    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=[
            '<tool_call><name>calc</name><arguments>{"expression": "2+2"}</arguments></tool_call>',
            "<message>it is 4</message>",
        ]
    )
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#prompt").value = "2+2?"
        await pilot.press("enter")
        await _drain_worker(pilot, app)

        texts = [b.text for b in app.query(MessageBubble)]
        # The transcript stays clean: no tool-activity lines leak into it.
        assert not any("calc" in t for t in texts)
        # Only the human turn and the final answer are chat bubbles.
        assert texts == ["2+2?", "it is 4"]
        # Agent activity lives on the right-hand flow graph instead: the turn
        # built user -> llm -> tool -> llm -> answer, one FlowNode per node.
        from coreybot.frontends.tui.flow import FlowPanel, FlowNode
        from coreybot.frontends.tui.flow import STATUS_OK
        panel = app.query_one(FlowPanel)
        assert len(panel.nodes()) >= 4
        assert any(n.kind == "answer" for n in panel.nodes())
        calc = next(n for n in panel.nodes() if "calc" in n.title)
        assert calc.status == STATUS_OK and "4" in calc.status_text
        assert len(app.query(FlowNode)) == len(panel.nodes())


async def test_error_during_turn_is_shown_and_history_rolled_back(fake_stream_provider):
    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(error=RuntimeError("boom"))
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#prompt").value = "trigger"
        await pilot.press("enter")
        await _drain_worker(pilot, app)

        # The user turn we could not answer is dropped -> only system remains.
        assert [m.role for m in app.agent.history] == [Role.SYSTEM]
        assert "boom" in list(app.query(MessageBubble))[-1].text


def _row_content(grid):
    """Return the content renderable of a MessageBubble grid's single row."""
    from rich.table import Table

    assert isinstance(grid, Table)
    # The grid has two columns (label, content); we want the content cell.
    return grid.columns[1]._cells[0]


def test_assistant_bubble_renders_markdown():
    """Assistant content is Rich Markdown; other roles stay plain Text.

    Both are laid out in a two-column ``Table.grid`` (label | content) so
    wrapped/multi-line content never invades the label column.
    """
    from rich.markdown import Markdown
    from rich.text import Text

    source = "use `x` and:\n```python\nprint(1)\n```"
    bot = MessageBubble(Role.ASSISTANT, source)
    assert isinstance(_row_content(bot.build_renderable(source)), Markdown)
    # Raw text is preserved for history/inspection.
    assert "```python" in bot.text

    you = MessageBubble(Role.USER, "plain `text`")
    assert isinstance(_row_content(you.build_renderable("plain `text`")), Text)


async def test_escape_interrupts_a_running_turn(fake_stream_provider):
    """Pressing Esc cancels an in-flight turn (Go-style context cancel)."""
    app = ChatApp(Config())
    # A slow reply so the turn is still running when we press Esc.
    app.agent.provider = fake_stream_provider(
        replies=["<message>too late</message>"], delay=5
    )
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#prompt").value = "hi"
        await pilot.press("enter")
        # Let the worker start and reach the awaiting provider call.
        for _ in range(50):
            await pilot.pause()
            if app._busy:
                break
        assert app._busy is True
        await pilot.press("escape")
        # Drain until the worker unwinds.
        for _ in range(200):
            await pilot.pause()
            if not app._busy:
                break
        assert app._busy is False
        # 'interrupted' is a flow-graph notice now, not a chat bubble.
        from coreybot.frontends.tui.flow import FlowPanel
        panel = app.query_one(FlowPanel)
        assert any("interrupted" in (n.detail or "") for n in panel.nodes())
        assert any(e.kind == "notice" and e.text == "interrupted"
                   for e in app.agent.telemetry)
        # The abandoned user turn was rolled back -> only the system line remains.
        assert [m.role for m in app.agent.history] == [Role.SYSTEM]


async def test_input_is_single_line_and_above_status_bar():
    """The prompt is a compact one-row input that stays above the bottom bar.

    Regression: a bordered 1-row input made its content row 0 high, spilling
    the typed text downward. The input must keep one real text row and sit
    strictly above the single bottom status bar.
    """
    from textual.widgets import Input
    from textual.containers import Horizontal

    app = ChatApp(Config())
    async with app.run_test(size=(70, 20)):
        prompt = app.query_one("#prompt", Input)
        status = app.query_one("#statusbar", Horizontal)
        assert prompt.region.height == 1
        # The input has a real text row (not eaten by a border).
        assert prompt.content_size.height >= 1
        # Prompt sits strictly above the status bar (no docking overlap).
        assert prompt.region.y + prompt.region.height <= status.region.y


async def test_wrapped_line_stays_in_content_column():
    """Long content soft-wraps under the content column, not the label."""
    from rich.console import Console

    app = ChatApp(Config())
    async with app.run_test(size=(60, 20)) as pilot:
        long_text = "x " * 60  # definitely wider than the chat panel
        bubble = app._add_bubble(Role.USER, long_text)
        await pilot.pause()
        width = app.query_one("#chat").content_size.width
        console = Console(width=max(width, 20), highlight=False, file=__import__("io").StringIO())
        console.print(bubble.build_renderable(long_text))
        lines = [ln for ln in console.file.getvalue().splitlines() if ln.strip()]
        # First line carries the label; wrapped lines must start with spaces
        # (i.e. the label column stays empty on continuation rows).
        assert "you" in lines[0]
        assert len(lines) > 1
        assert all(ln.startswith("     ") for ln in lines[1:])


async def test_unknown_command_reported():
    app = ChatApp(Config())
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#prompt").value = "/nope"
        await pilot.press("enter")
        await pilot.pause()
        assert "unknown command" in list(app.query(MessageBubble))[-1].text


async def test_bottom_bar_is_single_line_with_no_header_or_footer():
    """One bottom row only: no top Header and no separate Footer widget.

    The lone status bar carries the info that used to live in the Header (on
    the left) plus clickable key hints that used to live in the Footer (on the
    right) -- all on a single line.
    """
    from textual.widgets import Footer, Header
    from textual.containers import Horizontal
    from coreybot.frontends.tui.app import _KeyHint

    app = ChatApp(Config())
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        # Neither a Header nor a Footer widget is mounted.
        assert list(app.query(Header)) == []
        assert list(app.query(Footer)) == []
        bar = app.query_one("#statusbar", Horizontal)
        # It is the single bottom row (height 1) at the very bottom.
        assert bar.region.height == 1
        assert bar.region.y == app.size.height - 1
        # Left side carries the Header info (title/model + state).
        assert app.config.model in app._status_text.plain
        # Right side carries the (clickable) key hints as their own widgets.
        hints = list(app.query(_KeyHint))
        actions = {h._action for h in hints}
        assert actions == {"interrupt", "restore", "clear", "quit"}
        # Every hint sits on the same single bottom row.
        assert all(h.region.y == bar.region.y for h in hints)


async def test_heartbeat_animates_and_reflects_busy(fake_stream_provider):
    """The status spinner keeps moving (alive) and reads busy while a turn runs."""
    app = ChatApp(Config())
    # A slow reply so the turn is still running while we sample the spinner.
    app.agent.provider = fake_stream_provider(
        replies=["<message>done</message>"], delay=5
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert app._busy is False
        # Idle: the heartbeat still advances on its own timer.
        first = app._beat
        for _ in range(20):
            await pilot.pause()
            if app._beat != first:
                break
        assert app._beat != first  # animation is running without any activity

        # Start a turn -> the bar should switch to the 'working' state.
        app.query_one("#prompt").value = "hi"
        await pilot.press("enter")
        for _ in range(50):
            await pilot.pause()
            if app._busy:
                break
        assert app._busy is True
        # The redundant 'working'/'ready' TEXT was removed; the run state is now
        # conveyed only by the breathing LED (state key + colour), not words.
        assert app._status_state() == "working"
        assert "working" not in app._status_text.plain
        # Interrupt to unwind the worker cleanly.
        await pilot.press("escape")
        for _ in range(200):
            await pilot.pause()
            if not app._busy:
                break
        assert app._busy is False
        assert app._status_state() == "ready"
        assert "ready" not in app._status_text.plain


async def test_breathing_led_pulses_and_reflects_state():
    """The standby LED breathes (fades) and changes hue with the run state.

    The status glyph is a single dot whose brightness eases 0 -> 1 -> 0 on a
    triangle wave, like an old phone's notification light. Idle it is green and
    slow; while a turn runs it is amber and quicker. This checks the glyph, the
    fade (a range of brightnesses incl. near-off and near-full), and the hue --
    not exact timing.
    """
    def _rgb(text):
        # The LED colour is the Text's base style (Text(glyph, style=...)),
        # which stays a plain '#rrggbb' string; parse it into an (r, g, b).
        from rich.style import Style
        triplet = Style.parse(text.style).color.get_truecolor()
        return (triplet.red, triplet.green, triplet.blue)

    app = ChatApp(Config())
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        # The LED is the single leading glyph before the first space.
        glyph = app._status_text.plain.split(" ", 1)[0]
        assert glyph == "●"

        # Idle: sweep _beat across a full breath and confirm the brightness
        # actually varies -- from near-off up to near-full (a real fade).
        app._busy = False
        period = tui_app._LED_STATES["ready"]["period"]
        greens = []
        for beat in range(0, period):
            app._beat = beat
            r, g, b = _rgb(app._led_indicator())
            greens.append(g)
            # Idle hue is green-dominant whenever it is lit at all.
            assert g >= r and g >= b
        assert min(greens) <= 60      # dips to near-off (breath trough)
        assert max(greens) >= 200     # rises to near-full (breath peak)
        assert max(greens) - min(greens) >= 100  # genuine fade, not steady

        # Busy: peak hue is amber (red and green both high, blue low) and it
        # breathes on a shorter period than idle.
        app._busy = True
        busy_period = tui_app._LED_STATES["working"]["period"]
        assert busy_period < period
        app._beat = busy_period // 2   # peak of the triangle wave
        r, g, b = _rgb(app._led_indicator())
        assert r > 150 and g > 120 and b < r and b < g  # amber, not green

async def test_status_bar_key_hints_are_clickable():
    """Clicking the bottom key hints runs their actions (regression: not clickable).

    The old Footer rendered clickable keys; after moving to a custom status bar
    the hints must remain clickable via ``_KeyHint.on_click``.
    """
    from coreybot.frontends.tui.app import _KeyHint

    app = ChatApp(Config())
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        # Put something in the transcript, then click 'clear' -> it empties.
        app._add_bubble(Role.USER, "hello")
        app._add_bubble(Role.ASSISTANT, "hi")
        await pilot.pause()
        assert len(list(app.query(MessageBubble))) == 2
        clear_hint = next(h for h in app.query(_KeyHint) if h._action == "clear")
        r = clear_hint.region
        await pilot.click(offset=(r.x + 1, r.y))
        await pilot.pause()
        await pilot.pause()
        assert len(list(app.query(MessageBubble))) == 0


async def test_focus_stays_on_prompt_when_something_else_grabs_it():
    """Focus is pinned to the message input; nothing else keeps it.

    The input is the only place text is typed, so the app bounces focus back
    to ``#prompt`` whenever another widget (the flow canvas, a node, or Tab)
    takes it. See ``ChatApp.on_descendant_focus``.
    """
    from textual.widgets import Input
    from coreybot.frontends.tui.flow import FlowCanvas

    app = ChatApp(Config())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", Input)
        assert app.focused is prompt  # focused on mount

        # Programmatically hand focus to the (focusable) flow canvas.
        canvas = app.query_one(FlowCanvas)
        canvas.focus()
        for _ in range(5):
            await pilot.pause()
        assert app.focused is prompt  # bounced straight back

        # A real Tab press must not move focus off the input either.
        await pilot.press("tab")
        for _ in range(5):
            await pilot.pause()
        assert app.focused is prompt


async def test_chat_text_is_selectable_and_copyable_with_focus_on_input():
    """Chat bubbles yield selectable text so drag-select + Ctrl+C copies them.

    Regression: the bubble renders a two-column Table.grid, so Textual's
    default selection extraction returned None and nothing in the chat could
    be copied. ``MessageBubble.get_selection`` now yields the body text. With
    focus pinned to the input, Ctrl+C still copies a chat selection (the
    binding chain falls through Input.copy -> Screen.copy_text).
    """
    from textual.selection import Selection
    from textual.geometry import Offset

    app = ChatApp(Config())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._add_bubble(Role.USER, "plain user line")
        app._add_bubble(Role.ASSISTANT, "markdown answer line")
        await pilot.pause()

        # Both user and assistant bubbles expose their body for selection.
        whole = Selection.from_offsets(Offset(0, 0), Offset(999, 0))
        for bubble in app.query(MessageBubble):
            extracted, ending = bubble.get_selection(whole)
            assert extracted == bubble.text
            assert "\u2502" not in extracted  # never includes the 'you |' prefix

        # Simulate a completed drag-selection over the assistant bubble.
        answer = next(b for b in app.query(MessageBubble) if b.role is Role.ASSISTANT)
        app.screen.selections = {answer: whole}
        await pilot.pause()
        assert app.screen.get_selected_text() == "markdown answer line"

        # Focus is on #prompt, yet Ctrl+C copies the chat selection (not quit).
        from textual.widgets import Input
        assert isinstance(app.focused, Input)
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.is_running  # did not quit -- it copied
        assert app.clipboard == "markdown answer line"


async def test_chat_is_not_focusable_so_drag_select_is_not_cancelled():
    """Dragging to select in the chat must not bounce focus and cancel it.

    Regression: ``#chat`` (a VerticalScroll) was focusable, so pressing to
    start a selection focused it, which fired ``on_descendant_focus`` and
    yanked focus back to the input MID-DRAG, cancelling the selection
    ('a\u9009\u8f93\u5165\u7126\u70b9\u5c31\u8df3\u56de'). ``#chat`` is now
    ``can_focus=False``; a full drag builds and keeps a selection.
    """
    from textual.events import MouseDown, MouseMove, MouseUp
    from textual.widgets import Input

    def mk(cls, x, y):
        return cls(
            widget=None, x=x, y=y, delta_x=0, delta_y=0, button=1,
            shift=False, meta=False, ctrl=False, screen_x=x, screen_y=y, style=None,
        )

    app = ChatApp(Config())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # The chat scroll must not be focusable, and nothing focusable sits
        # over it (otherwise a press would steal focus and cancel the drag).
        chat = app.query_one("#chat")
        assert chat.can_focus is False

        app._add_bubble(Role.ASSISTANT, "SELECTME abcdefghij content row")
        await pilot.pause()
        await pilot.pause()
        bubble = next(iter(app.query(MessageBubble)))
        r = bubble.region
        y = r.y
        x0 = r.x + 8
        assert app.screen.get_focusable_widget_at(x0, y) is None

        screen = app.screen
        screen._forward_event(mk(MouseDown, x0, y))
        await pilot.pause()
        for dx in range(1, 18, 2):
            screen._forward_event(mk(MouseMove, x0 + dx, y))
            await pilot.pause()
        # Mid-drag the selection is live and focus has not been ripped away.
        assert screen._selecting is True
        assert screen.get_selected_text()
        screen._forward_event(mk(MouseUp, x0 + 18, y))
        await pilot.pause()
        await pilot.pause()
        assert screen.get_selected_text() == "SELECTME abcdefghij content row"
        # Copy it with Ctrl+C (focus is still on the input, yet it copies).
        assert isinstance(app.focused, Input)
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.is_running
        assert app.clipboard == "SELECTME abcdefghij content row"


async def test_ctrl_c_still_quits_when_nothing_is_selected():
    """With no selection, Ctrl+C keeps its quit behavior (chain falls through)."""
    app = ChatApp(Config())
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert app.screen.get_selected_text() is None
        await pilot.press("ctrl+c")
        await pilot.pause()
        await pilot.pause()
        assert app.is_running is False


async def test_inspector_modal_opens_takes_focus_and_copies(fake_stream_provider):
    """The node inspector opens full-screen, grabs focus, copies, then restores.

    Opening the modal is a deliberate exception to the focus-pinned-to-input
    rule: focus moves to the modal's TextArea. Copy puts the whole body on the
    clipboard. Closing returns focus to ``#prompt``.
    """
    from textual.widgets import Input, TextArea
    from coreybot.frontends.tui.app import InspectModal
    from coreybot.frontends.tui.flow import FlowPanel

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(replies=["<message>a long answer body</message>"])
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = "please explain the topic"
        await pilot.press("enter")
        for _ in range(60):
            await pilot.pause()
            if not app._busy:
                break
        panel = app.query_one(FlowPanel)
        key = next(k for k, v in panel._model.items() if v.source == "llm")

        panel.open_inspector(key)
        for _ in range(8):
            await pilot.pause()
        assert isinstance(app.screen, InspectModal)
        area = app.screen.query_one("#inspect-body", TextArea)
        assert app.focused is area  # focus moved to the modal
        assert "===== INPUT =====" in area.text
        assert "===== RESPONSE =====" in area.text
        assert "please explain the topic" in area.text

        # Copy the whole body to the clipboard.
        app.screen.action_copy()
        await pilot.pause()
        assert app.clipboard.startswith("===== INPUT =====")

        # Close -> focus returns to the input (policy resumes).
        await pilot.press("escape")
        for _ in range(8):
            await pilot.pause()
        assert len(app.screen_stack) == 1
        assert app.focused is app.query_one("#prompt", Input)


async def test_inspector_modal_buttons_show_shortcuts_and_work(fake_stream_provider):
    """The modal's footer buttons show their shortcut and act on click.

    'Copy (c)' copies the body; 'Close (Esc)' dismisses. The shortcut letter
    lives inside each button label so it is discoverable.
    """
    from textual.widgets import Input, Button
    from coreybot.frontends.tui.app import InspectModal
    from coreybot.frontends.tui.flow import FlowPanel

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(replies=["<message>a long answer body</message>"])
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = "please explain"
        await pilot.press("enter")
        for _ in range(60):
            await pilot.pause()
            if not app._busy:
                break
        panel = app.query_one(FlowPanel)
        key = next(k for k, v in panel._model.items() if v.source == "llm")

        panel.open_inspector(key)
        for _ in range(8):
            await pilot.pause()
        modal = app.screen
        assert isinstance(modal, InspectModal)
        # Shortcuts are shown inside the labels.
        assert str(modal.query_one("#inspect-copy", Button).label) == "Copy (c)"
        assert str(modal.query_one("#inspect-close", Button).label) == "Close (Esc)"

        # Clicking Copy copies the whole body to the clipboard.
        await pilot.click("#inspect-copy")
        await pilot.pause()
        assert app.clipboard.startswith("===== INPUT =====")

        # Clicking Close dismisses the modal (focus returns to the input).
        await pilot.click("#inspect-close")
        for _ in range(8):
            await pilot.pause()
        assert len(app.screen_stack) == 1
        assert app.focused is app.query_one("#prompt", Input)


async def test_all_modals_put_close_button_bottom_right(fake_stream_provider):
    """Every modal keeps its Close button in the bottom-right corner.

    The action row uses a flexible 1fr hint as the gap, so the Close button is
    shoved to the far right (flush with the box's inner edge) -- consistent with
    the app's status bar, whose exit/key hints live on the right. Primary
    actions stay on the left.
    """
    from textual.widgets import Input, Button
    from coreybot.frontends.tui.app import (
        InspectModal,
        SessionModal,
        RestoreModal,
        MessageBubble,
    )
    from coreybot.frontends.tui.flow import FlowPanel

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=["<message>a long answer body</message>", "<message>second</message>"]
    )
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = "please explain"
        await pilot.press("enter")
        await _drain_worker(pilot, app)

        def assert_close_bottom_right(modal, box_id, close_id, primary_id):
            box = modal.query_one(box_id)
            close = modal.query_one(close_id, Button)
            primary = modal.query_one(primary_id, Button)
            # Close hugs the box's right edge (a couple cells of border/padding).
            assert 0 <= box.region.right - close.region.right <= 3
            # The primary action stays on the left, left of Close.
            assert primary.region.x < close.region.x
            # Label/id contract is unchanged.
            assert str(close.label) == "Close (Esc)"

        # Inspector modal (from a flow llm node).
        panel = app.query_one(FlowPanel)
        key = next(k for k, v in panel._model.items() if v.source == "llm")
        panel.open_inspector(key)
        for _ in range(12):
            await pilot.pause()
        assert isinstance(app.screen, InspectModal)
        assert_close_bottom_right(
            app.screen, "#inspect-box", "#inspect-close", "#inspect-copy"
        )
        await pilot.press("escape")
        for _ in range(8):
            await pilot.pause()

        # Per-bubble session modal (Edit offered for a user message).
        bubble = next(b for b in app.query(MessageBubble) if b.role is Role.USER)
        app.push_screen(SessionModal(bubble.text, bubble.role, bubble.history_index))
        for _ in range(12):
            await pilot.pause()
        assert isinstance(app.screen, SessionModal)
        assert_close_bottom_right(
            app.screen, "#session-box", "#session-close", "#session-copy"
        )
        await pilot.press("escape")
        for _ in range(8):
            await pilot.pause()

        # History-tree restore modal.
        app.action_restore()
        for _ in range(15):
            await pilot.pause()
        assert isinstance(app.screen, RestoreModal)
        assert_close_bottom_right(
            app.screen, "#restore-box", "#restore-close", "#restore-session"
        )


async def test_clicking_flow_chart_keeps_focus_on_prompt():
    """Clicking the flow chart pane does not steal focus from the input."""
    from textual.widgets import Input
    from coreybot.frontends.tui.flow import FlowPanel

    app = ChatApp(Config())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", Input)
        panel = app.query_one(FlowPanel)
        reg = panel.region
        # Click empty background near the top-right of the flow pane.
        await pilot.click(offset=(reg.x + reg.width - 2, reg.y + 2))
        for _ in range(5):
            await pilot.pause()
        assert app.focused is prompt


async def test_clicking_interrupt_hint_cancels_turn(fake_stream_provider):
    """Clicking the 'Esc interrupt' hint cancels an in-flight turn like the key."""
    from coreybot.frontends.tui.app import _KeyHint

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=["<message>too late</message>"], delay=5
    )
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        app.query_one("#prompt").value = "hi"
        await pilot.press("enter")
        for _ in range(50):
            await pilot.pause()
            if app._busy:
                break
        assert app._busy is True
        interrupt_hint = next(h for h in app.query(_KeyHint) if h._action == "interrupt")
        r = interrupt_hint.region
        await pilot.click(offset=(r.x + 1, r.y))
        for _ in range(200):
            await pilot.pause()
            if not app._busy:
                break
        assert app._busy is False
        assert any(
            e.kind == "notice" and e.text == "interrupted"
            for e in app.agent.telemetry
        )


async def _run_two_turns(app, pilot):
    """Drive two plain-message turns so the transcript has 4 bubbles."""
    await pilot.pause()
    app.query_one("#prompt").value = "hello one"
    await pilot.press("enter")
    await _drain_worker(pilot, app)
    app.query_one("#prompt").value = "hello two"
    await pilot.press("enter")
    await _drain_worker(pilot, app)


async def test_bubbles_carry_their_history_index(fake_stream_provider):
    """Each bubble records the index of the message it represents.

    The user bubble points at the USER entry ``arun_turn`` appends; the
    answer bubble points at the ASSISTANT raw reply ``_drive`` appends. This
    mapping is what makes 'Restore to here' able to rewind precisely.
    """
    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=["<message>first answer</message>", "<message>second answer</message>"]
    )
    async with app.run_test(size=(100, 30)) as pilot:
        await _run_two_turns(app, pilot)
        bubbles = list(app.query(MessageBubble))
        pairs = [(b.role, b.text, b.history_index) for b in bubbles]
        assert pairs == [
            (Role.USER, "hello one", 1),
            (Role.ASSISTANT, "first answer", 2),
            (Role.USER, "hello two", 3),
            (Role.ASSISTANT, "second answer", 4),
        ]
        # The indices actually address those history entries.
        for _role, _text, idx in pairs:
            assert app.agent.history[idx].role == _role


async def test_first_click_selects_second_click_opens_modal(fake_stream_provider):
    """Two-step open: click 1 selects the bubble; click 2 opens the modal.

    Guards against opening the full-screen modal by accident. The first
    click only highlights the bubble (no screen pushed); a second click on
    the selected bubble pushes the ``SessionModal``.
    """
    from coreybot.frontends.tui.app import SessionModal

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(replies=["<message>an answer</message>"])
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one("#prompt").value = "hi"
        await pilot.press("enter")
        await _drain_worker(pilot, app)
        bubble = next(iter(app.query(MessageBubble)))

        # First click: selected, no modal.
        app.post_message(MessageBubble.SelectRequested(bubble))
        await pilot.pause()
        await pilot.pause()
        assert bubble.selected is True
        assert app._selected_bubble is bubble
        assert len(app.screen_stack) == 1

        # Second click: modal opens.
        app.post_message(MessageBubble.OpenRequested(bubble))
        for _ in range(8):
            await pilot.pause()
        assert isinstance(app.screen, SessionModal)


def _open_session_modal(app, bubble):
    """Build a SessionModal for ``bubble`` the way the app's handler does."""
    from coreybot.frontends.tui.app import SessionModal

    return SessionModal(bubble.text, bubble.role, bubble.history_index)


async def test_session_modal_shows_message_focused_and_copyable(fake_stream_provider):
    """The modal shows the message in a focused, selectable, copyable TextArea.

    Terminal-native selection is flaky, so the modal renders the message text
    in a read-only ``TextArea`` that takes focus (a deliberate exception to the
    focus-pinned-to-input policy) and can be selected + copied. Closing returns
    focus to the input.
    """
    from textual.widgets import Input, TextArea

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=["<message>first answer</message>", "<message>second answer</message>"]
    )
    async with app.run_test(size=(120, 34)) as pilot:
        await _run_two_turns(app, pilot)
        answer = next(b for b in app.query(MessageBubble) if b.text == "second answer")

        app.push_screen(_open_session_modal(app, answer))
        for _ in range(8):
            await pilot.pause()
        # Focused, selectable, read-only message text; no session tree anymore.
        area = app.screen.query_one("#session-body", TextArea)
        assert area.text == "second answer"
        assert app.focused is area
        assert area.read_only is True
        area.select_all()
        assert area.selected_text == "second answer"
        assert len(app.screen.query("#session-tree")) == 0

        app.screen.action_copy()
        await pilot.pause()
        assert app.clipboard == "second answer"

        await pilot.press("escape")
        for _ in range(8):
            await pilot.pause()
        assert len(app.screen_stack) == 1
        assert app.focused is app.query_one("#prompt", Input)


async def test_session_modal_buttons_show_shortcuts(fake_stream_provider):
    """Footer buttons show their shortcut letter in the label (Codex style)."""
    from textual.widgets import Button, Input
    from coreybot.frontends.tui.app import SessionModal

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(replies=["<message>done</message>"])
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = "hi"
        await pilot.press("enter")
        await _drain_worker(pilot, app)
        bubble = next(iter(app.query(MessageBubble)))

        app.push_screen(_open_session_modal(app, bubble))
        for _ in range(8):
            await pilot.pause()
        modal = app.screen
        assert str(modal.query_one("#session-copy", Button).label) == "Copy (c)"
        # Edit (rewind + resend) is offered for a USER message, with its label.
        assert str(modal.query_one("#session-edit", Button).label) == "Edit & resend (e)"
        # Restore was removed; no session tree / restore button remains.
        assert len(modal.query("#session-restore")) == 0
        assert str(modal.query_one("#session-close", Button).label) == "Close (Esc)"


async def test_edit_rewinds_to_before_message_and_prefills_input(fake_stream_provider):
    """Edit == rewind to just before this user message, then prefill input.

    Editing the 2nd user turn rewinds history to end right after turn 1 (the
    target message and everything after it drop from the current line), the
    transcript is rebuilt to that point, and the input is prefilled with the
    message so a modified version can be re-sent.
    """
    from textual.widgets import Input

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=["<message>first answer</message>", "<message>second answer</message>"]
    )
    async with app.run_test(size=(120, 34)) as pilot:
        await _run_two_turns(app, pilot)
        second_user = next(b for b in app.query(MessageBubble) if b.text == "hello two")

        app.push_screen(_open_session_modal(app, second_user))
        for _ in range(8):
            await pilot.pause()
        app.screen.action_edit()
        for _ in range(10):
            await pilot.pause()

        assert len(app.screen_stack) == 1
        # History rewound to JUST BEFORE 'hello two' (system + turn 1 only).
        assert [m.role for m in app.agent.history] == [
            Role.SYSTEM, Role.USER, Role.ASSISTANT
        ]
        assert app.agent.history[1].content == "hello one"
        # Transcript rebuilt to turn 1; input prefilled with the edited message.
        assert [b.text for b in app.query(MessageBubble)] == ["hello one", "first answer"]
        assert app.query_one("#prompt", Input).value == "hello two"
        assert app.focused is app.query_one("#prompt", Input)


async def test_edit_then_resend_branches_and_keeps_old_line(fake_stream_provider):
    """After Edit, re-sending a modified message forks the tree (old line kept)."""
    from textual.widgets import Input

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=[
            "<message>first answer</message>",
            "<message>second answer</message>",
            "<message>edited answer</message>",
        ]
    )
    async with app.run_test(size=(120, 34)) as pilot:
        await _run_two_turns(app, pilot)
        second_user = next(b for b in app.query(MessageBubble) if b.text == "hello two")
        app.push_screen(_open_session_modal(app, second_user))
        for _ in range(8):
            await pilot.pause()
        app.screen.action_edit()
        for _ in range(10):
            await pilot.pause()

        # Re-send a modified version -> a new branch off turn 1.
        app.query_one("#prompt", Input).value = "hello two EDITED"
        await pilot.press("enter")
        await _drain_worker(pilot, app)

        labels = [r.node.label for r in app.agent.sessions.rows()]
        # The original turn-2 node survives AND the edited branch exists.
        assert any(lbl == "hello two" for lbl in labels)
        assert any(lbl == "hello two EDITED" for lbl in labels)
        assert any(r.is_branch_point for r in app.agent.sessions.rows())
        assert [b.text for b in app.query(MessageBubble)] == [
            "hello one", "first answer", "hello two EDITED", "edited answer"
        ]


async def test_edit_absent_on_assistant_bubbles(fake_stream_provider):
    """Edit only makes sense for user messages; answers have no Edit button."""
    from textual.widgets import Input
    from coreybot.frontends.tui.app import SessionModal

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(replies=["<message>an answer</message>"])
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = "hi"
        await pilot.press("enter")
        await _drain_worker(pilot, app)
        answer = next(b for b in app.query(MessageBubble) if b.role is Role.ASSISTANT)
        app.push_screen(_open_session_modal(app, answer))
        for _ in range(8):
            await pilot.pause()
        # No Edit button for an assistant message; action_edit is a no-op.
        assert len(app.screen.query("#session-edit")) == 0
        app.screen.action_edit()
        await pilot.pause()
        assert isinstance(app.screen, SessionModal)  # still open, nothing happened


async def test_edit_while_turn_running_cancels_it_before_rewinding(fake_stream_provider):
    """Editing mid-turn cancels the in-flight turn and AWAITS it before rewinding.

    Regression: editing a message while a turn was still running left the old
    worker alive; its late response then landed in the freshly-rewound
    transcript as an orphan reply (no matching request) and committed a stray
    node. Edit must reuse the CancelToken: fire it, block on the worker until
    the turn unwinds, and only then rewind + prefill.
    """
    from textual.widgets import Input
    from coreybot.frontends.tui.app import SessionModal

    app = ChatApp(Config())
    # Turn 1 is fast; turn 2 is slow (still running when we edit).
    app.agent.provider = fake_stream_provider(
        replies=["<message>answer one</message>", "<message>never shown</message>"],
        delay=0,
    )
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = "hello one"
        await pilot.press("enter")
        await _drain_worker(pilot, app)

        # Make the SECOND turn hang mid-flight.
        app.agent.provider.delay = 5
        app.query_one("#prompt", Input).value = "hello two"
        await pilot.press("enter")
        for _ in range(50):
            await pilot.pause()
            if app._busy:
                break
        assert app._busy is True

        # Edit the first user message WHILE turn 2 is still running (via the
        # session modal's Edit path; the inline bubble link was removed).
        first_user = next(b for b in app.query(MessageBubble) if b.text == "hello one")
        app.post_message(
            SessionModal.EditRequested(first_user.text, first_user.history_index)
        )
        for _ in range(200):
            await pilot.pause()
            if not app._busy:
                break

        # The in-flight turn was cancelled and awaited: not busy, input re-enabled.
        assert app._busy is False
        # History rewound to just before "hello one" (root: system only). No
        # orphan assistant reply from the abandoned turn is present.
        assert [m.role for m in app.agent.history] == [Role.SYSTEM]
        # Transcript is empty (rewound); the edited text is prefilled.
        assert [b.text for b in app.query(MessageBubble)] == []
        assert app.query_one("#prompt", Input).value == "hello one"
        assert app.focused is app.query_one("#prompt", Input)
        # The abandoned turn did NOT commit a node: only root + turn 1 exist.
        labels = [r.node.label for r in app.agent.sessions.rows()]
        assert labels.count("hello two") == 0
        # Telemetry now reflects the rewound state (the transient "interrupted"
        # notice is dropped by the checkout, matching the clean-rewind intent):
        # no turn_start for the abandoned "hello two" survives.
        assert not any(
            e.kind == "turn_start" and getattr(e, "text", "") == "hello two"
            for e in app.agent.telemetry
        )


async def test_typing_lands_in_input_even_when_a_bubble_is_selected(fake_stream_provider):
    """Selecting a bubble must not steal typing: focus stays on the input.

    A selected bubble is only a visual highlight; the app keeps keyboard
    focus pinned to ``#prompt`` so characters typed after a selection still
    go into the message box.
    """
    from textual.widgets import Input

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(replies=["<message>ok</message>"])
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one("#prompt").value = "hi"
        await pilot.press("enter")
        await _drain_worker(pilot, app)
        bubble = next(iter(app.query(MessageBubble)))
        app.post_message(MessageBubble.SelectRequested(bubble))
        await pilot.pause()
        await pilot.pause()
        assert bubble.selected is True

        prompt = app.query_one("#prompt", Input)
        prompt.value = ""
        await pilot.press("a", "b", "c")
        assert app.focused is prompt
        assert prompt.value == "abc"


async def test_restore_hint_opens_session_tree_map_with_current_head_marked(
    fake_stream_provider,
):
    """The status bar has a clickable 'restore' hint that opens the node map.

    Restore is a first-class feature (separate from a bubble's Edit): a
    ``_KeyHint`` on the bottom bar opens a ``RestoreModal`` listing the WHOLE
    session tree, with the current head marked, so any commit can be picked.
    """
    from coreybot.frontends.tui.app import (
        _KeyHint,
        _SessionNodeBox,
        RestoreModal,
    )

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=["<message>answer one</message>", "<message>answer two</message>"]
    )
    async with app.run_test(size=(120, 34)) as pilot:
        await _run_two_turns(app, pilot)

        # A clickable 'restore' hint exists on the status bar.
        restore_hint = next(
            (h for h in app.query(_KeyHint) if h._action == "restore"), None
        )
        assert restore_hint is not None
        r = restore_hint.region
        await pilot.click(offset=(r.x + 1, r.y))
        for _ in range(15):
            await pilot.pause()

        # It opened the RestoreModal, which draws the WHOLE session tree as
        # telemetry-style boxes (one box per commit) on a pannable canvas.
        modal = app.screen
        assert isinstance(modal, RestoreModal)
        rows = app.agent.sessions.rows()
        boxes = list(modal.query(_SessionNodeBox))
        assert len(boxes) == len(rows)
        # It starts highlighting the CURRENT head (where we are now).
        assert modal._selected_node_id() == app.agent.sessions.head()


async def test_restore_scoped_session_restores_full_context(fake_stream_provider):
    """Select a node (no restore), then 's' rewinds history + telemetry to it.

    Restore is a scoped, explicit action: selecting a box only moves the
    cursor (the modal stays open), and it is the 's' key (Restore session)
    that rewinds the conversation. 'w' would additionally rewind files; here
    we assert the session scope.
    """
    from coreybot.frontends.tui.app import RestoreModal, _SessionCanvas
    from textual.widgets import Input

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=["<message>answer one</message>", "<message>answer two</message>"]
    )
    async with app.run_test(size=(120, 34)) as pilot:
        await _run_two_turns(app, pilot)
        assert [b.text for b in app.query(MessageBubble)] == [
            "hello one", "answer one", "hello two", "answer two"
        ]

        rows = app.agent.sessions.rows()
        app.action_restore()
        for _ in range(15):
            await pilot.pause()
        modal = app.screen
        canvas = modal.query_one("#restore-canvas", _SessionCanvas)
        # Select the FIRST user turn's box on the canvas.
        target_id = next(
            row.node.id for row in rows if row.node.label == "hello one"
        )
        canvas.select(target_id)
        await pilot.pause()
        # Selecting alone must NOT restore -- the modal is still up.
        assert app.screen is modal
        assert len(app.screen_stack) == 2
        # 's' = Restore session: dismiss the modal and rewind the conversation.
        await pilot.press("s")
        for _ in range(60):
            await pilot.pause()

        # Back on the main screen with the context rewound to just after turn 1.
        assert len(app.screen_stack) == 1
        assert [b.text for b in app.query(MessageBubble)] == [
            "hello one", "answer one"
        ]
        assert [m.role for m in app.agent.history] == [
            Role.SYSTEM, Role.USER, Role.ASSISTANT
        ]
        # Focus returns to the input (modal was a temporary focus exception).
        assert app.focused is app.query_one("#prompt", Input)
        # A 'restored session' notice is recorded in telemetry (-> flow).
        assert any(
            e.kind == "notice" and getattr(e, "text", "") == "restored session"
            for e in app.agent.telemetry
        )


async def test_restore_while_turn_running_cancels_it_before_checkout(
    fake_stream_provider,
):
    """Restoring mid-turn cancels the in-flight turn and AWAITS it first.

    A restore rewrites history/telemetry; a still-running turn would otherwise
    land an orphan reply (and a stray node) after the checkout. Restore reuses
    the same CancelToken flow as Edit: fire it, block on the worker, then
    checkout.
    """
    from coreybot.frontends.tui.app import RestoreModal
    from textual.widgets import Input

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=["<message>answer one</message>", "<message>never shown</message>"],
        delay=0,
    )
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = "hello one"
        await pilot.press("enter")
        await _drain_worker(pilot, app)

        # Make the SECOND turn hang mid-flight, then restore to root.
        app.agent.provider.delay = 5
        app.query_one("#prompt", Input).value = "hello two"
        await pilot.press("enter")
        for _ in range(80):
            await pilot.pause()
            if app._busy:
                break
        assert app._busy is True

        app.post_message(RestoreModal.RestoreRequested(app.agent.sessions.root_id))
        for _ in range(300):
            await pilot.pause()
            if not app._busy:
                break

        # The in-flight turn was cancelled and awaited before the checkout.
        assert app._busy is False
        assert not app.query_one("#prompt", Input).disabled
        # Context restored to the root (system only); no orphan reply/bubble.
        assert [m.role for m in app.agent.history] == [Role.SYSTEM]
        assert [b.text for b in app.query(MessageBubble)] == []
        # The abandoned turn committed no node (only root + turn 1 exist).
        labels = [r.node.label for r in app.agent.sessions.rows()]
        assert labels.count("hello two") == 0
        # No surviving turn_start for the abandoned "hello two".
        assert not any(
            e.kind == "turn_start" and getattr(e, "text", "") == "hello two"
            for e in app.agent.telemetry
        )


async def test_restore_is_non_destructive_old_branch_survives(fake_stream_provider):
    """Restore is a git checkout: the branch we left stays in the tree."""
    from coreybot.frontends.tui.app import RestoreModal

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=[
            "<message>answer one</message>",
            "<message>answer two</message>",
            "<message>answer three</message>",
        ]
    )
    async with app.run_test(size=(120, 34)) as pilot:
        await _run_two_turns(app, pilot)
        rows = app.agent.sessions.rows()
        node_one = next(r.node.id for r in rows if r.node.label == "hello one")

        # Restore to turn 1, then send a new turn -> it branches off turn 1.
        app.post_message(RestoreModal.RestoreRequested(node_one))
        for _ in range(60):
            await pilot.pause()
        app.query_one("#prompt").value = "hello three"
        await pilot.press("enter")
        await _drain_worker(pilot, app)

        labels = [r.node.label for r in app.agent.sessions.rows()]
        # The original turn-2 line survives AND the new branch exists.
        assert "hello two" in labels
        assert "hello three" in labels
        assert any(r.is_branch_point for r in app.agent.sessions.rows())
        assert [b.text for b in app.query(MessageBubble)] == [
            "hello one", "answer one", "hello three", "answer three"
        ]


async def test_restore_modal_draws_branching_tree_connectors(fake_stream_provider):
    """The restore map DRAWS the tree like the telemetry dashboard: boxes with
    connector lines, forks stepping right.

    After Edit-then-resend forks the session, the RestoreModal lays the commits
    out via ``graph_layout`` -- a linear line stacks straight down (col 0), a
    fork steps to a new column -- and the edge layer draws the orthogonal
    connectors between the boxes. So an abandoned line and the current branch
    are visually distinct (a column step + drawn branch), not a flat indent.
    """
    from coreybot.frontends.tui.app import (
        RestoreModal,
        SessionModal,
        _SessionCanvas,
        _SessionNodeBox,
    )

    VBAR = "\u2502"    # vertical run down the child's own lane
    HBAR = "\u2500"    # horizontal reach across empty lanes to the child
    ELBOW = "\u2514"   # bottom corner turning right into the child midpoint
    TOP_CORNER = "\u2510"  # top run (from the left) turns DOWN into the lane
    DIAG = "\u2572"    # (must be ABSENT -- connectors are straight lines)

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=[
            "<message>answer one</message>",
            "<message>answer two</message>",
            "<message>edited answer</message>",
        ]
    )
    async with app.run_test(size=(120, 34)) as pilot:
        await _run_two_turns(app, pilot)

        # Edit the FIRST user message and re-send a modified copy -> forks the
        # tree at the root (original line 'hello one -> hello two' survives).
        first = next(b for b in app.query(MessageBubble) if b.text == "hello one")
        app.post_message(SessionModal.EditRequested(first.text, first.history_index))
        for _ in range(80):
            await pilot.pause()
        app.query_one("#prompt").value = "hello one EDITED"
        await pilot.press("enter")
        await _drain_worker(pilot, app)

        # The fork is real: the layout has a branch point and steps to a new
        # column (a linear history would keep everything in column 0).
        layout = app.agent.sessions.graph_layout()
        assert any(node.is_branch_point for node in layout)
        assert max(node.col for node in layout) >= 1

        app.action_restore()
        for _ in range(15):
            await pilot.pause()
        modal = app.screen
        assert isinstance(modal, RestoreModal)

        # One box per commit, and the head box is marked (green `current`).
        boxes = {b._node_id: b for b in modal.query(_SessionNodeBox)}
        head = app.agent.sessions.head()
        assert head in boxes
        assert boxes[head].has_class("current")

        # The edge layer actually DRAWS orthogonal connectors between boxes.
        canvas = modal.query_one("#restore-canvas", _SessionCanvas)
        edge = canvas._edge_layer
        glyphs = set()
        for y in range(edge.size.height):
            glyphs |= set(edge.render_line(y).text)
        assert VBAR in glyphs
        assert HBAR in glyphs
        # The fork leaves the parent's right-border midpoint, turns DOWN the
        # child's own lane with a down+LEFT corner, and turns RIGHT into the
        # child with an up+right elbow -- pure straight lines, no diagonal.
        assert ELBOW in glyphs
        assert TOP_CORNER in glyphs
        assert DIAG not in glyphs

        # Both branches' user turns are present as boxes (labels carry the text).
        labels = {row.node.id: row.node.label for row in app.agent.sessions.rows()}
        drawn_labels = {labels.get(nid, "") for nid in boxes}
        assert "hello one EDITED" in drawn_labels
        assert "hello two" in drawn_labels


async def test_history_tree_connectors_are_planar_and_miss_box_interiors():
    """The drawn map is PLANAR: no connector crosses another, and none runs
    through a box interior.

    Builds a nested-fork tree. Because columns are never reused, every branch
    keeps its own lane and each fork's connector only traverses empty lanes,
    so (1) no horizontal run passes THROUGH a vertical rail (zero crossings)
    and (2) no connector glyph lands inside a box interior.
    """
    from coreybot.frontends.tui.app import (
        ChatApp,
        RestoreModal,
        _SessionCanvas,
        _SessionNodeBox,
    )
    from coreybot.core.config import Config
    from coreybot.runtime.session import SessionTree, Snapshot

    VBAR = "\u2502"
    HBAR = "\u2500"
    conn = set("\u2502\u2500\u2514\u250c\u2510\u251c\u2518\u252c\u2534\u253c")

    # root -> (A1 -> A2) mainline; a B fork that itself forks to C; and a D
    # fork off root. Every fork opens a FRESH lane (no reuse), which is what
    # keeps the picture planar.
    tree = SessionTree()
    root = tree.root_id
    tree.commit(Snapshot(), label="A1")
    tree.commit(Snapshot(), label="A2")
    tree._heads["main"] = root
    b1 = tree.commit(Snapshot(), label="B1")
    tree.commit(Snapshot(), label="B2")
    tree._heads["main"] = b1.id
    tree.commit(Snapshot(), label="C1")
    tree._heads["main"] = root
    tree.commit(Snapshot(), label="D1")

    # No lane reuse: each of the three forks (B, C, D) took its own column.
    layout = tree.graph_layout()
    cols = sorted({node.col for node in layout})
    assert cols == [0, 1, 2, 3]

    app = ChatApp(Config())
    async with app.run_test(size=(160, 44)) as pilot:
        await pilot.pause()
        app.push_screen(RestoreModal(tree, tree.head()))
        for _ in range(20):
            await pilot.pause()
        canvas = app.screen.query_one("#restore-canvas", _SessionCanvas)
        edge = canvas._edge_layer
        grid = [
            "".join(seg.text for seg in edge.render_line(y))
            for y in range(edge.size.height)
        ]

        # (1) Zero crossings: a horizontal cell with a vertical rail both
        # directly above AND below means one line passes THROUGH another.
        crossings = 0
        for y in range(1, len(grid) - 1):
            for x, ch in enumerate(grid[y]):
                if ch != HBAR:
                    continue
                up = grid[y - 1][x] if x < len(grid[y - 1]) else " "
                dn = grid[y + 1][x] if x < len(grid[y + 1]) else " "
                if up == VBAR and dn == VBAR:
                    crossings += 1
        assert crossings == 0

        # (2) No connector glyph inside a box interior.
        interiors = []
        for box in canvas.query(_SessionNodeBox):
            r = box.region
            interiors.append(
                (r.x + 1, r.x + r.width - 2, r.y + 1, r.y + r.height - 2)
            )
        ex = canvas.region.x - int(canvas.scroll_offset.x)
        ey = canvas.region.y - int(canvas.scroll_offset.y)
        hits = 0
        for ly, row in enumerate(grid):
            for lx, ch in enumerate(row):
                if ch not in conn:
                    continue
                sx = ex + lx
                sy = ey + ly
                for (x0, x1, y0, y1) in interiors:
                    if x0 <= sx <= x1 and y0 <= sy <= y1:
                        hits += 1
        assert hits == 0


async def test_history_tree_overlapping_forks_stay_planar_in_own_lanes():
    """Two forks alive over the same rows each get their OWN lane and never
    cross.

    ``root`` forks R (a second child) and its mainline ``m1`` also forks S;
    ``m1`` has a deep tail so both forks are alive over the same rows. With
    columns never reused they land in separate lanes, so the two connectors
    run side by side without crossing and each branch keeps a dedicated
    column for life.
    """
    from coreybot.frontends.tui.app import (
        ChatApp,
        RestoreModal,
        _SessionCanvas,
    )
    from coreybot.core.config import Config
    from coreybot.runtime.session import SessionTree, Snapshot

    VBAR = "\u2502"
    HBAR = "\u2500"

    tree = SessionTree()
    root = tree.root_id
    m1 = tree.commit(Snapshot(), label="m1 col0")
    tree.commit(Snapshot(), label="m1 tail a")
    tree.commit(Snapshot(), label="m1 tail b")
    tree._heads["main"] = root
    tree.commit(Snapshot(), label="root fork R")
    tree._heads["main"] = m1.id
    tree.commit(Snapshot(), label="m1 fork S")

    # Three lanes total (mainline + two forks), none reused.
    layout = tree.graph_layout()
    assert sorted({n.col for n in layout}) == [0, 1, 2]

    app = ChatApp(Config())
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        app.push_screen(RestoreModal(tree, tree.head()))
        for _ in range(24):
            await pilot.pause()
        canvas = app.screen.query_one("#restore-canvas", _SessionCanvas)
        edge = canvas._edge_layer
        grid = [
            "".join(seg.text for seg in edge.render_line(y))
            for y in range(edge.size.height)
        ]
        # Zero crossings: no horizontal passes through a vertical rail.
        crossings = 0
        for y in range(1, len(grid) - 1):
            for x, ch in enumerate(grid[y]):
                if ch != HBAR:
                    continue
                up = grid[y - 1][x] if x < len(grid[y - 1]) else " "
                dn = grid[y + 1][x] if x < len(grid[y + 1]) else " "
                if up == VBAR and dn == VBAR:
                    crossings += 1
        assert crossings == 0
        # Two independent vertical lanes are actually drawn (the two forks).
        lane_cols = set()
        for row in grid:
            for x, ch in enumerate(row):
                if ch == VBAR:
                    lane_cols.add(x)
        assert len(lane_cols) >= 2


def _edge_layer_ascii(canvas):
    """Render ONLY the session canvas's connector layer to a char grid.

    Returns the drawn connectors cropped to their non-blank bounding box as a
    list of strings -- a stable little "ASCII snapshot" of the wiring that is
    independent of box widths, padding and the surrounding modal chrome (which
    absolute screen coordinates are not). Blank => the layer drew nothing.
    """
    edge = canvas._edge_layer
    rows = [
        "".join(seg.text for seg in edge.render_line(y))
        for y in range(edge.size.height)
    ]
    marks = [
        (y, x)
        for y, row in enumerate(rows)
        for x, ch in enumerate(row)
        if ch != " "
    ]
    if not marks:
        return []
    y0 = min(y for y, _x in marks)
    y1 = max(y for y, _x in marks)
    x0 = min(x for _y, x in marks)
    x1 = max(x for _y, x in marks)
    return [rows[y][x0:x1 + 1].rstrip() for y in range(y0, y1 + 1)]


async def _restore_edge_ascii(tree, size=(120, 44)):
    """Open the RestoreModal for ``tree`` and snapshot its connector layer."""
    from coreybot.frontends.tui.app import RestoreModal, _SessionCanvas

    app = ChatApp(Config())
    app.agent.sessions = tree
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app.push_screen(RestoreModal(tree, tree.head()))
        for _ in range(24):
            await pilot.pause()
        canvas = app.screen.query_one("#restore-canvas", _SessionCanvas)
        return _edge_layer_ascii(canvas)


async def test_history_tree_linear_draws_no_connector_lines():
    """A linear history draws NOTHING on the connector layer.

    Same-column children are always the row directly below their parent, so
    the boxes touch and the stack itself shows the link -- drawing a line
    there would only stamp a corner into the child's label. So the wiring
    layer stays empty until a real fork appears.
    """
    from coreybot.runtime.session import SessionTree, Snapshot

    tree = SessionTree()
    tree.commit(Snapshot(), label="q1")
    tree.commit(Snapshot(), label="q2")
    assert await _restore_edge_ascii(tree) == []


def _assert_L_connector(rows):
    """Assert ``rows`` is a single planar, STRAIGHT (orthogonal) L whose two
    ends are vertically CENTERED on their bubbles.

    Shape (widths depend on box sizes, so only the SHAPE is pinned):
      * a TOP horizontal run of ``\u2500`` ending in a down+LEFT corner
        ``\u2510`` (leaves the parent's right-border MIDPOINT, turns DOWN);
      * a straight vertical ``\u2502`` down ONE lane;
      * a BOTTOM up+right corner ``\u2514`` (turns RIGHT into the child's
        left-border MIDPOINT), optionally followed by a short ``\u2500``.
    There is NO diagonal glyph -- the connector is pure box-drawing lines.
    """
    V = "\u2502"
    H = "\u2500"
    DL = "\u2510"
    UR = "\u2514"
    DIAG = "\u2572"
    assert len(rows) >= 3, rows
    # No diagonal anywhere: the connector is straight lines only.
    assert DIAG not in "".join(rows), rows
    # (1) Top run leaves horizontally and turns DOWN with a down+LEFT corner.
    top = rows[0]
    assert top.endswith(DL), rows
    assert set(top[:-1]) == {H}, rows       # ... otherwise all horizontal
    lane = len(top) - 1                      # the child lane column
    # (2) A straight vertical down that ONE lane between the two corners.
    for mid in rows[1:-1]:
        assert mid == " " * lane + V, rows
    # (3) Bottom corner turns RIGHT into the child (up+right), then only a
    #     short horizontal may follow (reaching the child border midpoint).
    bottom = rows[-1]
    assert bottom[:lane] == " " * lane, rows
    assert bottom[lane] == UR, rows
    assert set(bottom[lane + 1:]) <= {H}, rows


async def test_history_tree_single_fork_draws_a_planar_L():
    """One fork => one planar, STRAIGHT L: a horizontal leaves the parent's
    right-border midpoint and turns down the child's own lane, then an up+right
    corner turns into the child's left-border midpoint (no diagonal).
    """
    from coreybot.runtime.session import SessionTree, Snapshot

    tree = SessionTree()
    a = tree.commit(Snapshot(), label="a")
    tree._heads["main"] = a.id
    tree.commit(Snapshot(), label="fork-b")
    tree._heads["main"] = a.id
    tree.commit(Snapshot(), label="main-c")

    _assert_L_connector(await _restore_edge_ascii(tree))


async def test_history_tree_two_forks_draw_staggered_planar_lanes():
    """Two forks alive together draw as two separate planar, STRAIGHT Ls in
    DIFFERENT lanes (a staircase), never a crossing.
    """
    from coreybot.runtime.session import SessionTree, Snapshot

    V = "\u2502"
    H = "\u2500"
    DL = "\u2510"
    DIAG = "\u2572"
    UR = "\u2514"

    tree = SessionTree()
    root = tree.root_id
    m1 = tree.commit(Snapshot(), label="m1")
    tree.commit(Snapshot(), label="m1 tail a")
    tree.commit(Snapshot(), label="m1 tail b")
    tree._heads["main"] = root
    tree.commit(Snapshot(), label="root fork R")
    tree._heads["main"] = m1.id
    tree.commit(Snapshot(), label="m1 fork S")

    rows = await _restore_edge_ascii(tree)
    # Two down+LEFT corners AND two up+right corners (one per fork) at TWO
    # different lane columns -- i.e. two separate straight Ls in separate
    # lanes, not one merged spine. No diagonal glyph is ever drawn.
    assert DIAG not in "".join(rows)
    corner_cols = sorted(
        {x for row in rows for x, ch in enumerate(row) if ch == DL}
    )
    elbow_cols = sorted(
        {x for row in rows for x, ch in enumerate(row) if ch == UR}
    )
    assert len(corner_cols) == 2
    assert len(elbow_cols) == 2
    # Each fork's vertical runs down its OWN corner column (planar lanes).
    lane_cols = {x for row in rows for x, ch in enumerate(row) if ch == V}
    assert set(corner_cols) <= lane_cols
    # Zero crossings (a horizontal never passes through a vertical).
    crossings = 0
    for y in range(1, len(rows) - 1):
        for x, ch in enumerate(rows[y]):
            if ch != H:
                continue
            up = rows[y - 1][x] if x < len(rows[y - 1]) else " "
            dn = rows[y + 1][x] if x < len(rows[y + 1]) else " "
            if up == V and dn == V:
                crossings += 1
    assert crossings == 0


async def test_history_tree_clear_fork_wires_the_new_root_branch():
    """``clear`` forks a new root off the original; the map wires that new
    branch to its own lane as a single straight, planar L (same as any fork).
    """
    from coreybot.runtime.session import SessionTree, Snapshot

    tree = SessionTree()
    tree.commit(Snapshot(), label="q1")
    tree.commit(Snapshot(), label="q2")
    tree.new_root(Snapshot(), label="cleared")
    tree.commit(Snapshot(), label="q3")

    _assert_L_connector(await _restore_edge_ascii(tree))


async def test_history_tree_multi_child_parent_draws_one_comb_bar():
    """A parent with MANY children fans out from ONE shared horizontal bar (a
    git-style comb / T), not N separate stacked horizontals.

    ``root`` gets three children: a linear same-column child plus two forks.
    The two forks share a single bar on the parent's mid row that tees DOWN
    (``\u252c``) at the inner lane and ends with a down+LEFT corner
    (``\u2510``) at the farthest lane; each fork then drops its OWN lane and
    elbows (``\u2514``) into the child. The picture stays planar.
    """
    from coreybot.runtime.session import SessionTree, Snapshot

    V = "\u2502"
    H = "\u2500"
    DL = "\u2510"
    UR = "\u2514"
    TD = "\u252c"

    tree = SessionTree()
    root = tree.root_id
    tree.commit(Snapshot(), label="childA")          # linear (same column)
    tree._heads["main"] = root
    tree.commit(Snapshot(), label="childB")          # fork 1
    tree._heads["main"] = root
    tree.commit(Snapshot(), label="childC")          # fork 2

    rows = await _restore_edge_ascii(tree)
    blob = "".join(rows)
    # ONE shared bar: exactly one tee-down (the inner fork) and one down+LEFT
    # corner (the farthest fork), both on the SAME row -- not two separate
    # horizontals on two rows.
    assert blob.count(TD) == 1, rows
    assert blob.count(DL) == 1, rows
    tee_row = next(i for i, r in enumerate(rows) if TD in r)
    corner_row = next(i for i, r in enumerate(rows) if DL in r)
    assert tee_row == corner_row, rows
    bar = rows[tee_row]
    # The bar is a contiguous horizontal run: only line glyphs, tee at an
    # inner column, corner at the end.
    assert bar.endswith(DL), rows
    assert set(bar) <= {H, TD, DL}, rows
    assert bar.index(TD) < bar.index(DL), rows
    # Two forks => two elbows into two children, each in its own lane.
    assert blob.count(UR) == 2, rows
    lane_cols = {x for r in rows for x, ch in enumerate(r) if ch == V}
    assert len(lane_cols) == 2, rows
    # Zero crossings: no horizontal passes THROUGH a vertical rail.
    crossings = 0
    for y in range(1, len(rows) - 1):
        for x, ch in enumerate(rows[y]):
            if ch != H:
                continue
            up = rows[y - 1][x] if x < len(rows[y - 1]) else " "
            dn = rows[y + 1][x] if x < len(rows[y + 1]) else " "
            if up == V and dn == V:
                crossings += 1
    assert crossings == 0, rows


async def test_history_tree_comb_bar_is_centered_on_the_parent():
    """The comb's OUT endpoint (the shared bar) sits on the parent box's
    vertical MIDPOINT row, so the fan leaves the parent centered.
    """
    from coreybot.frontends.tui.app import (
        RestoreModal,
        _SessionCanvas,
        _SNODE_H,
    )
    from coreybot.runtime.session import SessionTree, Snapshot

    DL = "\u2510"
    TD = "\u252c"

    tree = SessionTree()
    root = tree.root_id
    tree.commit(Snapshot(), label="childA")
    tree._heads["main"] = root
    tree.commit(Snapshot(), label="childB")
    tree._heads["main"] = root
    tree.commit(Snapshot(), label="childC")

    app = ChatApp(Config())
    app.agent.sessions = tree
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        app.push_screen(RestoreModal(tree, tree.head()))
        for _ in range(24):
            await pilot.pause()
        canvas = app.screen.query_one("#restore-canvas", _SessionCanvas)
        edge = canvas._edge_layer
        rows = [
            "".join(seg.text for seg in edge.render_line(y))
            for y in range(edge.size.height)
        ]
        bar_row = next(y for y, r in enumerate(rows) if (TD in r or DL in r))
        # The bar row equals the root box's mid row (top + _SNODE_H // 2).
        top = canvas._pos[root][1]
        assert bar_row == top + _SNODE_H // 2, (bar_row, top)


async def test_restore_canvas_click_only_selects_and_scoped_restore_rewinds(
    fake_stream_provider,
):
    """A click NEVER restores -- it only selects. Restore is an explicit,
    scoped action (the ``s`` / ``w`` keys or their buttons).

    Arrow keys walk the selection cursor; clicking a box (even twice) only
    moves the cursor and keeps the modal open, so a stray double-click can
    never rewind the session or the workspace. Pressing ``s`` restores the
    SESSION scope to the selected node.
    """
    from coreybot.frontends.tui.app import (
        RestoreModal,
        _SessionCanvas,
        _SessionNodeBox,
    )
    from textual.widgets import Input

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=["<message>answer one</message>", "<message>answer two</message>"]
    )
    async with app.run_test(size=(120, 34)) as pilot:
        await _run_two_turns(app, pilot)
        rows = app.agent.sessions.rows()
        head = app.agent.sessions.head()
        root_id = app.agent.sessions.root_id

        app.action_restore()
        for _ in range(15):
            await pilot.pause()
        modal = app.screen
        assert isinstance(modal, RestoreModal)
        canvas = modal.query_one("#restore-canvas", _SessionCanvas)
        assert len(list(modal.query(_SessionNodeBox))) == len(rows)
        assert canvas._selected_id == head

        # Arrow-up walks the cursor toward the root (rows stack top-to-bottom).
        await pilot.press("up")
        await pilot.press("up")
        for _ in range(10):
            await pilot.pause()
        assert canvas._selected_id == root_id

        # Click the "hello one" box: it SELECTS but never restores, even on a
        # second click -- the modal stays open and nothing is rewound.
        target_id = next(r.node.id for r in rows if r.node.label == "hello one")
        box = canvas._boxes[target_id]
        for _ in range(6):
            await pilot.pause()
        r = box.region
        await pilot.click(offset=(r.x + 2, r.y + 1))
        for _ in range(6):
            await pilot.pause()
        assert canvas._selected_id == target_id
        assert len(app.screen_stack) == 2
        await pilot.click(offset=(r.x + 2, r.y + 1))   # second click: still no restore
        for _ in range(8):
            await pilot.pause()
        assert len(app.screen_stack) == 2               # modal still open
        assert [b.text for b in app.query(MessageBubble)] == [
            "hello one", "answer one", "hello two", "answer two"
        ]

        # Now restore EXPLICITLY (session scope) via the ``s`` key.
        await pilot.press("s")
        for _ in range(80):
            await pilot.pause()

        # Restored: back on the main screen, context rewound to turn 1, focus
        # returned to the prompt.
        assert len(app.screen_stack) == 1
        assert [b.text for b in app.query(MessageBubble)] == [
            "hello one", "answer one"
        ]
        assert app.focused is app.query_one("#prompt", Input)


async def test_restore_canvas_scroll_does_not_shadow_textual_scroll_to(
    fake_stream_provider,
):
    """Regression: the canvas helper must not shadow Textual's ``_scroll_to``.

    ``ScrollableContainer`` (via ``Widget``) already has a private
    ``_scroll_to(*, x, y, animate=...)`` that the scrollbar / mouse-wheel
    machinery calls. Naming our own node helper ``_scroll_to(node_id)`` used to
    override it with an incompatible signature, so *scrolling* the restore map
    raised ``TypeError: _scroll_to() got an unexpected keyword argument
    'animate'``. The helper is now ``_scroll_to_node`` and the inherited method
    must accept ``animate=``.
    """
    from coreybot.frontends.tui.app import RestoreModal, _SessionCanvas

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=["<message>answer one</message>", "<message>answer two</message>"]
    )
    async with app.run_test(size=(120, 20)) as pilot:
        await _run_two_turns(app, pilot)
        app.action_restore()
        for _ in range(15):
            await pilot.pause()
        modal = app.screen
        assert isinstance(modal, RestoreModal)
        canvas = modal.query_one("#restore-canvas", _SessionCanvas)

        # The inherited Textual scroll entry point still works (the crash path).
        canvas._scroll_to(y=1, animate=False)
        for _ in range(3):
            await pilot.pause()
        # And our node-centering helper is under its own, non-clashing name.
        canvas._scroll_to_node(app.agent.sessions.root_id)
        for _ in range(3):
            await pilot.pause()
        assert not hasattr(_SessionCanvas, "_scroll_to") or (
            "node_id" not in _SessionCanvas._scroll_to.__code__.co_varnames
        )


async def test_history_tree_shows_in_flight_turn_as_pending_node(
    fake_stream_provider,
):
    """A turn still in flight appears as a synthetic, non-restorable node.

    A commit only lands in the SessionTree when a turn FINISHES, so the map
    would otherwise omit the message you are waiting on. While ``_busy`` the
    canvas draws an extra dashed "pending" box hanging off the current head; it
    is NOT a real commit, so it is skipped by arrow-nav / clicks and cannot be
    restored to.
    """
    from coreybot.frontends.tui.app import (
        RestoreModal,
        _SessionCanvas,
        _SessionNodeBox,
        _SPENDING_ID,
    )
    from textual.widgets import Input

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=["<message>answer one</message>", "<message>never shown</message>"],
        delay=0,
    )
    async with app.run_test(size=(120, 30)) as pilot:
        # One finished turn -> one committed user node.
        app.query_one("#prompt", Input).value = "hello one"
        await pilot.press("enter")
        await _drain_worker(pilot, app)
        committed = len(app.agent.sessions.rows())

        # Start a second turn and make it hang mid-flight.
        app.agent.provider.delay = 5
        app.query_one("#prompt", Input).value = "still thinking"
        await pilot.press("enter")
        for _ in range(80):
            await pilot.pause()
            if app._busy:
                break
        assert app._busy is True
        assert app._pending_user_text == "still thinking"

        # Open the map WHILE busy: it draws committed nodes + one pending box.
        app.action_restore()
        for _ in range(20):
            await pilot.pause()
        modal = app.screen
        assert isinstance(modal, RestoreModal)
        canvas = modal.query_one("#restore-canvas", _SessionCanvas)
        for _ in range(6):
            await pilot.pause()
        assert canvas._pending is not None
        assert canvas._pending["label"] == "still thinking"
        # One more box than committed rows (the synthetic pending marker).
        assert len(list(modal.query(_SessionNodeBox))) == committed + 1

        # The pending marker is NOT selectable / restorable.
        assert _SPENDING_ID not in canvas._order()
        before = canvas._selected_id
        canvas.select(_SPENDING_ID)          # no-op: not a commit
        assert canvas._selected_id == before

        # Clean up: close the modal and cancel the hung turn.
        await pilot.press("escape")
        for _ in range(10):
            await pilot.pause()
        await app._cancel_active_turn()
        for _ in range(20):
            await pilot.pause()


async def test_history_tree_canvas_fills_the_modal(fake_stream_provider):
    """The canvas (edge layer) fills the modal pane, not just the graph bounds.

    Regression: without an ``on_resize`` reflow the virtual area kept a stale
    size and the map did not cover the pane (and lost drag slack). The edge
    layer must be at least as large as the canvas viewport in both axes.
    """
    from coreybot.frontends.tui.app import RestoreModal, _SessionCanvas

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=["<message>answer one</message>", "<message>answer two</message>"]
    )
    async with app.run_test(size=(120, 30)) as pilot:
        await _run_two_turns(app, pilot)
        app.action_restore()
        for _ in range(20):
            await pilot.pause()
        modal = app.screen
        assert isinstance(modal, RestoreModal)
        canvas = modal.query_one("#restore-canvas", _SessionCanvas)
        for _ in range(8):
            await pilot.pause()
        edge = canvas._edge_layer
        # The drawable area covers the whole viewport (with pan slack), so the
        # map is not squeezed into the small graph bounds.
        assert edge.size.width >= canvas.size.width
        assert edge.size.height >= canvas.size.height


async def test_history_tree_boxes_are_dense_and_content_sized(fake_stream_provider):
    """History-tree boxes stack densely and widen to fit wide utterances.

    Three fixes are pinned here: (1) the canvas hides its scrollbar chrome
    (size 0) so the bar never covers the tree; (2) boxes are absolutely
    positioned and stack flush (parent bottom border abutting child top
    border, a 3-cell stride) so connectors line up instead of drifting; (3)
    a box is content-sized -- a short utterance stays at the minimum width
    while a wide one stretches its bubble, capped at the canvas width - 10.
    """
    from coreybot.frontends.tui.app import (
        RestoreModal,
        _SessionCanvas,
        _SessionNodeBox,
        _SNODE_W,
        _SNODE_H,
        _SROW_GAP,
    )
    from textual.widgets import Input

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=["<message>a1</message>", "<message>a2</message>"]
    )
    wide = "x" * 200
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        for text in ("hi", wide):
            app.query_one("#prompt", Input).value = text
            await pilot.press("enter")
            await _drain_worker(pilot, app)

        app.action_restore()
        for _ in range(20):
            await pilot.pause()
        modal = app.screen
        assert isinstance(modal, RestoreModal)
        canvas = modal.query_one("#restore-canvas", _SessionCanvas)
        for _ in range(8):
            await pilot.pause()

        # (1) Scrollbar chrome takes zero cells -> it cannot cover the tree.
        assert int(canvas.styles.scrollbar_size_horizontal) == 0
        assert int(canvas.styles.scrollbar_size_vertical) == 0

        # (2) Boxes are absolutely positioned and stack flush: same-column
        # neighbours are one box-height apart (borders touch), so the row
        # stride is exactly _SNODE_H and connectors align with the boxes.
        assert _SNODE_H + _SROW_GAP == _SNODE_H
        for box in canvas.query(_SessionNodeBox):
            assert str(box.styles.position) == "absolute"
        ys = sorted({canvas._pos[nid][1] for nid in canvas._pos})
        deltas = {b - a for a, b in zip(ys, ys[1:])}
        assert deltas == {_SNODE_H}

        # (3) Content-sized widths: the short turn stays minimal, the wide
        # one stretches up to (but not past) the canvas width minus 10.
        cap = max(_SNODE_W, canvas.size.width - 10)
        widths = canvas._node_w
        labels = {row.node.id: row.node.label for row in app.agent.sessions.rows()}
        short_id = next(nid for nid, lbl in labels.items() if lbl == "hi")
        wide_id = next(nid for nid, lbl in labels.items() if lbl.startswith("x"))
        assert widths[short_id] == _SNODE_W
        assert widths[wide_id] == cap
        assert widths[wide_id] > widths[short_id]


async def test_clear_keeps_session_history_and_starts_new_root(fake_stream_provider):
    """Clear rewinds the live context but PRESERVES the session tree.

    ``action_clear`` empties the transcript + history (system only) and FORKS a
    fresh line off the original root, WITHOUT discarding earlier commits -- so
    the history-tree map shows the root split into the pre-clear branch and the
    new line, and a new turn grows under the fork (not the abandoned branch).
    """
    from coreybot.frontends.tui.app import RestoreModal, _SessionNodeBox
    from textual.widgets import Input

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=[
            "<message>answer one</message>",
            "<message>answer two</message>",
            "<message>fresh answer</message>",
        ]
    )
    async with app.run_test(size=(120, 30)) as pilot:
        await _run_two_turns(app, pilot)
        tree = app.agent.sessions
        before_nodes = len(tree)
        old_head = tree.head()

        # Clear: transcript + history go back to system-only.
        app.action_clear()
        for _ in range(15):
            await pilot.pause()
        assert [b.text for b in app.query(MessageBubble)] == []
        assert [m.role for m in app.agent.history] == [Role.SYSTEM]

        # The tree GREW (old branch survives) and the head is a fresh child
        # forked off the original root (so the root splits into two lines).
        assert len(tree) > before_nodes
        new_head = tree.head()
        assert not tree.get(new_head).is_root
        assert tree.get(new_head).parent == tree.root_id
        assert new_head != old_head
        assert tree.get(old_head) is not None
        labels = [r.node.label for r in tree.rows()]
        assert "hello one" in labels and "hello two" in labels

        # A turn after clear grows under the NEW root, not the old line.
        app.query_one("#prompt", Input).value = "fresh start"
        await pilot.press("enter")
        await _drain_worker(pilot, app)
        head_path = [n.label for n in tree.path(tree.head())]
        # The live line is rooted at the original root, runs through the
        # clear fork, and does NOT touch the abandoned pre-clear branch.
        assert head_path[0] == tree.get(tree.root_id).label
        assert tree.get(new_head).label in head_path
        assert "hello one" not in head_path and "hello two" not in head_path

        # The history-tree map shows the OLD branch AND the new line together,
        # but it HIDES the ``clear`` marker node (a clear restores no context,
        # so it is not a useful restore target). So the dashboard draws one
        # box per NON-clear commit: every real row except the cleared fork.
        from coreybot.runtime.session import CLEAR_LABEL

        app.action_restore()
        for _ in range(15):
            await pilot.pause()
        modal = app.screen
        assert isinstance(modal, RestoreModal)
        visible_rows = [r for r in tree.rows() if r.node.label != CLEAR_LABEL]
        boxes = list(modal.query(_SessionNodeBox))
        assert len(boxes) == len(visible_rows)
        # No box carries the clear marker's label...
        drawn_labels = [lay.node.label for lay in modal._layout]
        assert CLEAR_LABEL not in drawn_labels
        # ...yet both the abandoned pre-clear line and the fresh post-clear
        # line are still present (the post-clear turn re-attached to root).
        assert "hello one" in drawn_labels
        drawn_ids = [lay.node.id for lay in modal._layout]
        assert tree.head() in drawn_ids  # the fresh post-clear turn is shown
        await pilot.press("escape")
        for _ in range(8):
            await pilot.pause()

        # Restoring to the pre-clear head brings the old conversation back.
        app.post_message(RestoreModal.RestoreRequested(old_head))
        for _ in range(60):
            await pilot.pause()
        assert [b.text for b in app.query(MessageBubble)] == [
            "hello one", "answer one", "hello two", "answer two"
        ]


async def test_restore_transcript_hides_tool_call_xml(fake_stream_provider):
    """Rebuilding the transcript after restore must NOT show tool_call XML.

    Regression: an assistant history entry is the model's RAW turn -- either a
    ``<tool_call>`` (intermediate step) or a ``<message>`` (final answer). The
    live transcript only shows the final answer (tool steps go to the flow
    graph). ``_rebuild_transcript`` used to render EVERY assistant entry and,
    for a tool-call turn, fell back to the raw string -- leaking ``<tool_call>``
    tags into the chat. It must skip tool-call turns instead.
    """
    from coreybot.frontends.tui.app import RestoreModal
    from textual.widgets import Input

    app = ChatApp(Config())
    # Turn drives a tool call (calc is a builtin) then a final message.
    app.agent.provider = fake_stream_provider(
        replies=[
            "<tool_call><name>calc</name>"
            "<arguments><expression>2+3</expression></arguments></tool_call>",
            "<message>the answer is 5</message>",
        ]
    )
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = "what is 2+3"
        await pilot.press("enter")
        await _drain_worker(pilot, app)

        # Baseline: the live transcript is clean (human turn + final answer).
        assert [b.text for b in app.query(MessageBubble)] == [
            "what is 2+3", "the answer is 5"
        ]
        # The history really does contain an intermediate tool_call assistant
        # entry (otherwise this test would not exercise the bug).
        assert any(
            m.role == Role.ASSISTANT and "<tool_call" in m.content
            for m in app.agent.history
        )

        # Restore to the current head -> transcript is rebuilt from history.
        app.action_restore()
        for _ in range(10):
            await pilot.pause()
        app.screen.post_message(
            RestoreModal.RestoreRequested(app.agent.sessions.head())
        )
        for _ in range(60):
            await pilot.pause()

        bubbles = [b.text for b in app.query(MessageBubble)]
        # Same clean transcript -- no tool_call/message XML leaked in.
        assert bubbles == ["what is 2+3", "the answer is 5"]
        assert not any("<tool_call" in b or "<message" in b for b in bubbles)
# --- session manager: two tabs (session tree + global browser) ---------
def _seed_global_home(tmp_path):
    """Create a home dir with three saved rollouts; return (paths, files)."""
    from datetime import datetime

    from coreybot.core.message import Message
    from coreybot.core.paths import AgentPaths
    from coreybot.runtime.session import SessionTree, Snapshot
    from coreybot.runtime.session_store import save_tree

    paths = AgentPaths.resolve(tmp_path / "home")
    paths.ensure()

    def mk(session_id, when, users):
        tree = SessionTree(Snapshot(history=[Message.system("sys")]))
        hist = [Message.system("sys")]
        for text in users:
            hist = hist + [Message.user(text), Message.assistant("ok:" + text)]
            tree.commit(Snapshot(history=list(hist)), label=text)
        target = paths.session_file(session_id, when)
        save_tree(tree, target)
        return target

    files = {
        "cur": mk("cur00001", datetime(2024, 5, 5, 12, 0, 0), ["current run hello"]),
        "old": mk("old00002", datetime(2024, 1, 1, 8, 0, 0), ["banana pancakes"]),
        "mid": mk("mid00003", datetime(2024, 3, 3, 9, 0, 0), ["apple pie please"]),
    }
    return paths, files


async def _open_sessions_modal(app, pilot):
    app.action_restore()
    for _ in range(15):
        await pilot.pause()
    return app.screen


async def test_sessions_modal_has_two_tabs_tree_default(local_tmp_path):
    """The sessions modal hosts two TabPanes; the session tree is the default."""
    from textual.widgets import TabbedContent, TabPane

    from coreybot.frontends.tui.app import RestoreModal, _SessionCanvas

    paths, files = _seed_global_home(local_tmp_path)
    app = ChatApp(Config(), sessions_dir=str(paths.sessions_dir),
                  current_file=str(files["cur"]))
    async with app.run_test(size=(120, 34)) as pilot:
        modal = await _open_sessions_modal(app, pilot)
        assert isinstance(modal, RestoreModal)
        pane_ids = {p.id for p in modal.query(TabPane)}
        assert pane_ids == {"tab-tree", "tab-global"}
        tabs = modal.query_one("#restore-tabs", TabbedContent)
        assert tabs.active == "tab-tree"
        # The existing session-tree canvas is intact on tab 1.
        assert modal.query_one("#restore-canvas", _SessionCanvas)


async def test_sessions_modal_tabs_are_buttons_on_title_row(local_tmp_path):
    """Tabs are BUTTONS sharing the title row; the built-in tab bar is hidden.

    Compact contract: the ``#restore-title`` static and the two ``.tabbtn``
    buttons all sit on ONE height-1 row, and TabbedContent's own tab bar
    (``ContentTabs``) is collapsed to height 0 so it wastes no space.
    """
    from textual.widgets import Button, Static, TabbedContent
    from textual.widgets._tabbed_content import ContentTabs

    paths, files = _seed_global_home(local_tmp_path)
    app = ChatApp(Config(), sessions_dir=str(paths.sessions_dir),
                  current_file=str(files["cur"]))
    async with app.run_test(size=(120, 34)) as pilot:
        modal = await _open_sessions_modal(app, pilot)
        titlebar = modal.query_one("#restore-titlebar")
        title = modal.query_one("#restore-title", Static)
        btree = modal.query_one("#tabbtn-tree", Button)
        bglobal = modal.query_one("#tabbtn-global", Button)
        # The title row is a single compact line...
        assert titlebar.region.height == 1
        # ...and title + both tab buttons live on that same row.
        assert title.region.y == btree.region.y == bglobal.region.y
        # The tab buttons are Button widgets styled as tabs (class "tabbtn").
        assert "tabbtn" in btree.classes and "tabbtn" in bglobal.classes
        # The built-in TabbedContent tab bar is hidden (no extra tab row).
        assert modal.query_one(ContentTabs).region.height == 0
        # The active tab reads like a pressed (primary) button; the other is
        # default. Tree is the default active tab.
        assert btree.variant == "primary"
        assert bglobal.variant == "default"
        # The button LABELS must actually render (not clipped to 0 rows):
        # each tab button gets a real 1-row content box carrying its text.
        assert btree.size.height >= 1 and bglobal.size.height >= 1
        assert str(btree.render()) == "Session tree"
        assert str(bglobal.render()) == "All sessions"


async def test_sessions_modal_tab_buttons_switch_panes(local_tmp_path):
    """Clicking a tab button switches the pane and flips the active styling."""
    from textual.widgets import Button, ListView, TabbedContent

    paths, files = _seed_global_home(local_tmp_path)
    app = ChatApp(Config(), sessions_dir=str(paths.sessions_dir),
                  current_file=str(files["cur"]))
    async with app.run_test(size=(120, 34)) as pilot:
        modal = await _open_sessions_modal(app, pilot)
        btree = modal.query_one("#tabbtn-tree", Button)
        bglobal = modal.query_one("#tabbtn-global", Button)
        tabs = modal.query_one("#restore-tabs", TabbedContent)

        # Click "All sessions": pane switches, buttons flip variant.
        r = bglobal.region
        await pilot.click(offset=(r.x + 2, r.y))
        for _ in range(10):
            await pilot.pause()
        assert tabs.active == "tab-global"
        assert bglobal.variant == "primary"
        assert btree.variant == "default"
        assert len(modal.query_one("#global-list", ListView)) == 3

        # Click "Session tree": back to the tree pane.
        r = btree.region
        await pilot.click(offset=(r.x + 2, r.y))
        for _ in range(10):
            await pilot.pause()
        assert tabs.active == "tab-tree"
        assert btree.variant == "primary"
        assert bglobal.variant == "default"


async def test_global_tab_lists_all_sessions_newest_first(local_tmp_path):
    """Tab 2 lists every saved rollout, newest first, current one flagged."""
    from textual.widgets import ListView, TabbedContent

    paths, files = _seed_global_home(local_tmp_path)
    app = ChatApp(Config(), sessions_dir=str(paths.sessions_dir),
                  current_file=str(files["cur"]))
    async with app.run_test(size=(120, 34)) as pilot:
        modal = await _open_sessions_modal(app, pilot)
        modal.query_one("#restore-tabs", TabbedContent).active = "tab-global"
        for _ in range(10):
            await pilot.pause()
        listview = modal.query_one("#global-list", ListView)
        assert len(listview) == 3
        titles = [i.title for i in modal._filtered]
        assert titles == ["current run hello", "apple pie please", "banana pancakes"]
        assert modal._filtered[0].is_current is True


async def test_global_tab_search_filters_list(local_tmp_path):
    """The search box narrows the list by title/id/date substring."""
    from textual.widgets import TabbedContent

    paths, files = _seed_global_home(local_tmp_path)
    app = ChatApp(Config(), sessions_dir=str(paths.sessions_dir),
                  current_file=str(files["cur"]))
    async with app.run_test(size=(120, 34)) as pilot:
        modal = await _open_sessions_modal(app, pilot)
        modal.query_one("#restore-tabs", TabbedContent).active = "tab-global"
        for _ in range(8):
            await pilot.pause()
        modal.query_one("#global-search").value = "banana"
        for _ in range(8):
            await pilot.pause()
        assert [i.title for i in modal._filtered] == ["banana pancakes"]
        # Clearing the search restores the full list.
        modal.query_one("#global-search").value = ""
        for _ in range(8):
            await pilot.pause()
        assert len(modal._filtered) == 3


async def test_global_tab_preview_is_flat_by_time(local_tmp_path):
    """Highlighting a session previews its transcript flat, in time order."""
    from textual.widgets import ListView, TabbedContent, TextArea

    paths, files = _seed_global_home(local_tmp_path)
    app = ChatApp(Config(), sessions_dir=str(paths.sessions_dir),
                  current_file=str(files["cur"]))
    async with app.run_test(size=(120, 34)) as pilot:
        modal = await _open_sessions_modal(app, pilot)
        modal.query_one("#restore-tabs", TabbedContent).active = "tab-global"
        for _ in range(10):
            await pilot.pause()
        # Highlight the oldest (banana) session and read its preview.
        listview = modal.query_one("#global-list", ListView)
        idx = next(i for i, inf in enumerate(modal._filtered)
                   if inf.session_id == "old00002")
        listview.index = idx
        for _ in range(8):
            await pilot.pause()
        text = modal.query_one("#global-preview", TextArea).text
        assert "banana pancakes" in text
        assert "you: banana pancakes" in text
        assert "bot: ok:banana pancakes" in text


async def test_global_tab_open_loads_session_into_app(local_tmp_path):
    """Opening a session from tab 2 swaps the app onto that rollout."""
    from textual.widgets import ListView, TabbedContent

    paths, files = _seed_global_home(local_tmp_path)
    app = ChatApp(Config(), sessions_dir=str(paths.sessions_dir),
                  current_file=str(files["cur"]))
    async with app.run_test(size=(120, 34)) as pilot:
        modal = await _open_sessions_modal(app, pilot)
        modal.query_one("#restore-tabs", TabbedContent).active = "tab-global"
        for _ in range(10):
            await pilot.pause()
        listview = modal.query_one("#global-list", ListView)
        idx = next(i for i, inf in enumerate(modal._filtered)
                   if inf.session_id == "old00002")
        listview.index = idx
        await pilot.pause()
        modal._open_selected_global()
        for _ in range(40):
            await pilot.pause()
        # Modal closed and the agent now holds the opened session's history.
        assert len(app.screen_stack) == 1
        users = [m.content for m in app.agent.history if m.role is Role.USER]
        assert users == ["banana pancakes"]
        # Future turns persist to the opened rollout.
        assert app._current_session_file == str(files["old"])


async def test_scope_keys_do_not_restore_on_global_tab(local_tmp_path):
    """On the global tab, s/w are plain typing -- they must NOT restore/close."""
    from textual.widgets import TabbedContent

    paths, files = _seed_global_home(local_tmp_path)
    app = ChatApp(Config(), sessions_dir=str(paths.sessions_dir),
                  current_file=str(files["cur"]))
    async with app.run_test(size=(120, 34)) as pilot:
        modal = await _open_sessions_modal(app, pilot)
        modal.query_one("#restore-tabs", TabbedContent).active = "tab-global"
        for _ in range(8):
            await pilot.pause()
        stack_before = len(app.screen_stack)
        modal.action_restore_session()
        modal.action_restore_workspace()
        for _ in range(6):
            await pilot.pause()
        # Still open (no restore fired) -- scope actions are inert off the tree.
        assert len(app.screen_stack) == stack_before


async def test_sessions_modal_no_dir_shows_empty_browser(fake_stream_provider):
    """With no on-disk home wired, the global tab is empty but does not crash."""
    from textual.widgets import ListView, TabbedContent

    app = ChatApp(Config())  # no sessions_dir
    async with app.run_test(size=(120, 34)) as pilot:
        modal = await _open_sessions_modal(app, pilot)
        modal.query_one("#restore-tabs", TabbedContent).active = "tab-global"
        for _ in range(8):
            await pilot.pause()
        assert len(modal.query_one("#global-list", ListView)) == 0
        assert modal._filtered == []
async def test_dashboard_hides_clear_nodes(fake_stream_provider):
    """The session dashboard omits ``clear`` markers but keeps the post-clear line.

    A clear restores no context, so its marker node is not a useful restore
    target and is hidden from the tree map. The abandoned pre-clear branch and
    the fresh post-clear line both remain (the post-clear turn re-attaches to
    the original root), and the cursor lands on a VISIBLE node.
    """
    from coreybot.frontends.tui.app import RestoreModal, _SessionCanvas, _SessionNodeBox
    from coreybot.runtime.session import CLEAR_LABEL
    from textual.widgets import Input

    app = ChatApp(Config())
    app.agent.provider = fake_stream_provider(
        replies=["<message>a1</message>", "<message>a2</message>"]
    )
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        # One turn, then clear, then another turn on the fresh line.
        app.query_one("#prompt", Input).value = "before clear"
        await pilot.press("enter")
        await _drain_worker(pilot, app)
        app.action_clear()
        for _ in range(10):
            await pilot.pause()
        app.query_one("#prompt", Input).value = "after clear"
        await pilot.press("enter")
        await _drain_worker(pilot, app)

        tree = app.agent.sessions
        assert any(r.node.label == CLEAR_LABEL for r in tree.rows())  # it IS in the tree

        app.action_restore()
        for _ in range(15):
            await pilot.pause()
        modal = app.screen
        assert isinstance(modal, RestoreModal)
        # No clear marker is drawn; box count == non-clear rows.
        drawn = [lay.node.label for lay in modal._layout]
        assert CLEAR_LABEL not in drawn
        non_clear = [r for r in tree.rows() if r.node.label != CLEAR_LABEL]
        assert len(list(modal.query(_SessionNodeBox))) == len(non_clear)
        # Both lines survive on the map.
        assert "before clear" in drawn
        assert tree.head() in [lay.node.id for lay in modal._layout]
        # The cursor is on a visible node (never the hidden clear node).
        canvas = modal.query_one("#restore-canvas", _SessionCanvas)
        assert canvas._selected_id in canvas._boxes
