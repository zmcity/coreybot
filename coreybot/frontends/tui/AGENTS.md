# AGENTS.md -- TUI UI contract (FROZEN)

The Textual TUI (`app.py` + `flow.py`) is a **stabilized UI**. It was iterated on
heavily and is now the intended look & feel. Treat the invariants below as a
contract: keep them true, and keep the listed guard tests green. If a change must
break one, do it deliberately -- update this file **and** the affected tests in the
same change, and say so explicitly to the user.

Run the guards:
```
.\.venv\Scripts\python.exe -m pytest tests\test_tui.py tests\test_flow.py -q
```

## Widget tree & ids (do not rename/remove)
`ChatApp.compose` (see `coreybot/frontends/tui/app.py`):
- `Horizontal #body` → `VerticalScroll #chat` (width `2fr`) + `FlowPanel` (the
  flow chart, `Container id="flow"`, width `1fr`, `min-width: 28`).
- `Static #inputbar` -- a **single** slim separator row above the input.
- `Input #prompt` -- **single-line** input (Claude-Code style).
- `Horizontal #statusbar` -- the **single** bottom line: `Static #statusinfo`
  (`width: 1fr`, the animated info) + three `_KeyHint` widgets on the right.

## Hard invariants (each has a guard test in tests/test_tui.py)
- **No Header and no Footer widgets.** The top of the terminal is intentionally
  free so overflowing chat scrolls up. Guard:
  `test_bottom_bar_is_single_line_with_no_header_or_footer` (asserts `query(Header)`
  and `query(Footer)` are empty; `#statusbar` height==1 at the very bottom row).
- **Single bottom status row** carries title/model + state on the left and the
  clickable key hints on the right (no separate Footer). Same guard as above +
  `test_status_bar_key_hints_are_clickable`.
- **Key hints are clickable `_KeyHint` widgets**, actions exactly
  `{"interrupt", "restore", "clear", "quit"}`, all on the status row.
  `_KeyHint.on_click` must `await self.app.run_action(...)` (run_action is a
  coroutine). The `restore` hint (action name `restore`, but LABELLED
  `sessions`) opens the session-tree map (see the Restore section).
  Guards: `test_status_bar_key_hints_are_clickable`,
  `test_clicking_interrupt_hint_cancels_turn`,
  `test_restore_hint_opens_session_tree_map_with_current_head_marked`.
- **Input is one real text row, above the status bar** (`#prompt` and `#inputbar`
  each height 1; `#prompt` has `border: none` -- a border would eat the row and make
  text spill down into the status bar). Guard:
  `test_input_is_single_line_and_above_status_bar`.
- **Bindings**: `ctrl+c` quit, `ctrl+l` clear, `ctrl+r` restore, `escape`
  interrupt. `Esc` cancels the in-flight turn via the per-turn `CancelToken`;
  `Ctrl+R` opens the session-tree restore map. Guards:
  `test_escape_interrupts_a_running_turn`, `test_clicking_interrupt_hint_cancels_turn`.
- **Left chat = human conversation only.** User turns + final assistant answers.
  All agent activity (llm/tool/notice/system) goes to the flow chart via telemetry,
  NOT the chat. Assistant bubbles render **Markdown**. Guards:
  `test_banner_on_mount_goes_to_flow_not_chat`, `test_tool_call_shows_activity_then_answer`,
  `test_assistant_bubble_renders_markdown`, `test_wrapped_line_stays_in_content_column`.

## Focus policy (focus is pinned to the input)
Keyboard focus stays on `#prompt` **at all times** -- it is the only place text
is ever typed. `ChatApp.on_descendant_focus` bounces focus back to `#prompt`
whenever anything else takes it (clicking the flow pane / a node, or Tab), via
`_focus_prompt()` scheduled with `call_after_refresh`. Mouse-only interactions
(click a node to expand/collapse OR open its popup, drag the canvas to pan)
still work because
they do not need focus. Guards (tests/test_tui.py):
`test_focus_stays_on_prompt_when_something_else_grabs_it`,
`test_clicking_flow_chart_keeps_focus_on_prompt`. Do not add other focusable
input widgets or remove this redirect. **Exception:** while a modal screen is
open (`len(self.screen_stack) > 1`), the redirect stands down so the modal can
own focus -- see the inspector modal below.

## Status bar 'breathing LED' (retro standby light)
The 'alive' indicator is a **single breathing LED dot** `●` (NOT a spinner and
NOT signal bars -- both were tried and rejected). Implementation in `app.py`:
- Constants: `_LED_GLYPH`, `_LED_OFF_RGB`, `_LED_STATES` (`ready` = calm green, slow
  `period`; `working` = amber, quicker `period`).
- `_status_state()` returns the state key; add new states (e.g. an error red
  fast-blink) by adding a `_LED_STATES` entry + returning its key here.
- `_led_indicator()` eases brightness on a **triangle wave** of `_beat`, blending
  `_LED_OFF_RGB` → state rgb into a `#rrggbb` truecolor `Text` (a real fade, not
  hard on/off). `_tick_status` (0.12s interval) just advances `_beat` -- that motion
  is what proves the event loop is alive; keep it running idle and busy.
- `_render_status()` paints `#statusinfo`: LED + the current working directory
  (`os.getcwd()`, not a fixed product name) + `provider . model`
  ONLY -- there is NO `working`/`ready` word on the bar; the run state is shown
  purely by the LED's colour + speed. It stores `self._status_text` (tests assert
  on `.plain`, which must keep the model string and must NOT contain the words
  `working`/`ready`).
- Guards: `test_heartbeat_animates_and_reflects_busy` (`_beat` advances; state read
  via `_status_state()`, and the bar text has NO working/ready words),
  `test_breathing_led_pulses_and_reflects_state` (single dot; real brightness fade
  incl. near-off and near-full; idle green vs busy amber; busy period < idle
  period).

## Flow chart (right pane) -- `flow.py`
- Structure: `FlowPanel(Container id="flow")` → `FlowCanvas(ScrollableContainer)`
  → `_EdgeLayer(Static)` (draws connectors) + `FlowNode*` (`position: absolute`).
  It is a **projection of `Agent.telemetry`** (append-only, the source of truth), so
  prior turns persist across the session.
- **Connectors are CENTERED.** `_EdgeLayer.render_line` draws each edge from the
  parent's bottom-CENTER straight down to the child's top-CENTER -- a clean
  vertical exiting/entering each box at its horizontal middle (today's layout is
  one column, so parent+child centers share a column). The center column is
  `x + (width - 1) // 2` (the ROUNDED true center): for the fixed width-30 box
  (borders at cols 12..41, center 26.5) that is col 26. Do NOT use `width // 2`
  (col 27, a half-cell right -> reads as an off-center / corner-ish exit) and do
  NOT move it back to the left border (`x + 1`): both were reported as bugs. Only
  when a future DAG puts the child in a DIFFERENT column does it elbow (`\u2514`
  + horizontal) across at the child's top row. Guards:
  `test_edge_connector_runs_down_the_box_center_not_the_left_edge`,
  `test_edge_layer_renders_all_rows`.
- Public API (keep stable; tests use it): `set_history(events)` (full rebuild,
  follow), `append(event)` (incremental, follow), `clear()`, `toggle(key)`,
  `set_expanded(key, value)`, `expand_all()`, `collapse_all()`, `nodes()`, `edges()`,
  `resume_follow()`.
- **Grab-to-pan, no scrollbars.** CSS `overflow: auto auto; scrollbar-size: 0 0`.
  Pan by dragging the background only; `_CLICK_SLOP` distinguishes click vs drag.
- **Auto-follow**: `follow_tail` scrolls to the newest node while a turn runs; a
  manual drag pauses it; a new turn resumes it (`resume_follow`).
- **No boundary drift** (regression already fixed): panning clamps the target with
  `_clamp(...)` into `[0, max_scroll]` and applies `scroll_to(..., animate=False,
  immediate=True)`. `scroll_to`'s default is deferred + eased and WILL re-introduce
  glide-at-the-edge -- do not remove the clamp or `immediate=True`.
- **Free 2D pan even for tiny graphs**: virtual size is
  `max(content + _CANVAS_PAD, viewport + _CANVAS_PAD)` on both axes, guaranteeing
  `max_scroll > 0`. Layout `_relayout()` is a pure function of the model
  (`NODE_WIDTH`, `_H_GAP`, `_CANVAS_PAD` reserve DAG breadth for future multi-agent /
  parallel tool calls). Nodes are NOT individually draggable.
- Node model `GraphNode`: `full_detail`/`detail`/`expanded`/dynamic `height`;
  `notice` auto-expands; other non-inspectable kinds start collapsed with an
  inline caret. `expandable = collapsible and not inspectable` gates the inline
  expand affordance -- **inspectable nodes are popup-only (no caret, no inline
  expand)**. Internal dict is `_model` (NOT `_nodes`, which clashes with Textual).
- **Border colour encodes node TYPE.** `FlowNode.apply` adds a `src-<label>`
  class from the node's `source`, giving each step kind its own hue: llm =
  `$primary`, tool = `$success`, mcp = `$secondary`, skill = `$warning`, agent =
  `$accent` (user/answer/notice keep their kind hue). A `.fail` node overrides to
  red `$error` regardless of type. The border STYLE is always `round`.
- **Running-node border pulse + live timer.** A RUNNING node's *border* gently
  pulses toward an APPROXIMATE ACCENT of its OWN type colour -- a lighter shade of
  the same hue (`.src-*.blink` -> `$*-lighten-3`), NEVER a different colour like
  amber. The `.blink` class, toggled ~1.1Hz via `FlowPanel.pulse_on` in
  `FlowNode.apply`, is the brighter half. The border STYLE stays `round` in both
  phases so the edge never thickens/jumps -- it is a soft breath, not a strobe. The
  header shows a static running glyph plus an elapsed timer (no blinking dot).
  Timing lives on `GraphNode` (`started_at` stamped on `llm_call`/`tool_call`;
  `finished_at` on `llm_result`/`tool_result`; `duration(now)` is live while
  running, frozen after). `_format_duration`: **<2s -> ms, >=2s -> s**
  (>=10s drops the decimal). A single `set_interval(0.1, _tick)` runs ONLY while
  something is running (`_ensure_ticking`/`_stop_ticking`) and repaints only the
  running widgets via `FlowCanvas.repaint_running` (no reflow) -- 100ms is the
  fastest the ms timer updates, per spec, to stay cheap. `_freeze_running()`
  stamps `finished_at` on abandoned running nodes at `turn_end`/`notice` and after
  a `set_history` rebuild, so a timer never counts up forever. `_clock` is
  injectable for deterministic tests.
- Guards (tests/test_flow.py): `test_over_dragging_past_edge_clamps_without_drift`,
  `test_panning_background_moves_viewport_and_stops_follow`,
  `test_small_graph_can_pan_freely_in_both_axes`, `test_autofollow_scrolls_to_newest_node`,
  `test_new_turn_resumes_follow`, `test_nodes_are_not_individually_draggable`,
  `test_click_on_node_toggles_expansion`, `test_click_on_background_does_not_toggle`,
  `test_relayout_is_pure_and_non_overlapping`, `test_incremental_append_matches_full_set_history`,
  `test_set_history_is_idempotent`, `test_multiple_turns_preserve_context`,
  `test_format_duration_ms_under_2s_seconds_beyond`,
  `test_running_node_stamps_started_and_duration_is_live`,
  `test_finished_node_freezes_its_duration`, `test_interruption_freezes_a_running_node`,
  `test_set_history_freezes_historical_running_nodes`,
  `test_running_node_blinks_and_timer_ticks_then_stops`,
  `test_running_node_border_blinks_and_header_shows_timer`.

## Chat text selection & copy (focus does not block copy)
The left chat is drag-selectable and copyable even though focus stays on the
input:
- `MessageBubble` renders a two-column `Table.grid` (label + Markdown/Text), so
  Textual's default selection extraction returns `None` (nothing copyable).
  `MessageBubble.get_selection(selection)` overrides this to yield the raw
  message **body** (never the `you |` prefix); partial selections honour
  `selection.extract`, else the whole body is returned.
- **`#chat` must stay `can_focus=False`.** A `VerticalScroll` is focusable by
  default; if the chat can focus, pressing to start a selection focuses it,
  fires `on_descendant_focus`, and the redirect yanks focus back to the input
  MID-DRAG -- cancelling the selection (the reported "can't select, focus
  jumps back" bug). Keeping it non-focusable removes the churn; it still
  scrolls by mouse wheel.
- Copy uses Textual's built-in selection: drag to select, then `Ctrl+C`.
  Because `#prompt` (Input) is always focused, the binding chain is
  `Input.copy -> Screen.copy_text -> App.quit`. With no Input selection,
  `Input.copy` raises `SkipAction` and it falls through: **Ctrl+C copies a
  chat selection when one exists, and still quits when nothing is selected.**
  Do NOT rebind `ctrl+c` at the app level to something that swallows the
  fall-through, and do NOT make bubbles non-selectable.
- Guards (tests/test_tui.py): `test_chat_text_is_selectable_and_copyable_with_focus_on_input`,
  `test_chat_is_not_focusable_so_drag_select_is_not_cancelled`,
  `test_ctrl_c_still_quits_when_nothing_is_selected`.

## Node inspector modal (full input/response)
LLM telemetry carries the full model **input** and **response** so they can be
viewed in full:
- Agent side (`runtime/agent.py`): `llm_call` events carry the rendered prompt
  in `text` (via `_format_prompt(history)`); `llm_result` events carry the raw
  reply in `output` (its `text` stays the short kind label used as node status).
- Flow model (`flow.py`): `GraphNode` has `full_input` / `full_response`, plus
  `inspectable` and `inspect_sections()`. `_on_llm_call` stores the prompt,
  `_on_llm_result` stores the raw reply (also used as the inline expandable body).
- **Inspectable nodes are popup-only (one gesture, one place).** `node_content`
  appends only the inspect glyph `⤢` (NO inline caret) on an inspectable
  node's header. `FlowCanvas._handle_node_click` opens the inspector on a click
  ANYWHERE on such a node (header or body) via `FlowPanel.open_inspector(key)`,
  which posts `FlowPanel.InspectRequested(title, sections)`; it never inline-
  toggles. (Two different expand gestures in two spots was confusing.) Only
  non-inspectable nodes with a body inline-toggle (`FlowPanel.toggle`), and
  `toggle` / `set_expanded` / `expand_all` all skip inspectable nodes.
- Modal: `ChatApp.on_flow_panel_inspect_requested` pushes `InspectModal`
  (`ModalScreen`) -- covers ~90% (`width/height: 90%`), shows the labeled
  `INPUT`/`RESPONSE` in a read-only `TextArea` (scroll + selection), **takes
  focus** (the focus exception above), and has two footer buttons whose
  shortcut is shown in the label -- `Copy (c)` (copies the whole body via
  `App.copy_to_clipboard`) and `Close (Esc)`. Keys `c` / `Esc` / `q` still
  work; closing returns focus to `#prompt`. `Close` sits in the
  BOTTOM-RIGHT corner (the `#inspect-hint` is a flexible `1fr` gap that
  shoves it there) -- the SAME exit spot every modal uses.
- Guards: tests/test_flow.py -> `test_llm_node_captures_input_and_response`,
  `test_non_llm_nodes_are_not_inspectable`, `test_open_inspector_posts_message`,
  `test_inspectable_node_shows_button_glyph`,
  `test_inspectable_node_header_has_no_inline_caret`,
  `test_non_inspectable_node_with_body_still_shows_caret`,
  `test_clicking_inspectable_node_always_opens_inspector_never_toggles`;
  tests/test_tui.py ->
  `test_inspector_modal_opens_takes_focus_and_copies`,
  `test_all_modals_put_close_button_bottom_right` (Close is bottom-right in
  every modal).

## Per-bubble message modal (double-click a bubble)
Any chat bubble opens full-screen for reliable copy AND restoring the run to
any point/branch. Terminal-native drag-selection is kept (above) but flaky in
some terminals, so this modal is the dependable path.
- **Two-step open (guards against accidents).** The FIRST click on a bubble
  only *selects* it (`MessageBubble` posts `SelectRequested`; the app adds the
  `-selected` highlight and stores `_selected_bubble`). A SECOND click on the
  already-selected bubble posts `OpenRequested`, and `ChatApp` pushes
  `SessionModal`. Selecting is a visual highlight only.
- **Typing is never stolen.** A selected bubble does NOT take focus; clicks
  don't move focus and the app still pins focus to `#prompt`, so characters
  typed after a selection go into the input. This is required behaviour.
- **`history_index`.** Each bubble stores the index of the message it
  represents in `agent.history` (user bubble = `len(history)` recorded in
  `_send`; answer bubble = `len(history) - 1` in `_on_done`). System/help lines
  keep `-1`. Tool-result observations are `USER` messages but produce NO
  bubble, so the index is stored at creation -- never counted by 'nth bubble'.
- **Message modal (read / copy / edit).** `SessionModal` (`ModalScreen`, ~90%
  screen) shows ONE message in a read-only, *selectable* `TextArea`
  (`#session-body`) so it can be reliably read and copied even when
  terminal-native selection is flaky. It **takes focus** (the focus exception
  above). Footer buttons show the shortcut in the label: `Copy (c)`,
  `Edit & resend (e)` (user bubbles only), `Close (Esc)`; keys `c`/`e`/`Esc`/`q`
  also work. `Close` sits in the BOTTOM-RIGHT corner (same as every modal --
  the `1fr` hint is the gap). The bubble modal itself has NO in-modal tree / Restore control
  (that was removed as confusing); restore-to-any-node is now a first-class
  feature opened from the STATUS BAR instead (see the Restore section). The
  git-like session runtime backs both Edit and Restore (see below).
- **Edit == rewind-to-before + prefill.** Only offered for USER bubbles
  (`_can_edit = role is USER and history_index >= 0`; button `Edit & resend
  (e)`, absent on answers). `EditRequested(text, history_index)` ->
  `ChatApp.on_session_modal_edit_requested` -> the SHARED
  `ChatApp._edit_message(history_index, text)`, which checks out the session
  node ENDING JUST BEFORE that message (`_node_before_history_index`: snapshot
  history length == `history_index`; the first user turn falls back to the
  root), rebuilds the view, and prefills `#prompt` with the text. Re-sending
  then branches off that point (old line survives via the underlying session
  tree), matching how Claude/ChatGPT 'edit message' works.
- **Edit cancels an in-flight turn FIRST (and awaits it).** `_edit_message`
  is async; if a turn is still running it calls `_cancel_active_turn`
  (fires the turn's `CancelToken`, then `await worker.wait()`) BEFORE the
  rewind. The edit handler (`on_session_modal_edit_requested`) is async and
  `await`s it. Without this the abandoned turn's late
  `_on_done`/`_on_interrupted` would land an orphan reply in the rewound
  transcript and commit a stray node.
- **Session runtime** lives in `runtime/session.py` (`SessionTree`,
  `SessionNode`, `Snapshot`): append-only, per-`actor` branch heads (the
  multi-agent hook -- each agent moves only its own head, so concurrent
  commits diverge instead of clobbering). It is surfaced BOTH by the
  status-bar Restore map (checkout any node) and by Edit (`checkout_session`
  + branching commits), and it stays the substrate for future recovery
  features.
  `Snapshot.artifacts` is the reserved slot for future file snapshots
  (edited-file blobs, deletion tombstones); the tree does not assume what an
  artifact is, so file recovery can be layered in without touching it. `Agent`
  commits one node per turn (`arun_turn`). **`clear` does NOT discard the
  tree**: `Agent.reset` (behind `action_clear` / `/reset`) rewinds the live
  history/telemetry to system-only, then calls `SessionTree.new_root(...)`
  which FORKS a new child off the ORIGINAL root (`parent == root_id`, label
  `cleared`) and moves the head onto it -- so the root becomes a visible
  split: one child is the pre-clear conversation, the other the fresh line.
  The old branch stays in the store, so the history-tree map draws the fork
  and can restore to it; `rows` / `graph_rows` / `graph_layout` walk every
  root (`SessionTree.roots`, normally just the one). `root_id` still means
  the ORIGINAL (first) root, now a branch point after a clear. Guards:
  tests/test_session.py -> `test_new_root_forks_off_root_and_keeps_old_branch`,
  `test_rows_and_layout_fork_at_root_after_clear`; tests/test_tui.py ->
  `test_clear_keeps_session_history_and_starts_new_root`.
- Guards: tests/test_session.py (tree model) -> `test_commit_appends_child_and_moves_head`,
  `test_checkout_is_non_destructive_and_branches`,
  `test_each_actor_moves_only_its_own_head`, `test_snapshot_is_an_immutable_capture`,
  `test_rows_are_preorder_with_depth`; tests/test_tui.py ->
  `test_bubbles_carry_their_history_index`,
  `test_first_click_selects_second_click_opens_modal`,
  `test_session_modal_shows_message_focused_and_copyable`,
  `test_session_modal_buttons_show_shortcuts`,
  `test_edit_rewinds_to_before_message_and_prefills_input`,
  `test_edit_then_resend_branches_and_keeps_old_line`,
  `test_edit_absent_on_assistant_bubbles`,
  `test_typing_lands_in_input_even_when_a_bubble_is_selected`,
  `test_edit_while_turn_running_cancels_it_before_rewinding`.

## Restore-to-node (`sessions` map from the status bar)
Restore is a first-class, STANDALONE, SCOPED feature (distinct from a bubble's
Edit). It is deliberately EXPLICIT: selecting a node never restores -- the user
must choose a SCOPE (session vs workspace) so a stray click/Enter can never
silently rewind anything.

### Two tabs (BUTTON-STYLE, on the title row): session tree + global browser
`RestoreModal` hosts a `TabbedContent` (`id="restore-tabs"`), but its BUILT-IN
tab bar is HIDDEN (`TabbedContent > ContentTabs { display: none; height: 0 }`).
The tabs are instead two `Button`s styled like the modal's other buttons
(class `tabbtn`) sitting on the SAME compact height-1 row as the `sessions`
title: `Horizontal#restore-titlebar` = `Static#restore-title` (1fr) +
`Button#tabbtn-tree` + `Button#tabbtn-global`. Clicking one calls
`_switch_tab(tab_id)`, which sets `TabbedContent.active` and marks the active
button `variant="primary"` (the other `default`) so it reads like a pressed
tab; it also focuses the pane's natural widget (canvas / search). Do NOT
re-show `ContentTabs` or move the tabs off the title row -- the compact,
button-on-title-row look is the contract. NOTE: the tab buttons
MUST keep their label -- style them via `#restore-titlebar Button` (id-scoped,
like `#restore-actions Button`) AND pass `compact=True`. A class-only
selector (`.tabbtn`) loses to `Button`'s own nested `-style-default`
`border-top/bottom: tall` rules, so at `height: 1` the label gets ZERO
content rows and vanishes. Guard asserts the rendered label is non-empty.
- **Tab 1 `#tab-tree` (default): the SESSION TREE of THIS run** -- the
  `_SessionCanvas` described below, behavior UNCHANGED. `s`/`w` + the
  Restore buttons still scope-restore a picked node; the tree tab is the
  active tab on mount and the canvas keeps focus.
- **Tab 2 `#tab-global`: a GLOBAL browser over every saved rollout** under the
  home dir. Layout: `Input#global-search` on top, then `Horizontal#global-split`
  = `ListView#global-list` (2fr, left) + `TextArea#global-preview` (3fr, right).
  The list is `session_service.list_sessions(paths, current=...)` -- NEWEST
  FIRST, the live rollout flagged with a green `\u25cf` marker. The search box
  filters by title / id / timestamp substring (`_apply_filter`). Highlighting a
  row previews `flatten_session(path)` -- the conversation FLAT, in time order
  (NO tree). `Enter`/click/`_open_selected_global` posts
  `RestoreModal.OpenRequested(path)`; the app's async
  `on_restore_modal_open_requested` cancels+awaits any in-flight turn, then
  `load_tree`s that rollout into `agent.sessions`, replays its head snapshot
  into `history`/`telemetry`, and REBINDS `agent._session_saver` to keep
  writing to the opened file (`_current_session_file`).
- **`s`/`w` are INERT on the global tab** (`_restore` early-returns when
  `_on_global_tab()`), so those keys are plain typing into the search box and
  can never restore/close while browsing.
- **Plumbing**: `ChatApp.__init__(..., sessions_dir=None, current_file=None)`
  and `run_tui(..., sessions_dir=None, current_file=None)` carry the on-disk
  location; `__main__.py` fills them from `open_session().paths.sessions_dir` /
  `.session_file`. With no dir wired the global tab is simply empty (no crash).
- Guards (tests/test_tui.py): `test_sessions_modal_has_two_tabs_tree_default`,
  `test_sessions_modal_tabs_are_buttons_on_title_row`,
  `test_sessions_modal_tab_buttons_switch_panes`,
  `test_global_tab_lists_all_sessions_newest_first`,
  `test_global_tab_search_filters_list`,
  `test_global_tab_preview_is_flat_by_time`,
  `test_global_tab_open_loads_session_into_app`,
  `test_scope_keys_do_not_restore_on_global_tab`,
  `test_sessions_modal_no_dir_shows_empty_browser`; plus
  tests/test_session_service.py `test_list_sessions_*` / `test_flatten_session_*`.

- **Opened from the status bar**, not a bubble. The `restore` `_KeyHint`
  (labelled `sessions`; action stays `restore`) and the `Ctrl+R`
  binding call `ChatApp.action_restore`, which pushes
  `RestoreModal(sessions, head)` with the `SessionTree` (`agent.sessions`) and
  the current head id.
- **`RestoreModal` (`ModalScreen`, ~80%)** DRAWS the tree the SAME way the
  telemetry dashboard draws its flow: a pannable `_SessionCanvas`
  (`ScrollableContainer`) of `_SessionNodeBox` widgets, one round-bordered box
  per `SessionNode`, positioned from `SessionTree.graph_layout()`.
- **PLANAR layout (never reuse a column).** `graph_layout` assigns columns so
  a linear history stays in ONE column (boxes stack straight down -- no
  rightward creep) and every FORK opens a BRAND-NEW column to the right
  (`max column used so far + 1`). Columns are NEVER reused: each branch keeps
  its own lane for life, so a later child's lane is always further right than
  every lane opened before it. That is what makes the drawing PLANAR -- a
  connector only ever traverses not-yet-opened (empty) columns, so NO two
  connectors ever cross and no line cuts through a box. The trade is width
  (deep serial forks creep rightward), which is fine on the unbounded, pannable
  canvas. Guard: tests/test_session.py ->
  `test_graph_layout_never_reuses_a_column_so_the_tree_stays_planar`.
- **The dashboard HIDES `clear` marker nodes.** A `clear` restores no
  context, so its node is not a useful restore target. `RestoreModal`
  builds its layout via `graph_layout(hide_labels={CLEAR_LABEL})`, which
  makes matching nodes TRANSPARENT: they are not placed and their children
  re-attach to the nearest VISIBLE ancestor. So the post-clear line still
  shows, hanging off the ORIGINAL root as a fork (not off the hidden clear
  node); rows stay dense and the planar column rule still holds. The
  underlying `SessionTree` is UNCHANGED -- only this drawing view skips
  them (`rows`/`graph_rows` and persistence still include the clear node).
  The modal also clamps the initial cursor to a visible node: if the head
  is itself the hidden clear marker (right after Ctrl+L, before any send),
  the cursor falls back to its nearest visible ancestor so `s`/Enter can
  never restore to the clear node. Guards: tests/test_session.py ->
  `test_graph_layout_hide_labels_skips_clear_and_reparents_children`,
  `test_graph_layout_without_hide_labels_is_unchanged`; tests/test_tui.py ->
  `test_dashboard_hides_clear_nodes` (and the updated
  `test_clear_keeps_session_history_and_starts_new_root`).
- **Straight-line comb (T) connector renderer.** A `_SessionEdgeLayer` (lower
  layer) draws forks git-graph style with box-drawing lines ONLY -- glyphs
  `\u2502\u2500\u2510\u2514\u252c` (`\u2502` vertical, `\u2500` horizontal,
  `\u2510` down+LEFT corner, `\u2514` up+right elbow, `\u252c` tee-down; NO
  arrow head and NO diagonal `\u2572`). Straight lines are DELIBERATE: a
  diagonal lead-out was tried and REJECTED (`\u659c\u7ebf\u7528\u7684\u592a\u62c9\u4e86`); every
  endpoint must be vertically CENTERED on its bubble (OUT end on the parent's
  right-border MIDPOINT, IN end on the child's left-border MIDPOINT). Forks
  are grouped PER PARENT into a COMB (`_SessionCanvas._fork_groups()`, built
  from `_fork_edges()`): a parent with N fork children draws ONE horizontal
  BAR on its own mid row (`bar_row`) that leaves the parent's right border and
  runs to the FARTHEST child lane, teeing DOWN (`\u252c`) at each inner lane
  and ending with a `\u2510` down+LEFT corner at the last lane -- so N
  children fan out from one bar (a clean T/comb), NOT N horizontals stacked on
  one row (which was the old bug the user flagged as poor T support). Each
  child then drops its OWN (never-reused) lane with a straight `\u2502` and
  turns into the child's left-border midpoint with a `\u2514` elbow. A
  single-child fork degenerates to a plain L (`\u2500\u2500\u2510` bar +
  vertical + `\u2514`). The `\u2510` down+LEFT corner (NOT down+right
  `\u250c`) is used because the bar arrives from the LEFT and turns DOWN; a
  down+right corner would render mirror-flipped. LINEAR edges (child in the
  SAME column) are NOT drawn -- the touching boxes show the link (a
  same-column child is even skipped from the comb, so the bar/lanes never
  collide with it). Because each parent's bar sits ABOVE all its own child
  verticals and every child lane is never reused (columns step strictly
  right), no vertical is ever shared and no bar crosses a rail -- the picture
  is planar by construction; there is no rail-packing / lane-interval logic
  (the old `_lane_plan` / `_fork_rails` / `_iter_edges` / `_col_gap` are
  gone). The child's lane is `_lane_x(col) = col_x[col] - 1` (one cell LEFT of
  the box border, in the gutter) so the `\u2514` lands ON the border and the
  vertical never enters a box. `_compute_widths` sizes each column to its
  widest box and packs columns with a fixed `_SCOL_GAP` gutter. Guards:
  `test_history_tree_connectors_are_planar_and_miss_box_interiors` (cols step
  monotone right, ZERO crossings, ZERO box-interior hits),
  `test_history_tree_overlapping_forks_stay_planar_in_own_lanes`,
  `test_history_tree_single_fork_draws_a_planar_L`,
  `test_history_tree_two_forks_draw_staggered_planar_lanes`,
  `test_history_tree_multi_child_parent_draws_one_comb_bar` (ONE bar, one
  `\u252c` tee + one `\u2510` on the SAME row, two elbows, zero crossings),
  `test_history_tree_comb_bar_is_centered_on_the_parent`.
- **Node classes.** The head box gets the green `current` class (a filled dot
  `\u25cf` vs a ring `\u25cb` in the label), fork nodes get the `fork`
  class, and the selection cursor gets `-selected`.
- **Focus + input.** The modal **takes focus** (the same focus exception as the
  other modals): arrows move the cursor (row order), a click SELECTS a box, and
  a background drag pans the canvas.
- **Restore is SCOPED and explicit (no auto-restore).** Selecting a node --
  by click, a second click, or Enter -- ONLY moves the cursor;
  `on_session_canvas_selected` deliberately does nothing. Restore fires only
  from an explicit scope choice: the `s` key / `#restore-session` button
  (primary, `Restore session (s)`) posts
  `RestoreModal.RestoreRequested(node_id, scope="session")`; the `w` key /
  `#restore-workspace` button (warning, `Restore workspace (w)`) posts
  `scope="workspace"`. `RestoreRequested.scope` defaults to `"session"` (the
  safer scope). The `#restore-close` `Close` button sits in the BOTTOM-RIGHT
  corner (the `#restore-hint` `1fr` gap pushes it there), matching the other
  modals. There is NO Enter-to-restore binding and NO combined restore button
  any more -- only the two scoped buttons + `s`/`w` keys. Guards:
  `test_restore_canvas_click_only_selects_and_scoped_restore_rewinds`
  (click / 2nd click keep the modal open and rewind NOTHING; `s` restores),
  `test_all_modals_put_close_button_bottom_right` (asserts `#restore-session`).
  POSITIONING (critical): `_SessionNodeBox` MUST set `position: absolute`
  (like the flow `FlowNode`). `_SessionCanvas` is a `ScrollableContainer`
  (vertical layout), so without `absolute` a box's `offset` is added to its
  natural stacked position -- boxes drift and the edge layer (which draws at
  the computed `_pos`) misses them, leaving stray connector fragments and
  uneven gaps. With `absolute`, `offset` is true canvas coordinates and
  matches both `_pos` and the connectors. Guard:
  `test_history_tree_boxes_are_dense_and_content_sized`.
  DENSITY / WIDTH (`_SNODE_W=26`, `_SNODE_H=3`, `_SROW_GAP=0`, `_SCOL_GAP=4`):
  rows use a stride of `_SNODE_H + _SROW_GAP = 3`, so same-column boxes STACK
  FLUSH (a parent's bottom border abuts the child's top border -- both stay
  visible, no overlap-merge). Box width is content-sized, not fixed:
  `_compute_widths` measures each label and clamps to `[_SNODE_W,
  _max_box_w()]` where `_max_box_w = canvas width - 10` (modal inner width
  minus a margin), then packs each column to its widest box (`_col_x`), so a
  wide utterance stretches its bubble rightward (up to the cap) instead of
  being clipped. `agent._summarize` keeps labels generous (limit 160) so the
  box, not the label, bounds the visible width. The `#restore-canvas` hides
  scrollbar chrome (`overflow: auto auto; scrollbar-size: 0 0`, like
  `FlowCanvas`) so the bar never sits on top of the tree; pan by dragging.
  NOTE: `_SessionCanvas.Selected` sets `namespace="session_canvas"` so its
  handler resolves to `on_session_canvas_selected`; the leading underscore in
  the class name would otherwise derive `on__session_canvas_selected` (double
  underscore) and the handler would silently never route.
  NOTE: the node-centering helper is `_scroll_to_node`, NOT `_scroll_to` --
  `ScrollableContainer`/`Widget` already own a private
  `_scroll_to(*, x, y, animate=...)` that the scrollbar / mouse-wheel calls,
  so shadowing that name broke scrolling (`TypeError: unexpected keyword
  argument 'animate'`). Guard:
  `test_restore_canvas_scroll_does_not_shadow_textual_scroll_to`.
  NOTE (fill): `_SessionCanvas.on_resize` re-runs `_reflow`, which floors the
  edge-layer size at `viewport + _SPAD` in BOTH axes -- so the map FILLS the
  modal pane and keeps free drag-pan slack (like `FlowCanvas`). Without it the
  virtual area kept a stale/small size and the tree did not cover the pane.
  Guard: `test_history_tree_canvas_fills_the_modal`.
  NOTE (pending): a turn commits to the `SessionTree` only when it FINISHES,
  so an in-flight send has no node yet. `action_restore` passes
  `pending_text=self._pending_user_text` (set in `_send`, cleared in `_finish`)
  when `_busy`; the canvas then draws ONE synthetic dashed `pending` box
  (`_SPENDING_ID`) hanging off the current head. It is NOT a real commit: it is
  excluded from `_order()` and `_node_at`, so arrow-nav / clicks skip it and it
  can never be restored to. Guard:
  `test_history_tree_shows_in_flight_turn_as_pending_node`.
- **Restore cancels an in-flight turn FIRST (and awaits it).**
  `ChatApp.on_restore_modal_restore_requested` is async: it calls
  `_cancel_active_turn` (fire the `CancelToken`, then `await worker.wait()`)
  BEFORE touching history. On `scope=="workspace"` it calls
  `agent.restore_workspace(node_id)` (checkout + re-apply captured file
  artifacts, returns a count) and notices `restored session + workspace`
  (`(N files)` when any were applied); otherwise it calls
  `agent.checkout_session(node_id)` and notices `restored session`. Either way
  it then rebuilds the transcript (`_rebuild_transcript`), re-projects
  telemetry (`_flow.set_history`), and refocuses `#prompt`. Without the
  cancel+await the abandoned turn's late callback would drop an orphan reply
  into the restored transcript (exactly like the Edit case).
- **Transcript rebuild skips tool-call XML.** `_rebuild_transcript` redraws
  the left chat from `agent.history`, but an assistant history entry is the
  model's RAW turn -- either a `<tool_call>` (an intermediate step) or a
  `<message>` (the final answer). It must `parse_agent_response` each
  assistant entry and SKIP `is_tool_call` turns (they belong on the flow
  graph), rendering only the parsed `<message>` body via `_display_answer`.
  Otherwise a tool-call turn leaks its raw `<tool_call>` tags into a bubble
  (the live path never bubbles tool steps, so the rebuild must match).
- **Non-destructive** (git checkout): the branch you left survives, so a later
  send branches off the restored node. Backed by the same `SessionTree`
  runtime as Edit (per-`actor` heads keep multi-agent safe). `restore_workspace`
  reads `snapshot.artifacts["files"]` / `["deleted"]`; file capture is not
  wired yet, so today it safely degrades to a session-only rewind (count 0).
  Guards: tests/test_agent.py -> `test_checkout_session_rewinds_history_only`,
  `test_restore_workspace_rewinds_session_and_applies_file_artifacts`,
  `test_restore_workspace_with_no_artifacts_just_rewinds_session`.
- Guards: `test_restore_hint_opens_session_tree_map_with_current_head_marked`
  (box count == rows),
  `test_restore_scoped_session_restores_full_context` (select-only keeps the
  modal up; `s` rewinds and notices `restored session`),
  `test_restore_while_turn_running_cancels_it_before_checkout`,
  `test_restore_is_non_destructive_old_branch_survives`,
  `test_restore_modal_draws_branching_tree_connectors` (asserts `graph_layout`
  forks to `col>=1` and the edge layer draws the straight L glyphs),
  `test_restore_transcript_hides_tool_call_xml`,
  `test_history_tree_canvas_fills_the_modal` (fill),
  `test_history_tree_shows_in_flight_turn_as_pending_node` (pending marker);
  tests/test_session.py -> `test_graph_layout_keeps_a_linear_history_in_one_column`,
  `test_graph_layout_steps_forks_into_new_columns` (and the still-present
  `test_graph_rows_carry_git_graph_connectors_for_drawing`).

## When you DO need to change the UI
1. Read this file and the two guard test files first.
2. Make the change plus the matching test + doc update in one pass.
3. Run the guards above, then the full suite. Keep the count from regressing.
4. If you intentionally retire an invariant, update this contract and tell the user.
