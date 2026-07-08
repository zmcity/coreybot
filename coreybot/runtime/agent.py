"""The agent loop: chat + tool calling tied together.

This is where the framework stops being a chat wrapper and becomes an *agent*.
Each user turn runs a small internal loop:

    1. Ask the model for a response given the full history.
    2. Parse it (message or tool_call).
    3. If it is a tool_call: run the tool, append the tool result to history as
       a <tool_result> observation, and go back to step 1.
    4. If it is a message: return it to the caller (this ends the turn).

A ``max_steps`` guard prevents infinite tool loops. Observers can subscribe via
an ``on_event`` callback to display tool activity in a CLI/TUI.

The Agent owns the conversation ``history`` (its memory) and is transport- and
UI-agnostic: it talks to any ``LLMProvider`` and any ``ToolRegistry``.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Callable, List, Optional

from coreybot.core.cancel import CancelToken, CancelledError, run_cancellable
from coreybot.core.config import Config
from coreybot.core.message import Message, Role
from coreybot.llm.protocol import (
    AgentResponse,
    ResponseType,
    build_system_prompt,
    parse_agent_response,
)
from coreybot.llm.providers import LLMProvider, create_provider
from coreybot.runtime.session import CLEAR_LABEL, SessionTree, Snapshot
from coreybot.tools import ToolRegistry, get_registry


# Where an activity originates. This is the extension point for future
# injectors: MCP servers, skills, declarative sub-agents, etc. The flow panel
# renders an icon/color per source, so adding a new source needs no UI change.
class Source:
    LLM = "llm"
    TOOL = "tool"
    MCP = "mcp"
    SKILL = "skill"
    AGENT = "agent"
    SYSTEM = "system"


def _format_prompt(history: List[Message]) -> str:
    """Render the conversation the model is about to see as readable text.

    This is captured on each ``llm_call`` event so the UI can show the exact
    input that produced a response (see the flow chart's inspector modal).
    Each message becomes a ``[role]`` header followed by its content.
    """
    blocks = []
    for message in history:
        blocks.append(f"[{message.role.value}]\n{message.content}".rstrip())
    return "\n\n".join(blocks)


@dataclass
class AgentEvent:
    """Something worth showing the user during a turn.

    kind (lifecycle + activity):
    - ``turn_start`` / ``turn_end`` : bracket one user turn.
    - ``llm_call``   / ``llm_result``: a model round-trip (``ok`` on result).
    - ``tool_call``  / ``tool_result``: a tool invocation (``name``/``arguments``
      /``output``/``ok``).
    - ``notice``      : protocol warnings or loop notices (``text``).

    ``source`` says where the activity came from (see :class:`Source`); it is a
    plain string so third parties can introduce their own without code changes.
    """

    kind: str
    source: str = ""
    name: str = ""
    arguments: Optional[dict] = None
    output: str = ""
    ok: bool = True
    text: str = ""

    def __post_init__(self) -> None:
        if not self.source:
            self.source = _DEFAULT_SOURCE.get(self.kind, Source.SYSTEM)


_DEFAULT_SOURCE = {
    "turn_start": Source.SYSTEM,
    "turn_end": Source.SYSTEM,
    "llm_call": Source.LLM,
    "llm_result": Source.LLM,
    "tool_call": Source.TOOL,
    "tool_result": Source.TOOL,
    "notice": Source.SYSTEM,
}


EventHandler = Callable[[AgentEvent], None]


def _summarize(text: str, limit: int = 160) -> str:
    """A one-line label for a session commit (the user's utterance).

    Kept generous so a wide utterance can stretch its history-tree bubble;
    the box itself caps the visible width (to the modal width minus a
    margin) and ellipsizes anything longer, so this only bounds runaway
    labels, not the on-screen size.
    """
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat or "(empty)"
    return flat[: limit - 1] + "\u2026"


def _format_tool_result(name: str, output: str, ok: bool) -> str:
    """Render a tool outcome as a <tool_result> observation fed back to model."""
    status = "ok" if ok else "error"
    return (
        f"<tool_result name=\"{name}\" status=\"{status}\">\n"
        f"{output}\n"
        f"</tool_result>"
    )


class Agent:
    """Chat + tool-calling agent over a provider and a tool registry."""

    def __init__(
        self,
        config: Config,
        provider: Optional[LLMProvider] = None,
        registry: Optional[ToolRegistry] = None,
        max_steps: int = 6,
        session_saver: "Optional[Callable[[SessionTree], None]]" = None,
    ) -> None:
        self.config = config
        self.provider = provider if provider is not None else create_provider(config)
        self.registry = registry if registry is not None else get_registry()
        self.max_steps = max_steps
        self.system_prompt = build_system_prompt(
            config.system_prompt, self.registry.render_for_prompt()
        )
        self.history: List[Message] = [Message.system(self.system_prompt)]
        # Append-only session telemetry: every emitted AgentEvent, across all
        # turns. This is the durable source of truth a UI can *project* from
        # (e.g. the flow chart), which decouples rendering from the live loop.
        self.telemetry: List[AgentEvent] = []
        # Git-like session tree: every completed turn is committed as a
        # node, so any point (or branch) can be restored later. Seeded with
        # the initial system-only state as the root commit.
        self.sessions = SessionTree(self._snapshot())
        # Optional persistence hook: called with the tree after every commit /
        # new_root / checkout so the on-disk rollout stays current. Injected by
        # the entry point (bound to the resolved session file); defaults to a
        # no-op so headless/test agents need no disk. Persistence must never
        # break a turn, so ``_persist_sessions`` swallows I/O errors.
        self._session_saver = session_saver

    def reset(self) -> None:
        """Clear the conversation context, keeping the system prompt.

        This does NOT discard the session tree: the prior history stays in
        ``sessions`` as a sibling branch. ``SessionTree.new_root`` FORKS a
        fresh line off the original root, so that root becomes a visible
        split -- one child is the pre-clear conversation, the other the new
        line. ``clear`` therefore only rewinds the live conversation while
        the history-tree map can still restore to anything committed before
        it.
        """
        self.history = [Message.system(self.system_prompt)]
        self.telemetry = []
        self.sessions.new_root(self._snapshot(), label=CLEAR_LABEL)
        self._persist_sessions()

    def _snapshot(self) -> Snapshot:
        """Capture the current conversation + telemetry as an immutable node.

        Copied on capture so later mutation of the live agent never rewrites
        a committed snapshot. ``artifacts`` is left empty for now -- it is the
        reserved slot where future file snapshots (edited-file blobs, deletion
        tombstones) will live without changing the tree.
        """
        return Snapshot(
            history=list(self.history),
            telemetry=list(self.telemetry),
        )

    def _persist_sessions(self) -> None:
        """Save the session tree via the injected saver, if any.

        Best-effort: a persistence failure (disk full, permissions) must never
        abort a turn, so errors are swallowed. A no-op when no saver was
        injected (headless/test agents).
        """
        if self._session_saver is None:
            return
        try:
            self._session_saver(self.sessions)
        except Exception:
            pass

    def checkout_session(self, node_id: str, actor: str = "main") -> None:
        """Restore history + telemetry from a session node (git checkout).

        Non-destructive: the tree keeps every branch, so this can jump to any
        node -- a parent, a sibling branch, or a previously abandoned line --
        and later commits branch off from there.
        """
        node = self.sessions.checkout(node_id, actor=actor)
        self.history = list(node.snapshot.history)
        self.telemetry = list(node.snapshot.telemetry)
        self._persist_sessions()

    def restore_workspace(self, node_id: str, actor: str = "main") -> int:
        """Restore the session AND the on-disk workspace at ``node_id``.

        This is the wider, more dangerous sibling of :meth:`checkout_session`:
        besides the conversation it re-applies the file state captured in the
        node's ``snapshot.artifacts`` (edited-file blobs under ``"files"`` and
        deletion tombstones under ``"deleted"``). Snapshots do not yet capture
        files, so today ``artifacts`` is empty and only the session is
        restored -- but the scope is deliberately kept SEPARATE from a plain
        session restore so the UI never conflates "rewind the chat" with
        "rewind my files". Returns the number of workspace artifacts applied
        (0 until file capture lands), so a caller can report what happened.
        """
        self.checkout_session(node_id, actor=actor)
        node = self.sessions.get(node_id)
        if node is None:
            return 0
        artifacts = getattr(node.snapshot, "artifacts", None) or {}
        files = artifacts.get("files") or {}
        deleted = artifacts.get("deleted") or []
        applied = 0
        for path, blob in files.items():
            try:
                with open(path, "w", encoding="utf-8", newline="") as handle:
                    handle.write(blob)
                applied += 1
            except OSError:
                pass
        for path in deleted:
            try:
                os.remove(path)
                applied += 1
            except OSError:
                pass
        return applied

    async def arun_turn(
        self,
        user_input: str,
        on_event: Optional[EventHandler] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> AgentResponse:
        """Run one user turn to completion and return the final message.

        Internally loops over tool calls until the model produces a <message>
        or ``max_steps`` is reached. Pass a :class:`CancelToken` to make the
        whole turn interruptible (like a Go ``context.Context``): cancelling it
        stops the in-flight LLM/tool call and raises :class:`CancelledError`,
        which the caller surfaces as an "interrupted" notice rather than a crash.

        Transport errors from the provider propagate to the caller; tool
        failures are captured and fed back to the model rather than raised.
        """
        self._emit(on_event, AgentEvent(kind="turn_start", name="user", text=user_input))
        self.history.append(Message.user(user_input))
        response = await self._drive(on_event, cancel_token)
        self._emit(on_event, AgentEvent(kind="turn_end", ok=True, text=response.content))
        # Commit this turn as a restorable node (one commit per user turn).
        self.sessions.commit(self._snapshot(), label=_summarize(user_input))
        self._persist_sessions()
        return response

    def run_turn(
        self, user_input: str, on_event: Optional[EventHandler] = None
    ) -> AgentResponse:
        """Blocking convenience wrapper around :meth:`arun_turn`."""
        return asyncio.run(self.arun_turn(user_input, on_event))

    def _emit(self, on_event: Optional[EventHandler], event: AgentEvent) -> None:
        # Record first so telemetry is complete even if no observer is attached.
        self.telemetry.append(event)
        if on_event is not None:
            on_event(event)

    async def _drive(
        self, on_event: Optional[EventHandler], cancel_token: Optional[CancelToken]
    ) -> AgentResponse:
        for _ in range(self.max_steps):
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()
            prompt_text = _format_prompt(self.history)
            self._emit(
                on_event,
                AgentEvent(kind="llm_call", name=self.config.model, text=prompt_text),
            )
            result = await run_cancellable(
                self.provider.acomplete(self.history, cancel_token), cancel_token
            )
            raw_reply = result.text.strip()
            # Persist the model's raw turn so it sees its own structured output.
            self.history.append(Message(Role.ASSISTANT, raw_reply))

            response = parse_agent_response(raw_reply)
            _kind = "tool call" if response.is_tool_call else "message"
            self._emit(
                on_event,
                AgentEvent(
                    kind="llm_result", name=self.config.model, ok=True,
                    text=_kind, output=raw_reply,
                ),
            )

            if response.is_tool_call:
                if response.parse_error:
                    self._emit(on_event, AgentEvent(kind="notice", text=response.parse_error))
                await self._run_tool(response, on_event, cancel_token)
                continue  # feed the observation back into the model

            # It is a message -> the turn is done.
            if response.parse_error:
                self._emit(on_event, AgentEvent(kind="notice", text=response.parse_error))
            return response

        # Safety valve: too many tool steps without a final message.
        notice = f"stopped after {self.max_steps} steps without a final message"
        self._emit(on_event, AgentEvent(kind="notice", text=notice))
        return AgentResponse(
            type=ResponseType.MESSAGE,
            content="(the agent used too many tool steps without answering)",
            parse_error=notice,
        )

    async def _run_tool(
        self,
        response: AgentResponse,
        on_event: Optional[EventHandler],
        cancel_token: Optional[CancelToken],
    ) -> None:
        name = response.tool_name or ""
        arguments = response.tool_arguments or {}
        self._emit(
            on_event,
            AgentEvent(kind="tool_call", name=name, arguments=arguments),
        )

        tool_obj = self.registry.get(name)
        if tool_obj is None:
            available = ", ".join(self.registry.names()) or "(none)"
            output = f"unknown tool '{name}'. Available tools: {available}"
            ok = False
        else:
            # Tools are plain sync callables; run them off the event loop so a
            # slow tool neither blocks the UI nor ignores cancellation.
            outcome = await run_cancellable(
                asyncio.to_thread(tool_obj.call, arguments), cancel_token
            )
            output = outcome.output
            ok = outcome.ok

        self._emit(
            on_event,
            AgentEvent(kind="tool_result", name=name, output=output, ok=ok),
        )
        # Feed the observation back as a user turn the model will read next.
        observation = _format_tool_result(name, output, ok)
        self.history.append(Message.user(observation))