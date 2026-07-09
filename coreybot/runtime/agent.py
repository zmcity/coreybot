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
import time
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
from coreybot.security import OutboundBlocked, SecurityContext
from coreybot.security.journal import WorkspaceJournal
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
    # Optional structured execution record (e.g. for a tool_result): safety
    # decision, timing, and outcome. Surfaced verbatim in the flow inspector's
    # LOG section; empty for events that carry no log.
    log: str = ""

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
        security: Optional[SecurityContext] = None,
    ) -> None:
        self.config = config
        self.provider = provider if provider is not None else create_provider(config)
        self.registry = registry if registry is not None else get_registry()
        self.max_steps = max_steps
        # Security context (user info + secret store + outbound rule
        # pipeline). Optional so non-enterprise callers are unaffected; the
        # user-info block (never secrets) is folded into the system prompt.
        self.security = security
        # Pending workspace rollback data captured by the safety policy for
        # the current turn; folded into the next snapshot so the committed
        # session node becomes a restore point (see _snapshot / _run_tool).
        self._pending_artifacts: dict = {}
        self._last_safety_reason: str = ""
        self._last_safety_note: str = ""
        base_prompt = config.system_prompt
        if security is not None:
            user_block = security.system_prompt_block()
            if user_block:
                base_prompt = f"{base_prompt}\n\n{user_block}"
        self.system_prompt = build_system_prompt(
            base_prompt, self.registry.render_for_prompt()
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
        self._pending_artifacts = {}
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
            artifacts=dict(self._pending_artifacts),
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
        """Restore the session AND roll the on-disk workspace back to ``node_id``.

        This is the wider, more dangerous sibling of :meth:`checkout_session`:
        besides rewinding the conversation it *undoes* every workspace write made
        after ``node_id``. Each committed node carries, in ``snapshot.artifacts``,
        the pre-write originals captured by the safety policy during that turn
        (``"files"`` = original text to restore, ``"deleted"`` = paths that did
        not exist yet and must be removed). To reach ``node_id`` we walk the
        nodes strictly between the current head and ``node_id`` and apply their
        captured originals newest-first, which reproduces the on-disk state as of
        ``node_id`` byte-for-byte. Returns the number of file operations applied.

        The scope is deliberately kept SEPARATE from a plain session restore so
        the UI never conflates "rewind the chat" with "rewind my files".
        """
        head_before = self.sessions.head(actor)
        self.checkout_session(node_id, actor=actor)
        target = self.sessions.get(node_id)
        if target is None:
            return 0

        applied = 0
        restored: set = set()
        removed: set = set()

        def _apply(artifacts) -> None:
            nonlocal applied
            files = (artifacts or {}).get("files") or {}
            deleted = (artifacts or {}).get("deleted") or []
            for path, blob in files.items():
                if path in restored:
                    continue
                try:
                    with open(path, "w", encoding="utf-8", newline="") as handle:
                        handle.write(blob)
                    restored.add(path)
                    applied += 1
                except OSError:
                    pass
            for path in deleted:
                if path in removed:
                    continue
                removed.add(path)
                try:
                    os.remove(path)
                    applied += 1
                except OSError:
                    pass

        # (1) Apply the target node's OWN artifacts -- the state to reproduce at
        # this node (the long-standing restore contract).
        _apply(getattr(target.snapshot, "artifacts", None))

        # (2) Undo writes recorded by the turns AFTER the target on the old line.
        # The safety policy stores each turn's pre-write originals on that turn's
        # node, so replaying them newest-first rewinds the workspace to the target.
        path_ids = [n.id for n in self.sessions.path(head_before)]
        if node_id in path_ids:
            tail = path_ids[path_ids.index(node_id) + 1:]
        else:
            tail = path_ids  # different branch: undo the whole old line
        for nid in reversed(tail):
            node = self.sessions.get(nid)
            if node is not None:
                _apply(getattr(node.snapshot, "artifacts", None))
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
        # The snapshot folds in any workspace rollback data captured by the
        # safety policy this turn; reset the pending bag afterwards so it
        # attaches to exactly one node.
        self.sessions.commit(self._snapshot(), label=_summarize(user_input))
        self._pending_artifacts = {}
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

    def _vet_outbound(self, history: List[Message]) -> List[Message]:
        """Return a security-vetted copy of ``history`` to send to the LLM.

        With no security context this is the identity (returns the same list).
        Otherwise every message's content is run through the outbound rule
        pipeline: redactions produce rewritten copies, while a blocking rule
        raises :class:`OutboundBlocked` (handled by the caller). The system
        message is vetted too, so a secret accidentally placed in the prompt
        is caught. The original ``Message`` objects are never mutated.
        """
        if self.security is None:
            return history
        vetted: List[Message] = []
        for message in history:
            outcome = self.security.vet_outbound(message.content)
            if outcome.changed:
                vetted.append(
                    Message(message.role, outcome.text, dict(message.metadata))
                )
            else:
                vetted.append(message)
        return vetted

    async def _drive(
        self, on_event: Optional[EventHandler], cancel_token: Optional[CancelToken]
    ) -> AgentResponse:
        for _ in range(self.max_steps):
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()
            # Outbound security seam: vet the exact history about to be sent
            # to the LLM. Redaction rewrites a *copy* only -- the stored
            # history is never mutated -- and a blocking rule aborts the turn
            # with a notice instead of transmitting. The prompt captured into
            # telemetry is scrubbed too, so secrets never enter the durable
            # event stream / session rollout even when they are not sent.
            try:
                wire_history = self._vet_outbound(self.history)
            except OutboundBlocked as blocked:
                notice = f"outbound blocked: {blocked.reason} (rule '{blocked.rule}')"
                self._emit(on_event, AgentEvent(kind="notice", text=notice))
                return AgentResponse(
                    type=ResponseType.MESSAGE,
                    content="(request blocked by an outbound security rule)",
                    parse_error=notice,
                )
            prompt_text = _format_prompt(wire_history)
            if self.security is not None:
                prompt_text = self.security.redact(prompt_text)
            self._emit(
                on_event,
                AgentEvent(kind="llm_call", name=self.config.model, text=prompt_text),
            )
            result = await run_cancellable(
                self.provider.acomplete(wire_history, cancel_token), cancel_token
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

    def _safety_gate(
        self, name: str, arguments: dict, tool_obj, on_event: Optional[EventHandler]
    ) -> bool:
        """Consult the safety policy before executing ``tool_obj``.

        Returns True to proceed, False to refuse. On a recoverable call the
        pre-execution snapshot is merged into ``_pending_artifacts`` so the
        turn's commit becomes a rollback point; a partially reversible call is
        allowed but its compensation note is emitted for the audit trail; an
        unrecoverable, unapproved call is refused. With no safety policy set,
        this is a no-op that always proceeds (plain execution).
        """
        self._last_safety_reason = ""
        self._last_safety_note = ""
        if self.security is None or self.security.safety is None:
            return True
        profile = getattr(tool_obj, "safety", None)
        decision = self.security.safety.evaluate(name, arguments, profile)
        # Keep a compact, secret-free note for the tool's execution log
        # (recorded whether the call is allowed or refused).
        self._last_safety_note = (
            f"safety: {decision.reversibility.value} -> {decision.decision}"
            f" ({decision.reason})"
        )
        # Surface the class + rationale as telemetry (secret-free).
        self._emit(
            on_event,
            AgentEvent(
                kind="notice",
                text=(
                    f"safety[{decision.reversibility.value}] {name}: "
                    f"{decision.decision} -- {decision.reason}"
                ),
            ),
        )
        if not decision.allowed:
            self._last_safety_reason = (
                f"refused ({decision.reversibility.value}): {decision.reason}"
            )
            return False
        if decision.artifacts:
            self._pending_artifacts = WorkspaceJournal.merge_artifacts(
                self._pending_artifacts, decision.artifacts
            )
        if decision.compensation:
            self._emit(
                on_event,
                AgentEvent(kind="notice", text=f"compensation: {decision.compensation}"),
            )
        return True

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

        log_lines: List[str] = [f"tool: {name or '(unnamed)'}"]
        error: Optional[str] = None
        elapsed_ms: Optional[float] = None
        tool_log: str = ""

        tool_obj = self.registry.get(name)
        if tool_obj is None:
            available = ", ".join(self.registry.names()) or "(none)"
            output = f"unknown tool '{name}'. Available tools: {available}"
            ok = False
            log_lines.append("resolve: not found in registry")
        elif not self._safety_gate(name, arguments, tool_obj, on_event):
            # The safety policy refused this call (unrecoverable + not
            # approved). Do not execute; report the refusal back to the model
            # so it can choose a safer path. No workspace change happened.
            output = self._last_safety_reason or "blocked by safety policy"
            ok = False
            log_lines.append("resolve: found")
            if self._last_safety_note:
                log_lines.append(self._last_safety_note)
            log_lines.append("execution: skipped (blocked by safety policy)")
        else:
            log_lines.append("resolve: found")
            if self._last_safety_note:
                log_lines.append(self._last_safety_note)
            # Tools are plain sync callables; run them off the event loop so a
            # slow tool neither blocks the UI nor ignores cancellation.
            started = time.monotonic()
            outcome = await run_cancellable(
                asyncio.to_thread(tool_obj.call, arguments), cancel_token
            )
            elapsed_ms = (time.monotonic() - started) * 1000.0
            output = outcome.output
            ok = outcome.ok
            error = outcome.error
            tool_log = (getattr(outcome, "log", "") or "").strip()

        log_lines.append(f"outcome: {'ok' if ok else 'failed'}")
        if error:
            log_lines.append(f"error: {error}")
        log_lines.append(f"output: {len(output or '')} chars")
        if elapsed_ms is not None:
            log_lines.append(f"elapsed: {elapsed_ms:.1f} ms")
        if tool_log:
            log_lines.append("")
            log_lines.append(tool_log)

        self._emit(
            on_event,
            AgentEvent(
                kind="tool_result",
                name=name,
                output=output,
                ok=ok,
                log="\n".join(log_lines),
            ),
        )
        # Feed the observation back as a user turn the model will read next.
        observation = _format_tool_result(name, output, ok)
        self.history.append(Message.user(observation))