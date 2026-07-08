"""SafetyPolicy: turn a reversibility class into an execution decision.

This is the brain of the YOLO-with-recovery model. For each tool call it:

1. Classifies the call via :class:`~coreybot.security.reversibility.RiskEngine`.
2. For a recoverable side-effecting call, snapshots the affected files via the
   :class:`~coreybot.security.journal.WorkspaceJournal` *before* execution, so a
   rollback point exists.
3. Emits a :class:`SafetyDecision`:
   - ``FULL``    -> ALLOW, carrying the snapshot artifacts to commit.
   - ``PARTIAL`` -> ALLOW, carrying a compensation note for the audit trail.
   - ``NONE``    -> ask the approval handler; if it approves, ALLOW, else DENY.
                    With no handler (headless), NONE is DENIED by default.

The default posture is deliberately permissive: only genuinely unrecoverable
actions (``NONE``) can ever pause execution, so the common case runs untouched.

A snapshot that turns out *not* fully capturable (binary/oversized target)
downgrades a would-be ``FULL`` to ``NONE`` -- if we cannot guarantee rollback,
we do not pretend the action is safe.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from coreybot.security.capabilities import SafetyProfile, UNKNOWN_PROFILE
from coreybot.security.journal import SnapshotResult, WorkspaceJournal
from coreybot.security.reversibility import (
    Reversibility,
    RiskEngine,
    ToolCallRequest,
)


class Decision:
    """Terminal action the agent should take for a tool call."""

    ALLOW = "allow"
    DENY = "deny"


@dataclass
class ApprovalRequest:
    """Handed to an approval handler when a call is unrecoverable (``NONE``).

    Carries only non-sensitive, human-readable context so a UI can render a
    clear prompt. The handler returns ``True`` to permit the call.
    """

    tool: str
    reversibility: Reversibility
    reason: str
    arguments: Dict[str, object] = field(default_factory=dict)


# An approval handler decides whether an unrecoverable call may proceed.
# Sync signature keeps it usable from the thread-offloaded tool path; a TUI can
# bridge to its async prompt via a thread-safe call.
ApprovalHandler = Callable[[ApprovalRequest], bool]


@dataclass
class SafetyDecision:
    """The policy\'s verdict for one tool call.

    - ``decision``: ALLOW or DENY.
    - ``reversibility``: the assigned class (for telemetry/UI).
    - ``reason``: human-readable explanation (why denied / what class).
    - ``artifacts``: pre-execution snapshot to merge into the session node on a
      successful run, enabling rollback (empty unless FULL with captures).
    - ``compensation``: for PARTIAL calls, how the effect can be undone.
    - ``snapshot``: the raw :class:`SnapshotResult` (diagnostics/tests).
    """

    decision: str
    reversibility: Reversibility
    reason: str = ""
    artifacts: Dict[str, object] = field(default_factory=dict)
    compensation: str = ""
    snapshot: Optional[SnapshotResult] = None

    @property
    def allowed(self) -> bool:
        return self.decision == Decision.ALLOW

    @property
    def recoverable(self) -> bool:
        """True when a rollback point was captured for this call."""
        return bool(self.artifacts.get("files") or self.artifacts.get("deleted"))


def default_protected_roots() -> List[str]:
    """Paths that must never be treated as recoverable.

    Includes coreybot\'s own home (session storage / rollout) so a tool can never
    silently destroy the very state used to recover, plus obvious OS roots. The
    home is resolved the same way the rest of the app resolves it.
    """
    roots: List[str] = []
    try:
        from coreybot.core.paths import resolve_home

        roots.append(str(resolve_home()))
    except Exception:
        pass
    # Common irrecoverable system locations (best-effort, cross-platform-ish).
    system = os.environ.get("SystemRoot") or os.environ.get("windir")
    if system:
        roots.append(system)
    for candidate in ("/etc", "/bin", "/sbin", "/usr", "/boot", "/dev", "/System"):
        roots.append(candidate)
    return roots


class SafetyPolicy:
    """Classifies, snapshots, and decides per tool call."""

    def __init__(
        self,
        *,
        engine: Optional[RiskEngine] = None,
        journal: Optional[WorkspaceJournal] = None,
        approval_handler: Optional[ApprovalHandler] = None,
        workspace_root: Optional[str] = None,
        protected_roots: Optional[Sequence[str]] = None,
        confirm_partial: bool = False,
    ) -> None:
        self.engine = engine if engine is not None else RiskEngine()
        self.journal = journal if journal is not None else WorkspaceJournal()
        self.approval_handler = approval_handler
        self.workspace_root = os.path.abspath(workspace_root or os.getcwd())
        self.protected_roots = (
            list(protected_roots) if protected_roots is not None else default_protected_roots()
        )
        # When True, PARTIAL also requires approval (stricter). Default False:
        # PARTIAL runs but records its compensation note.
        self.confirm_partial = confirm_partial

    def set_approval_handler(self, handler: Optional[ApprovalHandler]) -> None:
        self.approval_handler = handler

    def _request(self, name: str, arguments: Dict[str, object], profile: SafetyProfile) -> ToolCallRequest:
        return ToolCallRequest(
            name=name,
            arguments=dict(arguments),
            profile=profile,
            workspace_root=self.workspace_root,
            protected_roots=self.protected_roots,
        )

    def _ask(self, req: ToolCallRequest, level: Reversibility, reason: str) -> bool:
        if self.approval_handler is None:
            return False  # headless: no gate keeper -> refuse the unrecoverable
        approval = ApprovalRequest(
            tool=req.name, reversibility=level, reason=reason, arguments=dict(req.arguments)
        )
        try:
            return bool(self.approval_handler(approval))
        except Exception:
            return False

    def evaluate(
        self, name: str, arguments: Dict[str, object], profile: Optional[SafetyProfile] = None
    ) -> SafetyDecision:
        """Classify + (if recoverable) snapshot + decide for one tool call."""
        profile = profile if profile is not None else UNKNOWN_PROFILE
        req = self._request(name, arguments, profile)
        level = self.engine.classify(req)

        if level is Reversibility.FULL:
            # Capture originals so the call can be rolled back. If any target is
            # not capturable, we cannot guarantee recovery -> escalate to NONE.
            snap = self.journal.snapshot(req.paths())
            if not snap.fully_capturable:
                reason = (
                    "target(s) not fully capturable for rollback: "
                    + ", ".join(os.path.basename(p) for p in snap.skipped)
                )
                return self._decide_none(req, reason, snapshot=snap)
            return SafetyDecision(
                decision=Decision.ALLOW,
                reversibility=level,
                reason="fully recoverable; snapshot captured",
                artifacts=snap.as_artifacts(),
                snapshot=snap,
            )

        if level is Reversibility.PARTIAL:
            compensation = profile.compensation or "effect is only partially reversible"
            if self.confirm_partial:
                ok = self._ask(req, level, compensation)
                if not ok:
                    return SafetyDecision(
                        decision=Decision.DENY, reversibility=level,
                        reason="partially reversible action declined",
                        compensation=compensation,
                    )
            return SafetyDecision(
                decision=Decision.ALLOW, reversibility=level,
                reason="partially reversible; compensation recorded",
                compensation=compensation,
            )

        # NONE
        return self._decide_none(req, "action is not reversible by any local means")

    def _decide_none(
        self, req: ToolCallRequest, reason: str, snapshot: Optional[SnapshotResult] = None
    ) -> SafetyDecision:
        ok = self._ask(req, Reversibility.NONE, reason)
        if ok:
            return SafetyDecision(
                decision=Decision.ALLOW, reversibility=Reversibility.NONE,
                reason=f"approved despite being unrecoverable ({reason})",
                snapshot=snapshot,
            )
        return SafetyDecision(
            decision=Decision.DENY, reversibility=Reversibility.NONE,
            reason=reason, snapshot=snapshot,
        )
