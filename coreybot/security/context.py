"""The session-long security context carried alongside the agent.

:class:`SecurityContext` bundles the three enterprise concerns this package
adds, and is the single object an application constructs and hands to the
:class:`~coreybot.runtime.agent.Agent`:

1. **User info** (:class:`UserInfo`) -- non-sensitive identity the assistant is
   *allowed* to know (id, display name, roles, and arbitrary safe attributes).
   It is rendered into the system prompt as a ``<user_context>`` block so the
   model can personalize and reason about permissions.
2. **Secrets** (:class:`~coreybot.security.secrets.SecretStore`) -- access
   tokens, PATs, scopes, env vars. These are held for the session but must NOT
   reach the model; they power tools/integrations and feed the outbound
   redactor. They are never serialized into snapshots or telemetry.
3. **Outbound rules** (:class:`~coreybot.security.pipeline.RulePipeline`) --
   validation applied to every message just before it is sent to the LLM.

The context also exposes :meth:`redact`, the single helper the agent uses to
scrub any text (prompt captures, telemetry) so a stored secret can never leak
into the durable event stream even if it is never sent to the model.

Crucially, a :class:`SecurityContext` is a *live* object, not session state: it
is intentionally excluded from :class:`~coreybot.runtime.session.Snapshot`, so
credentials never touch disk via the JSONL rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from coreybot.security.pipeline import PipelineOutcome, RulePipeline
from coreybot.security.rules import (
    PatternRedactRule,
    RedactSecretsRule,
    RuleLike,
)
from coreybot.security.secrets import SecretRef, SecretStore
from coreybot.security.policy import SafetyPolicy


@dataclass
class UserInfo:
    """Non-sensitive information about the end user the agent serves.

    Everything here is considered safe to send to the model. Do NOT put tokens
    or passwords in ``attributes`` -- those belong in the :class:`SecretStore`.
    """

    user_id: str = ""
    display_name: str = ""
    roles: List[str] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.user_id or self.display_name or self.roles or self.attributes)

    def render_block(self) -> str:
        """Render an XML ``<user_context>`` block for the system prompt.

        Returns an empty string when there is nothing to say, so the prompt is
        not cluttered for anonymous sessions.
        """
        if self.is_empty():
            return ""
        lines = ["<user_context>"]
        if self.user_id:
            lines.append(f"  <user_id>{self.user_id}</user_id>")
        if self.display_name:
            lines.append(f"  <display_name>{self.display_name}</display_name>")
        if self.roles:
            lines.append(f"  <roles>{', '.join(self.roles)}</roles>")
        for key in sorted(self.attributes):
            lines.append(f"  <{key}>{self.attributes[key]}</{key}>")
        lines.append("</user_context>")
        return "\n".join(lines)


def default_pipeline(store: SecretStore) -> RulePipeline:
    """Build the recommended default outbound pipeline for a secret store.

    Order matters: first mask any *known* secret value verbatim, then catch
    token-shaped strings the user never registered. Both are redactions (never
    block), which is the safe default -- an application can add ``BlockPatternRule``
    or ``MaxLengthRule`` for stricter policies.
    """
    return RulePipeline([RedactSecretsRule(store), PatternRedactRule()])


class SecurityContext:
    """Everything security-related that travels with a session."""

    def __init__(
        self,
        user: Optional[UserInfo] = None,
        secrets: Optional[SecretStore] = None,
        pipeline: Optional[RulePipeline] = None,
        *,
        auto_redact_telemetry: bool = True,
        safety: Optional[SafetyPolicy] = None,
    ) -> None:
        self.user = user if user is not None else UserInfo()
        self.secrets = secrets if secrets is not None else SecretStore()
        # If no explicit pipeline was given, default to redacting known secrets
        # plus token-shaped strings so a caller who only supplies secrets is
        # still protected out of the box.
        self.pipeline = pipeline if pipeline is not None else default_pipeline(self.secrets)
        self.auto_redact_telemetry = auto_redact_telemetry
        # Optional tool-execution safety policy (three-class reversibility
        # model + transactional snapshots). ``None`` means tool calls run
        # without the recover-or-confirm gate (the agent falls back to plain
        # execution), so this is purely additive.
        self.safety = safety

    # --- convenience builders ---------------------------------------------
    @classmethod
    def for_user(
        cls,
        user_id: str = "",
        display_name: str = "",
        roles: Optional[Sequence[str]] = None,
        attributes: Optional[Mapping[str, str]] = None,
    ) -> "SecurityContext":
        """Construct a context with just user info (no secrets/custom rules)."""
        info = UserInfo(
            user_id=user_id,
            display_name=display_name,
            roles=list(roles or []),
            attributes=dict(attributes or {}),
        )
        return cls(user=info)

    def add_secret(self, name: str, value: str, *, kind: str = "generic") -> SecretRef:
        """Store a secret and return a safe reference (no value)."""
        return self.secrets.put(name, value, kind=kind)

    def add_rule(self, rule: RuleLike) -> "SecurityContext":
        """Append an outbound rule to the pipeline (chainable)."""
        self.pipeline.add(rule)
        return self

    # --- system-prompt contribution ---------------------------------------
    def system_prompt_block(self) -> str:
        """The user-context block to append to the system prompt (may be empty)."""
        return self.user.render_block()

    # --- outbound + telemetry hooks ---------------------------------------
    def vet_outbound(self, text: str) -> PipelineOutcome:
        """Run the outbound pipeline over ``text`` (may raise OutboundBlocked)."""
        return self.pipeline.run(text)

    def redact(self, text: str) -> str:
        """Scrub known secrets from arbitrary text for safe display/telemetry.

        Unlike :meth:`vet_outbound` this never blocks and never applies policy
        rules -- it only removes secret *values* so nothing sensitive is written
        to the durable telemetry/session stream. A no-op when disabled or when
        there are no secrets.
        """
        if not self.auto_redact_telemetry or not self.secrets or not text:
            return text
        result = RedactSecretsRule(self.secrets).apply(text)
        return result.text

    def __repr__(self) -> str:
        # Never leak: only counts and the (safe) user id.
        who = self.user.user_id or "(anonymous)"
        return (
            f"SecurityContext(user={who!r}, secrets={len(self.secrets)}, "
            f"rules={len(self.pipeline)})"
        )
