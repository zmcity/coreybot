"""Enterprise security layer for coreybot.

Public API for carrying user information, safely storing/passing credentials
(access tokens, PATs, scopes, environment variables), and validating every
message sent to the LLM with a composable outbound rule pipeline.

Typical use::

    from coreybot.security import SecurityContext, SecretKind

    sec = SecurityContext.for_user(user_id="u123", display_name="Ada", roles=["admin"])
    sec.add_secret("github_pat", os.environ["GH_PAT"], kind=SecretKind.PAT)
    agent = Agent(config, security=sec)   # user block injected; outbound vetted

See :mod:`coreybot.security.secrets`, :mod:`coreybot.security.rules`,
:mod:`coreybot.security.pipeline`, and :mod:`coreybot.security.context`.
"""

from __future__ import annotations

from coreybot.security.context import (
    SecurityContext,
    UserInfo,
    default_pipeline,
)
from coreybot.security.pipeline import (
    OutboundBlocked,
    PipelineOutcome,
    RulePipeline,
)
from coreybot.security.rules import (
    Action,
    BlockPatternRule,
    MaxLengthRule,
    PatternRedactRule,
    RedactSecretsRule,
    Rule,
    RuleResult,
)
from coreybot.security.capabilities import (
    Capability,
    SafetyProfile,
    make_profile,
)
from coreybot.security.journal import (
    SnapshotResult,
    WorkspaceJournal,
)
from coreybot.security.policy import (
    ApprovalRequest,
    Decision,
    SafetyDecision,
    SafetyPolicy,
    default_protected_roots,
)
from coreybot.security.reversibility import (
    Reversibility,
    RiskEngine,
    ToolCallRequest,
)
from coreybot.security.secrets import (
    SecretKind,
    SecretRef,
    SecretStore,
    SecretValue,
)

__all__ = [
    # context
    "SecurityContext",
    "UserInfo",
    "default_pipeline",
    # pipeline
    "RulePipeline",
    "PipelineOutcome",
    "OutboundBlocked",
    # rules
    "Rule",
    "RuleResult",
    "Action",
    "RedactSecretsRule",
    "PatternRedactRule",
    "BlockPatternRule",
    "MaxLengthRule",
    # safety: capabilities / reversibility / journal / policy
    "Capability",
    "SafetyProfile",
    "make_profile",
    "Reversibility",
    "RiskEngine",
    "ToolCallRequest",
    "WorkspaceJournal",
    "SnapshotResult",
    "SafetyPolicy",
    "SafetyDecision",
    "ApprovalRequest",
    "Decision",
    "default_protected_roots",
    # secrets
    "SecretStore",
    "SecretValue",
    "SecretRef",
    "SecretKind",
]
