"""Outbound rule validation: vet every message *before* it reaches the LLM.

The agent loop sends the running conversation to the model on each step. In an
enterprise setting that outbound text is exactly where accidental disclosure or
policy violations happen -- a secret pasted by the user, a token echoed by a
tool, PII, or content that simply must never leave the process.

This module provides a small, composable rule system for that seam:

- A :class:`Rule` inspects the outbound text and returns a :class:`RuleResult`
  with one of three :class:`Action` outcomes:
    * ``ALLOW``  -- leave the text unchanged.
    * ``REDACT`` -- rewrite the text (e.g. mask a token) and continue.
    * ``BLOCK``  -- refuse to send; the pipeline raises :class:`OutboundBlocked`.
- Rules are ordinary callables, so adding one needs no base class and no edits
  to central code (same spirit as the ``@tool`` and provider registries).

Built-in rules cover the common cases: redacting known secrets from a
:class:`~coreybot.security.secrets.SecretStore`, masking token-shaped strings by
regex, blocking forbidden patterns, and capping length. Compose them with
:class:`~coreybot.security.pipeline.RulePipeline`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Pattern, Sequence

from coreybot.security.secrets import SecretStore


class Action:
    """What a rule decided to do with the outbound text."""

    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


@dataclass
class RuleResult:
    """Outcome of applying one rule to a piece of outbound text.

    - ``action``: one of :class:`Action`.
    - ``text``: the (possibly rewritten) text to carry forward. For ``ALLOW``
      it equals the input; for ``REDACT`` it is the masked version; for
      ``BLOCK`` it is irrelevant (the pipeline raises).
    - ``rule``: the rule name, for audit/telemetry.
    - ``reason``: human-readable explanation (why blocked / what was redacted).
    - ``hits``: number of substitutions made (for ``REDACT``), 0 otherwise.
    """

    action: str
    text: str
    rule: str = ""
    reason: str = ""
    hits: int = 0

    @classmethod
    def allow(cls, text: str, rule: str = "") -> "RuleResult":
        return cls(action=Action.ALLOW, text=text, rule=rule)

    @classmethod
    def redact(cls, text: str, *, rule: str = "", reason: str = "", hits: int = 0) -> "RuleResult":
        return cls(action=Action.REDACT, text=text, rule=rule, reason=reason, hits=hits)

    @classmethod
    def block(cls, text: str, *, rule: str = "", reason: str = "") -> "RuleResult":
        return cls(action=Action.BLOCK, text=text, rule=rule, reason=reason)


class Rule:
    """Base class for an outbound rule.

    Subclass and implement :meth:`apply`, or just pass any
    ``Callable[[str], RuleResult]`` to the pipeline -- both work. A ``name`` is
    used in audit records and telemetry notices.
    """

    name: str = "rule"

    def apply(self, text: str) -> RuleResult:  # pragma: no cover - interface
        raise NotImplementedError

    # Make instances directly callable so a Rule and a plain function are
    # interchangeable wherever a rule is expected.
    def __call__(self, text: str) -> RuleResult:
        return self.apply(text)


# The mask used when redacting; short and obviously non-sensitive.
REDACTED = "[REDACTED]"


class RedactSecretsRule(Rule):
    """Mask any *known* secret value that appears verbatim in the text.

    This is the most important defense: whatever the user stored in the
    :class:`SecretStore` (tokens, PATs, access tokens) is replaced with a label
    like ``[REDACTED:github_pat]`` before the text can reach the model. Longer
    secrets are masked first so one secret that contains another still redacts
    cleanly.
    """

    name = "redact_secrets"

    def __init__(self, store: SecretStore, *, label: bool = True) -> None:
        self._store = store
        self._label = label

    def apply(self, text: str) -> RuleResult:
        if not self._store or not text:
            return RuleResult.allow(text, rule=self.name)
        pairs = [(n, v) for n, v in self._store.secret_values()]
        # Redact longer values first to avoid partial overlaps.
        pairs.sort(key=lambda nv: len(nv[1]), reverse=True)
        hits = 0
        redacted_names: List[str] = []
        out = text
        for name, value in pairs:
            if value and value in out:
                replacement = f"[REDACTED:{name}]" if self._label else REDACTED
                count = out.count(value)
                out = out.replace(value, replacement)
                hits += count
                redacted_names.append(name)
        if hits:
            reason = "masked known secrets: " + ", ".join(sorted(set(redacted_names)))
            return RuleResult.redact(out, rule=self.name, reason=reason, hits=hits)
        return RuleResult.allow(text, rule=self.name)


# Token-shaped patterns worth masking even if not explicitly stored. These are
# conservative and target well-known credential formats.
_DEFAULT_TOKEN_PATTERNS: Sequence[Pattern[str]] = (
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),               # GitHub PAT (classic)
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),        # GitHub PAT (fine-grained)
    re.compile(r"\bgh[oprsu]_[A-Za-z0-9]{20,}\b"),          # other GitHub tokens
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),                 # OpenAI-style secret key
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),        # Slack token
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                    # AWS access key id
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{16,}\b"),  # bearer header value
)


class PatternRedactRule(Rule):
    """Mask substrings matching credential-shaped regular expressions.

    Defense in depth for secrets the user never registered (a token pasted mid
    sentence, a bearer header copied from a curl command). ``patterns`` defaults
    to a conservative built-in set; pass your own to extend or replace it.
    """

    name = "redact_patterns"

    def __init__(self, patterns: Optional[Sequence[Pattern[str]]] = None, *, mask: str = REDACTED) -> None:
        self._patterns = tuple(patterns) if patterns is not None else _DEFAULT_TOKEN_PATTERNS
        self._mask = mask

    def apply(self, text: str) -> RuleResult:
        if not text:
            return RuleResult.allow(text, rule=self.name)
        hits = 0
        out = text
        for pattern in self._patterns:
            out, n = pattern.subn(self._mask, out)
            hits += n
        if hits:
            return RuleResult.redact(
                out, rule=self.name, reason="masked token-shaped strings", hits=hits
            )
        return RuleResult.allow(text, rule=self.name)


class BlockPatternRule(Rule):
    """Refuse to send text that matches any forbidden pattern.

    Use for hard policy lines ("never let this classification leave"). A match
    causes the pipeline to raise :class:`OutboundBlocked` so the turn stops
    instead of quietly transmitting.
    """

    name = "block_patterns"

    def __init__(self, patterns: Sequence[Pattern[str]], *, reason: str = "forbidden content") -> None:
        if not patterns:
            raise ValueError("BlockPatternRule needs at least one pattern")
        self._patterns = tuple(patterns)
        self._reason = reason

    def apply(self, text: str) -> RuleResult:
        for pattern in self._patterns:
            match = pattern.search(text)
            if match:
                return RuleResult.block(
                    text, rule=self.name,
                    reason=f"{self._reason} (matched /{pattern.pattern}/)",
                )
        return RuleResult.allow(text, rule=self.name)


class MaxLengthRule(Rule):
    """Guard against oversized outbound payloads by truncating (or blocking)."""

    name = "max_length"

    def __init__(self, limit: int, *, block: bool = False) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self._limit = limit
        self._block = block

    def apply(self, text: str) -> RuleResult:
        if len(text) <= self._limit:
            return RuleResult.allow(text, rule=self.name)
        if self._block:
            return RuleResult.block(
                text, rule=self.name,
                reason=f"outbound text exceeds {self._limit} chars",
            )
        truncated = text[: self._limit] + "\n[...truncated by max_length rule...]"
        return RuleResult.redact(
            truncated, rule=self.name, reason=f"truncated to {self._limit} chars", hits=1
        )


# A rule is either a Rule instance or any callable(str) -> RuleResult.
RuleLike = Callable[[str], RuleResult]
