"""Compose outbound rules into a pipeline applied at the LLM boundary.

A :class:`RulePipeline` holds an ordered list of rules and runs them over a
piece of outbound text. Redactions are cumulative (each rule sees the previous
rule\'s output), and the first rule that returns :data:`Action.BLOCK` stops the
pipeline by raising :class:`OutboundBlocked`. The pipeline returns a
:class:`PipelineOutcome` recording the final text plus a per-rule audit trail,
so a UI/telemetry layer can show *what* was changed without seeing the secret.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from coreybot.security.rules import Action, RuleLike, RuleResult


class OutboundBlocked(RuntimeError):
    """Raised when a rule blocks text from being sent to the LLM.

    Carries the offending rule name and reason so the agent loop can surface a
    clear notice (and refuse the turn) instead of transmitting.
    """

    def __init__(self, rule: str, reason: str) -> None:
        super().__init__(f"outbound blocked by rule '{rule}': {reason}")
        self.rule = rule
        self.reason = reason


@dataclass
class PipelineOutcome:
    """Result of running the pipeline over one piece of text.

    - ``text``: the final, possibly-redacted text to send.
    - ``changed``: whether any rule rewrote the text.
    - ``trail``: the :class:`RuleResult` from each rule that did something
      (allow-with-no-change is omitted to keep the trail signal-rich).
    - ``total_hits``: sum of redaction substitutions across rules.
    """

    text: str
    changed: bool = False
    trail: List[RuleResult] = field(default_factory=list)
    total_hits: int = 0

    def redaction_summary(self) -> str:
        """One-line, secret-free summary of what changed (for notices)."""
        if not self.trail:
            return ""
        parts = [f"{r.rule}({r.hits})" if r.hits else r.rule for r in self.trail]
        return "redacted: " + ", ".join(parts)


class RulePipeline:
    """An ordered set of outbound rules applied at the LLM boundary."""

    def __init__(self, rules: Sequence[RuleLike] = ()) -> None:
        self._rules: List[RuleLike] = list(rules)

    def add(self, rule: RuleLike) -> "RulePipeline":
        """Append a rule (returns self so calls can be chained)."""
        self._rules.append(rule)
        return self

    def __len__(self) -> int:
        return len(self._rules)

    def __bool__(self) -> bool:
        return bool(self._rules)

    def run(self, text: str) -> PipelineOutcome:
        """Apply every rule in order; raise :class:`OutboundBlocked` on block.

        Redactions accumulate: each rule is given the output of the previous
        one. Rules that leave the text unchanged are not added to the trail.
        """
        current = text
        outcome = PipelineOutcome(text=text)
        for rule in self._rules:
            result = rule(current)
            name = result.rule or getattr(rule, "name", "rule")
            if result.action == Action.BLOCK:
                # Record nothing sensitive; just stop the send.
                raise OutboundBlocked(name, result.reason or "blocked")
            if result.action == Action.REDACT and result.text != current:
                current = result.text
                outcome.changed = True
                outcome.total_hits += result.hits
                outcome.trail.append(result)
            # ALLOW (or a redact that changed nothing): carry on unchanged.
        outcome.text = current
        return outcome

    def run_text(self, text: str) -> str:
        """Convenience: run the pipeline and return only the final text."""
        return self.run(text).text
