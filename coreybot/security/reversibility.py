"""Three-class reversibility model + the engine that assigns it.

Rather than a fuzzy numeric risk score, coreybot classifies every tool call by
*how recoverable its effect is* -- a concrete, actionable axis:

- :attr:`Reversibility.FULL` -- completely recoverable. The effect is confined
  to files inside the workspace that we snapshot before running, so it can be
  rolled back byte-for-byte. Read-only and restricted local work also qualify.
  Policy: run freely (YOLO).
- :attr:`Reversibility.PARTIAL` -- the action can be compensated but leaves a
  trace. A sent email can be recalled but recipients may have seen it; an
  external API write may have a reverse endpoint but the log remains. Policy:
  still run (YOLO), but *record the compensation note* for the audit trail.
- :attr:`Reversibility.NONE` -- not recoverable by any local means: formatting a
  disk, deleting paths outside the workspace, destroying coreybot's own storage
  (``~/.coreybot``), or any opaque/unknown effect. Policy: the only human gate.

The engine is a small, ordered set of rules. Each rule looks at a
:class:`ToolCallRequest` and either claims a class or abstains; the first claim
wins. **If no rule can prove the call is FULL or PARTIAL, it is NONE** -- the
conservative default, so an unclassifiable action is never treated as safe.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence

from coreybot.security.capabilities import Capability, SafetyProfile, UNKNOWN_PROFILE


class Reversibility(str, Enum):
    """How recoverable a tool call\'s effect is (the sole risk axis)."""

    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"

    def rank(self) -> int:
        """Higher = less recoverable (useful for ``max``/comparisons)."""
        return {"full": 0, "partial": 1, "none": 2}[self.value]


@dataclass
class ToolCallRequest:
    """Everything a rule needs to classify one tool invocation."""

    name: str
    arguments: Dict[str, object] = field(default_factory=dict)
    profile: SafetyProfile = UNKNOWN_PROFILE
    workspace_root: str = ""
    # Absolute-looking paths that must never be treated as recoverable
    # (coreybot\'s own home, system dirs). Touching any of these forces NONE.
    protected_roots: Sequence[str] = ()

    def paths(self) -> List[str]:
        return self.profile.paths_for(self.arguments)


# A classifier rule: return a Reversibility to claim, or None to abstain.
ReversibilityRule = Callable[[ToolCallRequest], Optional[Reversibility]]


# --- path helpers ----------------------------------------------------------
def _norm(path: str) -> str:
    try:
        return os.path.normcase(os.path.abspath(path))
    except Exception:
        return os.path.normcase(path or "")


def _is_within(path: str, root: str) -> bool:
    """True if ``path`` is inside ``root`` (both normalized)."""
    if not root:
        return False
    p, r = _norm(path), _norm(root)
    if p == r:
        return True
    return p.startswith(r.rstrip(os.sep) + os.sep)


def touches_protected(request: ToolCallRequest) -> bool:
    """True if any affected path escapes the workspace or hits a protected root."""
    for path in request.paths():
        for root in request.protected_roots:
            if _is_within(path, root):
                return True
        if request.workspace_root and not _is_within(path, request.workspace_root):
            # A write whose target is outside the workspace is not locally
            # recoverable (we only snapshot the workspace).
            return True
    return False


# --- built-in dangerous command patterns (for EXEC tools) ------------------
# These are heuristics: matching flags NONE (unrecoverable), not-matching does
# NOT prove safety (an EXEC tool with no match is still opaque -> handled by the
# default NONE fallback). Recovery, not detection, is the real safety net.
_DANGEROUS_COMMAND_PATTERNS: Sequence[re.Pattern] = (
    re.compile(r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*\s+|-[a-zA-Z]*f[a-zA-Z]*\s+).*", re.I),
    re.compile(r"\brm\s+-rf?\b", re.I),
    re.compile(r"\bmkfs(\.[a-z0-9]+)?\b", re.I),
    re.compile(r"\bdd\b.*\bof=/dev/", re.I),
    re.compile(r">\s*/dev/(sd|nvme|hd)", re.I),
    re.compile(r"\b(shutdown|reboot|halt|poweroff)\b", re.I),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.I),  # fork bomb
    re.compile(r"\b(format|del)\b.*\b[cC]:\\", re.I),               # windows fmt/del sys
    re.compile(r"\bcurl\b.*\|\s*(ba)?sh\b", re.I),                  # curl | sh
    re.compile(r"\bsudo\b", re.I),
)


def command_text(request: ToolCallRequest) -> str:
    """Best-effort extraction of a shell command string from arguments."""
    for key in ("command", "cmd", "script", "shell", "bash", "code"):
        value = request.arguments.get(key)
        if isinstance(value, str):
            return value
    return ""


def looks_dangerous_command(request: ToolCallRequest) -> bool:
    text = command_text(request)
    return bool(text) and any(p.search(text) for p in _DANGEROUS_COMMAND_PATTERNS)


# --- default rule set ------------------------------------------------------
def rule_explicit_hint(request: ToolCallRequest) -> Optional[Reversibility]:
    """Honor a tool\'s explicit ``reversible_hint`` override, if present."""
    hint = (request.profile.reversible_hint or "").lower()
    if hint in ("full", "partial", "none"):
        return Reversibility(hint)
    return None


def rule_protected_paths(request: ToolCallRequest) -> Optional[Reversibility]:
    """Anything touching protected/out-of-workspace paths is unrecoverable."""
    if touches_protected(request):
        return Reversibility.NONE
    return None


def rule_dangerous_exec(request: ToolCallRequest) -> Optional[Reversibility]:
    """EXEC tools whose command matches a destructive pattern are unrecoverable."""
    if request.profile.has(Capability.EXEC) and looks_dangerous_command(request):
        return Reversibility.NONE
    return None


def rule_destructive(request: ToolCallRequest) -> Optional[Reversibility]:
    """A destructive tool is recoverable only if all its targets are inside the
    workspace (already checked by :func:`rule_protected_paths`) *and* it declared
    those targets so we can snapshot them; otherwise it is unrecoverable."""
    if request.profile.has(Capability.DESTRUCTIVE):
        if request.paths():
            return Reversibility.FULL  # targets known + in-workspace -> snapshot
        return Reversibility.NONE      # destructive but opaque targets
    return None


def rule_external_side_effect(request: ToolCallRequest) -> Optional[Reversibility]:
    """External effects are PARTIAL when a compensation is declared, else NONE."""
    if request.profile.has(Capability.EXTERNAL_SIDE_EFFECT):
        if request.profile.compensation:
            return Reversibility.PARTIAL
        return Reversibility.NONE
    return None


def rule_fs_write(request: ToolCallRequest) -> Optional[Reversibility]:
    """In-workspace file writes are fully recoverable when their paths are known
    (so the journal can snapshot). A write with undeclared paths is opaque."""
    if request.profile.has(Capability.FS_WRITE):
        if request.paths():
            return Reversibility.FULL
        return Reversibility.NONE
    return None


def rule_read_only(request: ToolCallRequest) -> Optional[Reversibility]:
    """Pure reads / compute-only tools have no side effect to recover."""
    caps = request.profile.capabilities
    side_effecting = {
        Capability.FS_WRITE, Capability.EXEC, Capability.DESTRUCTIVE,
        Capability.EXTERNAL_SIDE_EFFECT,
    }
    if caps and not (caps & side_effecting):
        # Only reads and/or network GET-style capabilities remain.
        return Reversibility.FULL
    return None


DEFAULT_RULES: Sequence[ReversibilityRule] = (
    rule_explicit_hint,
    rule_protected_paths,
    rule_dangerous_exec,
    rule_destructive,
    rule_external_side_effect,
    rule_fs_write,
    rule_read_only,
)


class RiskEngine:
    """Assigns a :class:`Reversibility` to a tool call via ordered rules.

    The first rule that claims a class wins. If every rule abstains -- which
    includes any tool that declared no capabilities (opaque/unknown) -- the
    call is classified :attr:`Reversibility.NONE`. This is the safe default:
    we never assume an unclassifiable action is recoverable.
    """

    def __init__(self, rules: Optional[Sequence[ReversibilityRule]] = None) -> None:
        self._rules: List[ReversibilityRule] = list(rules if rules is not None else DEFAULT_RULES)

    def add_rule(self, rule: ReversibilityRule, *, first: bool = False) -> "RiskEngine":
        """Insert a custom rule (``first=True`` to give it priority)."""
        if first:
            self._rules.insert(0, rule)
        else:
            self._rules.append(rule)
        return self

    def classify(self, request: ToolCallRequest) -> Reversibility:
        for rule in self._rules:
            claimed = rule(request)
            if claimed is not None:
                return claimed
        return Reversibility.NONE
