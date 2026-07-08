"""Capabilities + per-tool safety profile.

The safety system decides how to run a tool from *what the tool can do*, not
from a hard-coded list of tool names. A tool declares its :class:`Capability`
set and (when it touches the filesystem or has an external side effect) a small
:class:`SafetyProfile` describing how to make it recoverable:

- ``affected_paths(arguments)`` -- which files a write/destructive tool will
  touch, so the journal can snapshot them *before* execution (enabling a
  byte-exact rollback afterwards).
- ``compensation`` -- for partially reversible actions (a sent email, an
  external API write), a human-readable note on how the effect can be undone /
  what trace it leaves. Recorded in telemetry so an operator has an audit trail.

All of this is optional: a tool that declares nothing is treated as opaque and
therefore *not known to be reversible*, which the risk engine maps to the
most conservative class. Existing tools need no changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional


class Capability:
    """What a tool is able to do. Plain strings so third parties can extend.

    - ``FS_READ``     : reads files.
    - ``FS_WRITE``    : creates or modifies files (reversible if snapshotted).
    - ``NETWORK``     : performs network I/O.
    - ``EXEC``        : runs external commands / shells.
    - ``DESTRUCTIVE`` : deletes or irreversibly overwrites resources.
    - ``EXTERNAL_SIDE_EFFECT`` : causes an effect outside this process that a
      local snapshot cannot undo (send email, external API write, git push).
    """

    FS_READ = "fs_read"
    FS_WRITE = "fs_write"
    NETWORK = "network"
    EXEC = "exec"
    DESTRUCTIVE = "destructive"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"


# A function that, given the tool arguments, returns the absolute/looking paths
# the call will write to (so they can be snapshotted before execution).
AffectedPaths = Callable[[Dict[str, Any]], List[str]]


@dataclass(frozen=True)
class SafetyProfile:
    """How to run a tool safely + how to recover from it.

    Attached to a tool via its spec. Every field is optional so declaring a
    profile is incremental.

    - ``capabilities``: the capability set (see :class:`Capability`).
    - ``affected_paths``: callable(args) -> list of paths the tool will modify.
      Used by the journal to snapshot originals for rollback. Only meaningful
      for ``FS_WRITE`` / ``DESTRUCTIVE`` tools.
    - ``compensation``: note describing how a partially reversible effect can be
      undone (for ``EXTERNAL_SIDE_EFFECT`` tools). Its presence is also a signal
      to the risk engine that the action is *partially* (not un-) recoverable.
    - ``reversible_hint``: optional explicit override of reversibility handling
      for unusual tools (``"full"`` / ``"partial"`` / ``"none"``). Prefer letting
      the risk engine infer; use only when capabilities cannot express intent.
    """

    capabilities: FrozenSet[str] = frozenset()
    affected_paths: Optional[AffectedPaths] = None
    compensation: str = ""
    reversible_hint: str = ""

    def has(self, capability: str) -> bool:
        return capability in self.capabilities

    def paths_for(self, arguments: Dict[str, Any]) -> List[str]:
        """Best-effort list of paths this call will modify (never raises)."""
        if self.affected_paths is None:
            return []
        try:
            paths = self.affected_paths(dict(arguments))
        except Exception:
            return []
        return [str(p) for p in (paths or []) if p]


def make_profile(
    *capabilities: str,
    affected_paths: Optional[AffectedPaths] = None,
    compensation: str = "",
    reversible_hint: str = "",
) -> SafetyProfile:
    """Convenience builder: ``make_profile(Capability.FS_WRITE, affected_paths=fn)``."""
    return SafetyProfile(
        capabilities=frozenset(capabilities),
        affected_paths=affected_paths,
        compensation=compensation,
        reversible_hint=reversible_hint,
    )


# The profile used when a tool declares nothing. Empty capabilities => opaque,
# which the risk engine treats conservatively (NONE) rather than assuming safe.
UNKNOWN_PROFILE = SafetyProfile()
