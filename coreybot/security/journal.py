"""Transactional workspace snapshots -- the recovery half of YOLO safety.

Before a recoverable-but-side-effecting tool runs, the journal captures enough
to *undo* it: the original bytes of every file the tool will modify, and a
tombstone for every target that does not yet exist (so "created a file" rolls
back to "delete it"). This capture is written into the agent\'s session snapshot
``artifacts`` under the exact shape ``restore_workspace`` already consumes::

    artifacts = {
        "files":   {abs_path: original_text, ...},   # restore these on rollback
        "deleted": [abs_path, ...],                    # remove these on rollback
    }

Because the capture lands in the committed session node, *any* later checkout of
that node re-applies it -- so an operator can rewind the workspace to any point
in the git-like session tree, exactly like rewinding the conversation.

Design notes:
- The journal never executes tools and never mutates the live workspace; it only
  *reads* originals. Applying a rollback is ``Agent.restore_workspace``.
- Binary or oversized files are skipped from content capture (a size guard keeps
  ``artifacts`` bounded); such a path is still recorded so the policy can refuse
  to treat the call as fully recoverable. See :meth:`WorkspaceJournal.snapshot`.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# Default cap on a single captured file (1 MiB). Larger files are not inlined
# into the session rollout to keep it small; callers learn via ``skipped``.
DEFAULT_MAX_CAPTURE_BYTES = 1024 * 1024


@dataclass
class SnapshotResult:
    """Outcome of snapshotting the paths a tool is about to modify.

    - ``files``: ``{path: original_text}`` for existing, capturable files.
    - ``deleted``: paths that do not exist yet (rollback = delete them).
    - ``skipped``: paths that exist but were not captured (binary/oversized/
      unreadable); their presence means the call is NOT fully recoverable.
    """

    files: Dict[str, str] = field(default_factory=dict)
    deleted: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)

    @property
    def fully_capturable(self) -> bool:
        """True when every affected path was captured or is a clean tombstone."""
        return not self.skipped

    def as_artifacts(self) -> Dict[str, object]:
        """Render into the ``Snapshot.artifacts`` shape (files + deleted)."""
        return {"files": dict(self.files), "deleted": list(self.deleted)}


class WorkspaceJournal:
    """Captures pre-execution originals so a tool call can be rolled back."""

    def __init__(self, *, max_capture_bytes: int = DEFAULT_MAX_CAPTURE_BYTES) -> None:
        self._max = max_capture_bytes

    def snapshot(self, paths: Sequence[str]) -> SnapshotResult:
        """Capture originals/tombstones for ``paths`` (never raises).

        For each path: if it exists and is a readable, small, UTF-8-decodable
        file, store its text; if it does not exist, record a tombstone; if it
        exists but cannot be safely captured (binary, too large, unreadable),
        record it as skipped so the policy can downgrade recoverability.
        """
        result = SnapshotResult()
        seen = set()
        for raw in paths:
            path = os.path.abspath(str(raw))
            if path in seen:
                continue
            seen.add(path)
            if not os.path.exists(path):
                result.deleted.append(path)
                continue
            if not os.path.isfile(path):
                # Directories/special files are not something we can restore by
                # content; treat as skipped (not fully recoverable).
                result.skipped.append(path)
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                result.skipped.append(path)
                continue
            if size > self._max:
                result.skipped.append(path)
                continue
            try:
                data = io.open(path, "rb").read()
                text = data.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                result.skipped.append(path)
                continue
            result.files[path] = text
        return result

    @staticmethod
    def merge_artifacts(
        base: Optional[Dict[str, object]], addition: Dict[str, object]
    ) -> Dict[str, object]:
        """Merge a new capture into an existing artifacts bag.

        Earlier originals win: if a file was already captured earlier in the
        turn, its *first* (oldest) snapshot is what a rollback should restore, so
        we do not overwrite it with a later, already-modified version.
        """
        merged_files: Dict[str, str] = {}
        merged_deleted: List[str] = []
        if base:
            merged_files.update(base.get("files") or {})  # type: ignore[arg-type]
            merged_deleted.extend(base.get("deleted") or [])  # type: ignore[arg-type]
        for path, text in (addition.get("files") or {}).items():  # type: ignore[union-attr]
            merged_files.setdefault(path, text)
        for path in (addition.get("deleted") or []):  # type: ignore[union-attr]
            if path not in merged_deleted:
                merged_deleted.append(path)
        return {"files": merged_files, "deleted": merged_deleted}
