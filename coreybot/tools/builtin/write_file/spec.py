"""Interface declaration for the ``write_file`` tool (see ``tool.py``).

Besides the model-facing contract, this SPEC carries a :class:`SafetyProfile`
so the safety policy knows the tool writes files and *which* file, enabling a
pre-execution snapshot and byte-exact rollback. Declaration-only (no logic):
the ``affected_paths`` callable is imported from ``paths.py``.
"""

from __future__ import annotations

from ...base import ToolSpec
from ....security.capabilities import Capability, make_profile
from .paths import affected_paths

SPEC = ToolSpec(
    name="write_file",
    description="Write UTF-8 text to a file, creating or overwriting it.",
    parameters={
        "path": "string -- path to the file to write",
        "content": "string -- the full text to write into the file",
    },
    safety=make_profile(Capability.FS_WRITE, affected_paths=affected_paths),
)
