"""Interface declaration for the ``bash`` tool (see ``tool.py``).

The SPEC also declares a :class:`SafetyProfile` marking the tool as
``EXEC`` + ``DESTRUCTIVE``: running an arbitrary shell command can do anything,
so the risk engine treats it as unrecoverable unless an approval handler
permits it (and immediately as ``NONE`` when the command text matches a known
destructive pattern such as ``rm -rf``). Declaration-only (no logic).
"""

from __future__ import annotations

from ...base import ToolSpec
from ....security.capabilities import Capability, make_profile

SPEC = ToolSpec(
    name="bash",
    description="Run a shell command and return its combined stdout/stderr and exit code.",
    parameters={
        "command": "string -- the shell command line to execute",
        "timeout": "number -- optional seconds before the command is killed (default 30)",
        "workdir": "string -- optional working directory to run the command in",
    },
    safety=make_profile(Capability.EXEC, Capability.DESTRUCTIVE),
)
