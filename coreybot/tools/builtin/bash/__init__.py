"""The ``bash`` builtin tool (run a shell command).

    from coreybot.tools.builtin.bash import run_bash, SPEC

This tool executes real shell commands and is therefore governed by the
safety policy: a command that matches a destructive pattern -- or any command
at all when no approval handler is installed -- is classified unrecoverable
(``NONE``) and refused. See ``spec.py`` for the declared capabilities.
"""

from __future__ import annotations

from .spec import SPEC
from .tool import run_bash

__all__ = ["SPEC", "run_bash"]
