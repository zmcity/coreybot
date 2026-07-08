"""Root-level pytest configuration.

Fixtures live in ``tests/conftest.py``. Colocated builtin tests (under
``coreybot/tools/builtin/*/tests``) need a couple of them -- most importantly
``local_tmp_path`` -- so we re-export the shared fixtures here at the repo root,
where pytest can discover them for the entire tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is importable so ``import tests.conftest`` works when
# pytest is launched from any working directory.
sys.path.insert(0, str(Path(__file__).parent))

# Some earlier workspace-restore tests created real scratch dirs under the
# repo (``ws_restore_<hex>``). On this machine the security software can lock
# such dirs so they are later un-listable (``scandir`` -> PermissionError) and
# CANNOT be deleted -- which would abort pytest collection. Skip any such
# leaked artifact dir so a stray one never breaks the suite. (Current tests
# monkeypatch the filesystem and never create these.)
collect_ignore_glob = ["ws_restore_*", "tests/ws_restore_*"]

from tests.conftest import (  # noqa: E402,F401
    fake_stream_provider,
    local_tmp_path,
    recording_async_json,
    recording_async_sse,
    recording_json,
    recording_sse,
    sample_messages,
)
