"""Path helper for the ``write_file`` safety profile (kept out of spec.py).

``spec.py`` must be logic-free (enforced by the builtin-conventions lint), so
the ``affected_paths`` callable the safety policy needs lives here and is
imported by ``spec.py`` to build the profile.
"""

from __future__ import annotations

from typing import Any, Dict, List


def affected_paths(arguments: Dict[str, Any]) -> List[str]:
    """Return the single path a ``write_file`` call will modify (for snapshot)."""
    path = arguments.get("path")
    return [str(path)] if path else []
