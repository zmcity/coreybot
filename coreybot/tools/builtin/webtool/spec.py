"""Interface declaration for the ``webtool`` tool (see ``tool.py``).

The SPEC declares a :class:`SafetyProfile` with ``NETWORK`` +
``EXTERNAL_SIDE_EFFECT`` and a compensation note. The external-side-effect
capability plus the note make the risk engine classify a call as ``PARTIAL``:
it still runs without interruption, but the note is recorded for the audit
trail (a request can mutate remote state we cannot roll back locally).
Declaration-only (no logic).
"""

from __future__ import annotations

from ...base import ToolSpec
from ....security.capabilities import Capability, make_profile

SPEC = ToolSpec(
    name="webtool",
    description="Fetch a web URL over HTTP(S) and return status, headers summary, and body text.",
    parameters={
        "url": "string -- the http(s) URL to request",
        "method": "string -- optional HTTP method, GET or POST (default GET)",
        "data": "string -- optional request body to send (implies POST if method omitted)",
        "timeout": "number -- optional seconds before the request is aborted (default 20)",
    },
    safety=make_profile(
        Capability.NETWORK,
        Capability.EXTERNAL_SIDE_EFFECT,
        compensation=(
            "an HTTP request may have changed remote state; review the target "
            "endpoint and issue a compensating request if needed"
        ),
    ),
)
