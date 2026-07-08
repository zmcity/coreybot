"""The ``webtool`` builtin tool (fetch a web resource over HTTP).

    from coreybot.tools.builtin.webtool import webtool, SPEC

Today this performs a plain HTTP request (GET/POST/...) via the standard
library. A network call is treated by the safety policy as *partially*
reversible: it runs freely (YOLO) but its compensation note is recorded, since
a request may have changed remote state that a local snapshot cannot undo.
Headless-browser rendering can be added later behind the same interface.
"""

from __future__ import annotations

from .spec import SPEC
from .tool import webtool

__all__ = ["SPEC", "webtool"]
