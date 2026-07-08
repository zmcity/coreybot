"""Leak-resistant storage for secrets (access tokens, PATs, scopes, env vars).

Enterprise agents routinely need to *hold* credentials during a session --
an access token to call an API on the user\'s behalf, a PAT, an OAuth scope
list, or a plain environment variable. The danger is not holding them; it is
that they trivially leak: into a log line, a traceback frame, the telemetry
stream, or (worst of all) the text sent to the LLM.

This module makes leaking hard *by default*:

- :class:`SecretValue` wraps a string whose ``repr()``/``str()`` never reveal
  the value. Printing it, logging it, or interpolating it into an f-string all
  yield a masked placeholder like ``Secret(github_pat, ****)``. The only way to
  read the real value is the explicit :meth:`SecretValue.reveal` call, which
  makes every intentional use greppable in code review.
- :class:`SecretStore` is a named collection of secrets with a category
  (``token`` / ``pat`` / ``access_token`` / ``scope`` / ``env``). It, too, has a
  masked ``repr``, and can enumerate the raw values *only* through an explicit
  method used by the outbound redactor (defense in depth).

Nothing here has third-party dependencies, and secrets are intentionally kept
out of every serializable structure (snapshots, telemetry) elsewhere in the
codebase -- see :mod:`coreybot.security.context`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple


# Recognized secret categories. Kept as plain strings (not an Enum) so callers
# and future integrations can introduce their own without editing this file.
class SecretKind:
    TOKEN = "token"
    PAT = "pat"
    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    SCOPE = "scope"
    ENV = "env"
    GENERIC = "generic"


_MASK = "****"


def _mask(value: str, *, keep: int = 0) -> str:
    """Return a masked form of ``value`` that reveals at most ``keep`` chars.

    ``keep`` shows a short non-sensitive prefix (useful for a UI hint such as
    ``ghp_...``) while never exposing enough to be usable. ``keep`` is clamped
    so we never reveal more than a quarter of a short secret.
    """
    if not value:
        return _MASK
    limit = max(0, min(keep, len(value) // 4))
    if limit <= 0:
        return _MASK
    return value[:limit] + _MASK


class SecretValue:
    """A string credential that resists accidental disclosure.

    The raw value is stored in a private slot and is deliberately excluded from
    ``repr``/``str``/formatting. Use :meth:`reveal` to obtain it on purpose.
    """

    __slots__ = ("_value", "name", "kind")

    def __init__(self, value: str, *, name: str = "secret", kind: str = SecretKind.GENERIC) -> None:
        if not isinstance(value, str):
            raise TypeError("SecretValue requires a str value")
        self._value = value
        self.name = name
        self.kind = kind

    def reveal(self) -> str:
        """Return the real secret value. Every call site should be reviewable."""
        return self._value

    def masked(self, *, keep: int = 0) -> str:
        """Return a safe, masked rendering (optionally keeping a short prefix)."""
        return _mask(self._value, keep=keep)

    def __len__(self) -> int:
        return len(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)

    def __eq__(self, other: object) -> bool:
        # Compare by secret value so lookups/dedup work, but only against other
        # SecretValues -- never against a plain str, to avoid a subtle path
        # where an attacker-controlled string is tested for equality.
        if isinstance(other, SecretValue):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("SecretValue", self._value))

    def __repr__(self) -> str:
        return f"Secret({self.name}, {self.masked()})"

    __str__ = __repr__

    # Guard against the most common accidental-leak vector: str.format / f-string
    # conversion. ``format(secret)`` returns the masked form, not the value.
    def __format__(self, spec: str) -> str:
        return self.masked()


@dataclass(frozen=True)
class SecretRef:
    """A non-sensitive descriptor of a stored secret (safe to log/show).

    Carries the ``name`` and ``kind`` plus a masked ``hint`` so a UI or audit
    log can reference a credential without ever touching its value.
    """

    name: str
    kind: str
    hint: str


class SecretStore:
    """A named, category-tagged collection of :class:`SecretValue` objects.

    Designed to be carried on a :class:`~coreybot.security.context.SecurityContext`
    for the whole session. It never renders its values and is never serialized.
    """

    def __init__(self) -> None:
        self._items: Dict[str, SecretValue] = {}

    # --- population --------------------------------------------------------
    def put(self, name: str, value: str, *, kind: str = SecretKind.GENERIC) -> SecretRef:
        """Store (or replace) a secret under ``name``; returns a safe ref."""
        if not name:
            raise ValueError("secret name must be non-empty")
        secret = SecretValue(value, name=name, kind=kind)
        self._items[name] = secret
        return self.ref(name)

    def put_secret(self, secret: SecretValue) -> SecretRef:
        """Store an already-wrapped :class:`SecretValue`."""
        if not secret.name:
            raise ValueError("SecretValue must have a name to be stored")
        self._items[secret.name] = secret
        return self.ref(secret.name)

    @classmethod
    def from_env(
        cls, names: Iterable[str], *, kind: str = SecretKind.ENV,
        environ: Optional[Mapping[str, str]] = None, required: bool = False,
    ) -> "SecretStore":
        """Build a store by reading environment variables by name.

        Missing variables are skipped unless ``required`` is set, in which case
        a :class:`KeyError` names the first missing one. The environment is read
        from ``environ`` (defaults to ``os.environ``) so tests need no globals.
        """
        env = os.environ if environ is None else environ
        store = cls()
        for name in names:
            if name in env:
                store.put(name, env[name], kind=kind)
            elif required:
                raise KeyError(f"required secret env var not set: {name}")
        return store

    # --- access ------------------------------------------------------------
    def get(self, name: str) -> Optional[SecretValue]:
        """Return the wrapped secret (still leak-resistant), or ``None``."""
        return self._items.get(name)

    def reveal(self, name: str) -> str:
        """Explicitly reveal a stored secret\'s value; raises if absent."""
        secret = self._items.get(name)
        if secret is None:
            raise KeyError(f"no secret named {name!r}")
        return secret.reveal()

    def ref(self, name: str) -> SecretRef:
        secret = self._items.get(name)
        if secret is None:
            raise KeyError(f"no secret named {name!r}")
        return SecretRef(name=secret.name, kind=secret.kind, hint=secret.masked(keep=4))

    def refs(self) -> List[SecretRef]:
        """Safe descriptors for every secret (for audit/UI). No values."""
        return [self.ref(name) for name in sorted(self._items)]

    def names(self) -> List[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    # --- redaction support -------------------------------------------------
    def secret_values(self) -> Iterator[Tuple[str, str]]:
        """Yield ``(name, raw_value)`` pairs for the outbound redactor ONLY.

        This is the single method that exposes every raw value at once; it exists
        so :mod:`coreybot.security.rules` can mask any secret that appears in text
        bound for the LLM. Keep call sites to the security pipeline.
        """
        for name, secret in self._items.items():
            raw = secret.reveal()
            if raw:
                yield name, raw

    def __repr__(self) -> str:
        return f"SecretStore({len(self._items)} secrets: {self.names()})"

    __str__ = __repr__
