from __future__ import annotations

import uuid
from pathlib import Path


_HEX = set("0123456789abcdef")


class AmbiguousPrefix(LookupError):
    """The given prefix matches multiple thread ids."""


class UnknownPrefix(LookupError):
    """No thread id matches the given prefix."""


def mint_thread_id() -> str:
    """Return a fresh 24-char lowercase hex id (uuid4 first 24 chars)."""
    return uuid.uuid4().hex[:24]


def validate_thread_id(s: str) -> bool:
    """True iff s is a syntactically valid thread id (24 lowercase hex chars)."""
    return len(s) == 24 and all(c in _HEX for c in s)


def resolve_prefix(threads_root: Path, prefix: str) -> str:
    """Resolve a thread-id prefix (>=4 chars) to a single full id.

    Raises ValueError if prefix is shorter than 4 chars.
    Raises UnknownPrefix if no thread directory matches.
    Raises AmbiguousPrefix if more than one matches; the exception message
    contains the matching ids so the caller can offer disambiguation.

    A full-length valid id is passed through if a directory with that name
    exists, otherwise the usual unknown/ambiguous rules apply.
    """
    if len(prefix) < 4:
        raise ValueError("thread-id prefix must be at least 4 characters")
    if not threads_root.is_dir():
        raise UnknownPrefix(f"no thread matches prefix {prefix!r}")
    matches = sorted(
        p.name for p in threads_root.iterdir()
        if p.is_dir() and validate_thread_id(p.name) and p.name.startswith(prefix)
    )
    if not matches:
        raise UnknownPrefix(f"no thread matches prefix {prefix!r}")
    if len(matches) > 1:
        raise AmbiguousPrefix(
            f"prefix {prefix!r} matches {len(matches)} threads: "
            + ", ".join(matches)
        )
    return matches[0]
