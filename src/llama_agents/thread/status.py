from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

Status = Literal["queued", "processing", "done", "failed"]
_VALID: tuple[Status, ...] = ("queued", "processing", "done", "failed")


def read_status(turn_dir: Path) -> str | None:
    """Return the status string, or None if the status file doesn't exist."""
    p = turn_dir / "status"
    if not p.is_file():
        return None
    return p.read_text(encoding="utf-8").strip()


def set_status(turn_dir: Path, status: Status) -> None:
    """Atomically write the status file. Raises ValueError on bad value."""
    if status not in _VALID:
        raise ValueError(f"invalid status: {status!r}; expected one of {_VALID}")
    turn_dir.mkdir(parents=True, exist_ok=True)
    p = turn_dir / "status"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(status + "\n", encoding="utf-8")
    os.replace(tmp, p)


def claim_for_processing(turn_dir: Path) -> bool:
    """Flip queued → processing atomically.

    Returns True iff the prior status was exactly 'queued' and the flip
    succeeded. Returns False if the status was anything else or absent.
    Not strictly atomic against another writer because we read before
    writing — but in a single-worker deployment (the only supported
    deployment) this is safe.
    """
    if read_status(turn_dir) != "queued":
        return False
    set_status(turn_dir, "processing")
    return True


def revert_processing_on_startup(threads_root: Path) -> int:
    """Walk threads/*/turns/*/status; revert any 'processing' to 'queued'.

    Returns the number of turns reverted. Safe to call on a missing root
    (returns 0).
    """
    if not threads_root.is_dir():
        return 0
    n = 0
    for thread_dir in threads_root.iterdir():
        if not thread_dir.is_dir():
            continue
        turns_dir = thread_dir / "turns"
        if not turns_dir.is_dir():
            continue
        for turn_dir in turns_dir.iterdir():
            if not turn_dir.is_dir():
                continue
            if read_status(turn_dir) == "processing":
                set_status(turn_dir, "queued")
                n += 1
    return n
