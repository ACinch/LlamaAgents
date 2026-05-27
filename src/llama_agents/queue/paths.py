from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

TerminalStatus = Literal["done", "failed"]
SUBDIRS = ("inbox", "processing", "done", "failed")


def ensure_dirs(root: Path) -> None:
    """Create the four queue subdirs under root. Idempotent."""
    root = Path(root)
    for name in SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)


def move_to_processing(root: Path, src: Path) -> Path | None:
    """Atomically move `src` (in inbox/) to processing/<same-name>.

    Returns the new path on success. Returns None if the destination
    already exists (lost race against another worker) or if the source
    is still locked by a writer (Windows PermissionError).
    """
    dst = Path(root) / "processing" / src.name
    if dst.exists():
        return None
    try:
        os.rename(src, dst)
    except FileExistsError:
        return None
    except PermissionError:
        # Windows: file is still being written, or another process holds it.
        return None
    return dst


def move_to_terminal(root: Path, src: Path, *, status: TerminalStatus) -> Path:
    """Move `src` (in processing/) into done/ or failed/.

    On name collision, appends a numeric suffix before the extension
    (e.g. `foo.md` -> `foo.1.md`, `foo.2.md`) so prior results are
    preserved.
    """
    target_dir = Path(root) / status
    target_dir.mkdir(parents=True, exist_ok=True)
    dst = target_dir / src.name
    if dst.exists():
        stem, suffix = src.stem, src.suffix
        i = 1
        while True:
            candidate = target_dir / f"{stem}.{i}{suffix}"
            if not candidate.exists():
                dst = candidate
                break
            i += 1
    os.replace(src, dst)
    return dst


def sweep_processing_to_inbox(root: Path) -> list[Path]:
    """Move every file in processing/ back to inbox/.

    Used on worker startup to recover files left behind by a prior crash
    or hard shutdown. Returns the list of new paths.
    """
    root = Path(root)
    proc = root / "processing"
    inbox = root / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    if not proc.is_dir():
        return []
    moved: list[Path] = []
    for entry in proc.iterdir():
        if not entry.is_file():
            continue
        dst = inbox / entry.name
        # If something with that name already exists in inbox (extremely
        # unlikely — only happens if a user manually re-dropped), append a
        # suffix to preserve both.
        if dst.exists():
            stem, suffix = entry.stem, entry.suffix
            i = 1
            while (inbox / f"{stem}.recovered{i}{suffix}").exists():
                i += 1
            dst = inbox / f"{stem}.recovered{i}{suffix}"
        os.replace(entry, dst)
        moved.append(dst)
    return moved
