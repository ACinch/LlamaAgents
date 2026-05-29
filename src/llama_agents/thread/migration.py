from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .ids import mint_thread_id
from .meta import ThreadMeta, write_meta, _now_iso
from .status import set_status

logger = logging.getLogger(__name__)

_LEGACY_DIRS = ("inbox", "processing", "done", "failed")
_LEGACY_TO_STATUS = {
    "inbox": "queued",
    "processing": "queued",  # crashed jobs go back to queued
    "done": "done",
    "failed": "failed",
}


def _is_primary_md(p: Path) -> bool:
    return (
        p.is_file()
        and p.suffix == ".md"
        and not p.stem.endswith(".prompt")
    )


def _migrate_one(legacy_dir: Path, md_path: Path, threads_root: Path,
                 status: str) -> bool:
    """Move one legacy job into a fresh thread. Returns True on success."""
    name = md_path.stem
    tid = mint_thread_id()
    turn1 = threads_root / tid / "turns" / "001"
    turn1.mkdir(parents=True, exist_ok=True)

    body = md_path.read_text(encoding="utf-8")
    sidecar_prompt = legacy_dir / f"{name}.prompt.md"
    sidecar_events = legacy_dir / f"{name}.events.jsonl"
    sidecar_error = legacy_dir / f"{name}.error.txt"

    # In a done/failed dir, the .md is the final answer; .prompt.md is the
    # original prompt. In inbox/processing the .md IS the prompt.
    if sidecar_prompt.is_file():
        prompt_text = sidecar_prompt.read_text(encoding="utf-8")
        result_text = body
    else:
        prompt_text = body
        result_text = ""

    (turn1 / "prompt.md").write_text(prompt_text, encoding="utf-8")
    if result_text:
        (turn1 / "result.md").write_text(result_text, encoding="utf-8")
    if sidecar_events.is_file():
        shutil.copy2(sidecar_events, turn1 / "events.jsonl")
    if sidecar_error.is_file():
        shutil.copy2(sidecar_error, turn1 / "error.txt")

    now = _now_iso()
    title = prompt_text.strip().splitlines()[0][:60] if prompt_text.strip() else name
    write_meta(threads_root, ThreadMeta(
        id=tid, title=title,
        created_at=now, updated_at=now,
        current_turn=1,
    ))
    set_status(turn1, status)

    # Remove source files
    md_path.unlink()
    for s in (sidecar_prompt, sidecar_events, sidecar_error):
        if s.is_file():
            s.unlink()
    return True


def migrate_legacy_queue_dirs(queue_root: Path) -> int:
    """One-shot migration of pre-thread inbox/processing/done/failed.

    Idempotent: re-running on an already-migrated tree finds nothing.
    Returns the count of files migrated.
    """
    queue_root = Path(queue_root)
    if not queue_root.is_dir():
        return 0

    threads_root = queue_root / "threads"
    threads_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for legacy_name in _LEGACY_DIRS:
        legacy_dir = queue_root / legacy_name
        if not legacy_dir.is_dir():
            continue
        status = _LEGACY_TO_STATUS[legacy_name]
        for md_path in list(legacy_dir.iterdir()):
            if not _is_primary_md(md_path):
                continue
            try:
                if _migrate_one(legacy_dir, md_path, threads_root, status):
                    total += 1
            except OSError as e:
                logger.warning(
                    "migrate(%s): failed to migrate %s: %s",
                    legacy_name, md_path.name, e,
                )
        # Remove the legacy folder if it's now empty
        try:
            legacy_dir.rmdir()
        except OSError:
            pass  # not empty (sidecar leftovers, partial failures)

    if total:
        logger.info("migrated %d legacy queue files into threads/", total)
    return total
