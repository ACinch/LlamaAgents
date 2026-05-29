from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ThreadMeta:
    id: str
    title: str
    created_at: str
    updated_at: str
    current_turn: int
    parent_thread_id: str | None = None
    parent_turn_idx: int | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _meta_path(threads_root: Path, thread_id: str) -> Path:
    return threads_root / thread_id / "meta.json"


def write_meta(threads_root: Path, meta: ThreadMeta) -> None:
    """Write meta.json atomically (tmp + replace) inside the thread folder."""
    p = _meta_path(threads_root, meta.id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(meta), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def read_meta(threads_root: Path, thread_id: str) -> ThreadMeta:
    p = _meta_path(threads_root, thread_id)
    if not p.is_file():
        raise FileNotFoundError(f"meta.json missing for thread {thread_id}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"meta.json for {thread_id} is malformed: {e}") from e
    return ThreadMeta(**data)


def update_meta(threads_root: Path, thread_id: str, **fields) -> ThreadMeta:
    """Read meta, apply field updates, bump updated_at, write back, return.

    Caller-supplied updated_at is overridden with the current time. Created_at
    and id are never overwritten by the field dict.
    """
    current = read_meta(threads_root, thread_id)
    data = asdict(current)
    for k, v in fields.items():
        if k in ("id", "created_at"):
            continue
        data[k] = v
    data["updated_at"] = _now_iso()
    new = ThreadMeta(**data)
    write_meta(threads_root, new)
    return new
