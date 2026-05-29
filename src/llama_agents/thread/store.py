from __future__ import annotations

from pathlib import Path

from .ids import mint_thread_id
from .meta import ThreadMeta, read_meta, update_meta, write_meta, _now_iso
from .status import read_status


class ThreadStore:
    """Filesystem-backed thread storage.

    Layout under root/:
        <thread_id>/meta.json
        <thread_id>/messages.jsonl
        <thread_id>/turns/<NNN>/{prompt.md, status, result.md, ...}
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    # ---------- thread lifecycle ----------

    def create_thread(
        self,
        *,
        title: str,
        parent_thread_id: str | None = None,
        parent_turn_idx: int | None = None,
    ) -> str:
        tid = mint_thread_id()
        now = _now_iso()
        meta = ThreadMeta(
            id=tid, title=title,
            created_at=now, updated_at=now,
            current_turn=1,
            parent_thread_id=parent_thread_id,
            parent_turn_idx=parent_turn_idx,
        )
        write_meta(self._root, meta)
        (self._root / tid / "turns" / "001").mkdir(parents=True, exist_ok=True)
        return tid

    def list_threads(self, limit: int | None = 20) -> list[ThreadMeta]:
        if not self._root.is_dir():
            return []
        metas: list[ThreadMeta] = []
        for d in self._root.iterdir():
            if not d.is_dir():
                continue
            try:
                metas.append(read_meta(self._root, d.name))
            except (FileNotFoundError, ValueError):
                continue
        metas.sort(key=lambda m: m.updated_at, reverse=True)
        if limit is not None:
            metas = metas[:limit]
        return metas

    # ---------- turn helpers ----------

    def turn_dir(self, thread_id: str, n: int) -> Path:
        return self._root / thread_id / "turns" / f"{n:03d}"

    def next_turn_dir(self, thread_id: str) -> tuple[Path, int]:
        """Bump current_turn, create the directory, return (path, idx)."""
        meta = read_meta(self._root, thread_id)
        new_idx = meta.current_turn + 1
        d = self.turn_dir(thread_id, new_idx)
        d.mkdir(parents=True, exist_ok=True)
        update_meta(self._root, thread_id, current_turn=new_idx)
        return d, new_idx

    def next_queued_turn(self) -> tuple[str, int] | None:
        """Find the oldest turn (across all threads) with status == queued.

        Returns (thread_id, turn_idx) or None.
        """
        if not self._root.is_dir():
            return None
        candidates: list[tuple[float, str, int]] = []
        for thread_dir in self._root.iterdir():
            if not thread_dir.is_dir():
                continue
            turns_dir = thread_dir / "turns"
            if not turns_dir.is_dir():
                continue
            for turn_dir in turns_dir.iterdir():
                if not turn_dir.is_dir() or not turn_dir.name.isdigit():
                    continue
                if read_status(turn_dir) != "queued":
                    continue
                status_file = turn_dir / "status"
                try:
                    mtime = status_file.stat().st_mtime
                except FileNotFoundError:
                    continue
                candidates.append((mtime, thread_dir.name, int(turn_dir.name)))
        if not candidates:
            return None
        candidates.sort()
        _, tid, idx = candidates[0]
        return tid, idx
