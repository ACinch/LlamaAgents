from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunker import chunk_markdown
from .db import VectorDB
from .types import BlobMeta, Embedder, RecalledChunk


def _new_id() -> str:
    return uuid.uuid4().hex[:24]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    def __init__(
        self,
        *,
        root: Path,
        embedder: Embedder,
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        retention_hours: int = 24,
    ) -> None:
        self._root = Path(root)
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._retention_hours = retention_hours
        self._db = VectorDB(self._root / "index.sqlite", dim=embedder.dim)
        self._active_runs: set[str] = set()

    async def init(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / "runs").mkdir(exist_ok=True)
        (self._root / "plans").mkdir(exist_ok=True)
        await self._db.init()

    def start_run(self, run_id: str) -> None:
        self._active_runs.add(run_id)

    async def end_run(self, run_id: str) -> None:
        self._active_runs.discard(run_id)
        if self._retention_hours == 0:
            await self._purge_run(run_id)
        elif self._retention_hours > 0:
            await self.gc_expired()

    async def gc_expired(self) -> int:
        expired = await self._db.list_expired_run_ids(
            now_iso=_now_iso(), retention_hours=self._retention_hours
        )
        for rid in expired:
            await self._purge_run(rid)
        return len(expired)

    async def _purge_run(self, run_id: str) -> None:
        await self._db.delete_blobs_for_run(run_id)
        rd = self._root / "runs" / run_id
        if rd.exists():
            shutil.rmtree(rd, ignore_errors=True)

    async def store_blob(
        self,
        *,
        kind: str,
        title: str,
        body: str,
        scope: str = "run",
        run_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        blob_id = _new_id()
        if scope == "run":
            assert run_id, "run_id required for scope='run'"
            dir_ = self._root / "runs" / run_id
        elif scope == "plans":
            dir_ = self._root / "plans"
        else:
            raise ValueError(f"bad scope: {scope}")
        dir_.mkdir(parents=True, exist_ok=True)
        fp = dir_ / f"{blob_id}.md"
        fp.write_text(body, encoding="utf-8")

        chunks = chunk_markdown(
            body, chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap
        )
        if not chunks:
            chunks = [body.strip() or "[empty]"]
        vecs = await self._embedder.embed(chunks)
        meta = BlobMeta(
            id=blob_id, scope=scope, run_id=run_id if scope == "run" else None,
            kind=kind, title=title, file_path=str(fp),
            metadata=metadata or {}, created_at=_now_iso(),
        )
        await self._db.insert_blob(
            meta,
            chunks=[(_new_id(), v, t) for v, t in zip(vecs, chunks)],
        )
        return blob_id

    async def store_plan(
        self, *, task: str, plan: str, accepted_attempt: int,
        run_id: str | None = None,
    ) -> str:
        body = (
            f"# Plan for: {task[:80]}\n\n"
            f"## Task\n{task}\n\n"
            f"## Accepted on attempt {accepted_attempt}\n\n"
            f"## Plan\n{plan}\n"
        )
        return await self.store_blob(
            kind="plan", scope="plans",
            title=task[:80], body=body,
            metadata={"task": task, "attempt": accepted_attempt,
                      "run_id": run_id},
        )

    async def recall(
        self,
        query: str,
        *,
        scope: str = "all",
        run_id: str | None = None,
        handle: str | None = None,
        k: int = 5,
        min_score: float | None = None,
    ) -> list[RecalledChunk]:
        [qvec] = await self._embedder.embed([query])
        hits = await self._db.search(
            qvec, scope=scope, run_id=run_id, blob_id=handle, k=k
        )
        out: list[RecalledChunk] = []
        for chunk_id, blob_id, score, text, title, kind, chunk_idx in hits:
            if min_score is not None and score < min_score:
                continue
            out.append(RecalledChunk(
                blob_id=blob_id, chunk_idx=chunk_idx, text=text,
                score=score, title=title, kind=kind,
            ))
        return out

    async def list_handles(
        self, *, scope: str, run_id: str | None = None
    ) -> list[BlobMeta]:
        return await self._db.list_blobs(scope=scope, run_id=run_id)

    async def close(self) -> None:
        await self._db.close()


class InertMemoryStore:
    """No-op store used when memory.enabled = false."""

    async def init(self) -> None: ...
    def start_run(self, run_id: str) -> None: ...
    async def end_run(self, run_id: str) -> None: ...
    async def gc_expired(self) -> int:
        return 0

    async def store_blob(self, **_: Any) -> str:
        return ""

    async def store_plan(self, **_: Any) -> str:
        return ""

    async def recall(self, *_: Any, **__: Any) -> list[RecalledChunk]:
        return []

    async def list_handles(self, **_: Any) -> list[BlobMeta]:
        return []

    async def close(self) -> None: ...
