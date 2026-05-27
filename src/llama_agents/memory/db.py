from __future__ import annotations

import asyncio
import json
import sqlite3
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .types import BlobMeta


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(buf: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(buf, dtype="<f4", count=dim)


class VectorDB:
    """SQLite-backed blob + chunk store with brute-force cosine search."""

    def __init__(self, path: Path, *, dim: int) -> None:
        self._path = Path(path)
        self._dim = dim
        self._lock = asyncio.Lock()
        self._conn: sqlite3.Connection | None = None

    async def init(self) -> None:
        async with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._path)
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS blobs (
                  id            TEXT PRIMARY KEY,
                  scope         TEXT NOT NULL,
                  run_id        TEXT,
                  kind          TEXT NOT NULL,
                  title         TEXT NOT NULL,
                  file_path     TEXT NOT NULL,
                  metadata_json TEXT,
                  created_at    TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_blobs_scope_run
                    ON blobs(scope, run_id);
                CREATE TABLE IF NOT EXISTS chunks (
                  id        TEXT PRIMARY KEY,
                  blob_id   TEXT NOT NULL REFERENCES blobs(id) ON DELETE CASCADE,
                  chunk_idx INTEGER NOT NULL,
                  text      TEXT NOT NULL,
                  embedding BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_blob ON chunks(blob_id);
                """
            )
            self._conn.commit()

    async def insert_blob(
        self,
        meta: BlobMeta,
        *,
        chunks: Iterable[tuple[str, list[float], str]],
    ) -> None:
        async with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "INSERT INTO blobs (id, scope, run_id, kind, title, file_path, "
                "metadata_json, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (meta.id, meta.scope, meta.run_id, meta.kind, meta.title,
                 meta.file_path, json.dumps(meta.metadata), meta.created_at),
            )
            for idx, (cid, vec, txt) in enumerate(chunks):
                if len(vec) != self._dim:
                    raise ValueError(
                        f"embedding dim mismatch: got {len(vec)}, expected {self._dim}"
                    )
                self._conn.execute(
                    "INSERT INTO chunks (id, blob_id, chunk_idx, text, embedding) "
                    "VALUES (?,?,?,?,?)",
                    (cid, meta.id, idx, txt, _pack(vec)),
                )
            self._conn.commit()

    async def delete_blob(self, blob_id: str) -> None:
        async with self._lock:
            assert self._conn is not None
            self._conn.execute("DELETE FROM blobs WHERE id = ?", (blob_id,))
            self._conn.commit()

    async def delete_blobs_for_run(self, run_id: str) -> list[str]:
        async with self._lock:
            assert self._conn is not None
            rows = self._conn.execute(
                "SELECT id, file_path FROM blobs WHERE run_id = ?", (run_id,)
            ).fetchall()
            self._conn.execute("DELETE FROM blobs WHERE run_id = ?", (run_id,))
            self._conn.commit()
            return [r[1] for r in rows]

    async def list_blobs(
        self, *, scope: str, run_id: str | None = None
    ) -> list[BlobMeta]:
        async with self._lock:
            assert self._conn is not None
            if scope == "run" and run_id is not None:
                rows = self._conn.execute(
                    "SELECT id, scope, run_id, kind, title, file_path, "
                    "metadata_json, created_at FROM blobs "
                    "WHERE scope = 'run' AND run_id = ? ORDER BY created_at",
                    (run_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, scope, run_id, kind, title, file_path, "
                    "metadata_json, created_at FROM blobs "
                    "WHERE scope = ? ORDER BY created_at",
                    (scope,),
                ).fetchall()
        return [
            BlobMeta(
                id=r[0], scope=r[1], run_id=r[2], kind=r[3], title=r[4],
                file_path=r[5],
                metadata=json.loads(r[6]) if r[6] else {},
                created_at=r[7],
            )
            for r in rows
        ]

    async def list_expired_run_ids(
        self, *, now_iso: str, retention_hours: int
    ) -> list[str]:
        if retention_hours < 0:
            return []
        now = datetime.fromisoformat(now_iso)
        cutoff = now - timedelta(hours=retention_hours)
        cutoff_iso = cutoff.isoformat()
        async with self._lock:
            assert self._conn is not None
            rows = self._conn.execute(
                "SELECT DISTINCT run_id FROM blobs "
                "WHERE scope = 'run' AND run_id IS NOT NULL "
                "AND created_at < ?",
                (cutoff_iso,),
            ).fetchall()
        return [r[0] for r in rows]

    async def search(
        self,
        query_vec: list[float],
        *,
        scope: str,
        run_id: str | None = None,
        blob_id: str | None = None,
        k: int = 5,
    ) -> list[tuple[str, str, float, str, str, str, int]]:
        """Return tuples of (chunk_id, blob_id, score, text, title, kind, chunk_idx)."""
        if len(query_vec) != self._dim:
            raise ValueError("query dim mismatch")
        async with self._lock:
            assert self._conn is not None
            sql = (
                "SELECT c.id, c.blob_id, c.text, c.embedding, c.chunk_idx, "
                "b.title, b.kind FROM chunks c JOIN blobs b ON b.id = c.blob_id "
                "WHERE 1=1"
            )
            params: list = []
            if blob_id is not None:
                sql += " AND c.blob_id = ?"
                params.append(blob_id)
            else:
                if scope == "run":
                    sql += " AND b.scope = 'run'"
                    if run_id is not None:
                        sql += " AND b.run_id = ?"
                        params.append(run_id)
                elif scope == "plans":
                    sql += " AND b.scope = 'plans'"
                elif scope == "all":
                    if run_id is not None:
                        sql += (
                            " AND (b.scope = 'plans' OR "
                            "(b.scope = 'run' AND b.run_id = ?))"
                        )
                        params.append(run_id)
            rows = self._conn.execute(sql, params).fetchall()

        if not rows:
            return []
        q = np.asarray(query_vec, dtype="<f4")
        qn = q / (np.linalg.norm(q) or 1.0)
        mat = np.stack([_unpack(r[3], self._dim) for r in rows])
        mat_n = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        scores = mat_n @ qn
        order = np.argsort(-scores)[:k]
        out: list[tuple[str, str, float, str, str, str, int]] = []
        for i in order:
            r = rows[int(i)]
            out.append(
                (r[0], r[1], float(scores[int(i)]), r[2], r[5], r[6], r[4])
            )
        return out

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
