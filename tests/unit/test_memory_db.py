import asyncio
import math
from pathlib import Path

import pytest

from llama_agents.memory.db import VectorDB
from llama_agents.memory.types import BlobMeta


def _vec(*xs: float) -> list[float]:
    n = math.sqrt(sum(x * x for x in xs)) or 1.0
    return [x / n for x in xs]


@pytest.mark.asyncio
async def test_insert_and_search_returns_top_k(tmp_path: Path):
    db = VectorDB(tmp_path / "i.sqlite", dim=3)
    await db.init()

    await db.insert_blob(
        BlobMeta(id="b1", scope="run", thread_id="r1", kind="user",
                 title="t1", file_path=str(tmp_path / "b1.md"),
                 created_at="2026-05-27T00:00:00"),
        chunks=[("c1a", _vec(1, 0, 0), "alpha"),
                ("c1b", _vec(0, 1, 0), "beta")],
    )
    await db.insert_blob(
        BlobMeta(id="b2", scope="plans", thread_id=None, kind="plan",
                 title="t2", file_path=str(tmp_path / "b2.md"),
                 created_at="2026-05-27T00:00:01"),
        chunks=[("c2a", _vec(0, 0, 1), "gamma")],
    )

    hits = await db.search(_vec(1, 0, 0), scope="all", thread_ids=["r1"], k=2)
    assert hits[0][0] == "c1a"
    assert hits[0][2] > 0.99

    hits_plans = await db.search(_vec(0, 0, 1), scope="plans", k=5)
    assert {h[0] for h in hits_plans} == {"c2a"}

    hits_b1 = await db.search(_vec(0, 1, 0), scope="all", blob_id="b1", k=5)
    assert {h[0] for h in hits_b1} == {"c1a", "c1b"}
    assert hits_b1[0][0] == "c1b"

    await db.close()


@pytest.mark.asyncio
async def test_blob_delete_cascades_chunks(tmp_path: Path):
    db = VectorDB(tmp_path / "i.sqlite", dim=2)
    await db.init()
    await db.insert_blob(
        BlobMeta(id="b1", scope="run", thread_id="r1", kind="user",
                 title="t", file_path=str(tmp_path / "b1.md"),
                 created_at="2026-05-27T00:00:00"),
        chunks=[("c", _vec(1, 0), "x")],
    )
    await db.delete_blob("b1")
    assert await db.search(_vec(1, 0), scope="all", k=5) == []
    await db.close()


@pytest.mark.asyncio
async def test_list_blobs_filters_by_scope_and_run(tmp_path: Path):
    db = VectorDB(tmp_path / "i.sqlite", dim=2)
    await db.init()
    for i, scope, thread_id in [
        ("a", "run", "r1"), ("b", "run", "r2"), ("c", "plans", None)
    ]:
        await db.insert_blob(
            BlobMeta(id=i, scope=scope, thread_id=thread_id, kind="x",
                     title="t", file_path=str(tmp_path / f"{i}.md"),
                     created_at="2026-05-27T00:00:00"),
            chunks=[("c" + i, _vec(1, 0), "x")],
        )

    r1 = [m.id for m in await db.list_blobs(scope="run", thread_id="r1")]
    assert r1 == ["a"]
    plans = [m.id for m in await db.list_blobs(scope="plans")]
    assert plans == ["c"]
    await db.close()


@pytest.mark.asyncio
async def test_list_expired_run_ids(tmp_path: Path):
    db = VectorDB(tmp_path / "i.sqlite", dim=2)
    await db.init()
    await db.insert_blob(
        BlobMeta(id="old", scope="run", thread_id="r_old", kind="x",
                 title="t", file_path=str(tmp_path / "old.md"),
                 created_at="2026-01-01T00:00:00"),
        chunks=[("c", _vec(1, 0), "x")],
    )
    await db.insert_blob(
        BlobMeta(id="new", scope="run", thread_id="r_new", kind="x",
                 title="t", file_path=str(tmp_path / "new.md"),
                 created_at="2099-01-01T00:00:00"),
        chunks=[("c2", _vec(0, 1), "y")],
    )
    expired = await db.list_expired_thread_ids(now_iso="2026-05-27T00:00:00",
                                               retention_hours=24)
    assert expired == ["r_old"]
    await db.close()


@pytest.mark.asyncio
async def test_schema_migration_renames_run_id_to_thread_id(tmp_path: Path):
    """Pre-create a database with the old run_id schema, then open it under
    the new code and verify the migration happened."""
    import sqlite3
    db_path = tmp_path / "i.sqlite"
    # Write the old-schema bytes by hand.
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE blobs (
          id            TEXT PRIMARY KEY,
          scope         TEXT NOT NULL,
          run_id        TEXT,
          kind          TEXT NOT NULL,
          title         TEXT NOT NULL,
          file_path     TEXT NOT NULL,
          metadata_json TEXT,
          created_at    TEXT NOT NULL
        );
        CREATE INDEX idx_blobs_scope_run ON blobs(scope, run_id);
        CREATE TABLE chunks (
          id        TEXT PRIMARY KEY,
          blob_id   TEXT NOT NULL REFERENCES blobs(id) ON DELETE CASCADE,
          chunk_idx INTEGER NOT NULL,
          text      TEXT NOT NULL,
          embedding BLOB NOT NULL
        );
        CREATE INDEX idx_chunks_blob ON chunks(blob_id);
        PRAGMA user_version = 0;
    """)
    conn.execute(
        "INSERT INTO blobs VALUES (?,?,?,?,?,?,?,?)",
        ("b1", "run", "old_run_id_value", "user", "t",
         str(tmp_path / "b1.md"), "{}", "2026-05-29T00:00:00"),
    )
    conn.commit()
    conn.close()

    db = VectorDB(db_path, dim=3)
    await db.init()

    # After init the column should be 'thread_id' and user_version = 1.
    raw = sqlite3.connect(db_path)
    cols = [r[1] for r in raw.execute("PRAGMA table_info(blobs)").fetchall()]
    assert "thread_id" in cols
    assert "run_id" not in cols
    ver = raw.execute("PRAGMA user_version").fetchone()[0]
    assert ver == 1
    # The existing row's data survived
    row = raw.execute("SELECT id, thread_id FROM blobs WHERE id = ?",
                      ("b1",)).fetchone()
    assert row == ("b1", "old_run_id_value")
    raw.close()
    await db.close()


@pytest.mark.asyncio
async def test_fresh_database_uses_thread_id_column(tmp_path: Path):
    db = VectorDB(tmp_path / "i.sqlite", dim=3)
    await db.init()
    import sqlite3
    raw = sqlite3.connect(tmp_path / "i.sqlite")
    cols = [r[1] for r in raw.execute("PRAGMA table_info(blobs)").fetchall()]
    assert "thread_id" in cols
    assert "run_id" not in cols
    ver = raw.execute("PRAGMA user_version").fetchone()[0]
    assert ver == 1
    raw.close()
    await db.close()
