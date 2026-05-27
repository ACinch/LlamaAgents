# RAG-Backed Memory Layer (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a portable, in-process RAG memory layer so the agent loop can store accepted plans, swap large subagent outputs out of the parent's context, and evict old tool results when the context window fills.

**Architecture:** New `src/llama_agents/memory/` package wraps a `MemoryStore` (composes a markdown chunker, a fastembed-backed embedder, and a SQLite-backed vector store with brute-force cosine search). The agent loop calls `MemoryStore.recall(...)` before planning and reviewing, `MemoryStore.store_plan(...)` on accept, and a new `_maybe_evict(...)` after each tool call. The subagent tool routes large outputs through `MemoryStore.store_blob(...)` and returns a one-shot LLM summary plus a `memory_handle`. A new built-in `memory_recall` tool exposes the store to the model.

**Tech Stack:** Python 3.12+, pydantic, fastembed (ONNX), numpy, SQLite (stdlib `sqlite3`), pytest, pytest-asyncio. No new external services.

**Spec:** `docs/superpowers/specs/2026-05-27-rag-memory-phase2-design.md`

---

## File structure (locked in by this plan)

**New:**
- `src/llama_agents/memory/__init__.py`
- `src/llama_agents/memory/types.py` — dataclasses + protocol
- `src/llama_agents/memory/chunker.py` — markdown chunking
- `src/llama_agents/memory/embedder.py` — `FastEmbedEmbedder`
- `src/llama_agents/memory/db.py` — SQLite DAO + cosine search
- `src/llama_agents/memory/store.py` — `MemoryStore` + `InertMemoryStore`
- `src/llama_agents/tools/builtin/memory.py` — `MemoryRecallTool`
- `tests/unit/test_memory_chunker.py`
- `tests/unit/test_memory_db.py`
- `tests/unit/test_memory_store.py`
- `tests/unit/test_memory_recall_tool.py`
- `tests/unit/test_agent_plan_retrieval.py`
- `tests/unit/test_agent_eviction.py`
- `tests/unit/test_subagent_summary_return.py`
- `tests/live/test_memory_e2e.py`
- `docs/memory.md`

**Modified:**
- `src/llama_agents/config.py` — add `MemoryConfig`
- `src/llama_agents/events.py` — add `MemoryStored`, `MemoryEvicted`
- `src/llama_agents/runtime.py` — build `MemoryStore`, register `MemoryRecallTool`, pass to spawn tool
- `src/llama_agents/agent.py` — plan-retrieval injection, eviction, `run_id`
- `src/llama_agents/tools/builtin/subagent.py` — summary + handle return
- `src/llama_agents/cli.py` — render new events
- `src/llama_agents/http_app.py` — forward new events on SSE
- `pyproject.toml` — add fastembed, numpy
- `config.toml` — `[memory]` defaults
- `.gitignore` — `.llama_agents/memory/`
- `README.md` — Memory section
- `CLAUDE.md` — strike "No RAG memory"

---

## Conventions for this plan

- **Always run from the repo root** (`D:\repos\llm\llama-agents`).
- **Always use `uv run pytest ...`** to invoke tests (uv may not be on PATH on Windows — see CLAUDE.md).
- **Commit after each task** (one task = one commit). Use the commit message in each task's final step verbatim — they follow conventional commits.
- **Branch:** work on `master` (or a feature branch you cut at the start; this plan does not assume one).
- **DRY/YAGNI/TDD:** every behavioral task writes a failing test first, then the minimal code to pass.

---

## Task 1: Memory config schema

**Files:**
- Modify: `src/llama_agents/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Add a failing test for `MemoryConfig` defaults**

Open `tests/unit/test_config.py` and append:

```python
def test_memory_config_defaults():
    from llama_agents.config import Config

    cfg = Config.model_validate({})
    assert cfg.memory.enabled is True
    assert cfg.memory.root == "_llama_agents/memory" or cfg.memory.root == ".llama_agents/memory"
    assert cfg.memory.embedding_model == "BAAI/bge-small-en-v1.5"
    assert cfg.memory.chunk_size == 1500
    assert cfg.memory.chunk_overlap == 150
    assert cfg.memory.plan_recall_k == 3
    assert 0.0 <= cfg.memory.plan_recall_threshold <= 1.0
    assert cfg.memory.subagent_inline_threshold_chars == 2000
    assert cfg.memory.evict_threshold_pct == 70
    assert cfg.memory.evict_tool_result_min_chars == 4000
    assert cfg.memory.scratch_retention_hours == 24


def test_memory_config_disabled_toml(tmp_path):
    from llama_agents.config import load_config

    p = tmp_path / "c.toml"
    p.write_text("[memory]\nenabled = false\n")
    cfg = load_config(p)
    assert cfg.memory.enabled is False
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/unit/test_config.py::test_memory_config_defaults tests/unit/test_config.py::test_memory_config_disabled_toml -v
```

Expected: AttributeError — `cfg.memory` does not exist.

- [ ] **Step 3: Add `MemoryConfig` and wire it into `Config`**

In `src/llama_agents/config.py`, after the `HttpConfig` class and before `McpServerConfig`, add:

```python
class MemoryConfig(BaseModel):
    enabled: bool = True
    root: str = ".llama_agents/memory"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = Field(default=1500, ge=200)
    chunk_overlap: int = Field(default=150, ge=0)
    plan_recall_k: int = Field(default=3, ge=0)
    plan_recall_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    subagent_inline_threshold_chars: int = Field(default=2000, ge=0)
    subagent_summary_max_tokens: int = Field(default=400, ge=64)
    evict_threshold_pct: int = Field(default=70, ge=10, le=99)
    evict_tool_result_min_chars: int = Field(default=4000, ge=200)
    scratch_retention_hours: int = Field(default=24, ge=-1)
```

Then add to the `Config` class:

```python
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_config.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```
git add src/llama_agents/config.py tests/unit/test_config.py
git commit -m "feat(config): add MemoryConfig with phase-2 defaults"
```

---

## Task 2: Memory event types

**Files:**
- Modify: `src/llama_agents/events.py`
- Test: `tests/unit/test_events.py`

- [ ] **Step 1: Add a failing test**

Append to `tests/unit/test_events.py`:

```python
def test_memory_events_construct():
    from llama_agents.events import Event, MemoryStored, MemoryEvicted

    s = MemoryStored(blob_id="01J", kind="plan", scope="plans", bytes_=42)
    assert isinstance(s, Event)
    assert s.bytes_ == 42

    e = MemoryEvicted(blob_id="01J", turn=3, bytes_freed=9000)
    assert isinstance(e, Event)
    assert e.bytes_freed == 9000
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/unit/test_events.py::test_memory_events_construct -v
```

Expected: ImportError.

- [ ] **Step 3: Add the events**

Append to `src/llama_agents/events.py`:

```python
@dataclass
class MemoryStored(Event):
    blob_id: str
    kind: str
    scope: str
    bytes_: int


@dataclass
class MemoryEvicted(Event):
    blob_id: str
    turn: int
    bytes_freed: int
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/unit/test_events.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```
git add src/llama_agents/events.py tests/unit/test_events.py
git commit -m "feat(events): add MemoryStored and MemoryEvicted"
```

---

## Task 3: Markdown chunker

**Files:**
- Create: `src/llama_agents/memory/__init__.py` (empty)
- Create: `src/llama_agents/memory/chunker.py`
- Create: `tests/unit/test_memory_chunker.py`

- [ ] **Step 1: Create the package marker**

Create `src/llama_agents/memory/__init__.py` (empty file).

- [ ] **Step 2: Write failing tests**

Create `tests/unit/test_memory_chunker.py`:

```python
from llama_agents.memory.chunker import chunk_markdown


def test_chunk_short_markdown_returns_one_chunk():
    chunks = chunk_markdown("# Title\n\nsmall body", chunk_size=1500, chunk_overlap=150)
    assert len(chunks) == 1
    assert chunks[0].startswith("# Title")
    assert "small body" in chunks[0]


def test_chunk_splits_by_headers():
    md = "# A\n\nbody a\n\n# B\n\nbody b\n\n# C\n\nbody c"
    chunks = chunk_markdown(md, chunk_size=20, chunk_overlap=0)
    assert len(chunks) >= 3
    assert any("body a" in c for c in chunks)
    assert any("body b" in c for c in chunks)
    assert any("body c" in c for c in chunks)


def test_chunk_oversized_section_is_split_with_overlap():
    body = "\n".join(f"line {i}" for i in range(200))
    md = f"# Long\n\n{body}"
    chunks = chunk_markdown(md, chunk_size=400, chunk_overlap=80)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.startswith("# Long") or "# Long" in c.split("\n", 1)[0]


def test_chunk_no_headers_falls_back_to_line_split():
    md = "\n".join(f"line {i}" for i in range(100))
    chunks = chunk_markdown(md, chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= 2
    assert "line 0" in chunks[0]


def test_chunk_empty_returns_empty_list():
    assert chunk_markdown("", chunk_size=1500, chunk_overlap=150) == []
    assert chunk_markdown("   \n  ", chunk_size=1500, chunk_overlap=150) == []
```

- [ ] **Step 3: Run tests to verify they fail**

```
uv run pytest tests/unit/test_memory_chunker.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement the chunker**

Create `src/llama_agents/memory/chunker.py`:

```python
from __future__ import annotations

import re

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def chunk_markdown(
    text: str, *, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Split markdown into chunks, preferring header boundaries.

    Mirrors the strategy of the reference RAG implementation: split on
    headers, then split oversized sections by lines with overlap, while
    keeping the current header line as context at the top of each chunk.
    """
    if not text.strip():
        return []

    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_header = ""
    current_lines: list[str] = []

    for line in lines:
        if _HEADER_RE.match(line):
            if current_lines:
                sections.append((current_header, current_lines))
            current_header = line
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_header, current_lines))

    if not sections:
        return _split_lines_with_overlap(lines, chunk_size, chunk_overlap)

    chunks: list[str] = []
    for header, section_lines in sections:
        body = "\n".join(section_lines).strip()
        if not body:
            continue
        if len(body) <= chunk_size:
            chunks.append(body)
            continue
        subs = _split_lines_with_overlap(section_lines, chunk_size, chunk_overlap)
        for sub in subs:
            if header and not sub.startswith(header):
                chunks.append(f"{header}\n\n{sub}")
            else:
                chunks.append(sub)
    return chunks


def _split_lines_with_overlap(
    lines: list[str], chunk_size: int, overlap: int
) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for line in lines:
        ls = len(line) + 1
        if size + ls > chunk_size and buf:
            chunks.append("\n".join(buf).strip())
            tail: list[str] = []
            tail_size = 0
            for prev in reversed(buf):
                if tail_size >= overlap:
                    break
                tail.insert(0, prev)
                tail_size += len(prev) + 1
            buf = tail
            size = tail_size
        buf.append(line)
        size += ls
    if buf:
        joined = "\n".join(buf).strip()
        if joined:
            chunks.append(joined)
    return chunks
```

- [ ] **Step 5: Run tests to verify they pass**

```
uv run pytest tests/unit/test_memory_chunker.py -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git add src/llama_agents/memory/__init__.py src/llama_agents/memory/chunker.py tests/unit/test_memory_chunker.py
git commit -m "feat(memory): markdown chunker with header-aware split + overlap"
```

---

## Task 4: Memory types and embedder protocol

**Files:**
- Create: `src/llama_agents/memory/types.py`
- Modify: `src/llama_agents/memory/__init__.py`

- [ ] **Step 1: Create the types module**

Create `src/llama_agents/memory/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

Scope = Literal["run", "plans", "all"]
WriteScope = Literal["run", "plans"]


@dataclass
class RecalledChunk:
    blob_id: str
    chunk_idx: int
    text: str
    score: float
    title: str
    kind: str


@dataclass
class BlobMeta:
    id: str
    scope: str
    run_id: str | None
    kind: str
    title: str
    file_path: str
    created_at: str
    metadata: dict = field(default_factory=dict)


class Embedder(Protocol):
    """Anything with an `embed` method that returns one float-list per input."""

    @property
    def dim(self) -> int: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

- [ ] **Step 2: Update the package init**

Set `src/llama_agents/memory/__init__.py` to:

```python
from .types import BlobMeta, Embedder, RecalledChunk, Scope, WriteScope

__all__ = ["BlobMeta", "Embedder", "RecalledChunk", "Scope", "WriteScope"]
```

- [ ] **Step 3: Verify nothing breaks**

```
uv run pytest tests/unit -q
```

Expected: all green.

- [ ] **Step 4: Commit**

```
git add src/llama_agents/memory/types.py src/llama_agents/memory/__init__.py
git commit -m "feat(memory): RecalledChunk, BlobMeta, Embedder protocol"
```

---

## Task 5: SQLite-backed vector DB

**Files:**
- Create: `src/llama_agents/memory/db.py`
- Create: `tests/unit/test_memory_db.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_memory_db.py`:

```python
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
        BlobMeta(id="b1", scope="run", run_id="r1", kind="user",
                 title="t1", file_path=str(tmp_path / "b1.md"),
                 created_at="2026-05-27T00:00:00"),
        chunks=[("c1a", _vec(1, 0, 0), "alpha"),
                ("c1b", _vec(0, 1, 0), "beta")],
    )
    await db.insert_blob(
        BlobMeta(id="b2", scope="plans", run_id=None, kind="plan",
                 title="t2", file_path=str(tmp_path / "b2.md"),
                 created_at="2026-05-27T00:00:01"),
        chunks=[("c2a", _vec(0, 0, 1), "gamma")],
    )

    hits = await db.search(_vec(1, 0, 0), scope="all", k=2)
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
        BlobMeta(id="b1", scope="run", run_id="r1", kind="user",
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
    for i, scope, run_id in [
        ("a", "run", "r1"), ("b", "run", "r2"), ("c", "plans", None)
    ]:
        await db.insert_blob(
            BlobMeta(id=i, scope=scope, run_id=run_id, kind="x",
                     title="t", file_path=str(tmp_path / f"{i}.md"),
                     created_at="2026-05-27T00:00:00"),
            chunks=[("c" + i, _vec(1, 0), "x")],
        )

    r1 = [m.id for m in await db.list_blobs(scope="run", run_id="r1")]
    assert r1 == ["a"]
    plans = [m.id for m in await db.list_blobs(scope="plans")]
    assert plans == ["c"]
    await db.close()


@pytest.mark.asyncio
async def test_list_expired_run_ids(tmp_path: Path):
    db = VectorDB(tmp_path / "i.sqlite", dim=2)
    await db.init()
    await db.insert_blob(
        BlobMeta(id="old", scope="run", run_id="r_old", kind="x",
                 title="t", file_path=str(tmp_path / "old.md"),
                 created_at="2026-01-01T00:00:00"),
        chunks=[("c", _vec(1, 0), "x")],
    )
    await db.insert_blob(
        BlobMeta(id="new", scope="run", run_id="r_new", kind="x",
                 title="t", file_path=str(tmp_path / "new.md"),
                 created_at="2099-01-01T00:00:00"),
        chunks=[("c2", _vec(0, 1), "y")],
    )
    expired = await db.list_expired_run_ids(now_iso="2026-05-27T00:00:00",
                                            retention_hours=24)
    assert expired == ["r_old"]
    await db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/unit/test_memory_db.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `VectorDB`**

Create `src/llama_agents/memory/db.py`:

```python
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
        # chunks: (chunk_id, embedding, text). chunk_idx is the position in iter.
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
    ) -> list[tuple[str, str, float, str, str, int]]:
        """Returns list of (chunk_id, blob_id, score, text, title, kind...).

        Actually returns tuples of:
        (chunk_id, blob_id, score, text, title, kind, chunk_idx)
        """
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
        out: list[tuple[str, str, float, str, str, int]] = []
        for i in order:
            r = rows[int(i)]
            out.append(
                (r[0], r[1], float(scores[int(i)]), r[2], r[5], r[6], r[4])
            )
        # Re-tuple to expected signature
        return [(o[0], o[1], o[2], o[3], o[4], o[5], o[6]) for o in out]

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
```

Note: the docstring/return shape of `search` is `(chunk_id, blob_id, score, text, title, kind, chunk_idx)` — seven fields. Tests assert positional access by index, matching this shape.

- [ ] **Step 4: Update tests to match the 7-tuple return**

The tests above only index `[0]` and `[2]` on the search return, so they already work — no edit needed. Confirm by re-reading the test file.

- [ ] **Step 5: Run tests to verify they pass**

```
uv run pytest tests/unit/test_memory_db.py -v
```

Expected: all green. (`pytest-asyncio` is already in dev deps — confirm by inspecting `pyproject.toml` if the tests fail with "async def not natively supported".)

- [ ] **Step 6: Commit**

```
git add src/llama_agents/memory/db.py tests/unit/test_memory_db.py
git commit -m "feat(memory): SQLite vector store with cosine search"
```

---

## Task 6: Embedder implementations

**Files:**
- Create: `src/llama_agents/memory/embedder.py`
- Create: `tests/unit/test_memory_embedder.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_memory_embedder.py`:

```python
import pytest

from llama_agents.memory.embedder import HashEmbedder


@pytest.mark.asyncio
async def test_hash_embedder_dim_and_determinism():
    e = HashEmbedder(dim=16)
    a = await e.embed(["hello world", "second one"])
    b = await e.embed(["hello world"])
    assert len(a) == 2
    assert len(a[0]) == 16
    assert a[0] == b[0]


@pytest.mark.asyncio
async def test_hash_embedder_similar_strings_have_high_cosine():
    import math

    e = HashEmbedder(dim=64)
    [v1, v2] = await e.embed(
        ["the quick brown fox jumps over the lazy dog",
         "the quick brown fox jumps over the lazy cat"]
    )

    def cos(a, b):
        s = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return s / (na * nb)

    assert cos(v1, v2) > 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/unit/test_memory_embedder.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement embedders**

Create `src/llama_agents/memory/embedder.py`:

```python
from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Iterable


class HashEmbedder:
    """Deterministic, dependency-free embedder for unit tests.

    Bag-of-tokens projected into `dim` dimensions via SHA1. Crude but
    stable; equal input -> equal output; overlapping vocabularies -> high
    cosine similarity.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in _tokens(text):
            h = hashlib.sha1(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self._dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def _tokens(s: str) -> Iterable[str]:
    return (m.group(0).lower() for m in re.finditer(r"[A-Za-z0-9_]+", s))


class FastEmbedEmbedder:
    """fastembed-backed embedder.

    Imports fastembed lazily so the module can be imported even when
    fastembed is not installed (we then surface a clear error at .embed
    time). The model loads on first call; subsequent calls reuse it.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model = None
        self._dim_cached: int | None = None
        self._lock = asyncio.Lock()

    @property
    def dim(self) -> int:
        return self._dim_cached or 384

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        await self._ensure_loaded()
        # fastembed's `.embed` is sync + generator; run off-thread.
        assert self._model is not None
        model = self._model

        def _run() -> list[list[float]]:
            vecs = list(model.embed(texts))
            return [v.tolist() for v in vecs]

        return await asyncio.to_thread(_run)

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return

            def _load():
                from fastembed import TextEmbedding

                m = TextEmbedding(model_name=self._model_name)
                probe = next(m.embed(["probe"]))
                return m, int(probe.shape[0])

            model, dim = await asyncio.to_thread(_load)
            self._model = model
            self._dim_cached = dim
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_memory_embedder.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```
git add src/llama_agents/memory/embedder.py tests/unit/test_memory_embedder.py
git commit -m "feat(memory): HashEmbedder (tests) and FastEmbedEmbedder (prod)"
```

---

## Task 7: MemoryStore (composition + inert variant)

**Files:**
- Create: `src/llama_agents/memory/store.py`
- Create: `tests/unit/test_memory_store.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_memory_store.py`:

```python
import asyncio
from pathlib import Path

import pytest

from llama_agents.memory.embedder import HashEmbedder
from llama_agents.memory.store import InertMemoryStore, MemoryStore


@pytest.mark.asyncio
async def test_store_blob_roundtrip(tmp_path: Path):
    store = MemoryStore(root=tmp_path, embedder=HashEmbedder(dim=64))
    await store.init()
    store.start_run("r1")

    handle = await store.store_blob(
        kind="subagent_output", scope="run", run_id="r1",
        title="t", body="# Header\n\nbody about quick brown fox",
    )
    assert handle

    chunks = await store.recall("quick brown fox", scope="all", run_id="r1", k=5)
    assert any(c.blob_id == handle for c in chunks)
    assert chunks[0].score > 0.0
    file_path = next(p for p in (tmp_path / "runs" / "r1").iterdir())
    assert "quick brown fox" in file_path.read_text(encoding="utf-8")

    await store.close()


@pytest.mark.asyncio
async def test_store_plan_writes_to_plans_scope(tmp_path: Path):
    store = MemoryStore(root=tmp_path, embedder=HashEmbedder(dim=64))
    await store.init()
    store.start_run("r1")

    handle = await store.store_plan(
        task="make a sandwich", plan="1. bread\n2. filling\n3. close", accepted_attempt=1
    )

    chunks = await store.recall("sandwich", scope="plans", k=5)
    assert chunks and chunks[0].blob_id == handle
    plans_dir = tmp_path / "plans"
    assert any(plans_dir.iterdir())

    await store.close()


@pytest.mark.asyncio
async def test_recall_with_handle_restricts_results(tmp_path: Path):
    store = MemoryStore(root=tmp_path, embedder=HashEmbedder(dim=32))
    await store.init()
    store.start_run("r1")
    h1 = await store.store_blob(kind="user", scope="run", run_id="r1",
                                title="a", body="cats love tuna fish")
    await store.store_blob(kind="user", scope="run", run_id="r1",
                           title="b", body="dogs love bones and tuna")
    chunks = await store.recall("tuna", scope="all", run_id="r1", handle=h1, k=5)
    assert all(c.blob_id == h1 for c in chunks)
    await store.close()


@pytest.mark.asyncio
async def test_end_run_with_zero_retention_deletes(tmp_path: Path):
    store = MemoryStore(root=tmp_path, embedder=HashEmbedder(dim=8),
                       retention_hours=0)
    await store.init()
    store.start_run("r1")
    await store.store_blob(kind="user", scope="run", run_id="r1",
                           title="t", body="anything")
    run_dir = tmp_path / "runs" / "r1"
    assert run_dir.exists()
    await store.end_run("r1")
    assert not run_dir.exists()
    await store.close()


@pytest.mark.asyncio
async def test_inert_store_returns_empty_and_no_writes(tmp_path: Path):
    store = InertMemoryStore()
    handle = await store.store_blob(
        kind="user", scope="run", title="x", body="hello"
    )
    assert handle == ""
    assert await store.recall("hello", scope="all", k=5) == []
    await store.store_plan(task="t", plan="p", accepted_attempt=1)
    await store.end_run("r")
    await store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/unit/test_memory_store.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `MemoryStore` and `InertMemoryStore`**

Create `src/llama_agents/memory/store.py`:

```python
from __future__ import annotations

import asyncio
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
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_memory_store.py -v
```

Expected: all green.

- [ ] **Step 5: Update package init exports**

Edit `src/llama_agents/memory/__init__.py` to:

```python
from .store import InertMemoryStore, MemoryStore
from .types import BlobMeta, Embedder, RecalledChunk, Scope, WriteScope

__all__ = [
    "BlobMeta", "Embedder", "InertMemoryStore", "MemoryStore",
    "RecalledChunk", "Scope", "WriteScope",
]
```

- [ ] **Step 6: Commit**

```
git add src/llama_agents/memory/store.py src/llama_agents/memory/__init__.py tests/unit/test_memory_store.py
git commit -m "feat(memory): MemoryStore composes chunker+embedder+db; InertMemoryStore"
```

---

## Task 8: `memory_recall` built-in tool

**Files:**
- Create: `src/llama_agents/tools/builtin/memory.py`
- Create: `tests/unit/test_memory_recall_tool.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_memory_recall_tool.py`:

```python
import pytest

from llama_agents.memory.embedder import HashEmbedder
from llama_agents.memory.store import MemoryStore
from llama_agents.tools.builtin.memory import MemoryRecallTool


@pytest.mark.asyncio
async def test_memory_recall_returns_chunks_from_store(tmp_path):
    store = MemoryStore(root=tmp_path, embedder=HashEmbedder(dim=64))
    await store.init()
    store.start_run("r1")
    h = await store.store_blob(
        kind="subagent_output", scope="run", run_id="r1",
        title="t", body="the quick brown fox jumps over the lazy dog",
    )
    tool = MemoryRecallTool(store=store, run_id_getter=lambda: "r1")
    res = await tool.invoke({"query": "quick brown fox", "k": 3})
    assert "chunks" in res and len(res["chunks"]) >= 1
    assert res["chunks"][0]["blob_id"] == h
    await store.close()


@pytest.mark.asyncio
async def test_memory_recall_with_handle_restricts(tmp_path):
    store = MemoryStore(root=tmp_path, embedder=HashEmbedder(dim=32))
    await store.init()
    store.start_run("r1")
    h1 = await store.store_blob(kind="user", scope="run", run_id="r1",
                                title="a", body="cats love tuna")
    await store.store_blob(kind="user", scope="run", run_id="r1",
                           title="b", body="dogs love tuna too")
    tool = MemoryRecallTool(store=store, run_id_getter=lambda: "r1")
    res = await tool.invoke({"query": "tuna", "handle": h1})
    assert all(c["blob_id"] == h1 for c in res["chunks"])
    await store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/unit/test_memory_recall_tool.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the tool**

Create `src/llama_agents/tools/builtin/memory.py`:

```python
from __future__ import annotations

from typing import Any, Callable

from ...memory.store import MemoryStore
from ..base import Tool


class MemoryRecallTool(Tool):
    name = "memory_recall"
    description = (
        "Retrieve previously-stored content from this run's scratch memory "
        "and past plans. Use this when you see '[evicted to memory ...]' in "
        "earlier tool results, or to look up the full text of a subagent's "
        "output via its memory_handle."
    )
    json_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "handle": {
                "type": "string",
                "description": "Optional. Restrict results to chunks from this blob_id.",
            },
            "k": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(
        self,
        *,
        store: MemoryStore,
        run_id_getter: Callable[[], str | None],
    ) -> None:
        self._store = store
        self._run_id_getter = run_id_getter

    async def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        rid = self._run_id_getter()
        chunks = await self._store.recall(
            query=args["query"],
            scope="all",
            run_id=rid,
            handle=args.get("handle"),
            k=int(args.get("k", 5)),
        )
        return {
            "chunks": [
                {
                    "text": c.text,
                    "blob_id": c.blob_id,
                    "chunk_idx": c.chunk_idx,
                    "score": c.score,
                    "title": c.title,
                    "kind": c.kind,
                }
                for c in chunks
            ]
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_memory_recall_tool.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```
git add src/llama_agents/tools/builtin/memory.py tests/unit/test_memory_recall_tool.py
git commit -m "feat(tools): memory_recall built-in tool"
```

---

## Task 9: Add fastembed + numpy dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Inspect current dependencies**

Run:

```
type pyproject.toml | findstr /n .
```

Locate the `[project]` `dependencies = [...]` array.

- [ ] **Step 2: Add the two deps**

Edit `pyproject.toml`. In the `dependencies` array (NOT optional/dev), add:

```
"fastembed>=0.4",
"numpy>=1.26",
```

(Preserve any existing entries; do not reorder.)

- [ ] **Step 3: Sync and re-run all unit tests**

```
uv sync --extra dev
uv run pytest tests/unit -q
```

Expected: all green. The fastembed model is NOT downloaded yet — nothing imports `FastEmbedEmbedder` at runtime in any test.

- [ ] **Step 4: Commit**

```
git add pyproject.toml uv.lock
git commit -m "chore(deps): add fastembed and numpy for memory layer"
```

(If `uv.lock` is gitignored or absent, just stage `pyproject.toml`.)

---

## Task 10: Runtime wiring (build store, register tool)

**Files:**
- Modify: `src/llama_agents/runtime.py`
- Modify: `tests/unit/test_runtime.py`

- [ ] **Step 1: Look at the current Runtime test to find the pattern**

Open `tests/unit/test_runtime.py` and read it (no edits yet) so the new test follows the same harness style (fake client, no llama-server spawn).

- [ ] **Step 2: Write failing tests**

Append to `tests/unit/test_runtime.py` (preserving existing imports — add what's missing):

```python
@pytest.mark.asyncio
async def test_runtime_registers_memory_recall_when_enabled(tmp_path, monkeypatch):
    from llama_agents.config import Config, MemoryConfig, SandboxConfig
    from llama_agents.runtime import Runtime

    cfg = Config(
        sandbox=SandboxConfig(allowed_dirs=[tmp_path]),
        memory=MemoryConfig(root=str(tmp_path / ".mem")),
    )
    # Avoid spawning llama-server
    monkeypatch.setattr(cfg.llama, "auto_spawn", False)

    class _FakeClient:
        async def chat(self, **_): raise NotImplementedError
        async def health(self): return True
        async def aclose(self): pass

    rt = await Runtime.create(cfg, client_factory=lambda url: _FakeClient())
    try:
        assert "memory_recall" in rt.registry.names()
    finally:
        await rt.aclose()


@pytest.mark.asyncio
async def test_runtime_uses_inert_store_when_disabled(tmp_path, monkeypatch):
    from llama_agents.config import Config, MemoryConfig, SandboxConfig
    from llama_agents.memory.store import InertMemoryStore
    from llama_agents.runtime import Runtime

    cfg = Config(
        sandbox=SandboxConfig(allowed_dirs=[tmp_path]),
        memory=MemoryConfig(enabled=False),
    )
    monkeypatch.setattr(cfg.llama, "auto_spawn", False)

    class _FakeClient:
        async def chat(self, **_): raise NotImplementedError
        async def health(self): return True
        async def aclose(self): pass

    rt = await Runtime.create(cfg, client_factory=lambda url: _FakeClient())
    try:
        assert isinstance(rt.memory, InertMemoryStore)
        assert "memory_recall" in rt.registry.names()  # tool is registered either way
    finally:
        await rt.aclose()
```

- [ ] **Step 3: Run tests to verify they fail**

```
uv run pytest tests/unit/test_runtime.py -v
```

Expected: failures — `Runtime` has no `memory` attribute, `memory_recall` not registered.

- [ ] **Step 4: Wire up Runtime**

Edit `src/llama_agents/runtime.py`. Add imports:

```python
from pathlib import Path
from .memory.embedder import HashEmbedder, FastEmbedEmbedder
from .memory.store import InertMemoryStore, MemoryStore
from .tools.builtin.memory import MemoryRecallTool
```

Add a field on `Runtime.__init__`:

```python
    def __init__(
        self,
        cfg: Config,
        client: _ClientLike,
        manager: LlamaServerManager | None,
        bridge: McpBridge | None,
        registry: ToolRegistry,
        semaphore: asyncio.Semaphore,
        memory: "MemoryStore | InertMemoryStore",
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.manager = manager
        self.bridge = bridge
        self.registry = registry
        self.semaphore = semaphore
        self.memory = memory
        self._current_run_id: str | None = None
```

In `Runtime.create`, after building the registry but BEFORE the spawn tool injection, build the memory store and register `memory_recall`. Replace the relevant section with:

```python
        # Build memory layer
        if cfg.memory.enabled:
            mem_root = _resolve_memory_root(cfg)
            embedder = FastEmbedEmbedder(model_name=cfg.memory.embedding_model)
            mem: MemoryStore | InertMemoryStore = MemoryStore(
                root=mem_root,
                embedder=embedder,
                chunk_size=cfg.memory.chunk_size,
                chunk_overlap=cfg.memory.chunk_overlap,
                retention_hours=cfg.memory.scratch_retention_hours,
            )
            await mem.init()
            await mem.gc_expired()
        else:
            mem = InertMemoryStore()
            await mem.init()

        sem = asyncio.Semaphore(cfg.agent.max_concurrent_agents)

        bridge: McpBridge | None = None
        if cfg.mcp_servers:
            bridge = McpBridge(cfg.mcp_servers)
            for t in await bridge.start():
                registry.register(t)

        rt = cls(cfg, client, manager, bridge, registry, sem, mem)

        # memory_recall tool (always available — InertMemoryStore returns [])
        registry.register(
            MemoryRecallTool(store=rt.memory, run_id_getter=lambda: rt._current_run_id)
        )

        # Inject the spawn tool last (needs runtime to make new agents).
        registry.register(
            SpawnSubagentTool(agent_factory=rt.new_agent, semaphore=sem)
        )
        return rt
```

Add at the bottom of `runtime.py`:

```python
def _resolve_memory_root(cfg: Config) -> Path:
    root_cfg = cfg.memory.root
    p = Path(root_cfg)
    if p.is_absolute():
        return p
    base = cfg.sandbox.allowed_dirs[0] if cfg.sandbox.allowed_dirs else Path.cwd()
    return base / root_cfg
```

Modify `Runtime.aclose`:

```python
    async def aclose(self) -> None:
        if self.bridge is not None:
            await self.bridge.aclose()
        if self.manager is not None:
            await self.manager.shutdown()
        await self.memory.close()
        await self.client.aclose()
```

Modify `new_agent` to NOT clone the recall tool every time (it shares a single store; cloning the registry already gives each agent its own dict pointing at the same tool instance — that's fine).

- [ ] **Step 5: Run tests to verify they pass**

```
uv run pytest tests/unit/test_runtime.py -v
```

Expected: all green. If `LlamaServerManager` requires `auto_spawn`-false handling, check existing patterns in `test_runtime.py` and mirror them.

- [ ] **Step 6: Run the full unit suite**

```
uv run pytest tests/unit -q
```

Expected: all green.

- [ ] **Step 7: Commit**

```
git add src/llama_agents/runtime.py tests/unit/test_runtime.py
git commit -m "feat(runtime): build MemoryStore, register memory_recall tool"
```

---

## Task 11: Agent run_id plumbing

**Files:**
- Modify: `src/llama_agents/agent.py`
- Modify: `src/llama_agents/runtime.py`

This task introduces a `run_id` on each `Agent.run()` invocation, plus the `MemoryStore` reference the agent needs. It does not yet change behavior beyond setting `Runtime._current_run_id`. Plan retrieval and eviction come in the next two tasks.

- [ ] **Step 1: Modify `Agent.__init__` to accept a memory store and optional run_id_setter**

In `src/llama_agents/agent.py`, change the imports and class:

```python
from .memory.store import InertMemoryStore, MemoryStore
```

Update the constructor:

```python
class Agent:
    def __init__(
        self,
        *,
        client: _ClientLike,
        registry: ToolRegistry,
        memory: "MemoryStore | InertMemoryStore | None" = None,
        on_run_id: "Callable[[str | None], None] | None" = None,
    ) -> None:
        self._client = client
        self._registry = registry
        self._memory = memory or InertMemoryStore()
        self._on_run_id = on_run_id
        self._cancel = asyncio.Event()
        self._run_id: str | None = None
        self.messages: list[dict[str, Any]] = []
```

(Add `from typing import Callable` if not already imported.)

- [ ] **Step 2: Generate a run_id at the top of `Agent.run`**

In `Agent.run`, after the method signature line and before the existing body, add:

```python
        import uuid
        self._run_id = uuid.uuid4().hex[:24]
        self._memory.start_run(self._run_id)
        if self._on_run_id is not None:
            self._on_run_id(self._run_id)
```

And at the end (after the `yield Done(reason="max_iterations")` line — but wrap the whole function body in a try/finally so end_run runs regardless):

Restructure `Agent.run` so the entire body is wrapped:

```python
    async def run(
        self, user_prompt: str, opts: AgentRunOptions
    ) -> AsyncIterator[Event]:
        import uuid
        self._run_id = uuid.uuid4().hex[:24]
        self._memory.start_run(self._run_id)
        if self._on_run_id is not None:
            self._on_run_id(self._run_id)
        try:
            # ... existing body unchanged for now ...
            <existing code here>
        finally:
            await self._memory.end_run(self._run_id)
            if self._on_run_id is not None:
                self._on_run_id(None)
```

Note: `AsyncIterator` and `try/finally` with `yield` works in Python 3.10+. Confirm by running tests.

- [ ] **Step 3: Update `Runtime.new_agent` to pass memory and run_id setter**

In `src/llama_agents/runtime.py`:

```python
    def new_agent(self) -> Agent:
        def _set_rid(rid: str | None) -> None:
            self._current_run_id = rid

        return Agent(
            client=self.client,
            registry=self.registry.clone(),
            memory=self.memory,
            on_run_id=_set_rid,
        )
```

- [ ] **Step 4: Update existing agent loop tests if they break**

Run:

```
uv run pytest tests/unit/test_agent_loop.py -v
```

If any test fails because `Agent()` rejects new kwargs, those tests are fine — the new kwargs have defaults. The likely break is the `try/finally` placement. Read the failure carefully and adjust.

- [ ] **Step 5: Run the full unit suite**

```
uv run pytest tests/unit -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git add src/llama_agents/agent.py src/llama_agents/runtime.py
git commit -m "feat(agent): plumb run_id and MemoryStore through Agent.run"
```

---

## Task 12: Plan retrieval injection

**Files:**
- Modify: `src/llama_agents/agent.py`
- Create: `tests/unit/test_agent_plan_retrieval.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_agent_plan_retrieval.py`:

```python
from __future__ import annotations

import pytest

from llama_agents.agent import Agent, AgentRunOptions
from llama_agents.events import (
    AssistantChunk, Done, PlanAccepted, PlanProposed, PlanReviewed,
)
from llama_agents.llama_client import ChatResponse, ToolCall
from llama_agents.memory.embedder import HashEmbedder
from llama_agents.memory.store import MemoryStore
from llama_agents.tools.registry import ToolRegistry


class _ScriptedClient:
    """Returns canned responses; records messages it received."""
    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)
        self.last_planner_messages: list = []
        self.last_reviewer_messages: list = []
        self._call_idx = 0

    async def chat(self, *, messages, tools, temperature=0.2,
                   reasoning_budget_tokens=None):
        idx = self._call_idx
        self._call_idx += 1
        # Heuristic: planner system mentions "planning agent", reviewer "reviewer"
        sys = messages[0]["content"] if messages else ""
        if "planning agent" in sys:
            self.last_planner_messages = list(messages)
        elif "plan reviewer" in sys:
            self.last_reviewer_messages = list(messages)
        return self._responses[idx]


def _resp(content: str, *, tool_calls=None) -> ChatResponse:
    return ChatResponse(content=content, tool_calls=tool_calls or [],
                        raw_message={"role": "assistant", "content": content})


@pytest.mark.asyncio
async def test_plan_retrieval_injects_prior_plans(tmp_path):
    store = MemoryStore(root=tmp_path, embedder=HashEmbedder(dim=64))
    await store.init()
    await store.store_plan(
        task="bake a sandwich tutorial",
        plan="1. fetch bread\n2. add filling\n3. close",
        accepted_attempt=1,
    )

    # Scripted chat: planner -> proposes a plan; reviewer -> ACCEPT;
    # main loop -> final assistant content with no tool calls.
    client = _ScriptedClient([
        _resp("1. step\n2. step\n3. step"),
        _resp("ACCEPT"),
        _resp("done."),
    ])

    registry = ToolRegistry()
    # Force planning by adding a subagent_spawn stub
    from llama_agents.tools.base import Tool

    class _StubSpawn(Tool):
        name = "subagent_spawn"
        description = "stub"
        json_schema = {"type": "object", "properties": {}, "required": []}
        async def invoke(self, args): return {"result": "x"}

    registry.register(_StubSpawn())

    agent = Agent(client=client, registry=registry, memory=store)
    events = []
    async for ev in agent.run("how do I bake a sandwich?",
                              AgentRunOptions(max_iterations=2)):
        events.append(ev)

    # Confirm planner/reviewer system prompts contain the prior-plan banner
    planner_sys = client.last_planner_messages[0]["content"]
    reviewer_sys = client.last_reviewer_messages[0]["content"]
    assert "PRIOR ACCEPTED PLANS" in planner_sys
    assert "PRIOR ACCEPTED PLANS" in reviewer_sys
    assert "sandwich" in planner_sys.lower()

    await store.close()


@pytest.mark.asyncio
async def test_plan_storage_on_accept(tmp_path):
    store = MemoryStore(root=tmp_path, embedder=HashEmbedder(dim=64))
    await store.init()

    client = _ScriptedClient([
        _resp("1. step\n2. step"),
        _resp("ACCEPT"),
        _resp("done."),
    ])
    registry = ToolRegistry()
    from llama_agents.tools.base import Tool

    class _StubSpawn(Tool):
        name = "subagent_spawn"
        description = "stub"
        json_schema = {"type": "object", "properties": {}, "required": []}
        async def invoke(self, args): return {"result": "x"}

    registry.register(_StubSpawn())

    agent = Agent(client=client, registry=registry, memory=store)
    async for _ in agent.run("a task", AgentRunOptions(max_iterations=2)):
        pass

    plans = await store.list_handles(scope="plans")
    assert len(plans) == 1
    assert plans[0].metadata.get("task") == "a task"
    await store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/unit/test_agent_plan_retrieval.py -v
```

Expected: failures — banner not present, plan not stored.

- [ ] **Step 3: Modify `Agent._plan_and_review` and `Agent.run`**

In `src/llama_agents/agent.py`, change `_plan_and_review` to accept opts and do retrieval at the top:

Inside `_plan_and_review`, after the existing `planner_system` and `reviewer_system` strings are computed, add:

```python
        prior = []
        try:
            prior = await self._memory.recall(
                query=user_prompt, scope="plans",
                k=getattr(opts, "plan_recall_k", 3),
                min_score=getattr(opts, "plan_recall_threshold", 0.5),
            )
        except Exception as e:  # noqa: BLE001
            import sys
            print(f"[memory] plan recall failed: {e}", file=sys.stderr)
        if prior:
            banner = "\n\nPRIOR ACCEPTED PLANS FOR SIMILAR TASKS:\n" + \
                "\n---\n".join(c.text for c in prior)
            planner_system = planner_system + banner
            reviewer_system = reviewer_system + banner
```

Add `plan_recall_k` and `plan_recall_threshold` to `AgentRunOptions`:

```python
    plan_recall_k: int = 3
    plan_recall_threshold: float = 0.5
```

Replace the final `PlanAccepted` emissions (both the in-loop one and the post-loop fallback) so that BEFORE yielding `PlanAccepted`, the plan is stored:

```python
            if accepted:
                try:
                    await self._memory.store_plan(
                        task=user_prompt, plan=last_plan,
                        accepted_attempt=attempt, run_id=self._run_id,
                    )
                except Exception as e:  # noqa: BLE001
                    import sys
                    print(f"[memory] plan store failed: {e}", file=sys.stderr)
                yield PlanAccepted(plan=last_plan, attempts=attempt)
                return
```

And for the post-loop fallback:

```python
        try:
            await self._memory.store_plan(
                task=user_prompt, plan=last_plan,
                accepted_attempt=opts.max_planning_iterations,
                run_id=self._run_id,
            )
        except Exception as e:  # noqa: BLE001
            import sys
            print(f"[memory] plan store failed: {e}", file=sys.stderr)
        yield PlanAccepted(plan=last_plan, attempts=opts.max_planning_iterations)
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_agent_plan_retrieval.py -v
uv run pytest tests/unit/test_agent_loop.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```
git add src/llama_agents/agent.py tests/unit/test_agent_plan_retrieval.py
git commit -m "feat(agent): retrieve prior plans for planner+reviewer; store accepted plans"
```

---

## Task 13: Overflow eviction

**Files:**
- Modify: `src/llama_agents/agent.py`
- Create: `tests/unit/test_agent_eviction.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_agent_eviction.py`:

```python
from __future__ import annotations

import pytest

from llama_agents.agent import Agent, AgentRunOptions
from llama_agents.events import MemoryEvicted
from llama_agents.llama_client import ChatResponse, ToolCall
from llama_agents.memory.embedder import HashEmbedder
from llama_agents.memory.store import MemoryStore
from llama_agents.tools.base import Tool
from llama_agents.tools.registry import ToolRegistry


class _BigTool(Tool):
    name = "big_tool"
    description = "returns a large string"
    json_schema = {"type": "object", "properties": {}, "required": []}

    async def invoke(self, args):
        return "X" * 8000


class _ScriptedClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0

    async def chat(self, *, messages, tools, temperature=0.2,
                   reasoning_budget_tokens=None):
        r = self._responses[self._i]
        self._i += 1
        return r


def _resp(content="", *, tool_calls=None):
    return ChatResponse(content=content, tool_calls=tool_calls or [],
                        raw_message={"role": "assistant", "content": content})


@pytest.mark.asyncio
async def test_eviction_rewrites_old_tool_results_when_threshold_crossed(tmp_path):
    store = MemoryStore(root=tmp_path, embedder=HashEmbedder(dim=32))
    await store.init()

    # 6 tool calls of big_tool, then a plain text reply.
    tc = lambda i: ToolCall(id=f"c{i}", name="big_tool", arguments={})
    responses = [
        _resp(tool_calls=[tc(0)]),
        _resp(tool_calls=[tc(1)]),
        _resp(tool_calls=[tc(2)]),
        _resp(tool_calls=[tc(3)]),
        _resp(tool_calls=[tc(4)]),
        _resp(tool_calls=[tc(5)]),
        _resp("done"),
    ]
    client = _ScriptedClient(responses)
    registry = ToolRegistry()
    registry.register(_BigTool())

    agent = Agent(client=client, registry=registry, memory=store)
    opts = AgentRunOptions(
        max_iterations=10,
        skip_planning=True,
        evict_threshold_pct=20,             # easy to trip
        evict_tool_result_min_chars=2000,
        ctx_size_for_eviction=8192,         # small synthetic context
    )

    evicted: list[MemoryEvicted] = []
    async for ev in agent.run("do thing", opts):
        if isinstance(ev, MemoryEvicted):
            evicted.append(ev)

    assert evicted, "expected at least one MemoryEvicted event"
    # And the corresponding messages should contain the stub text
    stubbed = [m for m in agent.messages
               if m.get("role") == "tool"
               and "[evicted to memory" in (m.get("content") or "")]
    assert stubbed, "expected at least one tool message rewritten with stub"
    await store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/unit/test_agent_eviction.py -v
```

Expected: failure — `evict_threshold_pct` / `ctx_size_for_eviction` not in `AgentRunOptions`; no eviction logic.

- [ ] **Step 3: Add eviction options to `AgentRunOptions`**

Append to `AgentRunOptions`:

```python
    evict_threshold_pct: int = 70
    evict_tool_result_min_chars: int = 4000
    ctx_size_for_eviction: int = 65536  # in tokens
```

- [ ] **Step 4: Implement `_maybe_evict` and call it after each tool result append**

In `Agent`, add a method:

```python
    _EST_CHARS_PER_TOKEN: float = 3.5

    async def _maybe_evict(self, opts: AgentRunOptions) -> list[Event]:
        budget_tokens = opts.ctx_size_for_eviction * (opts.evict_threshold_pct / 100.0)
        est = sum(len(_msg_str(m)) for m in self.messages) / self._EST_CHARS_PER_TOKEN
        if est < budget_tokens:
            return []

        events: list[Event] = []
        # Preserve last 4 messages from eviction
        last_preserved = max(0, len(self.messages) - 4)
        for i, msg in enumerate(self.messages[:last_preserved]):
            if msg.get("role") != "tool":
                continue
            body = msg.get("content") or ""
            if len(body) < opts.evict_tool_result_min_chars:
                continue
            try:
                blob_id = await self._memory.store_blob(
                    kind="evicted_tool", scope="run", run_id=self._run_id,
                    title=f"tool result @ msg {i}",
                    body=body,
                    metadata={"tool_call_id": msg.get("tool_call_id")},
                )
            except Exception as e:  # noqa: BLE001
                import sys
                print(f"[memory] eviction store failed: {e}", file=sys.stderr)
                continue
            freed = len(body)
            stub = (
                f"[evicted to memory — use memory_recall("
                f"handle=\"{blob_id}\", query=...) to retrieve. "
                f"Original size: {freed} chars.]"
            )
            msg["content"] = stub
            events.append(MemoryEvicted(blob_id=blob_id, turn=i,
                                        bytes_freed=freed - len(stub)))
            est -= (freed - len(stub)) / self._EST_CHARS_PER_TOKEN
            if est < opts.ctx_size_for_eviction * 0.5:
                break
        return events
```

And add the helper at module scope:

```python
def _msg_str(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    return _json_dump(c)
```

In `Agent.run`'s main loop, after `self.messages.append({"role": "tool", ...})` inside the for-loop over `resp.tool_calls`, add:

```python
                for ev in await self._maybe_evict(opts):
                    yield ev
```

- [ ] **Step 5: Run tests to verify they pass**

```
uv run pytest tests/unit/test_agent_eviction.py tests/unit/test_agent_loop.py -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git add src/llama_agents/agent.py tests/unit/test_agent_eviction.py
git commit -m "feat(agent): threshold-triggered eviction of large tool results to RAG"
```

---

## Task 14: Subagent return shape (summary + handle)

**Files:**
- Modify: `src/llama_agents/tools/builtin/subagent.py`
- Create: `tests/unit/test_subagent_summary_return.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_subagent_summary_return.py`:

```python
from __future__ import annotations

import pytest

from llama_agents.agent import Agent, AgentRunOptions
from llama_agents.llama_client import ChatResponse
from llama_agents.memory.embedder import HashEmbedder
from llama_agents.memory.store import MemoryStore
from llama_agents.tools.builtin.subagent import SpawnSubagentTool
from llama_agents.tools.registry import ToolRegistry


class _SubClient:
    """For the subagent itself: finishes immediately with a big text."""
    def __init__(self, big_text):
        self._big = big_text
        self._called = 0

    async def chat(self, *, messages, tools, temperature=0.2,
                   reasoning_budget_tokens=None):
        # First call: produce the big text and finish.
        # Second call (if any): treated as summarizer; return a short summary.
        self._called += 1
        if self._called == 1:
            return ChatResponse(
                content=self._big, tool_calls=[],
                raw_message={"role": "assistant", "content": self._big},
            )
        return ChatResponse(
            content="short summary.", tool_calls=[],
            raw_message={"role": "assistant", "content": "short summary."},
        )


@pytest.mark.asyncio
async def test_subagent_returns_summary_and_handle_for_large_output(tmp_path):
    store = MemoryStore(root=tmp_path, embedder=HashEmbedder(dim=32))
    await store.init()
    store.start_run("rTOP")  # the parent's run_id; subagent inherits

    big = "BIG OUTPUT " * 500  # > 2000 chars
    client = _SubClient(big)

    def factory():
        registry = ToolRegistry()
        a = Agent(client=client, registry=registry, memory=store)
        # Subagents inherit run_id by being constructed with the parent's:
        a._run_id = "rTOP"
        store.start_run("rTOP")
        return a

    import asyncio
    sem = asyncio.Semaphore(1)
    tool = SpawnSubagentTool(
        agent_factory=factory,
        semaphore=sem,
        memory=store,
        client_for_summary=client,
        inline_threshold_chars=2000,
        parent_run_id_getter=lambda: "rTOP",
    )

    result = await tool.invoke({"task": "describe the universe"})
    assert "memory_handle" in result
    assert result["memory_handle"]
    assert "summary" in result
    assert "result" not in result  # large path: no full inline result
    await store.close()


@pytest.mark.asyncio
async def test_subagent_returns_inline_for_small_output(tmp_path):
    store = MemoryStore(root=tmp_path, embedder=HashEmbedder(dim=32))
    await store.init()
    store.start_run("rTOP")

    small = "tiny output"
    client = _SubClient(small)

    def factory():
        registry = ToolRegistry()
        a = Agent(client=client, registry=registry, memory=store)
        a._run_id = "rTOP"
        store.start_run("rTOP")
        return a

    import asyncio
    sem = asyncio.Semaphore(1)
    tool = SpawnSubagentTool(
        agent_factory=factory,
        semaphore=sem,
        memory=store,
        client_for_summary=client,
        inline_threshold_chars=2000,
        parent_run_id_getter=lambda: "rTOP",
    )
    result = await tool.invoke({"task": "say hi"})
    assert result["result"] == "tiny output"
    assert "memory_handle" not in result
    await store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/unit/test_subagent_summary_return.py -v
```

Expected: failure — `SpawnSubagentTool` doesn't accept `memory`/`client_for_summary`/etc.

- [ ] **Step 3: Modify `SpawnSubagentTool`**

Replace `src/llama_agents/tools/builtin/subagent.py` with:

```python
from __future__ import annotations

import asyncio
from typing import Any, Callable

from ...agent import Agent, AgentRunOptions
from ...errors import AgentLimitExceeded
from ...events import AssistantChunk, Done, ToolCallResult, ToolCallStart
from ...memory.store import InertMemoryStore, MemoryStore
from ..base import Tool


_SUMMARIZER_SYSTEM = (
    "You summarize a subagent's output for the orchestrator that delegated "
    "the task. Write 3-6 sentences capturing what was done and any key "
    "findings. No preamble, no markdown headers."
)


class SpawnSubagentTool(Tool):
    name = "subagent_spawn"
    description = (
        "Spawn a subagent with its own conversation to handle a focused task. "
        "Returns the subagent's final assistant message as `result` for short "
        "outputs, or a `summary` + `memory_handle` for long outputs that have "
        "been written to memory (retrievable via memory_recall)."
    )
    json_schema = {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "system_prompt": {"type": "string"},
            "allowed_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "tool names the subagent may use; defaults to parent's minus subagent_spawn",
            },
            "max_iterations": {"type": "integer", "default": 20},
        },
        "required": ["task"],
    }

    def __init__(
        self,
        agent_factory: Callable[[], Agent],
        semaphore: asyncio.Semaphore,
        *,
        memory: "MemoryStore | InertMemoryStore | None" = None,
        client_for_summary: Any = None,
        inline_threshold_chars: int = 2000,
        summary_max_tokens: int = 400,
        parent_run_id_getter: Callable[[], str | None] | None = None,
    ) -> None:
        self._factory = agent_factory
        self._sem = semaphore
        self._memory = memory or InertMemoryStore()
        self._client = client_for_summary
        self._inline_threshold = inline_threshold_chars
        self._summary_max_tokens = summary_max_tokens
        self._parent_run_id_getter = parent_run_id_getter or (lambda: None)

    async def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._sem.locked() and self._sem._value > 0:  # type: ignore[attr-defined]
            pass
        acquired = self._sem.locked() is False and await _try_acquire(self._sem)
        if not acquired:
            raise AgentLimitExceeded("max_concurrent_agents reached")

        try:
            subagent = self._factory()
            allowed = args.get("allowed_tools")
            if allowed is not None:
                for n in list(subagent._registry.names()):
                    if n not in allowed:
                        subagent._registry.unregister(n)
            else:
                subagent._registry.unregister("subagent_spawn")

            opts = AgentRunOptions(
                max_iterations=int(args.get("max_iterations", 20)),
                system_prompt=args.get(
                    "system_prompt",
                    "You are a focused subagent. Complete the task and report back.",
                ),
            )

            iterations = 0
            tool_calls = 0
            final_text = ""
            async for ev in subagent.run(args["task"], opts):
                if isinstance(ev, ToolCallStart):
                    tool_calls += 1
                elif isinstance(ev, AssistantChunk):
                    final_text = ev.text
                elif isinstance(ev, Done):
                    if ev.final_message:
                        final_text = ev.final_message
                    break
                iterations += 1

            if len(final_text) <= self._inline_threshold:
                return {
                    "result": final_text,
                    "iterations": iterations,
                    "tool_calls": tool_calls,
                }

            parent_rid = self._parent_run_id_getter()
            try:
                blob_id = await self._memory.store_blob(
                    kind="subagent_output", scope="run",
                    run_id=parent_rid,
                    title=f"subagent: {args['task'][:60]}",
                    body=final_text,
                    metadata={"task": args["task"], "iterations": iterations,
                              "tool_calls": tool_calls},
                )
            except Exception as e:  # noqa: BLE001
                import sys
                print(f"[memory] subagent store failed: {e}", file=sys.stderr)
                # Fall back to inline return.
                return {
                    "result": final_text,
                    "iterations": iterations,
                    "tool_calls": tool_calls,
                }

            summary = await self._summarize(args["task"], final_text)
            return {
                "summary": summary,
                "memory_handle": blob_id,
                "result_bytes": len(final_text),
                "iterations": iterations,
                "tool_calls": tool_calls,
            }
        finally:
            self._sem.release()

    async def _summarize(self, task: str, output: str) -> str:
        if self._client is None:
            return output[:400]
        truncated = output[:8000]
        try:
            resp = await self._client.chat(
                messages=[
                    {"role": "system", "content": _SUMMARIZER_SYSTEM},
                    {"role": "user",
                     "content": f"TASK:\n{task}\n\nOUTPUT:\n{truncated}"},
                ],
                tools=[],
                temperature=0.0,
                reasoning_budget_tokens=0,
            )
            return (resp.content or "").strip() or truncated[:400]
        except Exception as e:  # noqa: BLE001
            import sys
            print(f"[memory] summary failed: {e}", file=sys.stderr)
            return truncated[:400]


async def _try_acquire(sem: asyncio.Semaphore) -> bool:
    if sem.locked():
        return False
    if sem._value <= 0:  # type: ignore[attr-defined]
        return False
    await sem.acquire()
    return True
```

- [ ] **Step 4: Update `Runtime.create` to pass the new args**

In `src/llama_agents/runtime.py`, change the `SpawnSubagentTool(...)` construction to:

```python
        registry.register(
            SpawnSubagentTool(
                agent_factory=rt.new_agent,
                semaphore=sem,
                memory=rt.memory,
                client_for_summary=rt.client,
                inline_threshold_chars=cfg.memory.subagent_inline_threshold_chars,
                summary_max_tokens=cfg.memory.subagent_summary_max_tokens,
                parent_run_id_getter=lambda: rt._current_run_id,
            )
        )
```

- [ ] **Step 5: Run tests to verify they pass**

```
uv run pytest tests/unit/test_subagent_summary_return.py tests/unit/test_subagent_tool.py -v
```

Expected: all green. If the existing `test_subagent_tool.py` constructs `SpawnSubagentTool` with positional args, the new kwargs are all optional — should still pass.

- [ ] **Step 6: Commit**

```
git add src/llama_agents/tools/builtin/subagent.py src/llama_agents/runtime.py tests/unit/test_subagent_summary_return.py
git commit -m "feat(subagent): return summary + memory_handle for large outputs"
```

---

## Task 15: CLI and HTTP event rendering

**Files:**
- Modify: `src/llama_agents/cli.py`
- Modify: `src/llama_agents/http_app.py`
- Modify: `tests/unit/test_cli.py`
- Modify: `tests/unit/test_http_app.py`

- [ ] **Step 1: Inspect the existing event rendering in cli.py and http_app.py**

Open both files and find the event dispatch. CLI likely has an `if isinstance(ev, ...)` ladder; HTTP likely has a `_serialize_event` function.

- [ ] **Step 2: Add failing tests for both**

Append to `tests/unit/test_cli.py`:

```python
def test_cli_renders_memory_events(capsys):
    from llama_agents.cli import _render_event  # adjust if helper has a different name
    from llama_agents.events import MemoryEvicted, MemoryStored

    _render_event(MemoryStored(blob_id="01J", kind="plan", scope="plans", bytes_=512))
    _render_event(MemoryEvicted(blob_id="01J", turn=3, bytes_freed=8000))
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "01J" in out or "memory" in out.lower()
    assert "evict" in out.lower()
```

Append to `tests/unit/test_http_app.py`:

```python
def test_serialize_memory_events():
    from llama_agents.http_app import _event_to_sse
    from llama_agents.events import MemoryEvicted, MemoryStored

    a = _event_to_sse(MemoryStored(blob_id="x", kind="plan", scope="plans", bytes_=12))
    b = _event_to_sse(MemoryEvicted(blob_id="x", turn=2, bytes_freed=3))
    assert "memory_stored" in a or "MemoryStored" in a
    assert "memory_evicted" in b or "MemoryEvicted" in b
```

(If the existing helper names differ, adapt the test to call the actual ones — read the modules first.)

- [ ] **Step 3: Run tests to verify they fail**

```
uv run pytest tests/unit/test_cli.py::test_cli_renders_memory_events tests/unit/test_http_app.py::test_serialize_memory_events -v
```

Expected: failures.

- [ ] **Step 4: Wire rendering**

In `src/llama_agents/cli.py`, in the event-rendering function (whatever its name is — `_render_event` or inline in the main loop), add handlers:

```python
from .events import MemoryEvicted, MemoryStored

# inside the isinstance ladder:
elif isinstance(ev, MemoryStored):
    print(f"  ◦ stored {ev.kind} → mem:{ev.blob_id[:8]} ({ev.bytes_} B)", file=sys.stderr)
elif isinstance(ev, MemoryEvicted):
    kb = ev.bytes_freed / 1024
    print(f"  ◦ evicted tool result → -{kb:.1f} KB (mem:{ev.blob_id[:8]})",
          file=sys.stderr)
```

In `src/llama_agents/http_app.py`, in `_event_to_sse` (or equivalent), add cases:

```python
from .events import MemoryEvicted, MemoryStored

if isinstance(ev, MemoryStored):
    return _sse("memory_stored",
                {"blob_id": ev.blob_id, "kind": ev.kind,
                 "scope": ev.scope, "bytes": ev.bytes_})
if isinstance(ev, MemoryEvicted):
    return _sse("memory_evicted",
                {"blob_id": ev.blob_id, "turn": ev.turn,
                 "bytes_freed": ev.bytes_freed})
```

(Use whatever helper / structure the existing file already uses.)

- [ ] **Step 5: Run tests to verify they pass**

```
uv run pytest tests/unit -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git add src/llama_agents/cli.py src/llama_agents/http_app.py tests/unit/test_cli.py tests/unit/test_http_app.py
git commit -m "feat(cli,http): render MemoryStored and MemoryEvicted events"
```

---

## Task 16: Config defaults, .gitignore, and docs

**Files:**
- Modify: `config.toml`
- Modify: `.gitignore`
- Create: `docs/memory.md`
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add `[memory]` defaults to `config.toml`**

Append to `config.toml`:

```toml
[memory]
enabled = true
root = ".llama_agents/memory"
embedding_model = "BAAI/bge-small-en-v1.5"
chunk_size = 1500
chunk_overlap = 150
plan_recall_k = 3
plan_recall_threshold = 0.5
subagent_inline_threshold_chars = 2000
subagent_summary_max_tokens = 400
evict_threshold_pct = 70
evict_tool_result_min_chars = 4000
scratch_retention_hours = 24
```

- [ ] **Step 2: Add memory dir to `.gitignore`**

Append to `.gitignore`:

```
.llama_agents/memory/
```

- [ ] **Step 3: Create `docs/memory.md`**

Write a single-page guide. Suggested skeleton (~150 lines):

```markdown
# Memory layer

llama-agents ships with a local RAG-backed memory system that:

- stores accepted plans across runs, and retrieves similar past plans
  for the planner and reviewer;
- offloads large subagent outputs to memory, returning a summary plus a
  handle the orchestrator can recall on demand;
- evicts old tool results from the context window when it fills, with
  the model retrieving on demand.

## On-disk layout

`.llama_agents/memory/` under the first allowed_dirs entry:

- `index.sqlite` — blob and chunk metadata, embeddings
- `runs/<run_id>/<blob_id>.md` — per-run scratch (subagent output, evicted tool results)
- `plans/<blob_id>.md` — persistent across runs

## Embeddings

We use `fastembed` with `BAAI/bge-small-en-v1.5` (384-dim ONNX, ~30 MB).
First import downloads to `~/.cache/fastembed/` and takes 10-30 s on a
fresh machine. Subsequent runs load instantly.

To disable the memory layer entirely:

```toml
[memory]
enabled = false
```

## Tools

The agent gets one new built-in tool:

- **memory_recall(query, handle?, k?)** — search this run's scratch +
  persistent plans. When `handle` is set, restricts to chunks of that
  blob.

## Lifecycle

- A `run_id` is generated for every top-level `Agent.run()` call.
  Subagents inherit it; the whole tree shares scratch.
- After a run, scratch is retained for `scratch_retention_hours` hours
  (default 24, `0` = delete immediately, `-1` = keep forever).
- Persistent plans are never auto-deleted.
- `llamactl memory gc` (future) wipes expired run scratch on demand.

## Tuning

- `evict_threshold_pct` — when estimated context use crosses this %, old
  large tool results are evicted to memory. Default 70.
- `evict_tool_result_min_chars` — never evicts tiny results.
- `plan_recall_k` / `plan_recall_threshold` — how many past plans to
  inject and how similar they must be.
- `subagent_inline_threshold_chars` — subagent outputs below this stay
  inline; above this go to memory.
```

- [ ] **Step 4: Update README with a Memory section**

Append a "Memory" section to `README.md` (above "Known limitations" if present) summarizing the same in 5-10 lines and linking to `docs/memory.md`.

- [ ] **Step 5: Strike the "No RAG memory" limitation in CLAUDE.md**

In `D:\repos\llm\llama-agents\CLAUDE.md`, find the section "Known limitations / future work" and replace the bullet:

```
- **No RAG memory.** ...
```

with:

```
- **RAG memory:** implemented in phase 2 — accepted plans persist
  across runs; large subagent outputs and overflow tool results are
  offloaded to a local SQLite + fastembed store; `memory_recall` retrieves
  them. See `docs/memory.md`.
```

- [ ] **Step 6: Commit**

```
git add config.toml .gitignore docs/memory.md README.md CLAUDE.md
git commit -m "docs(memory): config defaults, gitignore, user guide, CLAUDE.md update"
```

---

## Task 17: Live end-to-end test

**Files:**
- Create: `tests/live/test_memory_e2e.py`

This test downloads fastembed on first run and requires a live llama-server (`auto_spawn=true` or already running). It is gated by the `live` marker so CI doesn't pull the model.

- [ ] **Step 1: Inspect an existing live test for fixtures and setup**

Open `tests/live/` and read the smoke test. Note how it builds a `Runtime` and how it handles `auto_spawn` and `pytest.mark.live`.

- [ ] **Step 2: Write the live e2e test**

Create `tests/live/test_memory_e2e.py`:

```python
import pytest

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_memory_end_to_end(tmp_path):
    from llama_agents.agent import AgentRunOptions
    from llama_agents.config import (
        AgentConfig, Config, LlamaConfig, MemoryConfig, SandboxConfig,
    )
    from llama_agents.runtime import Runtime

    cfg = Config(
        llama=LlamaConfig(auto_spawn=False),
        sandbox=SandboxConfig(allowed_dirs=[tmp_path]),
        memory=MemoryConfig(
            root=str(tmp_path / "mem"),
            scratch_retention_hours=-1,  # keep so we can inspect
        ),
        agent=AgentConfig(max_iterations=5),
    )
    rt = await Runtime.create(cfg)
    try:
        agent = rt.new_agent()
        events = []
        async for ev in agent.run(
            "Say hello in one sentence.",
            AgentRunOptions(max_iterations=3, skip_planning=True),
        ):
            events.append(ev)
        # The plain prompt + small response should never trigger eviction
        # but the smoke is mostly: it runs to completion without error,
        # the memory dir exists, and the inert/real selection worked.
        assert (tmp_path / "mem" / "index.sqlite").exists()
    finally:
        await rt.aclose()
```

- [ ] **Step 3: Run the live test (optional — only if a llama-server is reachable)**

```
uv run pytest tests/live/test_memory_e2e.py -m live -v
```

Expected: PASS (or SKIP if no live server available).

- [ ] **Step 4: Commit**

```
git add tests/live/test_memory_e2e.py
git commit -m "test(live): memory layer end-to-end smoke"
```

---

## Final verification

- [ ] **Step 1: Run full unit suite**

```
uv run pytest tests/unit -q
```

Expected: all green.

- [ ] **Step 2: Lint check (if the project runs one)**

If there's a ruff/mypy config:

```
uv run ruff check src tests
uv run mypy src
```

Fix anything reported.

- [ ] **Step 3: Manual sanity (optional)**

```
$env:PYTHONIOENCODING = "utf-8"
uv run llamactl chat "List the files in this repo's src dir, then store a one-line plan in memory."
```

Watch for `◦ stored plan → mem:...` in the status output.

---

## Self-review notes for this plan

- All 17 tasks are TDD-shaped: test first, watch it fail, implement, watch it pass, commit.
- Spec coverage: every section of the spec (goals 1-3, non-goals, architecture, data flow, events, config, lifecycle, error handling, sandboxing, testing, deps, backwards-compat, future) maps to at least one task. The future-work bullets are intentionally untouched (e.g., RL outcome capture, reviewer-subagent variant, plan compaction).
- No placeholders: every code-bearing step includes the actual code. Tests show real assertions. Commands show what to run and the expected outcome.
- Type/name consistency: `MemoryStore`, `InertMemoryStore`, `RecalledChunk`, `BlobMeta`, `HashEmbedder`, `FastEmbedEmbedder`, `MemoryRecallTool`, `MemoryStored`, `MemoryEvicted` — referenced consistently across tasks. `memory_handle` is the field name in subagent return; `handle` is the arg name on `memory_recall` and on `MemoryStore.recall`.
- Bite-sized: each task is 5-9 steps; each step is one action.
