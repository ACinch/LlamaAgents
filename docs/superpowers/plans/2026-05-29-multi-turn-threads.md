# Multi-Turn Threads + Prompt History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-shot inbox/done/failed queue model with thread-centric storage where every submission belongs to a durable thread that supports follow-up turns and rerun-as-fork with parent memory inheritance.

**Architecture:** New `src/llama_agents/thread/` package (ids, meta, status, store, migration). Memory layer migrates `run_id` → `thread_id` via a one-shot SQLite schema migration. Agent gains `prior_messages` hydration. Queue worker scans `threads/*/turns/*/status=queued` instead of `inbox/`. New web pages `/activity` (rename of dashboard) + `/threads` + `/threads/{id}` replace the old job-detail views. CLI gains `--thread`, `--background`, and a `threads` subcommand group.

**Tech Stack:** Python 3.12+, pydantic, asyncio, SQLite (`pragma user_version` for schema migration), FastAPI + Jinja2 + HTMX, Typer for CLI, pytest + httpx + asgi-lifespan.

**Spec:** `docs/superpowers/specs/2026-05-29-multi-turn-threads-design.md`

---

## File structure (locked in by this plan)

**New module — thread storage primitives:**
- `src/llama_agents/thread/__init__.py`
- `src/llama_agents/thread/ids.py`
- `src/llama_agents/thread/meta.py`
- `src/llama_agents/thread/status.py`
- `src/llama_agents/thread/store.py`
- `src/llama_agents/thread/migration.py`

**New web templates:**
- `src/llama_agents/web/templates/activity.html` (rename of dashboard.html)
- `src/llama_agents/web/templates/threads.html`
- `src/llama_agents/web/templates/thread.html`

**New unit tests:**
- `tests/unit/test_thread_ids.py`
- `tests/unit/test_thread_meta.py`
- `tests/unit/test_thread_status.py`
- `tests/unit/test_thread_store.py`
- `tests/unit/test_thread_migration.py`
- `tests/unit/test_cli_threads.py`

**New live test + docs:**
- `tests/live/test_thread_e2e.py`
- `docs/threads.md`

**Modified:**
- `src/llama_agents/memory/db.py` (schema migration, `recall(thread_ids=...)`)
- `src/llama_agents/memory/store.py` (every `run_id` field renames; `recall` widens)
- `src/llama_agents/memory/types.py` (`BlobMeta.run_id` → `BlobMeta.thread_id`)
- `src/llama_agents/tools/builtin/memory.py` (ancestor chain → `recall`)
- `src/llama_agents/tools/builtin/subagent.py` (rename run_id propagation)
- `src/llama_agents/agent.py` (`_ACTIVE_RUN_ID` → `_ACTIVE_THREAD_ID`; `prior_messages` param)
- `src/llama_agents/runtime.py` (`Runtime.thread_store`)
- `src/llama_agents/queue/worker.py` (thread-scan pickup; turn-folder finalize)
- `src/llama_agents/queue/paths.py` (trimmed)
- `src/llama_agents/web/routes.py` (new routes; `_list_jobs` → `_list_turns`)
- `src/llama_agents/web/templates/base.html` (nav update)
- `src/llama_agents/cli.py` (`chat` flags; `threads` subcommand group)
- `tests/unit/test_queue_worker.py` (rewrite to use thread folders)
- `tests/unit/test_agent_loop.py` (hydration test, ancestor chain test, rename `run_id`)
- `tests/unit/test_memory_db.py` (schema migration test; `thread_ids` list semantics)
- `tests/unit/test_memory_store.py` (rename + widen recall)
- `tests/unit/test_web_routes.py` (rewrite for new routes)
- `CLAUDE.md` (module map; strike "no multi-turn" limitation)
- `docs/web.md`, `docs/install.md` (point at new nav)

**Deleted:**
- `src/llama_agents/web/templates/dashboard.html` (renamed → activity.html)
- `src/llama_agents/web/templates/job.html`
- `src/llama_agents/web/templates/_partials/job_list.html`
- `src/llama_agents/web/templates/_partials/job_row.html`

---

## Conventions for this plan

- **Always run from the repo root** (`D:\repos\llm\llama-agents`).
- **Always use `uv run pytest ...`** for tests. PowerShell prefix on Windows when `uv` isn't on PATH:
  ```
  $env:Path = "$env:USERPROFILE\AppData\Roaming\Python\Python314\Scripts;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; uv run pytest ...
  ```
- **Commit after each task** with the verbatim message in each task's final step.
- **Branch:** `main`.
- **TDD:** every behavioural task writes failing tests first, then minimal code to pass.
- No new external deps (still on jinja2, fastapi, sqlite, httpx, asgi-lifespan, typer).

---

## Task 1: `thread/ids.py` — id minting + prefix resolution

**Files:**
- Create: `src/llama_agents/thread/__init__.py` (empty)
- Create: `src/llama_agents/thread/ids.py`
- Create: `tests/unit/test_thread_ids.py`

- [ ] **Step 1: Create the package marker**

Create empty file `src/llama_agents/thread/__init__.py`.

- [ ] **Step 2: Write failing tests**

Create `tests/unit/test_thread_ids.py`:

```python
from pathlib import Path

import pytest

from llama_agents.thread.ids import (
    AmbiguousPrefix,
    UnknownPrefix,
    mint_thread_id,
    resolve_prefix,
    validate_thread_id,
)


def test_mint_thread_id_returns_24_hex():
    a = mint_thread_id()
    assert len(a) == 24
    assert all(c in "0123456789abcdef" for c in a)


def test_mint_thread_id_is_unique():
    ids = {mint_thread_id() for _ in range(100)}
    assert len(ids) == 100


def test_validate_accepts_24_hex():
    assert validate_thread_id("0123456789abcdef01234567") is True


def test_validate_rejects_wrong_length():
    assert validate_thread_id("0123") is False
    assert validate_thread_id("0" * 32) is False


def test_validate_rejects_non_hex():
    assert validate_thread_id("g" + "0" * 23) is False


def test_resolve_prefix_unique(tmp_path: Path):
    (tmp_path / "8c9f2bd6e041a3b5708141d9").mkdir()
    (tmp_path / "4e1a72fd0000000000000000").mkdir()
    assert resolve_prefix(tmp_path, "8c9f") == "8c9f2bd6e041a3b5708141d9"


def test_resolve_prefix_full_id_passthrough(tmp_path: Path):
    full = "8c9f2bd6e041a3b5708141d9"
    (tmp_path / full).mkdir()
    assert resolve_prefix(tmp_path, full) == full


def test_resolve_prefix_ambiguous_raises(tmp_path: Path):
    (tmp_path / "8c9f000000000000000000aa").mkdir()
    (tmp_path / "8c9f000000000000000000bb").mkdir()
    with pytest.raises(AmbiguousPrefix) as ei:
        resolve_prefix(tmp_path, "8c9f")
    assert "8c9f000000000000000000aa" in str(ei.value)
    assert "8c9f000000000000000000bb" in str(ei.value)


def test_resolve_prefix_unknown_raises(tmp_path: Path):
    (tmp_path / "8c9f000000000000000000aa").mkdir()
    with pytest.raises(UnknownPrefix):
        resolve_prefix(tmp_path, "ffff")


def test_resolve_prefix_too_short_raises(tmp_path: Path):
    (tmp_path / "8c9f000000000000000000aa").mkdir()
    with pytest.raises(ValueError, match="at least 4"):
        resolve_prefix(tmp_path, "8c")


def test_resolve_prefix_root_missing_raises_unknown(tmp_path: Path):
    with pytest.raises(UnknownPrefix):
        resolve_prefix(tmp_path / "nonexistent", "abcd")
```

- [ ] **Step 3: Run tests — expect ImportError**

```
$env:Path = "$env:USERPROFILE\AppData\Roaming\Python\Python314\Scripts;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; uv run pytest tests/unit/test_thread_ids.py -v
```

- [ ] **Step 4: Implement `ids.py`**

Create `src/llama_agents/thread/ids.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

```
uv run pytest tests/unit/test_thread_ids.py -v
```

Expected: 11 tests pass.

- [ ] **Step 6: Commit**

```
git add src/llama_agents/thread/__init__.py src/llama_agents/thread/ids.py tests/unit/test_thread_ids.py
git commit -m "feat(thread): id minting, validation, and prefix resolution"
```

---

## Task 2: `thread/meta.py` — ThreadMeta dataclass + JSON I/O

**Files:**
- Create: `src/llama_agents/thread/meta.py`
- Create: `tests/unit/test_thread_meta.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_thread_meta.py`:

```python
import json
from pathlib import Path

import pytest

from llama_agents.thread.meta import ThreadMeta, read_meta, update_meta, write_meta


def test_thread_meta_defaults():
    m = ThreadMeta(id="0123456789abcdef01234567", title="hi",
                   created_at="2026-05-29T10:00:00+00:00",
                   updated_at="2026-05-29T10:00:00+00:00",
                   current_turn=1)
    assert m.parent_thread_id is None
    assert m.parent_turn_idx is None
    assert m.current_turn == 1


def test_write_read_roundtrip(tmp_path: Path):
    m = ThreadMeta(
        id="aaaa" + "0" * 20, title="hello",
        created_at="2026-05-29T10:00:00+00:00",
        updated_at="2026-05-29T10:00:01+00:00",
        current_turn=2,
        parent_thread_id="bbbb" + "0" * 20,
        parent_turn_idx=1,
    )
    write_meta(tmp_path, m)
    got = read_meta(tmp_path, m.id)
    assert got == m


def test_read_meta_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_meta(tmp_path, "aaaa" + "0" * 20)


def test_read_meta_malformed_json_raises(tmp_path: Path):
    tid = "aaaa" + "0" * 20
    d = tmp_path / tid
    d.mkdir()
    (d / "meta.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        read_meta(tmp_path, tid)


def test_update_meta_preserves_unspecified_fields(tmp_path: Path):
    m = ThreadMeta(id="cccc" + "0" * 20, title="original",
                   created_at="2026-05-29T10:00:00+00:00",
                   updated_at="2026-05-29T10:00:00+00:00",
                   current_turn=1)
    write_meta(tmp_path, m)
    got = update_meta(tmp_path, m.id, title="renamed", current_turn=2)
    assert got.title == "renamed"
    assert got.current_turn == 2
    assert got.created_at == m.created_at
    # updated_at is bumped automatically
    assert got.updated_at != m.updated_at


def test_meta_json_field_order_stable(tmp_path: Path):
    """The on-disk JSON should be human-readable: pretty-printed."""
    m = ThreadMeta(id="dddd" + "0" * 20, title="x",
                   created_at="2026-05-29T10:00:00+00:00",
                   updated_at="2026-05-29T10:00:00+00:00",
                   current_turn=1)
    write_meta(tmp_path, m)
    text = (tmp_path / m.id / "meta.json").read_text(encoding="utf-8")
    parsed = json.loads(text)
    assert parsed["id"] == m.id
    assert "\n" in text  # pretty-printed (indent=2)
```

- [ ] **Step 2: Run tests — expect ImportError**

```
uv run pytest tests/unit/test_thread_meta.py -v
```

- [ ] **Step 3: Implement `meta.py`**

Create `src/llama_agents/thread/meta.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_thread_meta.py -v
```

Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```
git add src/llama_agents/thread/meta.py tests/unit/test_thread_meta.py
git commit -m "feat(thread): ThreadMeta dataclass + atomic JSON read/write/update"
```

---

## Task 3: `thread/status.py` — atomic status transitions

**Files:**
- Create: `src/llama_agents/thread/status.py`
- Create: `tests/unit/test_thread_status.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_thread_status.py`:

```python
from pathlib import Path

import pytest

from llama_agents.thread.status import (
    claim_for_processing,
    read_status,
    revert_processing_on_startup,
    set_status,
)


def test_read_status_returns_none_when_missing(tmp_path: Path):
    assert read_status(tmp_path) is None


def test_set_status_creates_file(tmp_path: Path):
    set_status(tmp_path, "queued")
    assert read_status(tmp_path) == "queued"


def test_set_status_overwrites_atomically(tmp_path: Path):
    set_status(tmp_path, "queued")
    set_status(tmp_path, "processing")
    set_status(tmp_path, "done")
    assert read_status(tmp_path) == "done"


def test_set_status_rejects_invalid_value(tmp_path: Path):
    with pytest.raises(ValueError, match="invalid status"):
        set_status(tmp_path, "running")


def test_claim_for_processing_flips_queued_to_processing(tmp_path: Path):
    set_status(tmp_path, "queued")
    assert claim_for_processing(tmp_path) is True
    assert read_status(tmp_path) == "processing"


def test_claim_for_processing_refuses_when_not_queued(tmp_path: Path):
    set_status(tmp_path, "processing")
    assert claim_for_processing(tmp_path) is False
    assert read_status(tmp_path) == "processing"


def test_claim_for_processing_refuses_when_missing(tmp_path: Path):
    assert claim_for_processing(tmp_path) is False


def test_revert_processing_on_startup_finds_and_resets(tmp_path: Path):
    # Build a fake threads tree: two threads, three turns total, two in
    # 'processing' from a prior crash.
    t1 = tmp_path / "aaaa" + "0" * 20 / "turns" / "001"
    t2 = tmp_path / "aaaa" + "0" * 20 / "turns" / "002"
    t3 = tmp_path / "bbbb" + "0" * 20 / "turns" / "001"
    for d in (t1, t2, t3):
        d.mkdir(parents=True)
    set_status(t1, "processing")
    set_status(t2, "done")
    set_status(t3, "processing")
    n = revert_processing_on_startup(tmp_path)
    assert n == 2
    assert read_status(t1) == "queued"
    assert read_status(t2) == "done"
    assert read_status(t3) == "queued"


def test_revert_processing_on_startup_with_empty_root_returns_zero(tmp_path: Path):
    empty = tmp_path / "nope"
    assert revert_processing_on_startup(empty) == 0
```

- [ ] **Step 2: Run tests — expect ImportError**

- [ ] **Step 3: Implement `status.py`**

Create `src/llama_agents/thread/status.py`:

```python
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
```

- [ ] **Step 4: Run tests — expect pass**

```
uv run pytest tests/unit/test_thread_status.py -v
```

Note: the `t1 = tmp_path / "aaaa" + "0" * 20 / "turns" / "001"` expression is operator-precedence sensitive. If pytest fails with a TypeError, wrap with parens: `t1 = tmp_path / ("aaaa" + "0" * 20) / "turns" / "001"`. Update the test if needed.

- [ ] **Step 5: Run unit suite to confirm no regression**

```
uv run pytest tests/unit -q
```

- [ ] **Step 6: Commit**

```
git add src/llama_agents/thread/status.py tests/unit/test_thread_status.py
git commit -m "feat(thread): atomic per-turn status transitions + crash recovery sweep"
```

---

## Task 4: `thread/store.py` part 1 — create/list/turn helpers

**Files:**
- Create: `src/llama_agents/thread/store.py`
- Create: `tests/unit/test_thread_store.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_thread_store.py`:

```python
import time
from pathlib import Path

import pytest

from llama_agents.thread.meta import read_meta
from llama_agents.thread.status import set_status
from llama_agents.thread.store import ThreadStore


def test_create_thread_writes_meta_and_turn1_folder(tmp_path: Path):
    store = ThreadStore(tmp_path)
    tid = store.create_thread(title="hello")
    assert (tmp_path / tid / "meta.json").is_file()
    assert (tmp_path / tid / "turns" / "001").is_dir()
    meta = read_meta(tmp_path, tid)
    assert meta.title == "hello"
    assert meta.current_turn == 1
    assert meta.parent_thread_id is None


def test_create_thread_records_parent(tmp_path: Path):
    store = ThreadStore(tmp_path)
    parent = store.create_thread(title="parent")
    child = store.create_thread(title="child", parent_thread_id=parent,
                                parent_turn_idx=2)
    cm = read_meta(tmp_path, child)
    assert cm.parent_thread_id == parent
    assert cm.parent_turn_idx == 2


def test_list_threads_empty(tmp_path: Path):
    assert ThreadStore(tmp_path).list_threads() == []


def test_list_threads_sorted_by_updated_at_desc(tmp_path: Path):
    store = ThreadStore(tmp_path)
    a = store.create_thread(title="first")
    time.sleep(0.01)
    b = store.create_thread(title="second")
    time.sleep(0.01)
    c = store.create_thread(title="third")
    metas = store.list_threads()
    assert [m.id for m in metas] == [c, b, a]


def test_list_threads_respects_limit(tmp_path: Path):
    store = ThreadStore(tmp_path)
    for _ in range(5):
        store.create_thread(title="t")
        time.sleep(0.005)
    assert len(store.list_threads(limit=3)) == 3
    assert len(store.list_threads(limit=None)) == 5


def test_turn_dir_returns_zero_padded(tmp_path: Path):
    store = ThreadStore(tmp_path)
    tid = store.create_thread(title="t")
    assert store.turn_dir(tid, 1).name == "001"
    assert store.turn_dir(tid, 17).name == "017"
    assert store.turn_dir(tid, 9999).name == "9999"


def test_next_turn_dir_increments_current_turn(tmp_path: Path):
    store = ThreadStore(tmp_path)
    tid = store.create_thread(title="t")
    # turn 1 already exists from create_thread; next is 2
    d, idx = store.next_turn_dir(tid)
    assert idx == 2
    assert d.name == "002"
    assert d.is_dir()
    assert read_meta(tmp_path, tid).current_turn == 2


def test_next_queued_turn_orders_by_mtime(tmp_path: Path):
    store = ThreadStore(tmp_path)
    a = store.create_thread(title="a")
    set_status(store.turn_dir(a, 1), "queued")
    time.sleep(0.02)
    b = store.create_thread(title="b")
    set_status(store.turn_dir(b, 1), "queued")
    result = store.next_queued_turn()
    assert result is not None
    tid, idx = result
    assert tid == a  # older mtime
    assert idx == 1


def test_next_queued_turn_returns_none_when_nothing_queued(tmp_path: Path):
    store = ThreadStore(tmp_path)
    tid = store.create_thread(title="x")
    set_status(store.turn_dir(tid, 1), "done")
    assert store.next_queued_turn() is None
```

- [ ] **Step 2: Run tests — expect ImportError**

- [ ] **Step 3: Implement store.py part 1**

Create `src/llama_agents/thread/store.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_thread_store.py -v
```

Expected: 9 tests pass.

- [ ] **Step 5: Commit**

```
git add src/llama_agents/thread/store.py tests/unit/test_thread_store.py
git commit -m "feat(thread): ThreadStore create/list/turn-dir/next-queued primitives"
```

---

## Task 5: `thread/store.py` part 2 — messages.jsonl + ancestor_chain

**Files:**
- Modify: `src/llama_agents/thread/store.py`
- Modify: `tests/unit/test_thread_store.py`

- [ ] **Step 1: Append failing tests**

Add to `tests/unit/test_thread_store.py`:

```python
def test_read_messages_returns_empty_when_no_file(tmp_path: Path):
    store = ThreadStore(tmp_path)
    tid = store.create_thread(title="t")
    assert store.read_messages(tid) == []


def test_append_then_read_roundtrip(tmp_path: Path):
    store = ThreadStore(tmp_path)
    tid = store.create_thread(title="t")
    store.append_messages(tid, [
        {"role": "system", "content": "you are X"},
        {"role": "user", "content": "hello"},
    ])
    store.append_messages(tid, [{"role": "assistant", "content": "hi"}])
    msgs = store.read_messages(tid)
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[2]["content"] == "hi"


def test_append_messages_empty_list_is_noop(tmp_path: Path):
    store = ThreadStore(tmp_path)
    tid = store.create_thread(title="t")
    store.append_messages(tid, [])
    assert store.read_messages(tid) == []


def test_ancestor_chain_linear(tmp_path: Path):
    store = ThreadStore(tmp_path)
    a = store.create_thread(title="root")
    b = store.create_thread(title="b", parent_thread_id=a, parent_turn_idx=1)
    c = store.create_thread(title="c", parent_thread_id=b, parent_turn_idx=1)
    assert store.ancestor_chain(a) == []
    assert store.ancestor_chain(b) == [a]
    assert store.ancestor_chain(c) == [b, a]


def test_ancestor_chain_capped_at_depth(tmp_path: Path):
    """Defensive cap: a malformed cyclic chain should not infinite-loop."""
    store = ThreadStore(tmp_path)
    a = store.create_thread(title="a")
    # Manually corrupt: make a's parent point to itself.
    from llama_agents.thread.meta import read_meta, write_meta
    m = read_meta(tmp_path, a)
    m.parent_thread_id = a
    m.parent_turn_idx = 1
    write_meta(tmp_path, m)
    # ancestor_chain should terminate (defensive cap) rather than spin.
    chain = store.ancestor_chain(a)
    assert len(chain) <= 32
```

- [ ] **Step 2: Run new tests — expect failure**

- [ ] **Step 3: Extend `store.py`**

Append to `src/llama_agents/thread/store.py`, inside `ThreadStore`:

```python
    # ---------- messages.jsonl ----------

    def _messages_path(self, thread_id: str) -> Path:
        return self._root / thread_id / "messages.jsonl"

    def read_messages(self, thread_id: str) -> list[dict]:
        import json as _json
        p = self._messages_path(thread_id)
        if not p.is_file():
            return []
        out: list[dict] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(_json.loads(line))
            except _json.JSONDecodeError:
                continue
        return out

    def append_messages(self, thread_id: str, new_msgs: list[dict]) -> None:
        import json as _json
        import os as _os
        if not new_msgs:
            return
        p = self._messages_path(thread_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        existing = p.read_text(encoding="utf-8") if p.is_file() else ""
        appended = "\n".join(_json.dumps(m, ensure_ascii=False) for m in new_msgs)
        tmp = p.with_suffix(".jsonl.tmp")
        tmp.write_text(
            existing + appended + "\n", encoding="utf-8"
        )
        _os.replace(tmp, p)

    # ---------- ancestor walking ----------

    _MAX_ANCESTOR_DEPTH = 32

    def ancestor_chain(self, thread_id: str) -> list[str]:
        """Return ancestor thread ids walking parent_thread_id, root first.

        Defensive depth cap of 32 prevents infinite loops on malformed
        cyclic metadata.
        """
        chain: list[str] = []
        seen: set[str] = {thread_id}
        cur = thread_id
        for _ in range(self._MAX_ANCESTOR_DEPTH):
            meta = read_meta(self._root, cur)
            parent = meta.parent_thread_id
            if not parent or parent in seen:
                break
            chain.append(parent)
            seen.add(parent)
            cur = parent
        return chain
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_thread_store.py -v
```

Expected: 14 tests pass total.

- [ ] **Step 5: Commit**

```
git add src/llama_agents/thread/store.py tests/unit/test_thread_store.py
git commit -m "feat(thread): messages.jsonl append/read + ancestor_chain (depth-capped)"
```

---

## Task 6: `thread/migration.py` — legacy queue → threads/

**Files:**
- Create: `src/llama_agents/thread/migration.py`
- Create: `tests/unit/test_thread_migration.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_thread_migration.py`:

```python
from pathlib import Path

from llama_agents.thread.migration import migrate_legacy_queue_dirs
from llama_agents.thread.meta import read_meta
from llama_agents.thread.status import read_status


def test_no_legacy_dirs_returns_zero(tmp_path: Path):
    assert migrate_legacy_queue_dirs(tmp_path) == 0


def test_migrate_inbox_creates_thread(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "task-1.md").write_text("the prompt body", encoding="utf-8")
    assert migrate_legacy_queue_dirs(tmp_path) == 1
    threads_root = tmp_path / "threads"
    threads = [p for p in threads_root.iterdir() if p.is_dir()]
    assert len(threads) == 1
    tid = threads[0].name
    meta = read_meta(threads_root, tid)
    assert meta.title.startswith("the prompt body"[:60])
    assert meta.current_turn == 1
    turn1 = threads_root / tid / "turns" / "001"
    assert (turn1 / "prompt.md").read_text(encoding="utf-8") == "the prompt body"
    assert read_status(turn1) == "queued"
    # source file removed
    assert not (inbox / "task-1.md").exists()


def test_migrate_done_with_sidecars(tmp_path: Path):
    done = tmp_path / "done"
    done.mkdir()
    (done / "x.md").write_text("FINAL ANSWER", encoding="utf-8")
    (done / "x.prompt.md").write_text("ORIGINAL PROMPT", encoding="utf-8")
    (done / "x.events.jsonl").write_text('{"type":"Done"}\n', encoding="utf-8")
    n = migrate_legacy_queue_dirs(tmp_path)
    assert n == 1
    threads_root = tmp_path / "threads"
    tid = [p.name for p in threads_root.iterdir() if p.is_dir()][0]
    turn1 = threads_root / tid / "turns" / "001"
    assert read_status(turn1) == "done"
    assert (turn1 / "result.md").read_text(encoding="utf-8") == "FINAL ANSWER"
    assert (turn1 / "prompt.md").read_text(encoding="utf-8") == "ORIGINAL PROMPT"
    assert "Done" in (turn1 / "events.jsonl").read_text(encoding="utf-8")


def test_migrate_failed_with_error_txt(tmp_path: Path):
    failed = tmp_path / "failed"
    failed.mkdir()
    (failed / "b.md").write_text("(no final answer)", encoding="utf-8")
    (failed / "b.prompt.md").write_text("trigger", encoding="utf-8")
    (failed / "b.events.jsonl").write_text("", encoding="utf-8")
    (failed / "b.error.txt").write_text("attempts: 1\n", encoding="utf-8")
    migrate_legacy_queue_dirs(tmp_path)
    threads_root = tmp_path / "threads"
    tid = [p.name for p in threads_root.iterdir() if p.is_dir()][0]
    turn1 = threads_root / tid / "turns" / "001"
    assert read_status(turn1) == "failed"
    assert (turn1 / "error.txt").read_text(encoding="utf-8") == "attempts: 1\n"


def test_migrate_is_idempotent(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "t.md").write_text("p", encoding="utf-8")
    n1 = migrate_legacy_queue_dirs(tmp_path)
    n2 = migrate_legacy_queue_dirs(tmp_path)
    assert n1 == 1
    assert n2 == 0


def test_migrate_processing_treated_as_queued(tmp_path: Path):
    """Processing files from a prior crash become queued so the worker
    picks them up again."""
    proc = tmp_path / "processing"
    proc.mkdir()
    (proc / "p.md").write_text("stuck job", encoding="utf-8")
    migrate_legacy_queue_dirs(tmp_path)
    threads_root = tmp_path / "threads"
    tid = [p.name for p in threads_root.iterdir() if p.is_dir()][0]
    turn1 = threads_root / tid / "turns" / "001"
    assert read_status(turn1) == "queued"


def test_migrate_ignores_sidecar_only_files(tmp_path: Path):
    """A .prompt.md or .events.jsonl without a matching .md is not migrated
    on its own."""
    done = tmp_path / "done"
    done.mkdir()
    (done / "orphan.prompt.md").write_text("strange", encoding="utf-8")
    n = migrate_legacy_queue_dirs(tmp_path)
    assert n == 0
```

- [ ] **Step 2: Run tests — expect ImportError**

- [ ] **Step 3: Implement `migration.py`**

Create `src/llama_agents/thread/migration.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_thread_migration.py -v
```

Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```
git add src/llama_agents/thread/migration.py tests/unit/test_thread_migration.py
git commit -m "feat(thread): idempotent legacy-queue migration into threads/"
```

---

## Task 7: Memory schema migration — `run_id` → `thread_id`

**Files:**
- Modify: `src/llama_agents/memory/db.py`
- Modify: `src/llama_agents/memory/types.py`
- Modify: `tests/unit/test_memory_db.py`

The column rename has to land before any callers rename their own parameters, otherwise the existing tests break en masse. We do the schema migration here and update existing callers in Task 8.

- [ ] **Step 1: Write a failing migration test**

Add to `tests/unit/test_memory_db.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect failure**

```
uv run pytest tests/unit/test_memory_db.py::test_schema_migration_renames_run_id_to_thread_id tests/unit/test_memory_db.py::test_fresh_database_uses_thread_id_column -v
```

- [ ] **Step 3: Update `BlobMeta` in `memory/types.py`**

Read `src/llama_agents/memory/types.py`. Find the `BlobMeta` dataclass and rename `run_id` → `thread_id`:

```python
@dataclass
class BlobMeta:
    id: str
    scope: str
    thread_id: str | None     # renamed from run_id
    kind: str
    title: str
    file_path: str
    metadata: dict
    created_at: str
```

- [ ] **Step 4: Update `VectorDB` in `memory/db.py`**

Rewrite the `init()` method to perform the migration and use the new column name:

```python
    async def init(self) -> None:
        async with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._path)
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA foreign_keys = ON")
            # Schema migration: rename run_id -> thread_id once, gated on
            # user_version. Fresh DBs create the table with thread_id
            # already.
            ver = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if ver < 1:
                self._migrate_to_v1()
            else:
                self._create_schema_v1_if_absent()
            self._conn.commit()

    def _create_schema_v1_if_absent(self) -> None:
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS blobs (
              id            TEXT PRIMARY KEY,
              scope         TEXT NOT NULL,
              thread_id     TEXT,
              kind          TEXT NOT NULL,
              title         TEXT NOT NULL,
              file_path     TEXT NOT NULL,
              metadata_json TEXT,
              created_at    TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_blobs_scope_thread
                ON blobs(scope, thread_id);
            CREATE TABLE IF NOT EXISTS chunks (
              id        TEXT PRIMARY KEY,
              blob_id   TEXT NOT NULL REFERENCES blobs(id) ON DELETE CASCADE,
              chunk_idx INTEGER NOT NULL,
              text      TEXT NOT NULL,
              embedding BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_blob ON chunks(blob_id);
            PRAGMA user_version = 1;
            """
        )

    def _migrate_to_v1(self) -> None:
        """Either create the v1 schema from scratch (no blobs table yet) or
        rename run_id -> thread_id in an existing v0 database."""
        assert self._conn is not None
        # Does the blobs table exist?
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='blobs'"
        ).fetchone()
        if row is None:
            self._create_schema_v1_if_absent()
            return
        # Existing v0 database; check whether it has run_id column.
        cols = [r[1] for r in self._conn.execute(
            "PRAGMA table_info(blobs)"
        ).fetchall()]
        if "run_id" in cols:
            self._conn.executescript(
                """
                ALTER TABLE blobs RENAME COLUMN run_id TO thread_id;
                DROP INDEX IF EXISTS idx_blobs_scope_run;
                CREATE INDEX IF NOT EXISTS idx_blobs_scope_thread
                    ON blobs(scope, thread_id);
                """
            )
        self._conn.execute("PRAGMA user_version = 1")
```

Then update the rest of `VectorDB` to use `thread_id` everywhere the old code used `run_id`. Find every reference (`run_id` in column names, SQL fragments, method signatures) and rename. Key signatures to update:

- `insert_blob(self, meta: BlobMeta, *, chunks: ...)` — the SQL `INSERT INTO blobs (... run_id ...) VALUES (...)` becomes `... thread_id ...`. The `meta.run_id` field access becomes `meta.thread_id`.
- `delete_blobs_for_run(self, run_id: str)` → `delete_blobs_for_thread(self, thread_id: str)`.
- `list_expired_run_ids(self, ...)` → `list_expired_thread_ids(self, ...)`. The SQL `SELECT DISTINCT run_id FROM blobs WHERE scope = 'run' AND run_id IS NOT NULL ...` becomes `SELECT DISTINCT thread_id FROM blobs WHERE scope = 'run' AND thread_id IS NOT NULL ...`.
- `list_blobs(self, *, scope, run_id=None)` → `list_blobs(self, *, scope, thread_id=None)`.
- `search(self, query_vec, *, scope, run_id=None, blob_id=None, k=5)` → `search(... thread_id=None ...)`. SQL clauses using `b.run_id` become `b.thread_id`. The OR branch in scope=="all" remains the same logic.

The blob-construction code in `list_blobs` constructs `BlobMeta(... run_id=r[2] ...)` — change to `thread_id=r[2]`.

- [ ] **Step 5: Update existing test_memory_db.py tests**

Find every usage of `run_id=` keyword in `tests/unit/test_memory_db.py` and replace with `thread_id=`. Find every `BlobMeta(... run_id=... )` and rename the keyword. Find every `db.list_expired_run_ids(...)` and rename to `list_expired_thread_ids`. Find every `db.delete_blobs_for_run(...)` and rename to `delete_blobs_for_thread`.

This is mechanical — about 12 call sites.

- [ ] **Step 6: Run memory db tests**

```
uv run pytest tests/unit/test_memory_db.py -v
```

Expected: all green, including the two new migration tests.

- [ ] **Step 7: Commit**

```
git add src/llama_agents/memory/db.py src/llama_agents/memory/types.py tests/unit/test_memory_db.py
git commit -m "feat(memory): rename run_id -> thread_id; SQLite migration v0 -> v1"
```

---

## Task 8: `MemoryStore.recall(thread_ids: list[str])`

**Files:**
- Modify: `src/llama_agents/memory/store.py`
- Modify: `src/llama_agents/memory/db.py`
- Modify: `tests/unit/test_memory_store.py`

The previous task renamed at the DB layer but `MemoryStore` still takes a single `run_id`. This task widens `recall` to accept a list and threads it through to `VectorDB.search`.

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_memory_store.py`:

```python
@pytest.mark.asyncio
async def test_recall_with_thread_ids_list_returns_union(tmp_path: Path):
    store = MemoryStore(root=tmp_path, embedder=HashEmbedder(dim=32))
    await store.init()
    store.start_run("thread_a")
    await store.store_blob(kind="user", scope="run", thread_id="thread_a",
                           title="A", body="alpha apple")
    store.start_run("thread_b")
    await store.store_blob(kind="user", scope="run", thread_id="thread_b",
                           title="B", body="alpha banana")

    hits_a = await store.recall("alpha", scope="all", thread_ids=["thread_a"])
    assert all("A" == c.title or "alpha apple" in c.text for c in hits_a)
    assert any("A" == c.title for c in hits_a)
    assert all("banana" not in c.text for c in hits_a)

    hits_both = await store.recall("alpha", scope="all",
                                   thread_ids=["thread_a", "thread_b"])
    titles = {c.title for c in hits_both}
    assert "A" in titles and "B" in titles


@pytest.mark.asyncio
async def test_recall_empty_thread_ids_returns_nothing(tmp_path: Path):
    store = MemoryStore(root=tmp_path, embedder=HashEmbedder(dim=32))
    await store.init()
    store.start_run("t")
    await store.store_blob(kind="user", scope="run", thread_id="t",
                           title="T", body="some text")
    # scope="all" but no thread_ids means "plans only" — there are no plans.
    hits = await store.recall("text", scope="all", thread_ids=[])
    assert hits == []
```

- [ ] **Step 2: Run tests — expect failure**

- [ ] **Step 3: Widen `VectorDB.search`**

In `src/llama_agents/memory/db.py`, change the `search` signature and SQL branch:

```python
    async def search(
        self,
        query_vec: list[float],
        *,
        scope: str,
        thread_ids: list[str] | None = None,
        blob_id: str | None = None,
        k: int = 5,
    ) -> list[tuple[str, str, float, str, str, str, int]]:
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
                    if thread_ids:
                        placeholders = ",".join("?" for _ in thread_ids)
                        sql += f" AND b.thread_id IN ({placeholders})"
                        params.extend(thread_ids)
                elif scope == "plans":
                    sql += " AND b.scope = 'plans'"
                elif scope == "all":
                    if thread_ids:
                        placeholders = ",".join("?" for _ in thread_ids)
                        sql += (
                            f" AND (b.scope = 'plans' OR "
                            f"(b.scope = 'run' AND b.thread_id IN ({placeholders})))"
                        )
                        params.extend(thread_ids)
                    else:
                        sql += " AND b.scope = 'plans'"
            rows = self._conn.execute(sql, params).fetchall()
        # ... rest of the function unchanged (the cosine-similarity numpy code)
```

- [ ] **Step 4: Widen `MemoryStore.recall`**

In `src/llama_agents/memory/store.py`, update `recall`:

```python
    async def recall(
        self,
        query: str,
        *,
        scope: str = "all",
        thread_ids: list[str] | None = None,
        handle: str | None = None,
        k: int = 5,
        min_score: float | None = None,
    ) -> list[RecalledChunk]:
        [qvec] = await self._embedder.embed([query])
        hits = await self._require_db().search(
            qvec, scope=scope, thread_ids=thread_ids, blob_id=handle, k=k,
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
```

Rename `MemoryStore.store_blob`'s `run_id` parameter → `thread_id`. Rename `store_plan`'s `run_id` → `thread_id`. Rename `start_run`/`end_run`/`_purge_run`/`gc_expired`/`list_handles` to take `thread_id` (start/end) or leave as-is where they walk metadata. Specifically:

- `start_run(self, run_id: str)` → `start_run(self, thread_id: str)` — also rename internal `_active_runs` to `_active_threads`.
- `end_run(self, run_id: str)` → `end_run(self, thread_id: str)`.
- `_purge_run(self, run_id: str)` → `_purge_thread(self, thread_id: str)`. The SQL helper call becomes `delete_blobs_for_thread`.
- `list_expired_run_ids` becomes `list_expired_thread_ids` (already renamed in Task 7).
- `gc_expired` calls `list_expired_thread_ids`.

Update `InertMemoryStore` to match the new signatures (no-op everywhere).

- [ ] **Step 5: Update existing `test_memory_store.py` callers**

Find every `recall(... run_id=X ...)` and replace with `thread_ids=[X]`. Find every `store_blob(... run_id=...)` and rename. Find every `start_run(X)` / `end_run(X)` — those keep the same shape but the keyword renames if any.

- [ ] **Step 6: Run memory tests**

```
uv run pytest tests/unit/test_memory_db.py tests/unit/test_memory_store.py -v
```

Expected: all green.

- [ ] **Step 7: Commit**

```
git add src/llama_agents/memory/store.py src/llama_agents/memory/db.py tests/unit/test_memory_store.py
git commit -m "feat(memory): recall takes thread_ids list; widen for ancestor inheritance"
```

---

## Task 9: Agent + tools rename `run_id` → `thread_id`; ancestor chain in memory_recall

**Files:**
- Modify: `src/llama_agents/agent.py`
- Modify: `src/llama_agents/tools/builtin/memory.py`
- Modify: `src/llama_agents/tools/builtin/subagent.py`
- Modify: `src/llama_agents/runtime.py`
- Modify: `tests/unit/test_agent_loop.py`

- [ ] **Step 1: Rename `_ACTIVE_RUN_ID` → `_ACTIVE_THREAD_ID` in agent.py**

In `src/llama_agents/agent.py`:

- `_ACTIVE_RUN_ID` → `_ACTIVE_THREAD_ID` (the contextvar at module level).
- `get_active_run_id()` → `get_active_thread_id()`.
- Inside `Agent.run`:
  - `self._run_id` → `self._thread_id` (the field).
  - The `run_id` kwarg on `run()` stays the public name for one task only — we'll rename to `thread_id` in Task 10 when we also add `prior_messages`. For now, just internally rename the field.
  - `self._memory.start_run(...)` keeps the call but with the new field name.
  - `_ACTIVE_RUN_ID.set(...)` → `_ACTIVE_THREAD_ID.set(...)`.

- [ ] **Step 2: Update memory_recall tool**

In `src/llama_agents/tools/builtin/memory.py`:

- The constructor takes `run_id_getter` — rename to `thread_id_getter`.
- The `invoke` method calls `self._store.recall(..., run_id=rid)` — change to `thread_ids=[rid]` (single-element list). The full ancestor chain widening happens in Task 11 when the queue worker has thread_store available; for now this is the minimum viable single-thread recall.

- [ ] **Step 3: Update subagent tool**

In `src/llama_agents/tools/builtin/subagent.py`:

- Find every `run_id` reference (the `parent_run_id` resolution, the call to `agent.run(..., run_id=...)`). Rename to `thread_id` where it refers to the active id. The arg passed to `Agent.run` stays `run_id=` (the public name will be renamed in Task 10).
- The `store_blob(... run_id=parent_rid)` call → `thread_id=parent_rid`.

- [ ] **Step 4: Update runtime.py**

In `src/llama_agents/runtime.py`:

- Update the `MemoryRecallTool` construction: `run_id_getter=get_active_run_id` → `thread_id_getter=get_active_thread_id`.

- [ ] **Step 5: Update tests**

In `tests/unit/test_agent_loop.py`:

- `from llama_agents.agent import _ACTIVE_RUN_ID` → `_ACTIVE_THREAD_ID` (find and rename in `test_subagent_summary_return.py` too).
- `_ACTIVE_RUN_ID.set(...)` → `_ACTIVE_THREAD_ID.set(...)`.

In `tests/unit/test_memory_recall_tool.py`:

- `MemoryRecallTool(store=..., run_id_getter=...)` → `thread_id_getter`.

- [ ] **Step 6: Run the full unit suite**

```
uv run pytest tests/unit -q
```

Expected: all green. The rename is mechanical; if anything breaks it's because a call site was missed — grep for `run_id_getter`, `_ACTIVE_RUN_ID`, `get_active_run_id` to find leftovers.

- [ ] **Step 7: Commit**

```
git add src/llama_agents/agent.py src/llama_agents/tools/builtin/memory.py src/llama_agents/tools/builtin/subagent.py src/llama_agents/runtime.py tests/unit/test_agent_loop.py tests/unit/test_memory_recall_tool.py tests/unit/test_subagent_summary_return.py
git commit -m "refactor(agent): rename _ACTIVE_RUN_ID to _ACTIVE_THREAD_ID; thread-aware recall"
```

---

## Task 10: Agent `prior_messages` hydration

**Files:**
- Modify: `src/llama_agents/agent.py`
- Modify: `tests/unit/test_agent_loop.py`

This is the multi-turn hydration step: the agent accepts a list of prior messages and seeds `self.messages` with them between the system prompt and the new user turn.

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_agent_loop.py`:

```python
async def test_agent_hydrates_prior_messages():
    """When prior_messages is non-empty, the agent's messages start with
    system + prior_messages + new user prompt."""
    prior = [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "first answer"},
    ]
    client = ScriptedClient([ChatResponse(content="second answer")])
    agent = Agent(client=client, registry=_registry_with_echo())
    events = await _collect(agent.run(
        "second turn",
        AgentRunOptions(max_iterations=3),
        prior_messages=prior,
    ))
    # The first chat call's messages should be: [system, prior..., new user]
    sent = client.calls[0]["messages"]
    roles = [m["role"] for m in sent]
    assert roles == ["system", "user", "assistant", "user"]
    assert sent[1]["content"] == "first turn"
    assert sent[2]["content"] == "first answer"
    assert sent[3]["content"] == "second turn"
    assert any(isinstance(e, Done) and e.reason == "finished" for e in events)


async def test_agent_with_empty_prior_messages_behaves_like_one_shot():
    client = ScriptedClient([ChatResponse(content="answer")])
    agent = Agent(client=client, registry=_registry_with_echo())
    events = await _collect(agent.run(
        "go",
        AgentRunOptions(max_iterations=3),
        prior_messages=[],
    ))
    sent = client.calls[0]["messages"]
    roles = [m["role"] for m in sent]
    assert roles == ["system", "user"]
    assert sent[1]["content"] == "go"
```

- [ ] **Step 2: Run tests — expect TypeError (kwarg not yet accepted)**

- [ ] **Step 3: Add `prior_messages` to `Agent.run`**

In `src/llama_agents/agent.py`, update the `Agent.run` signature:

```python
    async def run(
        self,
        user_prompt: str,
        opts: AgentRunOptions,
        *,
        thread_id: str | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Event]:
```

Rename the existing `run_id` kwarg to `thread_id`. Inside the body, change `self._thread_id = thread_id or uuid.uuid4().hex[:24]` (the field was already renamed in Task 9).

Find the messages initialization block:

```python
            self.messages = [
                {"role": "system", "content": opts.system_prompt},
                {"role": "user", "content": effective_prompt},
            ]
```

Replace with:

```python
            if prior_messages:
                self.messages = [
                    {"role": "system", "content": opts.system_prompt},
                    *prior_messages,
                    {"role": "user", "content": effective_prompt},
                ]
            else:
                self.messages = [
                    {"role": "system", "content": opts.system_prompt},
                    {"role": "user", "content": effective_prompt},
                ]
```

Note that `effective_prompt` may have been modified by the planning phase; the hydrated history goes between the system message and that.

- [ ] **Step 4: Update internal call sites of `run_id`**

Any existing test that calls `agent.run(..., run_id=...)` becomes `agent.run(..., thread_id=...)`. The subagent tool's call `subagent.run(args["task"], opts, run_id=parent_rid)` becomes `thread_id=parent_rid`.

Grep for `run_id=` to find remaining keyword usages of the renamed param.

- [ ] **Step 5: Run all unit tests**

```
uv run pytest tests/unit -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git add src/llama_agents/agent.py tests/unit/test_agent_loop.py src/llama_agents/tools/builtin/subagent.py
git commit -m "feat(agent): prior_messages hydration + rename run_id kwarg to thread_id"
```

---

## Task 11: Runtime exposes ThreadStore; memory_recall walks ancestor chain

**Files:**
- Modify: `src/llama_agents/runtime.py`
- Modify: `src/llama_agents/tools/builtin/memory.py`
- Modify: `tests/unit/test_runtime.py`
- Modify: `tests/unit/test_memory_recall_tool.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_runtime.py`:

```python
@pytest.mark.asyncio
async def test_runtime_exposes_thread_store(tmp_path: Path):
    from llama_agents.thread.store import ThreadStore
    cfg = Config(
        llama=LlamaConfig(auto_spawn=False),
        sandbox=SandboxConfig(allowed_dirs=[tmp_path]),
    )
    rt = await Runtime.create(cfg, client_factory=lambda url: FakeClient())
    try:
        assert isinstance(rt.thread_store, ThreadStore)
    finally:
        await rt.aclose()
```

Add to `tests/unit/test_memory_recall_tool.py`:

```python
@pytest.mark.asyncio
async def test_memory_recall_uses_ancestor_chain(tmp_path):
    """Forked thread should recall from parent's blobs."""
    from llama_agents.thread.store import ThreadStore

    store_root = tmp_path / "memory"
    threads_root = tmp_path / "threads"
    threads_root.mkdir()

    mem = MemoryStore(root=store_root, embedder=HashEmbedder(dim=32))
    await mem.init()
    thread_store = ThreadStore(threads_root)

    parent_id = thread_store.create_thread(title="parent")
    child_id = thread_store.create_thread(
        title="child", parent_thread_id=parent_id, parent_turn_idx=1,
    )

    mem.start_run(parent_id)
    await mem.store_blob(kind="user", scope="run", thread_id=parent_id,
                         title="parent-blob", body="quick brown fox")

    # The tool is configured with a thread_id_getter returning the CHILD;
    # it should still find the parent's blob via the ancestor chain.
    tool = MemoryRecallTool(
        store=mem, thread_id_getter=lambda: child_id,
        thread_store=thread_store,
    )
    res = await tool.invoke({"query": "quick brown fox", "k": 3})
    assert any(c["title"] == "parent-blob" for c in res["chunks"])
    await mem.close()
```

- [ ] **Step 2: Run tests — expect failure**

- [ ] **Step 3: Build `ThreadStore` in runtime**

In `src/llama_agents/runtime.py`, in `Runtime.create`:

- Add import: `from .thread.store import ThreadStore`.
- After resolving `mem_root` (memory root), also resolve a `threads_root` (same logic, but reading `cfg.queue.root / "threads"`). Use `_resolve_queue_root(cfg) / "threads"` so it lives alongside the queue.
- Construct `thread_store = ThreadStore(threads_root)`.
- Pass `thread_store` to the `Runtime` constructor: add a `thread_store: ThreadStore` parameter and a `self.thread_store = thread_store` assignment.
- Update the `MemoryRecallTool` construction to pass `thread_store=thread_store`.

- [ ] **Step 4: Update `MemoryRecallTool` to walk the ancestor chain**

In `src/llama_agents/tools/builtin/memory.py`, update the constructor and `invoke`:

```python
class MemoryRecallTool(Tool):
    name = "memory_recall"
    description = (
        "Retrieve previously-stored content from this thread's scratch "
        "memory, ancestor threads' memory, and past plans. Use this when "
        "you see '[evicted to memory ...]' in earlier tool results, or to "
        "look up the full text of a subagent's output via its memory_handle."
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
        store,
        thread_id_getter,
        thread_store=None,
    ) -> None:
        self._store = store
        self._thread_id_getter = thread_id_getter
        self._thread_store = thread_store

    async def invoke(self, args):
        tid = self._thread_id_getter()
        thread_ids: list[str] = []
        if tid:
            thread_ids.append(tid)
            if self._thread_store is not None:
                thread_ids.extend(self._thread_store.ancestor_chain(tid))
        chunks = await self._store.recall(
            query=args["query"], scope="all",
            thread_ids=thread_ids,
            handle=args.get("handle"),
            k=int(args.get("k", 5)),
        )
        return {
            "chunks": [
                {
                    "text": c.text, "blob_id": c.blob_id,
                    "chunk_idx": c.chunk_idx, "score": c.score,
                    "title": c.title, "kind": c.kind,
                }
                for c in chunks
            ]
        }
```

- [ ] **Step 5: Run tests**

```
uv run pytest tests/unit -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git add src/llama_agents/runtime.py src/llama_agents/tools/builtin/memory.py tests/unit/test_runtime.py tests/unit/test_memory_recall_tool.py
git commit -m "feat(thread): Runtime.thread_store; memory_recall widens across ancestor chain"
```

---

## Task 12: Queue worker — scan threads, finalize into turn folders

**Files:**
- Modify: `src/llama_agents/queue/worker.py`
- Modify: `tests/unit/test_queue_worker.py`

The largest single behavioral change. The worker's `_pick_one` switches from inbox scan to `thread_store.next_queued_turn()`. `_finalize` writes into the turn folder instead of `done/`/`failed/`. Startup gains the migration + processing-revert hooks.

- [ ] **Step 1: Rewrite the worker — read the spec section "Worker pickup" + "Worker `_finalize` change" before editing**

In `src/llama_agents/queue/worker.py`:

- Add imports:
  ```python
  from ..thread.migration import migrate_legacy_queue_dirs
  from ..thread.status import (
      claim_for_processing, revert_processing_on_startup, set_status,
  )
  from ..thread.store import ThreadStore
  ```
- The `JobQueueWorker.__init__` already calls `ensure_dirs(cfg.root)` + `sweep_processing_to_inbox(cfg.root)`. Replace the sweep with:
  ```python
  migrate_legacy_queue_dirs(cfg.root)
  revert_processing_on_startup(cfg.root / "threads")
  ```
- Replace the `_RuntimeLike` Protocol's expectations and pull `thread_store` from the runtime in `__init__`: change the constructor to accept it directly:
  ```python
  def __init__(
      self,
      runtime: _RuntimeLike,
      cfg: QueueConfig,
      *,
      thread_store: ThreadStore,
  ) -> None:
      self._rt = runtime
      self._cfg = cfg
      self._thread_store = thread_store
      self._stop = asyncio.Event()
      self._in_flight: set[asyncio.Task] = set()
      ensure_dirs(cfg.root)
      migrate_legacy_queue_dirs(cfg.root)
      revert_processing_on_startup(cfg.root / "threads")
  ```
- Rewrite `_pick_one`:
  ```python
  def _pick_one(self) -> tuple[str, int, Path] | None:
      """Find the oldest queued turn and atomically claim it.

      Returns (thread_id, turn_idx, turn_dir) on success.
      """
      candidate = self._thread_store.next_queued_turn()
      if candidate is None:
          return None
      thread_id, turn_idx = candidate
      turn_dir = self._thread_store.turn_dir(thread_id, turn_idx)
      if not claim_for_processing(turn_dir):
          # Lost the race; try again next poll
          return None
      logger.info("queue: picked up %s/turn-%d", thread_id, turn_idx)
      return thread_id, turn_idx, turn_dir
  ```
- Rewrite `_run_job` to accept `(thread_id, turn_idx, turn_dir)`:
  ```python
  async def _run_job(self, thread_id: str, turn_idx: int,
                     turn_dir: Path) -> None:
      attempt = 0
      while True:
          logger.info(
              "queue: running %s/turn-%d (attempt %d)",
              thread_id, turn_idx, attempt + 1,
          )
          result = await self._invoke_agent(thread_id, turn_idx, turn_dir)
          if self._should_retry(result, attempt):
              # ... (existing retry logic, unchanged)
              attempt += 1
              continue
          await self._finalize(thread_id, turn_idx, turn_dir, result,
                               attempt=attempt)
          return
  ```
- Rewrite `_invoke_agent` to read prompt + hydrate prior_messages:
  ```python
  async def _invoke_agent(self, thread_id: str, turn_idx: int,
                          turn_dir: Path) -> JobResult:
      prompt = (turn_dir / "prompt.md").read_text(encoding="utf-8")
      agent = self._rt.new_agent()
      opts = AgentRunOptions(max_iterations=self._cfg.max_iterations)
      prior = self._thread_store.read_messages(thread_id)

      events: list[dict[str, Any]] = []
      final_chunks: list[str] = []
      loop_error: LoopError | None = None
      new_messages: list[dict] = []
      async for ev in agent.run(prompt, opts, thread_id=thread_id,
                                prior_messages=prior):
          events.append(_serialize_event(ev))
          if isinstance(ev, AssistantChunk):
              final_chunks.append(ev.text)
          elif isinstance(ev, LoopError):
              loop_error = ev

      # The agent appends to its own self.messages; capture only the new
      # tail beyond `prior`.
      tail_start = 1 + len(prior) + 1  # [system, *prior, user]
      new_messages = list(agent.messages[tail_start:])
      success = loop_error is None
      return JobResult(
          success=success,
          final_text="\n\n".join(final_chunks) or "[no final answer]",
          events=events,
          loop_error=loop_error,
          prompt_text=prompt,
          new_messages=new_messages,
      )
  ```
- Add `new_messages: list[dict]` to the `JobResult` dataclass.
- Rewrite `_finalize`:
  ```python
  async def _finalize(self, thread_id: str, turn_idx: int,
                      turn_dir: Path, result: JobResult,
                      attempt: int) -> None:
      # Write side-cars BEFORE flipping status, so readers polling the
      # turn page never see a done turn with a missing result.md.
      (turn_dir / "result.md").write_text(result.final_text, encoding="utf-8")
      (turn_dir / "events.jsonl").write_text(
          "\n".join(json.dumps(e) for e in result.events) + "\n",
          encoding="utf-8",
      )
      # Append the new agent messages to the thread's messages.jsonl,
      # including the seed user turn we don't already have on disk.
      seed_user = {"role": "user", "content": result.prompt_text}
      self._thread_store.append_messages(
          thread_id, [seed_user, *result.new_messages]
      )

      status = "done" if result.success else "failed"
      if not result.success:
          err = result.loop_error
          (turn_dir / "error.txt").write_text(
              f"attempts: {attempt + 1}\n"
              f"error_type: {err.error_type if err else 'Unknown'}\n"
              f"message: {err.message if err else 'no LoopError captured'}\n",
              encoding="utf-8",
          )
      set_status(turn_dir, status)
      logger.info("queue: %s/turn-%d -> %s", thread_id, turn_idx, status)
  ```
- The existing `_fill_slots` and `run()` loop call `_pick_one` and `_run_job`. Their signatures change to pass `(thread_id, turn_idx, turn_dir)` through.

- [ ] **Step 2: Update `Runtime.create` to pass `thread_store` to the worker**

This is done in `http_app.py` lifespan (which instantiates the worker). Update the relevant line:

```python
worker = JobQueueWorker(rt, resolved_queue, thread_store=rt.thread_store)
```

- [ ] **Step 3: Rewrite the existing worker tests**

`tests/unit/test_queue_worker.py` currently stages files in `inbox/` and asserts on `done/<name>.md`. Every test needs to:

1. Create a `ThreadStore(tmp_path)` (or pass `cfg.queue.root` so the worker creates it).
2. Stage a queued turn via `store.create_thread(...)` + write `prompt.md` + `set_status(..., "queued")`.
3. Assert on `threads/<id>/turns/001/result.md` instead of `done/<name>.md`.

This rewrites ~12 tests. The shape is mechanical: each test's setup gets replaced with a `_stage_queued_turn(store, "prompt body")` helper. Add the helper at the top:

```python
from llama_agents.thread.status import set_status
from llama_agents.thread.store import ThreadStore


def _stage_queued_turn(store: ThreadStore, prompt: str) -> tuple[str, int]:
    """Create a one-turn thread with status=queued. Returns (thread_id, turn_idx)."""
    tid = store.create_thread(title=prompt[:60])
    td = store.turn_dir(tid, 1)
    (td / "prompt.md").write_text(prompt, encoding="utf-8")
    set_status(td, "queued")
    return tid, 1
```

Then every test that did `(tmp_path / "inbox" / "foo.md").write_text("...")` becomes:

```python
store = ThreadStore(tmp_path / "threads")
tid, _ = _stage_queued_turn(store, "...")
```

And assertions like `assert (tmp_path / "done" / "foo.md").exists()` become:

```python
turn_dir = store.turn_dir(tid, 1)
assert (turn_dir / "result.md").exists()
```

The `_StubRuntime` test double also needs a `thread_store` attribute pointing at the same store.

- [ ] **Step 4: Run the worker tests**

```
uv run pytest tests/unit/test_queue_worker.py -v
```

Expected: all green.

- [ ] **Step 5: Run the full unit suite**

```
uv run pytest tests/unit -q
```

Expected: most green; the web routes tests will be broken until Task 14. If `test_web_routes.py` failures appear, mark them with `@pytest.mark.skip(reason="rewritten in Task 14")` temporarily — the implementer should add the skip markers, run the suite, get a clean green, commit, then unskip in Task 14.

- [ ] **Step 6: Commit**

```
git add src/llama_agents/queue/worker.py tests/unit/test_queue_worker.py src/llama_agents/http_app.py
git commit -m "feat(queue): worker scans threads/, finalizes into turn folders, hydrates priors"
```

---

## Task 13: Web routes — `/activity`, `/threads`, `/threads/{id}`

**Files:**
- Modify: `src/llama_agents/web/routes.py`
- Modify: `src/llama_agents/web/templates/base.html`
- Create: `src/llama_agents/web/templates/threads.html`
- Create: `src/llama_agents/web/templates/thread.html`
- Rename: `src/llama_agents/web/templates/dashboard.html` → `activity.html`
- Modify: `tests/unit/test_web_routes.py`

- [ ] **Step 1: Update the nav in base.html**

In `src/llama_agents/web/templates/base.html`, change the nav block:

```html
<nav class="nav-tabs">
    <a href="/activity" class="{% if active == 'activity' %}active{% endif %}">Activity</a>
    <a href="/threads"  class="{% if active == 'threads'  %}active{% endif %}">Threads</a>
    <a href="/config"   class="{% if active == 'config'   %}active{% endif %}">Config</a>
</nav>
```

- [ ] **Step 2: Rename dashboard.html → activity.html (content swap)**

`git mv src/llama_agents/web/templates/dashboard.html src/llama_agents/web/templates/activity.html`.

Update the rendered list `<a>` hrefs in `activity.html`: the panel rows currently link to `/jobs/<status>/<name>` — change to `/threads/<thread_id>#turn-<idx>`. This needs a small server-side change to expose `thread_id` and `turn_idx` in the row context (Task 14 covers this; for now, just point the links at `/threads/<tid>` and accept incomplete data).

- [ ] **Step 3: Write `threads.html`**

Create `src/llama_agents/web/templates/threads.html`:

```html
{% extends "base.html" %}
{% block title %}Llama Agents — Threads{% endblock %}
{% block content %}
<div class="page-header">
    <h1>Threads</h1>
</div>
<div class="card">
    {% if threads %}
    <ul class="thread-list">
        {% for t in threads %}
        <li>
            <a href="/threads/{{ t.id }}">
                <div class="thread-title">{{ t.title }}</div>
                <div class="thread-meta muted">
                    {{ t.current_turn }} turn{{ "s" if t.current_turn != 1 else "" }}
                    · {{ t.updated_at | fmt_ts }}
                </div>
            </a>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">No threads yet. Submit a prompt from the Activity page.</div>
    {% endif %}
</div>
{% endblock %}
```

- [ ] **Step 4: Write `thread.html`**

Create `src/llama_agents/web/templates/thread.html`:

```html
{% extends "base.html" %}
{% block title %}Llama Agents — {{ thread.title }}{% endblock %}
{% block content %}
<p><a href="/threads" class="back-link">← Threads</a></p>
<div class="page-header">
    <h1 id="thread-title" contenteditable="true"
        data-thread-id="{{ thread.id }}">{{ thread.title }}</h1>
    {% if thread.parent_thread_id %}
    <div class="subtitle">
        Forked from <a href="/threads/{{ thread.parent_thread_id }}">{{ thread.parent_thread_id[:8] }}</a>
        at turn {{ thread.parent_turn_idx }}
    </div>
    {% endif %}
</div>

{% for turn in turns %}
<div class="card section" id="turn-{{ turn.idx }}">
    <h2 class="with-icon">
        Turn {{ turn.idx }}
        <span class="muted">— {{ turn.status }}</span>
        {% if turn.status in ("done", "failed") %}
        <form method="post" action="/api/threads/{{ thread.id }}/rerun/{{ turn.idx }}"
              style="margin-left: auto; display: inline;">
            <button type="submit">⟳ Rerun</button>
        </form>
        {% endif %}
    </h2>

    <h3>Prompt</h3>
    <pre class="prompt-block">{{ turn.prompt }}</pre>

    {% if turn.events %}
    <details>
        <summary>Events ({{ turn.events | length }})</summary>
        {% for ev in turn.events %}
            <details class="event">
                <summary>
                    <span class="badge {{ ev._color }}">{{ ev.type }}</span>
                    <span class="muted">{{ ev.ts | fmt_ts }}</span>
                </summary>
                <pre>{{ ev._raw }}</pre>
            </details>
        {% endfor %}
    </details>
    {% endif %}

    {% if turn.result %}
    <h3>Final answer</h3>
    <pre class="prompt-block">{{ turn.result }}</pre>
    {% endif %}

    {% if turn.error %}
    <h3>Error</h3>
    <pre class="error-block">{{ turn.error }}</pre>
    {% endif %}
</div>
{% endfor %}

{% if can_continue %}
<div class="card section">
    <h2>Continue</h2>
    <form method="post" action="/api/threads/{{ thread.id }}/continue">
        <p><textarea name="body" required placeholder="Your follow-up..."></textarea></p>
        <p><button type="submit" class="full-btn primary">Send</button></p>
    </form>
</div>
{% else %}
<div class="card section">
    <div class="empty-state">Continue available once the current turn finishes.</div>
</div>
{% endif %}

<script>
    // Inline title editor — PATCH /api/threads/{id} on blur
    (function () {
        const el = document.getElementById("thread-title");
        if (!el) return;
        let original = el.textContent.trim();
        el.addEventListener("blur", async function () {
            const cur = el.textContent.trim();
            if (cur === original || !cur) {
                el.textContent = original;
                return;
            }
            const r = await fetch("/api/threads/" + el.dataset.threadId, {
                method: "PATCH",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({"title": cur}),
            });
            if (r.ok) {
                original = cur;
            } else {
                el.textContent = original;
            }
        });
    })();
</script>
{% endblock %}
```

- [ ] **Step 5: Add the routes**

In `src/llama_agents/web/routes.py`:

Add at the top:

```python
from ..thread.store import ThreadStore
from ..thread.meta import read_meta
from ..thread.status import read_status
```

In `register_routes`, build a thread_store reference once:

```python
threads_root = Path(cfg.queue.root) / "threads"
thread_store = ThreadStore(threads_root)
```

Add routes:

```python
    @app.get("/", response_class=HTMLResponse)
    async def root_redirect():
        from fastapi.responses import RedirectResponse as _R
        return _R(url="/activity", status_code=302)

    @app.get("/activity", response_class=HTMLResponse)
    async def activity(request: Request):
        return templates.TemplateResponse(
            request, "activity.html",
            {"presets": _load_presets(repo_root), "active": "activity"},
        )

    @app.get("/threads", response_class=HTMLResponse)
    async def threads_index(request: Request):
        return templates.TemplateResponse(
            request, "threads.html",
            {"threads": thread_store.list_threads(limit=200),
             "active": "threads"},
        )

    @app.get("/threads/{thread_id}", response_class=HTMLResponse)
    async def thread_detail(request: Request, thread_id: str):
        from ..thread.ids import validate_thread_id
        if not validate_thread_id(thread_id):
            raise HTTPException(status_code=404, detail="invalid thread id")
        try:
            meta = read_meta(threads_root, thread_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="thread not found")
        turns = []
        for n in range(1, meta.current_turn + 1):
            td = thread_store.turn_dir(thread_id, n)
            if not td.is_dir():
                continue
            prompt = (td / "prompt.md").read_text(encoding="utf-8") \
                if (td / "prompt.md").is_file() else ""
            result = (td / "result.md").read_text(encoding="utf-8") \
                if (td / "result.md").is_file() else ""
            error = (td / "error.txt").read_text(encoding="utf-8") \
                if (td / "error.txt").is_file() else ""
            events = _read_events(td / "events.jsonl")
            turns.append({
                "idx": n, "status": read_status(td) or "unknown",
                "prompt": prompt, "result": result,
                "error": error, "events": events,
            })
        latest_status = turns[-1]["status"] if turns else ""
        can_continue = latest_status in ("done", "failed")
        return templates.TemplateResponse(
            request, "thread.html",
            {"thread": meta, "turns": turns,
             "can_continue": can_continue, "active": "threads"},
        )
```

Update `_list_jobs` to query the thread store instead of folder scanning. Rename to `_list_turns`:

```python
def _list_turns(thread_store: ThreadStore, status: str, *,
                limit: int | None = None) -> list[dict]:
    """Find turns across all threads matching the given status."""
    if status not in _VALID_STATUSES:
        raise HTTPException(status_code=404, detail="unknown status")
    rows: list[dict] = []
    root = thread_store.root
    if not root.is_dir():
        return []
    for thread_dir in root.iterdir():
        if not thread_dir.is_dir():
            continue
        turns_dir = thread_dir / "turns"
        if not turns_dir.is_dir():
            continue
        try:
            meta = read_meta(root, thread_dir.name)
        except (FileNotFoundError, ValueError):
            continue
        for turn_dir in turns_dir.iterdir():
            if not turn_dir.is_dir() or not turn_dir.name.isdigit():
                continue
            if read_status(turn_dir) != status:
                continue
            try:
                mtime = (turn_dir / "status").stat().st_mtime
            except FileNotFoundError:
                continue
            rows.append({
                "thread_id": thread_dir.name,
                "turn_idx": int(turn_dir.name),
                "title": meta.title,
                "mtime": mtime,
            })
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    if limit is not None:
        rows = rows[:limit]
    return rows
```

Update the `/api/jobs/{status}` route to use `_list_turns`:

```python
    @app.get("/api/jobs/{status}", response_class=HTMLResponse)
    async def jobs_partial(request: Request, status: str):
        limit = 50 if status in ("done", "failed") else None
        rows = _list_turns(thread_store, status, limit=limit)
        return templates.TemplateResponse(
            request, "_partials/turn_list.html",
            {"status": status, "rows": rows},
        )
```

Create `src/llama_agents/web/templates/_partials/turn_list.html`:

```html
{% if rows %}
    {% for r in rows %}
    <li>
        <a href="/threads/{{ r.thread_id }}#turn-{{ r.turn_idx }}">
            {{ r.title }} — turn {{ r.turn_idx }}
        </a>
        <span class="muted">{{ r.mtime | age }} ago</span>
    </li>
    {% endfor %}
{% else %}
    <div class="empty-state">Nothing here</div>
{% endif %}
```

- [ ] **Step 6: Delete the old job templates**

```
git rm src/llama_agents/web/templates/job.html src/llama_agents/web/templates/_partials/job_list.html src/llama_agents/web/templates/_partials/job_row.html
```

(The implementer should also remove the old `/jobs/{status}/{name}` route handler from `routes.py`.)

- [ ] **Step 7: Rewrite the affected web tests**

`tests/unit/test_web_routes.py` has tests that hit `/`, `/jobs/...`, `/api/jobs/...`. Update:

- The `test_root_redirects_to_activity` (new): `GET /` → 302 to `/activity`.
- `test_activity_page` (new): `GET /activity` → 200 with all four bucket headings.
- `test_threads_list_empty`, `test_threads_list_shows_thread`, `test_thread_detail_renders_turns` (new).
- Remove the old `test_job_detail_*` tests entirely; they were testing the now-deleted `/jobs/{status}/{name}` route.

This is a substantial rewrite (~10 tests removed/replaced). The pattern for staging is from Task 12:

```python
store = ThreadStore(cfg.queue.root / "threads")
tid = store.create_thread(title="Test thread")
(store.turn_dir(tid, 1) / "prompt.md").write_text("the prompt", encoding="utf-8")
set_status(store.turn_dir(tid, 1), "done")
(store.turn_dir(tid, 1) / "result.md").write_text("the answer", encoding="utf-8")
```

- [ ] **Step 8: Run web tests**

```
uv run pytest tests/unit/test_web_routes.py -v
```

Expected: all green.

- [ ] **Step 9: Run full unit suite**

```
uv run pytest tests/unit -q
```

Expected: all green.

- [ ] **Step 10: Commit**

```
git add src/llama_agents/web/ tests/unit/test_web_routes.py
git commit -m "feat(web): /activity + /threads + /threads/{id}; drop job.html"
```

---

## Task 14: `/api/threads/{id}/continue` + PATCH endpoint

**Files:**
- Modify: `src/llama_agents/web/routes.py`
- Modify: `tests/unit/test_web_routes.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_web_routes.py`:

```python
@pytest.mark.asyncio
async def test_continue_appends_turn(cfg, config_path):
    from llama_agents.thread.store import ThreadStore
    from llama_agents.thread.status import set_status
    threads_root = cfg.queue.root / "threads"
    threads_root.mkdir(parents=True)
    store = ThreadStore(threads_root)
    tid = store.create_thread(title="t")
    (store.turn_dir(tid, 1) / "prompt.md").write_text("first", encoding="utf-8")
    set_status(store.turn_dir(tid, 1), "done")

    app = create_app(cfg, client_factory=lambda url: _FakeClient(),
                     config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(f"/api/threads/{tid}/continue",
                              data={"body": "follow-up"})
            assert r.status_code in (303, 200)
    assert (store.turn_dir(tid, 2) / "prompt.md").read_text(encoding="utf-8") == "follow-up"
    assert (store.turn_dir(tid, 2) / "status").read_text(encoding="utf-8").strip() == "queued"


@pytest.mark.asyncio
async def test_continue_refuses_when_prior_running(cfg, config_path):
    from llama_agents.thread.store import ThreadStore
    from llama_agents.thread.status import set_status
    threads_root = cfg.queue.root / "threads"
    threads_root.mkdir(parents=True)
    store = ThreadStore(threads_root)
    tid = store.create_thread(title="t")
    (store.turn_dir(tid, 1) / "prompt.md").write_text("first", encoding="utf-8")
    set_status(store.turn_dir(tid, 1), "processing")

    app = create_app(cfg, client_factory=lambda url: _FakeClient(),
                     config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(f"/api/threads/{tid}/continue",
                              data={"body": "x"})
            assert r.status_code == 409
    # turn 2 was not created
    assert not (store.turn_dir(tid, 2)).exists()


@pytest.mark.asyncio
async def test_patch_thread_updates_title(cfg, config_path):
    from llama_agents.thread.store import ThreadStore
    from llama_agents.thread.meta import read_meta
    threads_root = cfg.queue.root / "threads"
    threads_root.mkdir(parents=True)
    store = ThreadStore(threads_root)
    tid = store.create_thread(title="original")

    app = create_app(cfg, client_factory=lambda url: _FakeClient(),
                     config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.patch(f"/api/threads/{tid}",
                               json={"title": "renamed"})
            assert r.status_code == 200
    assert read_meta(threads_root, tid).title == "renamed"
```

- [ ] **Step 2: Run tests — expect 404**

- [ ] **Step 3: Implement the routes**

Add to `routes.py` inside `register_routes`:

```python
    @app.post("/api/threads/{thread_id}/continue")
    async def continue_thread(thread_id: str, body: str = Form(...)):
        from ..thread.ids import validate_thread_id
        if not validate_thread_id(thread_id):
            raise HTTPException(status_code=404, detail="invalid thread id")
        try:
            meta = read_meta(threads_root, thread_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="thread not found")
        latest_status = read_status(thread_store.turn_dir(thread_id, meta.current_turn))
        if latest_status in ("queued", "processing"):
            return PlainTextResponse(
                f"thread has an active turn (turn {meta.current_turn})",
                status_code=409,
            )
        new_dir, new_idx = thread_store.next_turn_dir(thread_id)
        (new_dir / "prompt.md").write_text(body, encoding="utf-8")
        set_status(new_dir, "queued")
        return RedirectResponse(
            url=f"/threads/{thread_id}#turn-{new_idx}", status_code=303,
        )

    @app.patch("/api/threads/{thread_id}")
    async def patch_thread(thread_id: str, payload: dict = Body(...)):
        from ..thread.ids import validate_thread_id
        from ..thread.meta import update_meta
        if not validate_thread_id(thread_id):
            raise HTTPException(status_code=404, detail="invalid thread id")
        if "title" not in payload:
            raise HTTPException(status_code=400, detail="title required")
        title = str(payload["title"]).strip()
        if not title:
            raise HTTPException(status_code=400, detail="title cannot be empty")
        try:
            update_meta(threads_root, thread_id, title=title)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="thread not found")
        return {"ok": True, "title": title}
```

Add `Body` to the imports: `from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile`.

- [ ] **Step 4: Run tests — expect pass**

```
uv run pytest tests/unit/test_web_routes.py -v
```

- [ ] **Step 5: Commit**

```
git add src/llama_agents/web/routes.py tests/unit/test_web_routes.py
git commit -m "feat(web): POST /api/threads/{id}/continue + PATCH /api/threads/{id}"
```

---

## Task 15: `/api/threads/{id}/rerun/{turn}` fork endpoint

**Files:**
- Modify: `src/llama_agents/web/routes.py`
- Modify: `tests/unit/test_web_routes.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_web_routes.py`:

```python
@pytest.mark.asyncio
async def test_rerun_forks_thread(cfg, config_path):
    from llama_agents.thread.store import ThreadStore
    from llama_agents.thread.status import set_status
    from llama_agents.thread.meta import read_meta
    threads_root = cfg.queue.root / "threads"
    threads_root.mkdir(parents=True)
    store = ThreadStore(threads_root)
    parent_id = store.create_thread(title="parent")
    (store.turn_dir(parent_id, 1) / "prompt.md").write_text("orig", encoding="utf-8")
    set_status(store.turn_dir(parent_id, 1), "done")

    app = create_app(cfg, client_factory=lambda url: _FakeClient(),
                     config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(f"/api/threads/{parent_id}/rerun/1",
                              data={"body": "edited prompt"},
                              follow_redirects=False)
            assert r.status_code == 303
            location = r.headers["location"]
    new_tid = location.split("/threads/")[1].split("#")[0]
    assert new_tid != parent_id
    new_meta = read_meta(threads_root, new_tid)
    assert new_meta.parent_thread_id == parent_id
    assert new_meta.parent_turn_idx == 0  # fork of turn 1 starts before turn 1
    assert (store.turn_dir(new_tid, 1) / "prompt.md").read_text(encoding="utf-8") == "edited prompt"


@pytest.mark.asyncio
async def test_rerun_without_edit_reuses_original_prompt(cfg, config_path):
    from llama_agents.thread.store import ThreadStore
    from llama_agents.thread.status import set_status
    threads_root = cfg.queue.root / "threads"
    threads_root.mkdir(parents=True)
    store = ThreadStore(threads_root)
    parent_id = store.create_thread(title="parent")
    (store.turn_dir(parent_id, 1) / "prompt.md").write_text("original text",
                                                            encoding="utf-8")
    set_status(store.turn_dir(parent_id, 1), "done")

    app = create_app(cfg, client_factory=lambda url: _FakeClient(),
                     config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(f"/api/threads/{parent_id}/rerun/1",
                              follow_redirects=False)
            assert r.status_code == 303
            location = r.headers["location"]
    new_tid = location.split("/threads/")[1].split("#")[0]
    assert (store.turn_dir(new_tid, 1) / "prompt.md").read_text(encoding="utf-8") == "original text"
```

- [ ] **Step 2: Run tests — expect 404**

- [ ] **Step 3: Implement the rerun route**

Add to `register_routes`:

```python
    @app.post("/api/threads/{thread_id}/rerun/{turn_idx}")
    async def rerun_turn(
        thread_id: str, turn_idx: int,
        body: str | None = Form(None),
    ):
        from ..thread.ids import validate_thread_id
        if not validate_thread_id(thread_id):
            raise HTTPException(status_code=404, detail="invalid thread id")
        try:
            meta = read_meta(threads_root, thread_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="thread not found")
        if turn_idx < 1 or turn_idx > meta.current_turn:
            raise HTTPException(status_code=404, detail="turn not found")
        # Read original prompt for fallback
        orig_path = thread_store.turn_dir(thread_id, turn_idx) / "prompt.md"
        original_prompt = orig_path.read_text(encoding="utf-8") if orig_path.is_file() else ""

        new_prompt = (body or "").strip() or original_prompt
        if not new_prompt:
            raise HTTPException(status_code=400, detail="rerun prompt is empty")

        # Fork: parent_turn_idx = turn_idx - 1 (fork point is BEFORE the
        # reran turn — so turn 1 inherits no parent messages)
        new_tid = thread_store.create_thread(
            title=new_prompt.strip().splitlines()[0][:60] or meta.title,
            parent_thread_id=thread_id,
            parent_turn_idx=turn_idx - 1,
        )
        (thread_store.turn_dir(new_tid, 1) / "prompt.md").write_text(
            new_prompt, encoding="utf-8",
        )
        set_status(thread_store.turn_dir(new_tid, 1), "queued")
        return RedirectResponse(
            url=f"/threads/{new_tid}#turn-1", status_code=303,
        )
```

- [ ] **Step 4: Run tests — expect pass**

- [ ] **Step 5: Run full suite**

```
uv run pytest tests/unit -q
```

- [ ] **Step 6: Commit**

```
git add src/llama_agents/web/routes.py tests/unit/test_web_routes.py
git commit -m "feat(web): POST /api/threads/{id}/rerun/{turn} forks with parent linkage"
```

---

## Task 16: Update `/api/submit` to create a new thread

**Files:**
- Modify: `src/llama_agents/web/routes.py`
- Modify: `tests/unit/test_web_routes.py`

The web's submit form currently writes a file into `inbox/`. After the cutover, it should create a thread instead.

- [ ] **Step 1: Update failing tests**

Find every test in `test_web_routes.py` that asserts on `cfg.queue.root / "inbox" / <name>.md`. Change to:

```python
# Find the newest thread; its turn 1 should have the submitted prompt.
threads_root = cfg.queue.root / "threads"
thread_dirs = [p for p in threads_root.iterdir() if p.is_dir()]
assert len(thread_dirs) == 1
tid = thread_dirs[0].name
turn1 = thread_dirs[0] / "turns" / "001"
assert (turn1 / "prompt.md").read_text(encoding="utf-8") == "hello world"
assert (turn1 / "status").read_text(encoding="utf-8").strip() == "queued"
```

The duplicate-rejection test no longer applies (each submit creates a new thread). Replace it with a test that confirms two submits create two threads.

- [ ] **Step 2: Update the `/api/submit` route**

In `register_routes` rewrite the body of `submit`:

```python
    @app.post("/api/submit")
    async def submit(
        request: Request,
        file: UploadFile | None = File(None),
        filename: str | None = Form(None),
        body: str | None = Form(None),
    ):
        accepted = list(cfg.queue.accepted_extensions)
        if file is not None and file.filename:
            name = _validate_name(file.filename, accepted)
            if name is None:
                return PlainTextResponse(
                    f"invalid filename or extension: {file.filename!r}",
                    status_code=400,
                )
            content_bytes = await file.read()
            content = content_bytes.decode("utf-8", errors="replace")
            title = name
        else:
            content = body or ""
            if not content.strip():
                return PlainTextResponse("body required", status_code=400)
            title = (filename or "").strip() or content.strip().splitlines()[0][:60]

        tid = thread_store.create_thread(title=title)
        turn1 = thread_store.turn_dir(tid, 1)
        (turn1 / "prompt.md").write_text(content, encoding="utf-8")
        set_status(turn1, "queued")
        return RedirectResponse(url=f"/threads/{tid}", status_code=303)
```

The legacy `os.replace`-into-inbox dance goes away.

- [ ] **Step 3: Run web tests**

```
uv run pytest tests/unit/test_web_routes.py -v
```

- [ ] **Step 4: Commit**

```
git add src/llama_agents/web/routes.py tests/unit/test_web_routes.py
git commit -m "feat(web): /api/submit creates a thread instead of writing inbox/"
```

---

## Task 17: CLI — `chat --thread` and `chat --background`

**Files:**
- Modify: `src/llama_agents/cli.py`
- Create: `tests/unit/test_cli_threads.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_cli_threads.py`:

```python
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llama_agents.cli import app


@pytest.fixture
def cfg_file(tmp_path: Path) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(
        f'[llama]\nauto_spawn = false\n'
        f'[sandbox]\nallowed_dirs = ["{tmp_path.as_posix()}"]\n'
        f'[queue]\nenabled = true\nroot = "{(tmp_path / "q").as_posix()}"\n',
        encoding="utf-8",
    )
    return p


def test_chat_background_creates_queued_thread(cfg_file, tmp_path, monkeypatch):
    """`chat --background` writes a thread with status=queued and exits 0."""
    runner = CliRunner()
    result = runner.invoke(app, [
        "chat", "--config", str(cfg_file), "--background", "hello world",
    ])
    assert result.exit_code == 0, result.stdout
    threads_root = tmp_path / "q" / "threads"
    threads = [p for p in threads_root.iterdir() if p.is_dir()]
    assert len(threads) == 1
    turn1 = threads[0] / "turns" / "001"
    assert (turn1 / "prompt.md").read_text(encoding="utf-8") == "hello world"
    assert (turn1 / "status").read_text(encoding="utf-8").strip() == "queued"
    # Thread id printed on stdout
    assert threads[0].name in result.stdout


def test_chat_thread_continue_refuses_when_prior_running(cfg_file, tmp_path):
    from llama_agents.thread.store import ThreadStore
    from llama_agents.thread.status import set_status

    threads_root = tmp_path / "q" / "threads"
    threads_root.mkdir(parents=True)
    store = ThreadStore(threads_root)
    tid = store.create_thread(title="t")
    (store.turn_dir(tid, 1) / "prompt.md").write_text("first", encoding="utf-8")
    set_status(store.turn_dir(tid, 1), "processing")

    runner = CliRunner()
    result = runner.invoke(app, [
        "chat", "--config", str(cfg_file),
        "--thread", tid[:8], "--background", "follow-up",
    ])
    assert result.exit_code == 3
    assert "active turn" in result.stdout.lower() or "active turn" in result.stderr.lower()
```

- [ ] **Step 2: Run tests — expect failure**

- [ ] **Step 3: Update `cli.py`**

Read `src/llama_agents/cli.py`. Update the `chat` command:

```python
@app.command()
def chat(
    config: Path = typer.Option(_default_config_path, "--config", "-c"),
    prompt: str = typer.Argument(..., help="Prompt to send to the agent."),
    max_iterations: int = typer.Option(20, "--max-iterations"),
    thread: str | None = typer.Option(
        None, "--thread", "-t",
        help="Continue an existing thread (full id or unique prefix >=4 chars).",
    ),
    background: bool = typer.Option(
        False, "--background",
        help="Submit the turn with status=queued and exit immediately. "
             "The queue worker (running in `llamactl serve`) picks it up.",
    ),
) -> None:
    """Run a single agent turn against the configured llama-server."""
    from .thread.ids import AmbiguousPrefix, UnknownPrefix, resolve_prefix
    from .thread.meta import read_meta
    from .thread.status import read_status, set_status
    from .thread.store import ThreadStore
    from .runtime import _resolve_queue_root

    cfg = load_config(config)
    queue_root = _resolve_queue_root(cfg)
    threads_root = queue_root / "threads"
    threads_root.mkdir(parents=True, exist_ok=True)
    store = ThreadStore(threads_root)

    # Resolve --thread prefix
    if thread is not None:
        try:
            thread_id = resolve_prefix(threads_root, thread)
        except UnknownPrefix as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=2)
        except AmbiguousPrefix as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=2)
        meta = read_meta(threads_root, thread_id)
        latest_status = read_status(store.turn_dir(thread_id, meta.current_turn))
        if latest_status in ("queued", "processing"):
            console.print(
                f"[red]thread {thread_id[:8]} has an active turn ({meta.current_turn})[/red]"
            )
            raise typer.Exit(code=3)
        # New turn in existing thread
        turn_dir, turn_idx = store.next_turn_dir(thread_id)
    else:
        # New thread
        thread_id = store.create_thread(
            title=prompt.strip().splitlines()[0][:60] or "untitled",
        )
        turn_dir = store.turn_dir(thread_id, 1)
        turn_idx = 1

    (turn_dir / "prompt.md").write_text(prompt, encoding="utf-8")

    if background:
        set_status(turn_dir, "queued")
        console.print(f"Thread: {thread_id} (turn {turn_idx})")
        return

    # In-process run (synchronous)
    set_status(turn_dir, "processing")
    asyncio.run(_run_chat_in_process(cfg, thread_id, turn_idx, turn_dir,
                                     prompt, max_iterations, store))


async def _run_chat_in_process(cfg, thread_id, turn_idx, turn_dir, prompt,
                               max_iterations, store):
    import json as _json
    from .thread.status import set_status as _set_status
    rt = await Runtime.create(cfg)
    try:
        agent = rt.new_agent()
        opts = AgentRunOptions(max_iterations=max_iterations)
        prior = store.read_messages(thread_id)
        events: list[dict] = []
        final_chunks: list[str] = []
        had_error = False
        async for ev in agent.run(prompt, opts, thread_id=thread_id,
                                  prior_messages=prior):
            _render_event(ev)
            from .queue.worker import _serialize_event
            events.append(_serialize_event(ev))
            if isinstance(ev, AssistantChunk):
                final_chunks.append(ev.text)
            elif isinstance(ev, LoopError):
                had_error = True
        result_text = "\n\n".join(final_chunks) or "[no final answer]"
        (turn_dir / "result.md").write_text(result_text, encoding="utf-8")
        (turn_dir / "events.jsonl").write_text(
            "\n".join(_json.dumps(e) for e in events) + "\n", encoding="utf-8",
        )
        # Append new messages
        tail_start = 1 + len(prior) + 1
        new_msgs = list(agent.messages[tail_start:])
        seed_user = {"role": "user", "content": prompt}
        store.append_messages(thread_id, [seed_user, *new_msgs])
        _set_status(turn_dir, "failed" if had_error else "done")
        console.print(f"\n[dim]Thread: {thread_id}[/dim]")
    finally:
        await rt.aclose()
```

Remove the old `_run_chat` function (replaced by `_run_chat_in_process`).

- [ ] **Step 4: Run tests — expect pass**

```
uv run pytest tests/unit/test_cli_threads.py -v
```

- [ ] **Step 5: Commit**

```
git add src/llama_agents/cli.py tests/unit/test_cli_threads.py
git commit -m "feat(cli): chat --thread + --background; in-process turn writes to thread store"
```

---

## Task 18: `threads` subcommand group

**Files:**
- Modify: `src/llama_agents/cli.py`
- Modify: `tests/unit/test_cli_threads.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_cli_threads.py`:

```python
def test_threads_list_shows_threads(cfg_file, tmp_path):
    from llama_agents.thread.store import ThreadStore
    threads_root = tmp_path / "q" / "threads"
    threads_root.mkdir(parents=True)
    store = ThreadStore(threads_root)
    a = store.create_thread(title="alpha")
    b = store.create_thread(title="beta")

    runner = CliRunner()
    result = runner.invoke(app, ["threads", "--config", str(cfg_file), "list"])
    assert result.exit_code == 0
    assert a[:8] in result.stdout
    assert b[:8] in result.stdout
    assert "alpha" in result.stdout
    assert "beta" in result.stdout


def test_threads_show_renders_turns(cfg_file, tmp_path):
    from llama_agents.thread.store import ThreadStore
    from llama_agents.thread.status import set_status
    threads_root = tmp_path / "q" / "threads"
    threads_root.mkdir(parents=True)
    store = ThreadStore(threads_root)
    tid = store.create_thread(title="t")
    (store.turn_dir(tid, 1) / "prompt.md").write_text("ask X", encoding="utf-8")
    (store.turn_dir(tid, 1) / "result.md").write_text("answer Y", encoding="utf-8")
    set_status(store.turn_dir(tid, 1), "done")

    runner = CliRunner()
    result = runner.invoke(app, [
        "threads", "--config", str(cfg_file), "show", tid[:8],
    ])
    assert result.exit_code == 0
    assert "ask X" in result.stdout
    assert "answer Y" in result.stdout
```

- [ ] **Step 2: Add the subcommand group**

In `src/llama_agents/cli.py`, append:

```python
threads_app = typer.Typer(no_args_is_help=True, help="Manage threads.")
app.add_typer(threads_app, name="threads")


@threads_app.command("list")
def threads_list(
    config: Path = typer.Option(_default_config_path, "--config", "-c"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """List threads, newest first."""
    from .thread.store import ThreadStore
    from .runtime import _resolve_queue_root

    cfg = load_config(config)
    threads_root = _resolve_queue_root(cfg) / "threads"
    store = ThreadStore(threads_root)
    metas = store.list_threads(limit=limit)
    if not metas:
        console.print("[dim]No threads yet.[/dim]")
        return
    # 4-column table
    console.print(f"[bold]{'ID':<10}{'Title':<50}{'Turns':>6}  {'Updated'}[/bold]")
    for m in metas:
        title = (m.title[:47] + "...") if len(m.title) > 50 else m.title
        console.print(f"{m.id[:8]:<10}{title:<50}{m.current_turn:>6}  {m.updated_at}")


@threads_app.command("show")
def threads_show(
    config: Path = typer.Option(_default_config_path, "--config", "-c"),
    thread: str = typer.Argument(...),
    full: bool = typer.Option(False, "--full"),
) -> None:
    """Render every turn in a thread."""
    from .thread.ids import resolve_prefix, AmbiguousPrefix, UnknownPrefix
    from .thread.meta import read_meta
    from .thread.status import read_status
    from .thread.store import ThreadStore
    from .runtime import _resolve_queue_root

    cfg = load_config(config)
    threads_root = _resolve_queue_root(cfg) / "threads"
    store = ThreadStore(threads_root)
    try:
        tid = resolve_prefix(threads_root, thread)
    except (UnknownPrefix, AmbiguousPrefix) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2)
    meta = read_meta(threads_root, tid)
    console.print(f"[bold]{meta.title}[/bold] ([dim]{tid}[/dim])")
    for n in range(1, meta.current_turn + 1):
        td = store.turn_dir(tid, n)
        status = read_status(td) or "unknown"
        console.print(f"\n[bold]── Turn {n} — {status} ──[/bold]")
        prompt_p = td / "prompt.md"
        if prompt_p.is_file():
            console.print("[dim]Prompt:[/dim]")
            console.print(prompt_p.read_text(encoding="utf-8"))
        result_p = td / "result.md"
        if result_p.is_file():
            console.print("[dim]Result:[/dim]")
            console.print(result_p.read_text(encoding="utf-8"))
        error_p = td / "error.txt"
        if error_p.is_file():
            console.print(f"[red]{error_p.read_text(encoding='utf-8')}[/red]")


@threads_app.command("rerun")
def threads_rerun(
    config: Path = typer.Option(_default_config_path, "--config", "-c"),
    thread: str = typer.Argument(...),
    turn: int = typer.Argument(...),
    edit: str | None = typer.Option(None, "--edit"),
) -> None:
    """Fork a thread by rerunning turn N, optionally with an edited prompt."""
    from .thread.ids import resolve_prefix, AmbiguousPrefix, UnknownPrefix
    from .thread.meta import read_meta
    from .thread.status import set_status
    from .thread.store import ThreadStore
    from .runtime import _resolve_queue_root

    cfg = load_config(config)
    threads_root = _resolve_queue_root(cfg) / "threads"
    store = ThreadStore(threads_root)
    try:
        parent_id = resolve_prefix(threads_root, thread)
    except (UnknownPrefix, AmbiguousPrefix) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2)
    parent_meta = read_meta(threads_root, parent_id)
    if turn < 1 or turn > parent_meta.current_turn:
        console.print(f"[red]turn {turn} out of range[/red]")
        raise typer.Exit(code=2)

    orig_prompt = (store.turn_dir(parent_id, turn) / "prompt.md").read_text(
        encoding="utf-8",
    )
    new_prompt = (edit or "").strip() or orig_prompt
    new_tid = store.create_thread(
        title=new_prompt.strip().splitlines()[0][:60],
        parent_thread_id=parent_id,
        parent_turn_idx=turn - 1,
    )
    (store.turn_dir(new_tid, 1) / "prompt.md").write_text(new_prompt, encoding="utf-8")
    set_status(store.turn_dir(new_tid, 1), "queued")
    console.print(f"Forked → thread {new_tid} (queued; will run via worker or `chat -t {new_tid[:8]}`)")
```

- [ ] **Step 3: Run tests — expect pass**

```
uv run pytest tests/unit/test_cli_threads.py -v
```

- [ ] **Step 4: Run full suite**

```
uv run pytest tests/unit -q
```

- [ ] **Step 5: Commit**

```
git add src/llama_agents/cli.py tests/unit/test_cli_threads.py
git commit -m "feat(cli): threads list/show/rerun subcommand group"
```

---

## Task 19: Live e2e test

**Files:**
- Create: `tests/live/test_thread_e2e.py`

- [ ] **Step 1: Read the existing live test for harness pattern**

Open `tests/live/test_queue_e2e.py` to refresh the harness style.

- [ ] **Step 2: Write the live test**

Create `tests/live/test_thread_e2e.py`:

```python
import asyncio
import time
from pathlib import Path

import pytest

from llama_agents.config import Config, QueueConfig, SandboxConfig, load_config
from llama_agents.queue.worker import JobQueueWorker
from llama_agents.runtime import Runtime
from llama_agents.thread.status import read_status
from llama_agents.thread.store import ThreadStore


@pytest.mark.live
@pytest.mark.asyncio
async def test_thread_continue_e2e(tmp_path: Path):
    """Submit a turn; wait for completion; submit a follow-up; wait; assert
    the second turn's response shows the agent had prior context."""
    base_cfg = load_config("config.toml")
    cfg = Config.model_validate({
        **base_cfg.model_dump(),
        "sandbox": SandboxConfig(
            allowed_dirs=[tmp_path],
            shell_allowlist=base_cfg.sandbox.shell_allowlist,
        ).model_dump(),
        "queue": QueueConfig(
            enabled=True, root=tmp_path / "q",
            poll_interval_seconds=0.5, max_concurrent=1,
            max_retries=0, retry_backoff_seconds=0.0,
            max_iterations=8, drain_timeout_seconds=10.0,
        ).model_dump(),
    })

    rt = await Runtime.create(cfg)
    worker = JobQueueWorker(rt, cfg.queue, thread_store=rt.thread_store)
    task = asyncio.create_task(worker.run())
    try:
        # Turn 1
        tid = rt.thread_store.create_thread(title="my secret number is 42")
        td1 = rt.thread_store.turn_dir(tid, 1)
        (td1 / "prompt.md").write_text(
            "Remember: my secret number is 42. Reply with just OK.",
            encoding="utf-8",
        )
        from llama_agents.thread.status import set_status
        set_status(td1, "queued")

        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if read_status(td1) == "done":
                break
            await asyncio.sleep(0.5)
        else:
            pytest.fail("turn 1 did not complete within 60s")

        # Turn 2 — ask about it
        td2, _ = rt.thread_store.next_turn_dir(tid)
        (td2 / "prompt.md").write_text(
            "What was my secret number?", encoding="utf-8",
        )
        set_status(td2, "queued")

        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if read_status(td2) == "done":
                break
            await asyncio.sleep(0.5)
        else:
            pytest.fail("turn 2 did not complete within 60s")

        result = (td2 / "result.md").read_text(encoding="utf-8")
        # Soft assertion: the response should mention 42
        assert "42" in result, f"expected '42' in result, got: {result[:200]}"
    finally:
        await worker.drain(timeout=5.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await rt.aclose()
```

- [ ] **Step 3: Run live test**

```
uv run pytest tests/live/test_thread_e2e.py -m live -v
```

Expected: PASS (if llama-server is reachable). If the live test fails for non-infrastructure reasons (e.g. the model genuinely doesn't carry context through the turn boundary), investigate; do not weaken the assertion.

- [ ] **Step 4: Commit**

```
git add tests/live/test_thread_e2e.py
git commit -m "test(live): two-turn thread e2e — agent carries context across turns"
```

---

## Task 20: Docs

**Files:**
- Create: `docs/threads.md`
- Modify: `CLAUDE.md`
- Modify: `docs/web.md`
- Modify: `docs/install.md`

- [ ] **Step 1: Write `docs/threads.md`**

Create `docs/threads.md`:

````markdown
# Threads

Every submission to llama-agents lives in a **thread**. A thread has
an id, a title, and one or more **turns**. A one-shot prompt is just
a thread with a single turn.

## Layout

Under `<queue_root>/threads/<thread_id>/`:

- `meta.json` — title, timestamps, current_turn, optional parent link
- `messages.jsonl` — running conversation, one OpenAI-shaped message per line
- `turns/<NNN>/`
  - `prompt.md` — the user's submission for this turn
  - `status` — `queued` | `processing` | `done` | `failed`
  - `result.md`, `events.jsonl`, `error.txt` as appropriate

## Continuing a thread

Web: click any thread, scroll to the **Continue** form at the bottom,
type a follow-up, hit Send.

CLI: `uv run llamactl chat --thread <id-or-prefix> "follow-up"`.

The agent's prior conversation (messages + tool calls + tool results)
is hydrated as the context for the new turn.

## Rerunning a turn

Web: each turn block has a **⟳ Rerun** button. Optionally edit the
prompt, hit Submit. The original thread is preserved; you get a new
thread that's a fork at the rerun point.

CLI: `uv run llamactl threads rerun <id-or-prefix> <turn> [--edit "..."]`.
Without `--edit`, the original prompt is reused verbatim — useful for
retrying after a transient error.

## How rerun-as-fork inherits memory

When thread B is a fork of thread A, B's agent can recall everything
A stored in scratch memory. The memory layer walks the parent chain
(`meta.parent_thread_id`) up to a depth of 32 and includes all
ancestor thread ids in the recall query.

## Thread ids and prefixes

Thread ids are 24 lowercase hex characters. Most commands accept a
prefix of at least 4 characters as long as it uniquely matches one
thread. The web URLs always use the full id.

## Migration from the old queue model

The first time you start `llamactl serve` (or run any other entry
point) after upgrading, any files left in the old
`inbox/`/`processing/`/`done/`/`failed/` folders are automatically
migrated into single-turn threads. The migration is idempotent and
safe to re-run; partial failures leave the source files in place for
inspection.
````

- [ ] **Step 2: Update `CLAUDE.md`**

In the "Module map" table, add rows:

```
| `thread/ids.py` | thread id minting + prefix resolution. |
| `thread/meta.py` | ThreadMeta dataclass + JSON I/O. |
| `thread/status.py` | atomic per-turn status transitions. |
| `thread/store.py` | ThreadStore: create, list, turn_dir, messages, ancestor chain. |
| `thread/migration.py` | one-shot legacy inbox/done/failed → threads/. |
```

Update the `queue/worker.py` row to reflect the thread integration. Update the `web/routes.py` row.

Remove the "no multi-turn conversation continuity" entry from "Known limitations".

- [ ] **Step 3: Update `docs/web.md`**

Change references to `/` (dashboard) → `/activity`. Add a Threads section pointing at `docs/threads.md`. Update the page list.

- [ ] **Step 4: Update `docs/install.md`**

Add one line noting the auto-migration on first boot for users upgrading from a pre-thread checkout.

- [ ] **Step 5: Regression check**

```
uv run pytest tests/unit -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git add docs/threads.md "D:\repos\llm\llama-agents\CLAUDE.md" docs/web.md docs/install.md
git commit -m "docs(threads): user guide, module map update, web/install pointers"
```

---

## Done criteria

After Task 20:

- `uv run pytest tests/unit -q` is green; the suite includes ~45 new thread-related tests.
- `uv run pytest tests/live -m live` includes `test_thread_continue_e2e` and it passes against a real llama-server.
- A fresh checkout: `uv run llamactl init` → `uv run llamactl serve` → open `/activity`, submit a prompt, see it run, click into the thread, type a follow-up, see the agent respond with awareness of the first turn.
- A pre-thread checkout: `uv run llamactl serve` migrates any existing inbox/done/failed files into threads on first boot, logged at INFO.
- `llamactl threads list / show / rerun` work; `llamactl chat --thread <prefix>` continues a thread; `llamactl chat --background` enqueues without blocking.
- `CLAUDE.md` no longer claims "no multi-turn" as a limitation.
