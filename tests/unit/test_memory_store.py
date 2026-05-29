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
        kind="subagent_output", scope="run", thread_id="r1",
        title="t", body="# Header\n\nbody about quick brown fox",
    )
    assert handle

    chunks = await store.recall("quick brown fox", scope="all", thread_ids=["r1"], k=5)
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
    h1 = await store.store_blob(kind="user", scope="run", thread_id="r1",
                                title="a", body="cats love tuna fish")
    await store.store_blob(kind="user", scope="run", thread_id="r1",
                           title="b", body="dogs love bones and tuna")
    chunks = await store.recall("tuna", scope="all", thread_ids=["r1"], handle=h1, k=5)
    assert all(c.blob_id == h1 for c in chunks)
    await store.close()


@pytest.mark.asyncio
async def test_end_run_with_zero_retention_deletes(tmp_path: Path):
    store = MemoryStore(root=tmp_path, embedder=HashEmbedder(dim=8),
                       retention_hours=0)
    await store.init()
    store.start_run("r1")
    await store.store_blob(kind="user", scope="run", thread_id="r1",
                           title="t", body="anything")
    run_dir = tmp_path / "runs" / "r1"
    assert run_dir.exists()
    await store.end_run("r1")
    assert not run_dir.exists()
    await store.close()


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
    store = MemoryStore(root=tmp_path, embedder=HashEmbedder(dim=8))
    await store.init()
    store.start_run("t")
    await store.store_blob(kind="user", scope="run", thread_id="t",
                           title="T", body="some text")
    # scope="all" but no thread_ids means "plans only" — there are no plans.
    hits = await store.recall("text", scope="all", thread_ids=[])
    assert hits == []


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
