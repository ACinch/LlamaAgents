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
        kind="subagent_output", scope="run", thread_id="r1",
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
    h1 = await store.store_blob(kind="user", scope="run", thread_id="r1",
                                title="a", body="cats love tuna")
    await store.store_blob(kind="user", scope="run", thread_id="r1",
                           title="b", body="dogs love tuna too")
    tool = MemoryRecallTool(store=store, run_id_getter=lambda: "r1")
    res = await tool.invoke({"query": "tuna", "handle": h1})
    assert all(c["blob_id"] == h1 for c in res["chunks"])
    await store.close()
