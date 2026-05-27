from __future__ import annotations

import asyncio

import pytest

from llama_agents.agent import Agent, AgentRunOptions, _ACTIVE_RUN_ID
from llama_agents.llama_client import ChatResponse
from llama_agents.memory.embedder import HashEmbedder
from llama_agents.memory.store import MemoryStore
from llama_agents.tools.builtin.subagent import SpawnSubagentTool
from llama_agents.tools.registry import ToolRegistry


class _SubClient:
    """For the subagent itself: finishes immediately with a big text.
    Second call (if any) is treated as the summarizer."""
    def __init__(self, big_text):
        self._big = big_text
        self._called = 0

    async def chat(self, *, messages, tools, temperature=0.2,
                   reasoning_budget_tokens=None):
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
    store.start_run("rTOP")

    big = "BIG OUTPUT " * 500  # > 2000 chars
    client = _SubClient(big)

    def factory():
        registry = ToolRegistry()
        a = Agent(client=client, registry=registry, memory=store)
        a._run_id = "rTOP"
        store.start_run("rTOP")
        return a

    sem = asyncio.Semaphore(1)
    tool = SpawnSubagentTool(
        agent_factory=factory,
        semaphore=sem,
        memory=store,
        client_for_summary=client,
        inline_threshold_chars=2000,
    )

    token = _ACTIVE_RUN_ID.set("rTOP")
    try:
        result = await tool.invoke({"task": "describe the universe"})
    finally:
        _ACTIVE_RUN_ID.reset(token)
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

    sem = asyncio.Semaphore(1)
    tool = SpawnSubagentTool(
        agent_factory=factory,
        semaphore=sem,
        memory=store,
        client_for_summary=client,
        inline_threshold_chars=2000,
    )
    token = _ACTIVE_RUN_ID.set("rTOP")
    try:
        result = await tool.invoke({"task": "say hi"})
    finally:
        _ACTIVE_RUN_ID.reset(token)
    assert result["result"] == "tiny output"
    assert "memory_handle" not in result
    await store.close()
