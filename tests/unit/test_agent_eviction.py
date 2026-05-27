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
        evict_threshold_pct=20,
        evict_tool_result_min_chars=2000,
        ctx_size_for_eviction=8192,
    )

    evicted: list[MemoryEvicted] = []
    async for ev in agent.run("do thing", opts):
        if isinstance(ev, MemoryEvicted):
            evicted.append(ev)

    assert evicted, "expected at least one MemoryEvicted event"
    stubbed = [m for m in agent.messages
               if m.get("role") == "tool"
               and "[evicted to memory" in (m.get("content") or "")]
    assert stubbed, "expected at least one tool message rewritten with stub"
    await store.close()
