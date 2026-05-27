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

    client = _ScriptedClient([
        _resp("1. step\n2. step\n3. step"),
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
    events = []
    async for ev in agent.run("how do I bake a sandwich?",
                              AgentRunOptions(max_iterations=2,
                                              plan_recall_threshold=0.0)):
        events.append(ev)

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
