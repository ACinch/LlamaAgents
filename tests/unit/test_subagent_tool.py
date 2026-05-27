import asyncio
from typing import Any

import pytest

from llama_agents.agent import Agent, AgentRunOptions
from llama_agents.errors import AgentLimitExceeded
from llama_agents.llama_client import ChatResponse, ToolCall
from llama_agents.tools.builtin.subagent import SpawnSubagentTool
from llama_agents.tools.registry import ToolRegistry


class ScriptedClient:
    def __init__(self, scripts: dict[str, list[ChatResponse]]):
        self.scripts = scripts
        self.session_for_prompt: dict[str, str] = {}

    async def chat(self, *, messages, tools, temperature=0.2, reasoning_budget_tokens=None):
        # Route by the first user message text.
        for m in messages:
            if m["role"] == "user":
                key = m["content"]
                break
        script = self.scripts[key]
        return script.pop(0)


async def test_subagent_returns_final_message():
    client = ScriptedClient({
        "do thing": [ChatResponse(content="subagent done")],
    })
    parent_registry = ToolRegistry()
    semaphore = asyncio.Semaphore(5)

    def factory() -> Agent:
        return Agent(client=client, registry=ToolRegistry())

    spawn = SpawnSubagentTool(agent_factory=factory, semaphore=semaphore)
    parent_registry.register(spawn)

    result = await spawn.invoke({"task": "do thing"})
    assert result["result"] == "subagent done"
    assert result["iterations"] >= 1


async def test_subagent_respects_concurrency_cap():
    client = ScriptedClient({"x": [ChatResponse(content="ok")]})
    semaphore = asyncio.Semaphore(1)
    # Take the only slot.
    await semaphore.acquire()

    def factory() -> Agent:
        return Agent(client=client, registry=ToolRegistry())

    spawn = SpawnSubagentTool(agent_factory=factory, semaphore=semaphore)
    with pytest.raises(AgentLimitExceeded):
        await spawn.invoke({"task": "x"})

    semaphore.release()


async def test_subagent_strips_spawn_unless_allowed():
    """Subagent's own registry should NOT contain spawn unless requested."""
    client = ScriptedClient({"t": [ChatResponse(content="ok")]})
    sem = asyncio.Semaphore(5)

    captured: list[Agent] = []

    def factory() -> Agent:
        a = Agent(client=client, registry=ToolRegistry())
        captured.append(a)
        return a

    spawn = SpawnSubagentTool(agent_factory=factory, semaphore=sem)
    await spawn.invoke({"task": "t"})
    assert "subagent_spawn" not in captured[0]._registry.names()
