import asyncio
from typing import Any

import pytest

from llama_agents.agent import Agent, AgentRunOptions
from llama_agents.events import AssistantChunk, Done, ToolCallResult, ToolCallStart
from llama_agents.errors import MaxIterationsExceeded
from llama_agents.llama_client import ChatResponse, ToolCall
from llama_agents.tools.base import Tool
from llama_agents.tools.registry import ToolRegistry


class ScriptedClient:
    """Returns a predefined sequence of ChatResponses."""

    def __init__(self, script: list[ChatResponse]):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, *, messages, tools, temperature=0.2):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self.script.pop(0)


class StubEcho(Tool):
    name = "echo"
    description = "echo"
    json_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def invoke(self, args):
        return args["text"]


def _registry_with_echo():
    reg = ToolRegistry()
    reg.register(StubEcho())
    return reg


async def _collect(agen):
    return [e async for e in agen]


async def test_finishes_when_model_returns_plain_message():
    client = ScriptedClient([
        ChatResponse(content="hello world"),
    ])
    agent = Agent(client=client, registry=_registry_with_echo())
    events = await _collect(agent.run("hi", AgentRunOptions(max_iterations=5)))
    assert any(isinstance(e, AssistantChunk) and e.text == "hello world" for e in events)
    assert isinstance(events[-1], Done) and events[-1].reason == "finished"


async def test_dispatches_tool_then_finishes():
    client = ScriptedClient([
        ChatResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "ping"})],
        ),
        ChatResponse(content="all done"),
    ])
    agent = Agent(client=client, registry=_registry_with_echo())
    events = await _collect(agent.run("go", AgentRunOptions(max_iterations=5)))
    assert any(isinstance(e, ToolCallStart) and e.name == "echo" for e in events)
    assert any(isinstance(e, ToolCallResult) and e.ok and e.content == "ping" for e in events)
    assert isinstance(events[-1], Done) and events[-1].reason == "finished"


async def test_tool_error_fed_back_to_model():
    class Boom(Tool):
        name = "boom"
        description = "fails"
        json_schema = {"type": "object", "properties": {}}
        async def invoke(self, args):
            raise RuntimeError("kaboom")

    reg = ToolRegistry()
    reg.register(Boom())
    client = ScriptedClient([
        ChatResponse(content=None, tool_calls=[ToolCall(id="c1", name="boom", arguments={})]),
        ChatResponse(content="recovered"),
    ])
    agent = Agent(client=client, registry=reg)
    events = await _collect(agent.run("go", AgentRunOptions(max_iterations=5)))
    bad = [e for e in events if isinstance(e, ToolCallResult)]
    assert bad and bad[0].ok is False and "kaboom" in str(bad[0].content)
    assert isinstance(events[-1], Done) and events[-1].reason == "finished"


async def test_max_iterations():
    looping = ChatResponse(
        content=None,
        tool_calls=[ToolCall(id="c", name="echo", arguments={"text": "x"})],
    )
    client = ScriptedClient([looping, looping, looping])
    agent = Agent(client=client, registry=_registry_with_echo())
    events = await _collect(agent.run("go", AgentRunOptions(max_iterations=2)))
    assert isinstance(events[-1], Done) and events[-1].reason == "max_iterations"


async def test_cancellation_stops_loop():
    client = ScriptedClient([
        ChatResponse(content=None, tool_calls=[ToolCall(id="c", name="echo", arguments={"text": "x"})]),
        ChatResponse(content="never reached"),
    ])
    agent = Agent(client=client, registry=_registry_with_echo())
    agent.cancel()  # pre-cancel
    events = await _collect(agent.run("go", AgentRunOptions(max_iterations=5)))
    assert isinstance(events[-1], Done) and events[-1].reason == "cancelled"
