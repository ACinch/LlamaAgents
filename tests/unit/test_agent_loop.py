import asyncio
from typing import Any

import pytest

from llama_agents.agent import Agent, AgentRunOptions
from llama_agents.events import (
    AssistantChunk,
    Done,
    PlanAccepted,
    PlanProposed,
    PlanReviewed,
    ToolCallResult,
    ToolCallStart,
)
from llama_agents.errors import MaxIterationsExceeded
from llama_agents.llama_client import ChatResponse, ToolCall
from llama_agents.tools.base import Tool
from llama_agents.tools.registry import ToolRegistry


class ScriptedClient:
    """Returns a predefined sequence of ChatResponses."""

    def __init__(self, script: list[ChatResponse]):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, *, messages, tools, temperature=0.2, reasoning_budget_tokens=None):
        self.calls.append({"messages": list(messages), "tools": tools, "reasoning_budget_tokens": reasoning_budget_tokens})
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


class _SpawnStub(Tool):
    name = "subagent_spawn"
    description = "stub"
    json_schema = {"type": "object", "properties": {}}

    async def invoke(self, args):
        return {"result": "stub"}


def _orchestrator_registry():
    reg = ToolRegistry()
    reg.register(StubEcho())
    reg.register(_SpawnStub())
    return reg


async def test_planning_skipped_when_no_subagent_spawn_in_registry():
    """Subagent-style registry (no spawn tool) must NOT trigger planning."""
    client = ScriptedClient([ChatResponse(content="direct answer")])
    agent = Agent(client=client, registry=_registry_with_echo())
    events = await _collect(agent.run("hi", AgentRunOptions(max_iterations=3)))
    assert not any(isinstance(e, (PlanProposed, PlanReviewed, PlanAccepted)) for e in events)


async def test_planning_skipped_when_opts_skip_planning_true():
    client = ScriptedClient([ChatResponse(content="direct answer")])
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(
        agent.run("hi", AgentRunOptions(max_iterations=3, skip_planning=True))
    )
    assert not any(isinstance(e, (PlanProposed, PlanReviewed, PlanAccepted)) for e in events)


async def test_planning_runs_when_orchestrator_registry_has_spawn():
    client = ScriptedClient([
        ChatResponse(content="1. do X\n2. do Y"),    # planner draft
        ChatResponse(content="ACCEPT"),                # reviewer
        ChatResponse(content="all done"),              # main loop final reply
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(agent.run("orchestrate me", AgentRunOptions(max_iterations=3)))
    plans = [e for e in events if isinstance(e, PlanProposed)]
    reviews = [e for e in events if isinstance(e, PlanReviewed)]
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(plans) == 1 and plans[0].attempt == 1
    assert len(reviews) == 1 and reviews[0].accepted is True
    assert len(accepted) == 1 and accepted[0].attempts == 1


async def test_planning_iterates_on_reject_then_accepts():
    client = ScriptedClient([
        ChatResponse(content="bad plan"),                              # draft 1
        ChatResponse(content="REJECT: step 2 names a tool that does not exist"),
        ChatResponse(content="1. echo hi\n2. done"),                   # draft 2
        ChatResponse(content="ACCEPT"),                                # accepted
        ChatResponse(content="done"),                                  # main loop final
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(agent.run("orchestrate", AgentRunOptions(max_iterations=3)))
    plans = [e for e in events if isinstance(e, PlanProposed)]
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(plans) == 2
    assert len(accepted) == 1 and accepted[0].attempts == 2


async def test_planning_gives_up_after_max_iterations_and_uses_last_plan():
    client = ScriptedClient([
        ChatResponse(content="plan A"),
        ChatResponse(content="REJECT: bad"),
        ChatResponse(content="plan B"),
        ChatResponse(content="REJECT: still bad"),
        ChatResponse(content="plan C"),
        ChatResponse(content="REJECT: worse"),
        ChatResponse(content="answer"),  # main loop after exhausting retries
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(
        agent.run(
            "orchestrate",
            AgentRunOptions(max_iterations=2, max_planning_iterations=3),
        )
    )
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(accepted) == 1
    assert accepted[0].plan == "plan C"
    assert accepted[0].attempts == 3


async def test_cancellation_stops_loop():
    client = ScriptedClient([
        ChatResponse(content=None, tool_calls=[ToolCall(id="c", name="echo", arguments={"text": "x"})]),
        ChatResponse(content="never reached"),
    ])
    agent = Agent(client=client, registry=_registry_with_echo())
    agent.cancel()  # pre-cancel
    events = await _collect(agent.run("go", AgentRunOptions(max_iterations=5)))
    assert isinstance(events[-1], Done) and events[-1].reason == "cancelled"
