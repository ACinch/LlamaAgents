import asyncio
from typing import Any

import pytest

from llama_agents.agent import Agent, AgentRunOptions
from llama_agents.events import (
    AssistantChunk,
    Done,
    LoopError,
    PlanAccepted,
    PlanProposed,
    PlanReviewed,
    ReviewerVerdict,
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
        ChatResponse(content="ACCEPT"),                # reviewer 0
        ChatResponse(content="ACCEPT"),                # reviewer 1
        ChatResponse(content="ACCEPT"),                # reviewer 2
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
        ChatResponse(content="REJECT: step 2 names a tool that does not exist"),
        ChatResponse(content="REJECT: step 2 names a tool that does not exist"),
        ChatResponse(content="1. echo hi\n2. done"),                   # draft 2
        ChatResponse(content="ACCEPT"),                                # reviewer 0
        ChatResponse(content="ACCEPT"),                                # reviewer 1
        ChatResponse(content="ACCEPT"),                                # reviewer 2
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
        ChatResponse(content="REJECT: bad"),
        ChatResponse(content="REJECT: bad"),
        ChatResponse(content="plan B"),
        ChatResponse(content="REJECT: still bad"),
        ChatResponse(content="REJECT: still bad"),
        ChatResponse(content="REJECT: still bad"),
        ChatResponse(content="plan C"),
        ChatResponse(content="REJECT: worse"),
        ChatResponse(content="REJECT: worse"),
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


def test_agent_run_options_has_reviewer_defaults():
    from llama_agents.agent import AgentRunOptions

    opts = AgentRunOptions()
    assert opts.reviewer_count == 3
    assert opts.reviewer_temperature == 0.5
    # Backwards compat: existing fields untouched
    assert opts.max_planning_iterations == 3
    assert opts.max_iterations == 20


def test_normalize_verdict_accepts_simple_accept():
    from llama_agents.agent import _normalize_verdict
    accepted, feedback = _normalize_verdict("ACCEPT")
    assert accepted is True
    assert feedback == ""


def test_normalize_verdict_accepts_with_checklist_preamble():
    from llama_agents.agent import _normalize_verdict
    raw = (
        "Tool validity: PASS\n"
        "Scope match: PASS\n"
        "Context safety: PASS\n"
        "Specificity: PASS\n"
        "Executability: PASS\n"
        "\n"
        "ACCEPT"
    )
    accepted, feedback = _normalize_verdict(raw)
    assert accepted is True


def test_normalize_verdict_rejects_with_reason():
    from llama_agents.agent import _normalize_verdict
    raw = "Tool validity: FAIL — step 3 names rag_query, no such tool.\nREJECT: step 3 names a non-existent tool"
    accepted, feedback = _normalize_verdict(raw)
    assert accepted is False
    assert "step 3" in feedback


def test_normalize_verdict_handles_lowercase():
    from llama_agents.agent import _normalize_verdict
    accepted, feedback = _normalize_verdict("accept")
    assert accepted is True


def test_normalize_verdict_malformed_counts_as_reject_with_truncated_raw():
    from llama_agents.agent import _normalize_verdict
    raw = "the plan seems okay i guess " * 50  # no ACCEPT/REJECT prefix
    accepted, feedback = _normalize_verdict(raw)
    assert accepted is False
    assert len(feedback) <= 500
    assert "the plan seems okay" in feedback


def test_normalize_verdict_empty_string_rejects():
    from llama_agents.agent import _normalize_verdict
    accepted, feedback = _normalize_verdict("")
    assert accepted is False


def test_normalize_verdict_trailing_whitespace_stripped():
    from llama_agents.agent import _normalize_verdict
    accepted, feedback = _normalize_verdict("ACCEPT\n\n  \n")
    assert accepted is True


async def test_three_accepts_accepts_plan_emits_three_verdicts():
    client = ScriptedClient([
        ChatResponse(content="1. echo hi\n2. done"),  # planner
        ChatResponse(content="ACCEPT"),                 # reviewer 0
        ChatResponse(content="ACCEPT"),                 # reviewer 1
        ChatResponse(content="ACCEPT"),                 # reviewer 2
        ChatResponse(content="all done"),               # main loop final
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(agent.run("orchestrate", AgentRunOptions(max_iterations=3)))
    rvs = [e for e in events if isinstance(e, ReviewerVerdict)]
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(rvs) == 3
    assert all(rv.accepted for rv in rvs)
    assert len(accepted) == 1 and accepted[0].attempts == 1


async def test_majority_accept_wins():
    client = ScriptedClient([
        ChatResponse(content="plan"),                   # planner
        ChatResponse(content="ACCEPT"),                 # reviewer 0
        ChatResponse(content="REJECT: nope"),           # reviewer 1
        ChatResponse(content="ACCEPT"),                 # reviewer 2
        ChatResponse(content="done"),
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(agent.run("orchestrate", AgentRunOptions(max_iterations=3)))
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(accepted) == 1 and accepted[0].attempts == 1


async def test_majority_reject_triggers_retry():
    client = ScriptedClient([
        ChatResponse(content="plan 1"),
        ChatResponse(content="REJECT: a"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="REJECT: b"),
        ChatResponse(content="plan 2"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="done"),
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(agent.run("orchestrate", AgentRunOptions(max_iterations=3)))
    plans = [e for e in events if isinstance(e, PlanProposed)]
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(plans) == 2
    assert len(accepted) == 1 and accepted[0].attempts == 2


async def test_three_rejects_collates_distinct_feedback():
    """Three distinct rejection reasons should appear as a bulleted list
    in the planner's retry user message."""
    client = ScriptedClient([
        ChatResponse(content="plan 1"),
        ChatResponse(content="REJECT: missing tool"),
        ChatResponse(content="REJECT: scope drift"),
        ChatResponse(content="REJECT: vague step 2"),
        ChatResponse(content="plan 2"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="done"),
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    await _collect(agent.run("orchestrate", AgentRunOptions(max_iterations=3)))
    # The retry user message is the LAST message in the planner's second
    # call's messages. Inspect via client.calls.
    # Order of calls: planner(0), reviewer 0(1), reviewer 1(2), reviewer 2(3),
    # planner(4 — second attempt), reviewer 0(5), ...
    retry_user_msg = client.calls[4]["messages"][-1]["content"]
    assert "missing tool" in retry_user_msg
    assert "scope drift" in retry_user_msg
    assert "vague step 2" in retry_user_msg
    # Bulleted format when >1 distinct reasons
    assert retry_user_msg.count("- ") >= 3


async def test_three_rejects_dedupes_identical_feedback():
    """Three reviewers giving identical feedback should produce only one
    bullet in the retry message."""
    client = ScriptedClient([
        ChatResponse(content="plan 1"),
        ChatResponse(content="REJECT: same problem"),
        ChatResponse(content="REJECT: same problem"),
        ChatResponse(content="REJECT: same problem"),
        ChatResponse(content="plan 2"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="done"),
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    await _collect(agent.run("orchestrate", AgentRunOptions(max_iterations=3)))
    retry_user_msg = client.calls[4]["messages"][-1]["content"]
    # Only one occurrence of the reason
    assert retry_user_msg.count("same problem") == 1


async def test_reviewer_count_one_consumes_single_call_per_attempt():
    """Backwards-compat: reviewer_count=1 falls back to single-reviewer flow."""
    client = ScriptedClient([
        ChatResponse(content="plan"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="done"),
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(agent.run(
        "orchestrate",
        AgentRunOptions(max_iterations=3, reviewer_count=1),
    ))
    rvs = [e for e in events if isinstance(e, ReviewerVerdict)]
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(rvs) == 1
    assert len(accepted) == 1


class _PartialFailureClient:
    """Returns scripted responses, but raises on a specific call index."""

    def __init__(self, script: list, fail_indices: set[int], exc: Exception):
        self._script = list(script)
        self._fail = fail_indices
        self._exc = exc
        self._i = -1
        self.calls: list[dict[str, Any]] = []

    async def chat(self, *, messages, tools, temperature=0.2, reasoning_budget_tokens=None):
        self._i += 1
        self.calls.append({"messages": list(messages), "tools": tools})
        if self._i in self._fail:
            raise self._exc
        return self._script.pop(0)


async def test_one_reviewer_exception_treated_as_reject_vote():
    """1 reviewer raises + 2 ACCEPT = 2/3 majority, plan accepted."""
    from llama_agents.errors import LlamaUnreachable

    # Script: planner (idx 0), reviewer-0 raises (idx 1), reviewer-1 ACCEPT (idx 2),
    # reviewer-2 ACCEPT (idx 3), main loop final (idx 4)
    script = [
        ChatResponse(content="plan"),    # planner (idx 0)
        ChatResponse(content="ACCEPT"),  # reviewer-1 (idx 2 after failure)
        ChatResponse(content="ACCEPT"),  # reviewer-2 (idx 3)
        ChatResponse(content="done"),    # main loop (idx 4)
    ]
    client = _PartialFailureClient(script, {1}, LlamaUnreachable("conn refused"))
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(agent.run("orchestrate", AgentRunOptions(max_iterations=3)))
    rvs = [e for e in events if isinstance(e, ReviewerVerdict)]
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    # One verdict is False (the exception), two are True
    falses = [r for r in rvs if not r.accepted]
    trues = [r for r in rvs if r.accepted]
    assert len(falses) == 1
    assert "LlamaUnreachable" in falses[0].feedback
    assert len(trues) == 2
    assert len(accepted) == 1


async def test_all_reviewers_exception_emits_loop_error_and_no_plan_accepted():
    from llama_agents.errors import LlamaUnreachable

    # Script: planner (idx 0), all 3 reviewers raise (idx 1,2,3), main loop (idx 4)
    # When all reviewers fail, LoopError is emitted but main loop still runs.
    script = [
        ChatResponse(content="plan"),  # planner (idx 0)
        ChatResponse(content="done"),  # main loop (idx 4 after 3 reviewer failures)
    ]
    client = _PartialFailureClient(script, {1, 2, 3}, LlamaUnreachable("server down"))
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(agent.run("orchestrate", AgentRunOptions(max_iterations=3)))
    errors = [e for e in events if isinstance(e, LoopError)]
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(errors) == 1
    assert errors[0].error_type == "LlamaUnreachable"
    assert len(accepted) == 0


class _SlowReviewerClient:
    """First call returns the planner response immediately; subsequent calls
    sleep so a cancel() can fire while they're in flight."""

    def __init__(self, planner_response, delay: float = 0.5):
        self._planner = planner_response
        self._delay = delay
        self._called_planner = False
        self.calls: list[dict[str, Any]] = []

    async def chat(self, *, messages, tools, temperature=0.2, reasoning_budget_tokens=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self._called_planner:
            self._called_planner = True
            return self._planner
        await asyncio.sleep(self._delay)
        return ChatResponse(content="ACCEPT")


async def test_cancellation_mid_review_returns_without_plan_accepted():
    client = _SlowReviewerClient(
        planner_response=ChatResponse(content="plan"),
        delay=0.3,
    )
    agent = Agent(client=client, registry=_orchestrator_registry())
    # Schedule a cancel while reviewers are sleeping.
    async def _runner():
        await asyncio.sleep(0.1)
        agent.cancel()
    cancel_task = asyncio.create_task(_runner())
    events = await _collect(agent.run("orchestrate", AgentRunOptions(max_iterations=3)))
    await cancel_task
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(accepted) == 0


async def test_reviewer_count_zero_yields_loop_error_no_crash():
    """Regression: reviewer_count=0 used to crash via StopIteration. Now
    surfaces a clean LoopError. (Main loop still runs after the planning
    error and consumes one more ChatResponse.)"""
    client = ScriptedClient([
        ChatResponse(content="plan"),  # planner; 0 reviewer calls follow
        ChatResponse(content="done"),  # main loop after LoopError
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(agent.run(
        "orchestrate",
        AgentRunOptions(max_iterations=3, reviewer_count=0),
    ))
    errors = [e for e in events if isinstance(e, LoopError)]
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(errors) == 1
    assert errors[0].error_type == "ValueError"
    assert "reviewer_count" in errors[0].message
    assert len(accepted) == 0
