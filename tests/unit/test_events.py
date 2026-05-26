from llama_agents.events import (
    AssistantChunk,
    ToolCallStart,
    ToolCallResult,
    LoopError,
    Done,
    Event,
)


def test_event_types_are_dataclasses_and_share_base():
    e1 = AssistantChunk(text="hi")
    e2 = ToolCallStart(call_id="c", name="t", arguments={"x": 1})
    e3 = ToolCallResult(call_id="c", ok=True, content="ok")
    e4 = LoopError(error_type="X", message="m")
    e5 = Done(reason="finished")
    for e in (e1, e2, e3, e4, e5):
        assert isinstance(e, Event)
