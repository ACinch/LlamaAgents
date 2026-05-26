from __future__ import annotations

import asyncio
from typing import Any, Callable

from ...agent import Agent, AgentRunOptions
from ...errors import AgentLimitExceeded
from ...events import AssistantChunk, Done, ToolCallResult, ToolCallStart
from ..base import Tool


class SpawnSubagentTool(Tool):
    name = "subagent_spawn"
    description = (
        "Spawn a subagent with its own conversation to handle a focused task. "
        "Returns the subagent's final assistant message as `result`."
    )
    json_schema = {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "system_prompt": {"type": "string"},
            "allowed_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "tool names the subagent may use; defaults to parent's minus subagent_spawn",
            },
            "max_iterations": {"type": "integer", "default": 20},
        },
        "required": ["task"],
    }

    def __init__(
        self,
        agent_factory: Callable[[], Agent],
        semaphore: asyncio.Semaphore,
    ) -> None:
        self._factory = agent_factory
        self._sem = semaphore

    async def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._sem.locked() and self._sem._value > 0:  # type: ignore[attr-defined]
            pass  # there's room
        acquired = self._sem.locked() is False and await _try_acquire(self._sem)
        if not acquired:
            raise AgentLimitExceeded("max_concurrent_agents reached")

        try:
            subagent = self._factory()
            allowed = args.get("allowed_tools")
            if allowed is not None:
                for n in list(subagent._registry.names()):
                    if n not in allowed:
                        subagent._registry.unregister(n)
            else:
                subagent._registry.unregister("subagent_spawn")

            opts = AgentRunOptions(
                max_iterations=int(args.get("max_iterations", 20)),
                system_prompt=args.get(
                    "system_prompt",
                    "You are a focused subagent. Complete the task and report back.",
                ),
            )

            iterations = 0
            tool_calls = 0
            final_text = ""
            async for ev in subagent.run(args["task"], opts):
                if isinstance(ev, ToolCallStart):
                    tool_calls += 1
                elif isinstance(ev, AssistantChunk):
                    final_text = ev.text
                elif isinstance(ev, Done):
                    if ev.final_message:
                        final_text = ev.final_message
                    break
                iterations += 1
            return {
                "result": final_text,
                "iterations": iterations,
                "tool_calls": tool_calls,
            }
        finally:
            self._sem.release()


async def _try_acquire(sem: asyncio.Semaphore) -> bool:
    """Non-blocking acquire; returns False if no slot available."""
    if sem.locked():
        return False
    # asyncio.Semaphore doesn't expose try-acquire; use _value (best effort).
    if sem._value <= 0:  # type: ignore[attr-defined]
        return False
    await sem.acquire()
    return True
