from __future__ import annotations

import asyncio
from typing import Any, Callable

from ...agent import Agent, AgentRunOptions
from ...errors import AgentLimitExceeded
from ...events import AssistantChunk, Done, ToolCallResult, ToolCallStart
from ...memory.store import InertMemoryStore, MemoryStore
from ..base import Tool


_SUMMARIZER_SYSTEM = (
    "You summarize a subagent's output for the orchestrator that delegated "
    "the task. Write 3-6 sentences capturing what was done and any key "
    "findings. No preamble, no markdown headers."
)


class SpawnSubagentTool(Tool):
    name = "subagent_spawn"
    description = (
        "Spawn a subagent with its own conversation to handle a focused task. "
        "Returns the subagent's final assistant message as `result` for short "
        "outputs, or a `summary` + `memory_handle` for long outputs that have "
        "been written to memory (retrievable via memory_recall)."
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
        *,
        memory: "MemoryStore | InertMemoryStore | None" = None,
        client_for_summary: Any = None,
        inline_threshold_chars: int = 2000,
        summary_max_tokens: int = 400,
        parent_run_id_getter: Callable[[], str | None] | None = None,
    ) -> None:
        self._factory = agent_factory
        self._sem = semaphore
        self._memory = memory or InertMemoryStore()
        self._client = client_for_summary
        self._inline_threshold = inline_threshold_chars
        self._summary_max_tokens = summary_max_tokens
        self._parent_run_id_getter = parent_run_id_getter or (lambda: None)

    async def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._sem.locked() and self._sem._value > 0:  # type: ignore[attr-defined]
            pass
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

            if len(final_text) <= self._inline_threshold:
                return {
                    "result": final_text,
                    "iterations": iterations,
                    "tool_calls": tool_calls,
                }

            parent_rid = self._parent_run_id_getter()
            try:
                blob_id = await self._memory.store_blob(
                    kind="subagent_output", scope="run",
                    run_id=parent_rid,
                    title=f"subagent: {args['task'][:60]}",
                    body=final_text,
                    metadata={"task": args["task"], "iterations": iterations,
                              "tool_calls": tool_calls},
                )
            except Exception as e:  # noqa: BLE001
                import sys
                print(f"[memory] subagent store failed: {e}", file=sys.stderr)
                return {
                    "result": final_text,
                    "iterations": iterations,
                    "tool_calls": tool_calls,
                }

            summary = await self._summarize(args["task"], final_text)
            return {
                "summary": summary,
                "memory_handle": blob_id,
                "result_bytes": len(final_text),
                "iterations": iterations,
                "tool_calls": tool_calls,
            }
        finally:
            self._sem.release()

    async def _summarize(self, task: str, output: str) -> str:
        if self._client is None:
            return output[:400]
        truncated = output[:8000]
        try:
            resp = await self._client.chat(
                messages=[
                    {"role": "system", "content": _SUMMARIZER_SYSTEM},
                    {"role": "user",
                     "content": f"TASK:\n{task}\n\nOUTPUT:\n{truncated}"},
                ],
                tools=[],
                temperature=0.0,
                reasoning_budget_tokens=0,
            )
            return (resp.content or "").strip() or truncated[:400]
        except Exception as e:  # noqa: BLE001
            import sys
            print(f"[memory] summary failed: {e}", file=sys.stderr)
            return truncated[:400]


async def _try_acquire(sem: asyncio.Semaphore) -> bool:
    """Non-blocking acquire; returns False if no slot available."""
    if sem.locked():
        return False
    if sem._value <= 0:  # type: ignore[attr-defined]
        return False
    await sem.acquire()
    return True
