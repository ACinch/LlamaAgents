from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

from .errors import LlamaAgentsError
from .events import (
    AssistantChunk,
    Done,
    Event,
    LoopError,
    ToolCallResult,
    ToolCallStart,
)
from .llama_client import ChatResponse
from .tools.registry import ToolRegistry


class _ClientLike(Protocol):
    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = ...,
    ) -> ChatResponse: ...


@dataclass
class AgentRunOptions:
    max_iterations: int = 20
    system_prompt: str = (
        "You are a careful coding agent. Use tools to read files, run commands, "
        "and query the RAG when helpful. When finished, reply in plain text."
    )
    temperature: float = 0.2


class Agent:
    def __init__(
        self,
        *,
        client: _ClientLike,
        registry: ToolRegistry,
    ) -> None:
        self._client = client
        self._registry = registry
        self._cancel = asyncio.Event()
        self.messages: list[dict[str, Any]] = []

    def cancel(self) -> None:
        self._cancel.set()

    async def run(
        self, user_prompt: str, opts: AgentRunOptions
    ) -> AsyncIterator[Event]:
        self.messages = [
            {"role": "system", "content": opts.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for _ in range(opts.max_iterations):
            if self._cancel.is_set():
                yield Done(reason="cancelled")
                return

            try:
                resp = await self._client.chat(
                    messages=self.messages,
                    tools=self._registry.schemas(),
                    temperature=opts.temperature,
                )
            except LlamaAgentsError as e:
                yield LoopError(error_type=type(e).__name__, message=str(e))
                yield Done(reason="error")
                return

            self.messages.append(
                resp.raw_message
                or {"role": "assistant", "content": resp.content}
            )

            if not resp.tool_calls:
                if resp.content:
                    yield AssistantChunk(text=resp.content)
                yield Done(reason="finished", final_message=resp.content)
                return

            for call in resp.tool_calls:
                yield ToolCallStart(
                    call_id=call.id, name=call.name, arguments=call.arguments
                )
                try:
                    result = await self._registry.invoke(call.name, call.arguments)
                    ok, content = True, result
                except Exception as e:  # noqa: BLE001 — feed all tool errors back
                    ok, content = False, f"{type(e).__name__}: {e}"
                yield ToolCallResult(call_id=call.id, ok=ok, content=content)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": _as_tool_text(ok, content),
                    }
                )

        yield Done(reason="max_iterations")


def _as_tool_text(ok: bool, content: Any) -> str:
    if ok:
        return content if isinstance(content, str) else _json_dump(content)
    return _json_dump({"error": str(content)})


def _json_dump(x: Any) -> str:
    import json

    try:
        return json.dumps(x, default=str)
    except (TypeError, ValueError):
        return str(x)
