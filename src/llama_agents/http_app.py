from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .agent import AgentRunOptions
from .config import Config
from .events import AssistantChunk, Done, LoopError, ToolCallResult, ToolCallStart
from .runtime import Runtime


class ChatRequest(BaseModel):
    prompt: str
    max_iterations: int | None = None
    system_prompt: str | None = None


def create_app(
    cfg: Config,
    *,
    client_factory: Callable | None = None,
) -> FastAPI:
    runtime_box: dict[str, Runtime] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime_box["rt"] = await Runtime.create(cfg, client_factory=client_factory)
        try:
            yield
        finally:
            await runtime_box["rt"].aclose()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    async def chat(req: ChatRequest):
        rt = runtime_box["rt"]
        agent = rt.new_agent()
        opts = AgentRunOptions(
            max_iterations=req.max_iterations or cfg.agent.max_iterations,
            system_prompt=req.system_prompt or AgentRunOptions().system_prompt,
        )

        async def gen():
            async for ev in agent.run(req.prompt, opts):
                yield _serialize(ev)

        return EventSourceResponse(gen())

    return app


def _serialize(ev: Any) -> dict[str, str]:
    if isinstance(ev, AssistantChunk):
        return {"event": "assistant_chunk", "data": json.dumps({"text": ev.text})}
    if isinstance(ev, ToolCallStart):
        return {
            "event": "tool_call_start",
            "data": json.dumps(
                {"call_id": ev.call_id, "name": ev.name, "arguments": ev.arguments}
            ),
        }
    if isinstance(ev, ToolCallResult):
        return {
            "event": "tool_call_result",
            "data": json.dumps(
                {"call_id": ev.call_id, "ok": ev.ok, "content": str(ev.content)}
            ),
        }
    if isinstance(ev, LoopError):
        return {
            "event": "error",
            "data": json.dumps({"type": ev.error_type, "message": ev.message}),
        }
    if isinstance(ev, Done):
        return {
            "event": "done",
            "data": json.dumps(
                {"reason": ev.reason, "final_message": ev.final_message}
            ),
        }
    return {"event": "unknown", "data": "{}"}
