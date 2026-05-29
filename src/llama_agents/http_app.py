from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .agent import AgentRunOptions
from .config import Config
from .events import (
    AssistantChunk, Done, LoopError, MemoryEvicted, MemoryStored,
    ReviewerVerdict, ToolCallResult, ToolCallStart,
)
from .queue.worker import JobQueueWorker
from .runtime import Runtime, _resolve_queue_root
from .web.routes import register_routes


class ChatRequest(BaseModel):
    prompt: str
    max_iterations: int | None = None
    system_prompt: str | None = None


def create_app(
    cfg: Config,
    *,
    client_factory: Callable | None = None,
    config_path: Path | None = None,
) -> FastAPI:
    runtime_box: dict[str, Runtime] = {}
    worker_box: dict[str, JobQueueWorker | None] = {"worker": None}
    worker_task_box: dict[str, asyncio.Task | None] = {"task": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        rt = await Runtime.create(cfg, client_factory=client_factory)
        runtime_box["rt"] = rt
        if cfg.queue.enabled:
            resolved_queue = cfg.queue.model_copy(update={"root": _resolve_queue_root(cfg)})
            worker = JobQueueWorker(rt, resolved_queue, thread_store=rt.thread_store)
            worker_box["worker"] = worker
            worker_task_box["task"] = asyncio.create_task(worker.run())
        try:
            yield
        finally:
            worker = worker_box["worker"]
            task = worker_task_box["task"]
            if worker is not None:
                await worker.drain(timeout=cfg.queue.drain_timeout_seconds)
            if task is not None:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            await rt.aclose()

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

    if config_path is not None:
        register_routes(app, cfg, config_path=config_path)

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
    if isinstance(ev, MemoryStored):
        return {
            "event": "memory_stored",
            "data": json.dumps(
                {"blob_id": ev.blob_id, "kind": ev.kind, "scope": ev.scope, "bytes": ev.bytes_}
            ),
        }
    if isinstance(ev, MemoryEvicted):
        return {
            "event": "memory_evicted",
            "data": json.dumps(
                {"blob_id": ev.blob_id, "turn": ev.turn, "bytes_freed": ev.bytes_freed}
            ),
        }
    if isinstance(ev, ReviewerVerdict):
        return {
            "event": "reviewer_verdict",
            "data": json.dumps({
                "attempt": ev.attempt,
                "reviewer_idx": ev.reviewer_idx,
                "accepted": ev.accepted,
                "feedback": ev.feedback,
            }),
        }
    return {"event": "unknown", "data": "{}"}
