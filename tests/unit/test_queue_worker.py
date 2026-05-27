import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from llama_agents.agent import Agent
from llama_agents.config import QueueConfig
from llama_agents.llama_client import ChatResponse
from llama_agents.queue.paths import ensure_dirs
from llama_agents.queue.worker import JobQueueWorker
from llama_agents.tools.registry import ToolRegistry


class _ScriptedClient:
    """Returns a fixed ChatResponse on every call."""

    def __init__(self, response: ChatResponse):
        self._response = response

    async def chat(self, *, messages, tools, temperature=0.2,
                   reasoning_budget_tokens=None):
        return self._response


class _StubRuntime:
    """Minimal runtime: hands out fresh Agents with a scripted client."""

    def __init__(self, client_factory):
        self._client_factory = client_factory

    def new_agent(self) -> Agent:
        return Agent(client=self._client_factory(), registry=ToolRegistry())


@pytest.fixture
def queue_cfg(tmp_path: Path) -> QueueConfig:
    return QueueConfig(
        enabled=True,
        root=tmp_path,
        poll_interval_seconds=0.05,
        max_concurrent=1,
        max_retries=0,
        retry_backoff_seconds=0.0,
        max_iterations=5,
        drain_timeout_seconds=2.0,
    )


async def _wait_until(predicate, timeout=2.0):
    loop = asyncio.get_running_loop()
    start = loop.time()
    while loop.time() - start < timeout:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return False


@pytest.mark.asyncio
async def test_happy_path_moves_inbox_to_done_with_outputs(queue_cfg, tmp_path):
    ensure_dirs(queue_cfg.root)
    (tmp_path / "inbox" / "foo.md").write_text("say hello")

    rt = _StubRuntime(lambda: _ScriptedClient(ChatResponse(content="hi there")))
    worker = JobQueueWorker(rt, queue_cfg)
    task = asyncio.create_task(worker.run())
    try:
        ok = await _wait_until(lambda: (tmp_path / "done" / "foo.md").exists())
        assert ok, "job never landed in done/"
    finally:
        await worker.drain(timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert (tmp_path / "done" / "foo.md").read_text() == "hi there"
    events_path = tmp_path / "done" / "foo.events.jsonl"
    assert events_path.exists()
    lines = events_path.read_text().splitlines()
    types = [json.loads(line)["type"] for line in lines]
    assert "Done" in types
    assert not (tmp_path / "inbox" / "foo.md").exists()
    assert not (tmp_path / "processing" / "foo.md").exists()


@pytest.mark.asyncio
async def test_ignored_extensions_are_skipped(queue_cfg, tmp_path):
    ensure_dirs(queue_cfg.root)
    (tmp_path / "inbox" / "skip.tmp").write_text("ignore me")
    (tmp_path / "inbox" / "take.md").write_text("do it")

    rt = _StubRuntime(lambda: _ScriptedClient(ChatResponse(content="ok")))
    worker = JobQueueWorker(rt, queue_cfg)
    task = asyncio.create_task(worker.run())
    try:
        ok = await _wait_until(lambda: (tmp_path / "done" / "take.md").exists())
        assert ok
    finally:
        await worker.drain(timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # The .tmp file is still in inbox, untouched.
    assert (tmp_path / "inbox" / "skip.tmp").exists()
    assert not (tmp_path / "processing" / "skip.tmp").exists()
    assert not (tmp_path / "done" / "skip.tmp").exists()
