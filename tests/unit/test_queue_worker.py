import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from llama_agents.agent import Agent
from llama_agents.config import QueueConfig
from llama_agents.llama_client import ChatResponse, ToolCall
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


class _ErroringClient:
    """Raises a given exception on every call to chat()."""

    def __init__(self, exc: Exception):
        self._exc = exc

    async def chat(self, **_):
        raise self._exc


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


@pytest.mark.asyncio
async def test_non_infra_error_lands_in_failed(queue_cfg, tmp_path):
    from llama_agents.errors import LlamaProtocolError

    ensure_dirs(queue_cfg.root)
    (tmp_path / "inbox" / "boom.md").write_text("trigger")

    rt = _StubRuntime(lambda: _ErroringClient(LlamaProtocolError("bad shape")))
    worker = JobQueueWorker(rt, queue_cfg)
    task = asyncio.create_task(worker.run())
    try:
        ok = await _wait_until(lambda: (tmp_path / "failed" / "boom.md").exists())
        assert ok
    finally:
        await worker.drain(timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    err_text = (tmp_path / "failed" / "boom.error.txt").read_text()
    assert "LlamaProtocolError" in err_text
    assert "bad shape" in err_text
    assert (tmp_path / "failed" / "boom.events.jsonl").exists()


class _FlakyClient:
    """Fails N times with the given exception, then returns the response."""

    def __init__(self, exc: Exception, fail_times: int, success_response):
        self._exc = exc
        self._fail_times = fail_times
        self._response = success_response
        self.calls = 0

    async def chat(self, **_):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return self._response


@pytest.mark.asyncio
async def test_infra_error_retries_then_succeeds(tmp_path):
    from llama_agents.errors import LlamaUnreachable

    cfg = QueueConfig(
        enabled=True, root=tmp_path,
        poll_interval_seconds=0.05, max_concurrent=1,
        max_retries=2, retry_backoff_seconds=0.0,
        max_iterations=5, drain_timeout_seconds=2.0,
    )
    ensure_dirs(cfg.root)
    (tmp_path / "inbox" / "retry.md").write_text("go")

    # Share a single client across new_agent() calls so .calls accumulates.
    flaky = _FlakyClient(
        LlamaUnreachable("conn refused"),
        fail_times=1,
        success_response=ChatResponse(content="finally"),
    )
    rt = _StubRuntime(lambda: flaky)
    worker = JobQueueWorker(rt, cfg)
    task = asyncio.create_task(worker.run())
    try:
        ok = await _wait_until(
            lambda: (tmp_path / "done" / "retry.md").exists(), timeout=3.0
        )
        assert ok
    finally:
        await worker.drain(timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert (tmp_path / "done" / "retry.md").read_text() == "finally"
    assert flaky.calls == 2


@pytest.mark.asyncio
async def test_infra_error_terminates_after_max_retries(tmp_path):
    from llama_agents.errors import LlamaUnreachable

    cfg = QueueConfig(
        enabled=True, root=tmp_path,
        poll_interval_seconds=0.05, max_concurrent=1,
        max_retries=1, retry_backoff_seconds=0.0,
        max_iterations=5, drain_timeout_seconds=2.0,
    )
    ensure_dirs(cfg.root)
    (tmp_path / "inbox" / "dead.md").write_text("go")

    rt = _StubRuntime(lambda: _ErroringClient(LlamaUnreachable("nope")))
    worker = JobQueueWorker(rt, cfg)
    task = asyncio.create_task(worker.run())
    try:
        ok = await _wait_until(
            lambda: (tmp_path / "failed" / "dead.md").exists(), timeout=3.0
        )
        assert ok
    finally:
        await worker.drain(timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    err = (tmp_path / "failed" / "dead.error.txt").read_text()
    assert "attempts: 2" in err  # initial + 1 retry
    assert "LlamaUnreachable" in err


class _ToolLoopClient:
    """Forces an infinite tool-call loop so max_iterations triggers."""

    async def chat(self, **_):
        return ChatResponse(
            content=None,
            tool_calls=[ToolCall(id="x", name="nonexistent", arguments={})],
        )


@pytest.mark.asyncio
async def test_max_iterations_counts_as_success(queue_cfg, tmp_path):
    ensure_dirs(queue_cfg.root)
    (tmp_path / "inbox" / "loop.md").write_text("loop forever")

    rt = _StubRuntime(lambda: _ToolLoopClient())
    worker = JobQueueWorker(rt, queue_cfg)
    task = asyncio.create_task(worker.run())
    try:
        ok = await _wait_until(lambda: (tmp_path / "done" / "loop.md").exists())
        assert ok
    finally:
        await worker.drain(timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert (tmp_path / "done" / "loop.md").read_text() == "[no final answer]"
    types = [
        json.loads(l)["type"]
        for l in (tmp_path / "done" / "loop.events.jsonl")
        .read_text().splitlines()
    ]
    assert "Done" in types


class _SlowClient:
    """Sleeps for a configurable duration before responding."""

    def __init__(self, delay: float, response: ChatResponse):
        self._delay = delay
        self._response = response

    async def chat(self, **_):
        await asyncio.sleep(self._delay)
        return self._response


@pytest.mark.asyncio
async def test_concurrency_cap_is_respected(tmp_path):
    cfg = QueueConfig(
        enabled=True, root=tmp_path,
        poll_interval_seconds=0.02, max_concurrent=2,
        max_retries=0, retry_backoff_seconds=0.0,
        max_iterations=5, drain_timeout_seconds=5.0,
    )
    ensure_dirs(cfg.root)
    for name in ("a.md", "b.md", "c.md"):
        (tmp_path / "inbox" / name).write_text("go")

    rt = _StubRuntime(lambda: _SlowClient(0.3, ChatResponse(content="ok")))
    worker = JobQueueWorker(rt, cfg)
    task = asyncio.create_task(worker.run())

    observed: list[int] = []
    try:
        # Sample the in-flight set size while jobs are running.
        for _ in range(20):
            observed.append(len(worker._in_flight))  # noqa: SLF001
            await asyncio.sleep(0.05)
        ok = await _wait_until(
            lambda: all((tmp_path / "done" / n).exists() for n in ("a.md", "b.md", "c.md")),
            timeout=5.0,
        )
        assert ok
    finally:
        await worker.drain(timeout=2.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert max(observed) <= 2, f"saw in-flight={max(observed)} (cap=2)"
    # At least one sample should have hit the cap.
    assert max(observed) >= 2


class _BlockingClient:
    """Never returns — useful for testing cancellation."""

    async def chat(self, **_):
        await asyncio.Event().wait()  # blocks forever
        raise RuntimeError("unreachable")


@pytest.mark.asyncio
async def test_drain_cancels_in_flight_and_leaves_file_in_processing(tmp_path):
    cfg = QueueConfig(
        enabled=True, root=tmp_path,
        poll_interval_seconds=0.02, max_concurrent=1,
        max_retries=0, retry_backoff_seconds=0.0,
        max_iterations=5, drain_timeout_seconds=0.1,
    )
    ensure_dirs(cfg.root)
    (tmp_path / "inbox" / "stuck.md").write_text("never returns")

    rt = _StubRuntime(lambda: _BlockingClient())
    worker = JobQueueWorker(rt, cfg)
    task = asyncio.create_task(worker.run())
    try:
        ok = await _wait_until(
            lambda: (tmp_path / "processing" / "stuck.md").exists(),
            timeout=1.0,
        )
        assert ok
    finally:
        await worker.drain(timeout=0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # File remains in processing/, NOT in done/ or failed/.
    assert (tmp_path / "processing" / "stuck.md").exists()
    assert not (tmp_path / "done" / "stuck.md").exists()
    assert not (tmp_path / "failed" / "stuck.md").exists()


@pytest.mark.asyncio
async def test_new_worker_sweeps_processing_back_into_inbox(tmp_path):
    cfg = QueueConfig(
        enabled=True, root=tmp_path,
        poll_interval_seconds=0.05, max_concurrent=1,
        max_retries=0, retry_backoff_seconds=0.0,
        max_iterations=5, drain_timeout_seconds=2.0,
    )
    ensure_dirs(cfg.root)
    # Simulate a prior crash: a file left in processing/.
    (tmp_path / "processing" / "recovered.md").write_text("was stuck")

    rt = _StubRuntime(lambda: _ScriptedClient(ChatResponse(content="done now")))
    worker = JobQueueWorker(rt, cfg)
    task = asyncio.create_task(worker.run())
    try:
        ok = await _wait_until(
            lambda: (tmp_path / "done" / "recovered.md").exists(),
            timeout=2.0,
        )
        assert ok
    finally:
        await worker.drain(timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert (tmp_path / "done" / "recovered.md").read_text() == "done now"


@pytest.mark.asyncio
async def test_finalize_writes_prompt_sidecar_to_done(queue_cfg, tmp_path):
    ensure_dirs(queue_cfg.root)
    (tmp_path / "inbox" / "foo.md").write_text("please pong")

    rt = _StubRuntime(lambda: _ScriptedClient(ChatResponse(content="pong")))
    worker = JobQueueWorker(rt, queue_cfg)
    task = asyncio.create_task(worker.run())
    try:
        ok = await _wait_until(lambda: (tmp_path / "done" / "foo.md").exists())
        assert ok
    finally:
        await worker.drain(timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    prompt_sidecar = tmp_path / "done" / "foo.prompt.md"
    assert prompt_sidecar.is_file()
    assert prompt_sidecar.read_text(encoding="utf-8") == "please pong"


@pytest.mark.asyncio
async def test_finalize_writes_prompt_sidecar_to_failed(queue_cfg, tmp_path):
    from llama_agents.errors import LlamaProtocolError

    ensure_dirs(queue_cfg.root)
    (tmp_path / "inbox" / "boom.md").write_text("trigger error")

    rt = _StubRuntime(lambda: _ErroringClient(LlamaProtocolError("bad shape")))
    worker = JobQueueWorker(rt, queue_cfg)
    task = asyncio.create_task(worker.run())
    try:
        ok = await _wait_until(lambda: (tmp_path / "failed" / "boom.md").exists())
        assert ok
    finally:
        await worker.drain(timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    prompt_sidecar = tmp_path / "failed" / "boom.prompt.md"
    assert prompt_sidecar.is_file()
    assert prompt_sidecar.read_text(encoding="utf-8") == "trigger error"
