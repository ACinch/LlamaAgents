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
from llama_agents.thread.status import read_status, set_status
from llama_agents.thread.store import ThreadStore
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

    def __init__(self, client_factory, thread_store: ThreadStore):
        self._client_factory = client_factory
        self.thread_store = thread_store

    def new_agent(self) -> Agent:
        return Agent(client=self._client_factory(), registry=ToolRegistry())


def _stage_queued_turn(store: ThreadStore, prompt: str) -> tuple[str, int]:
    """Create a one-turn thread with status=queued. Returns (thread_id, turn_idx)."""
    tid = store.create_thread(title=prompt[:60] or "untitled")
    td = store.turn_dir(tid, 1)
    (td / "prompt.md").write_text(prompt, encoding="utf-8")
    set_status(td, "queued")
    return tid, 1


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
async def test_happy_path_writes_result_and_events(queue_cfg, tmp_path):
    ensure_dirs(queue_cfg.root)
    store = ThreadStore(queue_cfg.root / "threads")
    tid, _ = _stage_queued_turn(store, "say hello")
    turn_dir = store.turn_dir(tid, 1)

    rt = _StubRuntime(lambda: _ScriptedClient(ChatResponse(content="hi there")), store)
    worker = JobQueueWorker(rt, queue_cfg, thread_store=store)
    task = asyncio.create_task(worker.run())
    try:
        ok = await _wait_until(lambda: (turn_dir / "result.md").exists())
        assert ok, "job never wrote result.md"
    finally:
        await worker.drain(timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert (turn_dir / "result.md").read_text(encoding="utf-8") == "hi there"
    events_path = turn_dir / "events.jsonl"
    assert events_path.exists()
    lines = events_path.read_text().splitlines()
    types = [json.loads(line)["type"] for line in lines]
    assert "Done" in types
    assert read_status(turn_dir) == "done"


@pytest.mark.asyncio
async def test_non_infra_error_lands_in_failed(queue_cfg, tmp_path):
    from llama_agents.errors import LlamaProtocolError

    ensure_dirs(queue_cfg.root)
    store = ThreadStore(queue_cfg.root / "threads")
    tid, _ = _stage_queued_turn(store, "trigger")
    turn_dir = store.turn_dir(tid, 1)

    rt = _StubRuntime(lambda: _ErroringClient(LlamaProtocolError("bad shape")), store)
    worker = JobQueueWorker(rt, queue_cfg, thread_store=store)
    task = asyncio.create_task(worker.run())
    try:
        ok = await _wait_until(lambda: read_status(turn_dir) == "failed")
        assert ok
    finally:
        await worker.drain(timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    err_text = (turn_dir / "error.txt").read_text()
    assert "LlamaProtocolError" in err_text
    assert "bad shape" in err_text
    assert (turn_dir / "events.jsonl").exists()


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
    store = ThreadStore(cfg.root / "threads")
    tid, _ = _stage_queued_turn(store, "go")
    turn_dir = store.turn_dir(tid, 1)

    # Share a single client across new_agent() calls so .calls accumulates.
    flaky = _FlakyClient(
        LlamaUnreachable("conn refused"),
        fail_times=1,
        success_response=ChatResponse(content="finally"),
    )
    rt = _StubRuntime(lambda: flaky, store)
    worker = JobQueueWorker(rt, cfg, thread_store=store)
    task = asyncio.create_task(worker.run())
    try:
        ok = await _wait_until(
            lambda: read_status(turn_dir) == "done", timeout=3.0
        )
        assert ok
    finally:
        await worker.drain(timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert (turn_dir / "result.md").read_text(encoding="utf-8") == "finally"
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
    store = ThreadStore(cfg.root / "threads")
    tid, _ = _stage_queued_turn(store, "go")
    turn_dir = store.turn_dir(tid, 1)

    rt = _StubRuntime(lambda: _ErroringClient(LlamaUnreachable("nope")), store)
    worker = JobQueueWorker(rt, cfg, thread_store=store)
    task = asyncio.create_task(worker.run())
    try:
        ok = await _wait_until(
            lambda: read_status(turn_dir) == "failed", timeout=3.0
        )
        assert ok
    finally:
        await worker.drain(timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    err = (turn_dir / "error.txt").read_text()
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
    store = ThreadStore(queue_cfg.root / "threads")
    tid, _ = _stage_queued_turn(store, "loop forever")
    turn_dir = store.turn_dir(tid, 1)

    rt = _StubRuntime(lambda: _ToolLoopClient(), store)
    worker = JobQueueWorker(rt, queue_cfg, thread_store=store)
    task = asyncio.create_task(worker.run())
    try:
        ok = await _wait_until(lambda: read_status(turn_dir) == "done")
        assert ok
    finally:
        await worker.drain(timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert (turn_dir / "result.md").read_text(encoding="utf-8") == "[no final answer]"
    types = [
        json.loads(l)["type"]
        for l in (turn_dir / "events.jsonl").read_text().splitlines()
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
    store = ThreadStore(cfg.root / "threads")
    turn_dirs = []
    for i in range(3):
        tid, _ = _stage_queued_turn(store, f"go {i}")
        turn_dirs.append(store.turn_dir(tid, 1))

    rt = _StubRuntime(lambda: _SlowClient(0.3, ChatResponse(content="ok")), store)
    worker = JobQueueWorker(rt, cfg, thread_store=store)
    task = asyncio.create_task(worker.run())

    observed: list[int] = []
    try:
        # Sample the in-flight set size while jobs are running.
        for _ in range(20):
            observed.append(len(worker._in_flight))  # noqa: SLF001
            await asyncio.sleep(0.05)
        ok = await _wait_until(
            lambda: all(read_status(td) == "done" for td in turn_dirs),
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
async def test_drain_cancels_in_flight_and_leaves_turn_processing(tmp_path):
    cfg = QueueConfig(
        enabled=True, root=tmp_path,
        poll_interval_seconds=0.02, max_concurrent=1,
        max_retries=0, retry_backoff_seconds=0.0,
        max_iterations=5, drain_timeout_seconds=0.1,
    )
    ensure_dirs(cfg.root)
    store = ThreadStore(cfg.root / "threads")
    tid, _ = _stage_queued_turn(store, "never returns")
    turn_dir = store.turn_dir(tid, 1)

    rt = _StubRuntime(lambda: _BlockingClient(), store)
    worker = JobQueueWorker(rt, cfg, thread_store=store)
    task = asyncio.create_task(worker.run())
    try:
        ok = await _wait_until(
            lambda: read_status(turn_dir) == "processing",
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

    # Turn remains in processing, NOT done or failed.
    assert read_status(turn_dir) == "processing"
    assert not (turn_dir / "result.md").exists()


@pytest.mark.asyncio
async def test_new_worker_reverts_processing_turns_to_queued(tmp_path):
    """On startup, any turn left in 'processing' (from a prior crash) is
    reverted to 'queued' so it can be picked up again."""
    cfg = QueueConfig(
        enabled=True, root=tmp_path,
        poll_interval_seconds=0.05, max_concurrent=1,
        max_retries=0, retry_backoff_seconds=0.0,
        max_iterations=5, drain_timeout_seconds=2.0,
    )
    ensure_dirs(cfg.root)
    store = ThreadStore(cfg.root / "threads")
    # Stage a turn but manually set status to "processing" (simulating a crash).
    tid = store.create_thread(title="was stuck")
    td = store.turn_dir(tid, 1)
    (td / "prompt.md").write_text("was stuck", encoding="utf-8")
    set_status(td, "processing")

    rt = _StubRuntime(lambda: _ScriptedClient(ChatResponse(content="done now")), store)
    worker = JobQueueWorker(rt, cfg, thread_store=store)
    task = asyncio.create_task(worker.run())
    try:
        ok = await _wait_until(
            lambda: read_status(td) == "done",
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

    assert (td / "result.md").read_text(encoding="utf-8") == "done now"


@pytest.mark.asyncio
async def test_worker_fork_inherits_parent_messages(queue_cfg, tmp_path):
    """A fork thread submitted to the worker should run with the parent's
    messages hydrated, not an empty context."""
    from llama_agents.thread.status import set_status
    from llama_agents.thread.store import ThreadStore

    threads_root = queue_cfg.root / "threads"
    threads_root.mkdir(parents=True)
    store = ThreadStore(threads_root)

    # Parent thread with one completed turn whose messages were already
    # appended to messages.jsonl
    parent = store.create_thread(title="parent")
    store.append_messages(parent, [
        {"role": "user", "content": "say x"},
        {"role": "assistant", "content": "x"},
    ])
    set_status(store.turn_dir(parent, 1), "done")

    # Fork
    child = store.create_thread(
        title="child fork", parent_thread_id=parent, parent_turn_idx=1,
    )
    (store.turn_dir(child, 1) / "prompt.md").write_text("now what?", encoding="utf-8")
    set_status(store.turn_dir(child, 1), "queued")

    # Capture what the client sees when chat() is called
    seen_messages: list[list[dict]] = []

    class _CaptureClient:
        async def chat(self, *, messages, tools, temperature=0.2,
                       reasoning_budget_tokens=None):
            seen_messages.append(list(messages))
            return ChatResponse(content="forked reply")

    rt = _StubRuntime(lambda: _CaptureClient(), store)
    worker = JobQueueWorker(rt, queue_cfg, thread_store=store)
    task = asyncio.create_task(worker.run())
    try:
        ok = await _wait_until(
            lambda: read_status(store.turn_dir(child, 1)) == "done",
            timeout=2.0,
        )
        assert ok, "fork turn never completed"
    finally:
        await worker.drain(timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # First call to the agent's client should have the parent's messages
    # already present.
    assert seen_messages, "client was never called"
    first = seen_messages[0]
    contents = [m.get("content") for m in first]
    assert "say x" in contents
    assert "x" in contents
    assert "now what?" in contents


@pytest.mark.asyncio
async def test_legacy_inbox_files_migrated_on_startup(tmp_path):
    """Pre-thread inbox/ .md files are migrated into threads/ on first run."""
    cfg = QueueConfig(
        enabled=True, root=tmp_path,
        poll_interval_seconds=0.05, max_concurrent=1,
        max_retries=0, retry_backoff_seconds=0.0,
        max_iterations=5, drain_timeout_seconds=2.0,
    )
    # Simulate a legacy inbox file (pre-thread-store era).
    (tmp_path / "inbox").mkdir(parents=True)
    (tmp_path / "inbox" / "old.md").write_text("legacy prompt", encoding="utf-8")

    store = ThreadStore(cfg.root / "threads")
    rt = _StubRuntime(lambda: _ScriptedClient(ChatResponse(content="migrated")), store)
    # Constructing the worker triggers migration.
    worker = JobQueueWorker(rt, cfg, thread_store=store)
    # inbox/ should now be gone (migrated into threads/).
    assert not (tmp_path / "inbox" / "old.md").exists()

    task = asyncio.create_task(worker.run())
    try:
        # Wait for the migrated turn to complete.
        ok = await _wait_until(
            lambda: any(
                read_status(td) == "done"
                for t_dir in (tmp_path / "threads").iterdir()
                if t_dir.is_dir()
                for td in (t_dir / "turns").iterdir()
                if td.is_dir()
            ),
            timeout=3.0,
        )
        assert ok, "migrated job never completed"
    finally:
        await worker.drain(timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
