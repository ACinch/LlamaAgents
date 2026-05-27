import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from llama_agents.config import Config, LlamaConfig, AgentConfig, QueueConfig, SandboxConfig
from llama_agents.http_app import create_app
from llama_agents.llama_client import ChatResponse


class FakeClient:
    async def chat(self, *, messages, tools, temperature=0.2, reasoning_budget_tokens=None):
        return ChatResponse(content="hello from fake")

    async def health(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


class _FakeClient:
    async def chat(self, **_):
        return ChatResponse(content="ok")

    async def health(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        llama=LlamaConfig(auto_spawn=False),
        agent=AgentConfig(),
        sandbox=SandboxConfig(allowed_dirs=[tmp_path], shell_allowlist=["python"]),
    )


def test_health_endpoint(cfg: Config):
    app = create_app(cfg, client_factory=lambda url: FakeClient())
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_chat_stream_endpoint_emits_sse(cfg: Config):
    app = create_app(cfg, client_factory=lambda url: FakeClient())
    with TestClient(app) as c:
        with c.stream("POST", "/chat", json={"prompt": "hi"}) as r:
            assert r.status_code == 200
            body = "".join(r.iter_text())
            assert "hello from fake" in body
            assert "event: done" in body


def test_serialize_memory_events():
    import json
    from llama_agents.http_app import _serialize
    from llama_agents.events import MemoryEvicted, MemoryStored

    stored = _serialize(MemoryStored(blob_id="abc123", kind="tool_result", scope="turn", bytes_=1024))
    assert stored["event"] == "memory_stored"
    data = json.loads(stored["data"])
    assert data["blob_id"] == "abc123"
    assert data["kind"] == "tool_result"
    assert data["bytes"] == 1024

    evicted = _serialize(MemoryEvicted(blob_id="abc123", turn=2, bytes_freed=512))
    assert evicted["event"] == "memory_evicted"
    data = json.loads(evicted["data"])
    assert data["blob_id"] == "abc123"
    assert data["bytes_freed"] == 512


@pytest.mark.asyncio
async def test_lifespan_starts_queue_worker_when_enabled(tmp_path: Path):
    from asgi_lifespan import LifespanManager

    cfg = Config(
        llama=LlamaConfig(auto_spawn=False),
        sandbox=SandboxConfig(allowed_dirs=[tmp_path]),
        queue=QueueConfig(
            enabled=True, root=tmp_path / "q",
            poll_interval_seconds=0.05, max_concurrent=1,
            drain_timeout_seconds=1.0, max_iterations=3,
        ),
    )
    app = create_app(cfg, client_factory=lambda url: _FakeClient())
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Drop a job, wait for it to land in done/.
            (tmp_path / "q" / "inbox").mkdir(parents=True, exist_ok=True)
            (tmp_path / "q" / "inbox" / "hi.md").write_text("hello")
            for _ in range(40):
                if (tmp_path / "q" / "done" / "hi.md").exists():
                    break
                await asyncio.sleep(0.05)
            else:
                pytest.fail("job never landed in done/")
            assert (tmp_path / "q" / "done" / "hi.md").read_text() == "ok"


@pytest.mark.asyncio
async def test_lifespan_resolves_relative_queue_root_against_sandbox(tmp_path: Path):
    """Regression test for I3: relative cfg.queue.root must be resolved
    against cfg.sandbox.allowed_dirs[0], not against the process cwd."""
    from asgi_lifespan import LifespanManager
    from llama_agents.config import MemoryConfig

    cfg = Config(
        llama=LlamaConfig(auto_spawn=False),
        sandbox=SandboxConfig(allowed_dirs=[tmp_path]),
        queue=QueueConfig(
            enabled=True,
            root=Path(".llama_agents/queue"),  # relative — must resolve against tmp_path
            poll_interval_seconds=0.05, max_concurrent=1,
            drain_timeout_seconds=1.0, max_iterations=3,
        ),
    )
    app = create_app(cfg, client_factory=lambda url: _FakeClient())
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test"):
            resolved_inbox = tmp_path / ".llama_agents" / "queue" / "inbox"
            # Wait for the worker's __init__ to have created the folder.
            for _ in range(40):
                if resolved_inbox.is_dir():
                    break
                await asyncio.sleep(0.05)
            assert resolved_inbox.is_dir(), (
                f"queue root not resolved against sandbox; expected {resolved_inbox}"
            )


@pytest.mark.asyncio
async def test_lifespan_does_not_start_worker_when_disabled(tmp_path: Path):
    from asgi_lifespan import LifespanManager

    cfg = Config(
        llama=LlamaConfig(auto_spawn=False),
        sandbox=SandboxConfig(allowed_dirs=[tmp_path]),
        queue=QueueConfig(enabled=False, root=tmp_path / "q"),
    )
    app = create_app(cfg, client_factory=lambda url: _FakeClient())
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Even if a job is present, it must NOT be processed.
            (tmp_path / "q" / "inbox").mkdir(parents=True, exist_ok=True)
            (tmp_path / "q" / "inbox" / "ignored.md").write_text("nope")
            await asyncio.sleep(0.3)
            assert (tmp_path / "q" / "inbox" / "ignored.md").exists()
            assert not (tmp_path / "q" / "done" / "ignored.md").exists()
