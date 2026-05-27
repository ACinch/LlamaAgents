from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llama_agents.config import Config, LlamaConfig, AgentConfig, SandboxConfig
from llama_agents.http_app import create_app
from llama_agents.llama_client import ChatResponse


class FakeClient:
    async def chat(self, *, messages, tools, temperature=0.2, reasoning_budget_tokens=None):
        return ChatResponse(content="hello from fake")

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
