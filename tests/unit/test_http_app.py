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
