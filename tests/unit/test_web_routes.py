import asyncio
from pathlib import Path

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from llama_agents.config import Config, LlamaConfig, QueueConfig, SandboxConfig
from llama_agents.http_app import create_app
from llama_agents.llama_client import ChatResponse


class _FakeClient:
    async def chat(self, **_): return ChatResponse(content="ok")
    async def health(self): return True
    async def aclose(self): pass


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        llama=LlamaConfig(auto_spawn=False),
        sandbox=SandboxConfig(allowed_dirs=[tmp_path]),
        queue=QueueConfig(enabled=False, root=tmp_path / "q"),
    )


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    p = tmp_path / "config.toml"
    p.write_text('[llama]\nserver_url = "http://127.0.0.1:8080"\n', encoding="utf-8")
    return p


@pytest.mark.asyncio
async def test_dashboard_route_returns_200_and_contains_buckets(cfg, config_path):
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/")
            assert r.status_code == 200
            for word in ("Inbox", "Processing", "Done", "Failed"):
                assert word in r.text


@pytest.mark.asyncio
async def test_static_htmx_is_served(cfg, config_path):
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/static/htmx.min.js")
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("application/javascript") \
                or r.headers["content-type"].startswith("text/javascript")
