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


def _seed_queue_dirs(root: Path) -> None:
    for sub in ("inbox", "processing", "done", "failed"):
        (root / sub).mkdir(parents=True, exist_ok=True)


@pytest.mark.asyncio
async def test_api_jobs_inbox_lists_staged_file(cfg, config_path, tmp_path):
    _seed_queue_dirs(cfg.queue.root)
    (cfg.queue.root / "inbox" / "alpha.md").write_text("hi")

    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/api/jobs/inbox")
            assert r.status_code == 200
            assert "alpha.md" in r.text


@pytest.mark.asyncio
async def test_api_jobs_processing_empty_returns_empty_list(cfg, config_path):
    _seed_queue_dirs(cfg.queue.root)
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/api/jobs/processing")
            assert r.status_code == 200
            # No <li> rows expected.
            assert "<li" not in r.text


@pytest.mark.asyncio
async def test_api_jobs_done_filters_sidecar_files(cfg, config_path):
    _seed_queue_dirs(cfg.queue.root)
    done = cfg.queue.root / "done"
    (done / "foo.md").write_text("answer")
    (done / "foo.prompt.md").write_text("the prompt")
    (done / "foo.events.jsonl").write_text("{}")

    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/api/jobs/done")
            assert r.status_code == 200
            assert r.text.count("foo.md") >= 1  # appears in the link and link text
            assert "foo.prompt.md" not in r.text
            assert "foo.events.jsonl" not in r.text


@pytest.mark.asyncio
async def test_api_jobs_unknown_status_returns_404(cfg, config_path):
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/api/jobs/elsewhere")
            assert r.status_code == 404


@pytest.mark.asyncio
async def test_dashboard_wires_htmx_polling_for_each_bucket(cfg, config_path):
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/")
            assert r.status_code == 200
            for status in ("inbox", "processing", "done", "failed"):
                assert f'hx-get="/api/jobs/{status}"' in r.text
            assert 'hx-trigger="load, every 2s"' in r.text
