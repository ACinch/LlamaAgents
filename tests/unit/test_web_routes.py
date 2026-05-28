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


@pytest.mark.asyncio
async def test_submit_multipart_file_lands_in_inbox(cfg, config_path):
    _seed_queue_dirs(cfg.queue.root)
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(
                "/api/submit",
                files={"file": ("hi.md", b"hello world", "text/markdown")},
            )
            assert r.status_code in (303, 200)
    landed = cfg.queue.root / "inbox" / "hi.md"
    assert landed.is_file()
    assert landed.read_text(encoding="utf-8") == "hello world"


@pytest.mark.asyncio
async def test_submit_textarea_lands_in_inbox(cfg, config_path):
    _seed_queue_dirs(cfg.queue.root)
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(
                "/api/submit",
                data={"filename": "t.md", "body": "hello"},
            )
            assert r.status_code in (303, 200)
    assert (cfg.queue.root / "inbox" / "t.md").read_text(encoding="utf-8") == "hello"


@pytest.mark.asyncio
async def test_submit_textarea_default_filename(cfg, config_path):
    import re
    _seed_queue_dirs(cfg.queue.root)
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.post("/api/submit", data={"filename": "", "body": "auto"})
    matches = list((cfg.queue.root / "inbox").glob("task-*.md"))
    assert len(matches) == 1
    assert re.match(r"task-\d+\.md", matches[0].name)
    assert matches[0].read_text(encoding="utf-8") == "auto"


@pytest.mark.asyncio
async def test_submit_rejects_bad_extension(cfg, config_path):
    _seed_queue_dirs(cfg.queue.root)
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(
                "/api/submit",
                files={"file": ("bad.exe", b"NOPE", "application/octet-stream")},
            )
            assert r.status_code == 400


@pytest.mark.asyncio
async def test_submit_rejects_path_traversal(cfg, config_path):
    _seed_queue_dirs(cfg.queue.root)
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(
                "/api/submit",
                data={"filename": "../escape.md", "body": "nope"},
            )
            assert r.status_code == 400


@pytest.mark.asyncio
async def test_submit_rejects_duplicate(cfg, config_path):
    _seed_queue_dirs(cfg.queue.root)
    (cfg.queue.root / "inbox" / "dupe.md").write_text("existing")
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(
                "/api/submit",
                data={"filename": "dupe.md", "body": "new"},
            )
            assert r.status_code == 400
    assert (cfg.queue.root / "inbox" / "dupe.md").read_text(encoding="utf-8") == "existing"


@pytest.mark.asyncio
async def test_job_detail_inbox_shows_prompt_body(cfg, config_path):
    _seed_queue_dirs(cfg.queue.root)
    (cfg.queue.root / "inbox" / "foo.md").write_text("the actual prompt")
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/jobs/inbox/foo.md")
            assert r.status_code == 200
            assert "the actual prompt" in r.text
            assert "no events recorded" in r.text.lower()


@pytest.mark.asyncio
async def test_job_detail_done_shows_prompt_events_and_answer(cfg, config_path):
    import json as _json
    _seed_queue_dirs(cfg.queue.root)
    done = cfg.queue.root / "done"
    (done / "foo.md").write_text("FINAL ANSWER")
    (done / "foo.prompt.md").write_text("ORIGINAL PROMPT")
    events = [
        {"type": "ToolCallStart", "ts": "2026-05-27T10:00:00+00:00",
         "call_id": "c1", "name": "fs_read_file", "arguments": {"path": "x"}},
        {"type": "Done", "ts": "2026-05-27T10:00:05+00:00",
         "reason": "finished", "final_message": "FINAL ANSWER"},
    ]
    (done / "foo.events.jsonl").write_text(
        "\n".join(_json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )

    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/jobs/done/foo.md")
            assert r.status_code == 200
            assert "ORIGINAL PROMPT" in r.text
            assert "FINAL ANSWER" in r.text
            assert "ToolCallStart" in r.text
            assert "Done" in r.text


@pytest.mark.asyncio
async def test_job_detail_failed_shows_error(cfg, config_path):
    _seed_queue_dirs(cfg.queue.root)
    failed = cfg.queue.root / "failed"
    (failed / "boom.md").write_text("[no final answer]")
    (failed / "boom.prompt.md").write_text("trigger")
    (failed / "boom.events.jsonl").write_text("")
    (failed / "boom.error.txt").write_text(
        "attempts: 1\nerror_type: LlamaProtocolError\nmessage: bad shape\n"
    )

    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/jobs/failed/boom.md")
            assert r.status_code == 200
            assert "LlamaProtocolError" in r.text
            assert "bad shape" in r.text


@pytest.mark.asyncio
async def test_job_detail_missing_returns_404(cfg, config_path):
    _seed_queue_dirs(cfg.queue.root)
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/jobs/done/missing.md")
            assert r.status_code == 404


@pytest.mark.asyncio
async def test_job_detail_invalid_status_returns_404(cfg, config_path):
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/jobs/elsewhere/foo.md")
            assert r.status_code == 404


@pytest.mark.asyncio
async def test_job_detail_rejects_path_traversal_in_name(cfg, config_path):
    _seed_queue_dirs(cfg.queue.root)
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/jobs/done/..%2Fconfig.toml")
            assert r.status_code in (400, 404)


@pytest.mark.asyncio
async def test_config_view_returns_toml_content(cfg, config_path):
    config_path.write_text(
        '[llama]\nserver_url = "http://127.0.0.1:9999"\n', encoding="utf-8"
    )
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/config")
            assert r.status_code == 200
            assert "[llama]" in r.text
            assert "127.0.0.1:9999" in r.text


@pytest.mark.asyncio
async def test_config_view_re_reads_file_per_request(cfg, config_path):
    """Edit-in-place should surface without restart."""
    config_path.write_text('[llama]\nserver_url = "first"\n', encoding="utf-8")
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r1 = await ac.get("/config")
            assert "first" in r1.text
            config_path.write_text('[llama]\nserver_url = "second"\n', encoding="utf-8")
            r2 = await ac.get("/config")
            assert "second" in r2.text


@pytest.mark.asyncio
async def test_list_jobs_skips_files_that_vanish_between_iterdir_and_stat(cfg, config_path, tmp_path, monkeypatch):
    """If the worker moves a file out from under us, the list endpoint
    should skip it rather than 500."""
    _seed_queue_dirs(cfg.queue.root)
    # Stage one real file (alive) and one that 'vanishes' via monkeypatched stat.
    (cfg.queue.root / "inbox" / "alive.md").write_text("present")
    ghost = cfg.queue.root / "inbox" / "ghost.md"
    ghost.write_text("about to vanish")

    real_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        if self.name == "ghost.md":
            raise FileNotFoundError(str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)

    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/api/jobs/inbox")
            assert r.status_code == 200
            assert "alive.md" in r.text
            assert "ghost.md" not in r.text


@pytest.mark.asyncio
async def test_dashboard_lists_presets_from_docs_examples(cfg, tmp_path: Path):
    # Stage a fake docs/examples next to the config so _load_presets finds them.
    config_path = tmp_path / "config.toml"
    config_path.write_text('[llama]\nserver_url = "x"\n', encoding="utf-8")
    examples = tmp_path / "docs" / "examples"
    examples.mkdir(parents=True)
    (examples / "alpha-task.md").write_text("# First example\n\nbody A\n", encoding="utf-8")
    (examples / "beta-task.md").write_text("# Second example\n\nbody B\n", encoding="utf-8")
    (examples / "README.md").write_text("# README — should be filtered\n", encoding="utf-8")
    (examples / "_meta.md").write_text("# Meta — should be filtered\n", encoding="utf-8")

    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/")
            assert r.status_code == 200
            assert "First example" in r.text
            assert "Second example" in r.text
            assert "alpha-task" in r.text       # the id appears as option value
            assert "preset-picker" in r.text    # the <select> id
            assert "presets-data" in r.text     # the JSON data block
            assert "README — should be filtered" not in r.text
            assert "Meta — should be filtered" not in r.text


@pytest.mark.asyncio
async def test_dashboard_omits_preset_picker_when_no_examples(cfg, config_path):
    """No docs/examples folder near the config -> dropdown not rendered."""
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/")
            assert r.status_code == 200
            assert "preset-picker" not in r.text


def test_event_style_includes_reviewer_verdict():
    from llama_agents.web.routes import _EVENT_STYLE
    assert "ReviewerVerdict" in _EVENT_STYLE
    color, summary_key = _EVENT_STYLE["ReviewerVerdict"]
    assert color == "teal"
    assert summary_key == "accepted"
