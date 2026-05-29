import asyncio
from pathlib import Path

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from llama_agents.config import Config, LlamaConfig, QueueConfig, SandboxConfig
from llama_agents.http_app import create_app
from llama_agents.llama_client import ChatResponse
from llama_agents.thread.store import ThreadStore
from llama_agents.thread.status import set_status


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


# ---------- / redirect ----------

@pytest.mark.asyncio
async def test_root_redirects_to_activity(cfg, config_path):
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/", follow_redirects=False)
            assert r.status_code == 302
            assert r.headers["location"] == "/activity"


# ---------- /activity ----------

@pytest.mark.asyncio
async def test_activity_route_returns_200_and_contains_buckets(cfg, config_path):
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/activity")
            assert r.status_code == 200
            for word in ("Inbox", "Processing", "Done", "Failed"):
                assert word in r.text


@pytest.mark.asyncio
async def test_activity_wires_htmx_polling_for_each_bucket(cfg, config_path):
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/activity")
            assert r.status_code == 200
            for status in ("queued", "processing", "done", "failed"):
                assert f'hx-get="/api/jobs/{status}"' in r.text
            assert 'hx-trigger="load, every 2s"' in r.text


# ---------- /threads ----------

@pytest.mark.asyncio
async def test_threads_index_empty(cfg, config_path):
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/threads")
            assert r.status_code == 200
            assert "No threads yet" in r.text


@pytest.mark.asyncio
async def test_threads_index_lists_thread(cfg, config_path):
    store = ThreadStore(cfg.queue.root / "threads")
    store.create_thread(title="My first thread")

    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/threads")
            assert r.status_code == 200
            assert "My first thread" in r.text


# ---------- /threads/{id} ----------

@pytest.mark.asyncio
async def test_thread_detail_renders_turn_content(cfg, config_path):
    store = ThreadStore(cfg.queue.root / "threads")
    tid = store.create_thread(title="Test thread")
    td = store.turn_dir(tid, 1)
    td.mkdir(parents=True, exist_ok=True)
    (td / "prompt.md").write_text("THE PROMPT", encoding="utf-8")
    (td / "result.md").write_text("THE ANSWER", encoding="utf-8")
    set_status(td, "done")

    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get(f"/threads/{tid}")
            assert r.status_code == 200
            assert "THE PROMPT" in r.text
            assert "THE ANSWER" in r.text


@pytest.mark.asyncio
async def test_thread_detail_invalid_id_returns_404(cfg, config_path):
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/threads/not-a-valid-id-at-all")
            assert r.status_code == 404


@pytest.mark.asyncio
async def test_thread_detail_missing_thread_returns_404(cfg, config_path):
    # Valid format but doesn't exist
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/threads/aabbccddeeff00112233aabb")
            assert r.status_code == 404


# ---------- /api/jobs/{status} — now queries thread turns ----------

@pytest.mark.asyncio
async def test_api_jobs_done_lists_done_turns(cfg, config_path):
    store = ThreadStore(cfg.queue.root / "threads")
    tid = store.create_thread(title="alpha")
    td = store.turn_dir(tid, 1)
    set_status(td, "done")

    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/api/jobs/done")
            assert r.status_code == 200
            assert "alpha" in r.text or tid in r.text


@pytest.mark.asyncio
async def test_api_jobs_processing_empty_returns_empty_state(cfg, config_path):
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/api/jobs/processing")
            assert r.status_code == 200
            assert "<li" not in r.text


@pytest.mark.asyncio
async def test_api_jobs_unknown_status_returns_404(cfg, config_path):
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/api/jobs/elsewhere")
            assert r.status_code == 404


@pytest.mark.asyncio
async def test_api_jobs_queued_lists_queued_turns(cfg, config_path):
    store = ThreadStore(cfg.queue.root / "threads")
    tid = store.create_thread(title="queued-thread")
    td = store.turn_dir(tid, 1)
    set_status(td, "queued")

    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/api/jobs/queued")
            assert r.status_code == 200
            assert "queued-thread" in r.text or tid in r.text


# ---------- /static ----------

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


# ---------- /api/submit (Task 16 will wire to thread store) ----------

@pytest.mark.asyncio
async def test_submit_multipart_file_lands_in_thread(cfg, config_path):
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(
                "/api/submit",
                files={"file": ("hi.md", b"hello world", "text/markdown")},
                follow_redirects=False,
            )
            assert r.status_code == 303
            location = r.headers["location"]
    # Extract thread ID from redirect location
    tid = location.split("/threads/")[1]
    threads_root = cfg.queue.root / "threads"
    turn1 = threads_root / tid / "turns" / "001"
    assert (turn1 / "prompt.md").is_file()
    assert (turn1 / "prompt.md").read_text(encoding="utf-8") == "hello world"
    assert (turn1 / "status").read_text(encoding="utf-8").strip() == "queued"


@pytest.mark.asyncio
async def test_submit_textarea_lands_in_thread(cfg, config_path):
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(
                "/api/submit",
                data={"filename": "t.md", "body": "hello"},
                follow_redirects=False,
            )
            assert r.status_code == 303
            location = r.headers["location"]
    # Extract thread ID from redirect location
    tid = location.split("/threads/")[1]
    threads_root = cfg.queue.root / "threads"
    turn1 = threads_root / tid / "turns" / "001"
    assert (turn1 / "prompt.md").is_file()
    assert (turn1 / "prompt.md").read_text(encoding="utf-8") == "hello"
    assert (turn1 / "status").read_text(encoding="utf-8").strip() == "queued"


@pytest.mark.asyncio
async def test_submit_textarea_default_filename(cfg, config_path):
    from llama_agents.thread.meta import read_meta
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(
                "/api/submit",
                data={"filename": "", "body": "auto"},
                follow_redirects=False,
            )
            assert r.status_code == 303
            location = r.headers["location"]
    # Extract thread ID from redirect location
    tid = location.split("/threads/")[1]
    threads_root = cfg.queue.root / "threads"
    turn1 = threads_root / tid / "turns" / "001"
    # Title should be derived from first line of body
    meta = read_meta(threads_root, tid)
    assert meta.title == "auto"
    assert (turn1 / "prompt.md").read_text(encoding="utf-8") == "auto"


@pytest.mark.asyncio
async def test_submit_rejects_bad_extension(cfg, config_path):
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
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(
                "/api/submit",
                data={"filename": "../escape.md", "body": "nope"},
            )
            assert r.status_code == 400


# ---------- /api/threads/{id}/rerun/{turn} (Task 15) ----------

@pytest.mark.asyncio
async def test_rerun_forks_thread(cfg, config_path):
    from llama_agents.thread.meta import read_meta
    threads_root = cfg.queue.root / "threads"
    threads_root.mkdir(parents=True)
    store = ThreadStore(threads_root)
    parent_id = store.create_thread(title="parent")
    (store.turn_dir(parent_id, 1) / "prompt.md").write_text("orig", encoding="utf-8")
    set_status(store.turn_dir(parent_id, 1), "done")

    app = create_app(cfg, client_factory=lambda url: _FakeClient(),
                     config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(f"/api/threads/{parent_id}/rerun/1",
                              data={"body": "edited prompt"},
                              follow_redirects=False)
            assert r.status_code == 303
            location = r.headers["location"]
    new_tid = location.split("/threads/")[1].split("#")[0]
    assert new_tid != parent_id
    new_meta = read_meta(threads_root, new_tid)
    assert new_meta.parent_thread_id == parent_id
    assert new_meta.parent_turn_idx == 0  # fork of turn 1 starts before turn 1
    assert (store.turn_dir(new_tid, 1) / "prompt.md").read_text(encoding="utf-8") == "edited prompt"


@pytest.mark.asyncio
async def test_rerun_without_edit_reuses_original_prompt(cfg, config_path):
    from llama_agents.thread.meta import read_meta
    threads_root = cfg.queue.root / "threads"
    threads_root.mkdir(parents=True)
    store = ThreadStore(threads_root)
    parent_id = store.create_thread(title="parent")
    (store.turn_dir(parent_id, 1) / "prompt.md").write_text("original text",
                                                            encoding="utf-8")
    set_status(store.turn_dir(parent_id, 1), "done")

    app = create_app(cfg, client_factory=lambda url: _FakeClient(),
                     config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(f"/api/threads/{parent_id}/rerun/1",
                              follow_redirects=False)
            assert r.status_code == 303
            location = r.headers["location"]
    new_tid = location.split("/threads/")[1].split("#")[0]
    assert (store.turn_dir(new_tid, 1) / "prompt.md").read_text(encoding="utf-8") == "original text"


@pytest.mark.asyncio
async def test_submit_creates_distinct_threads_for_same_filename(cfg, config_path):
    """Two submits with the same filename should create two different threads."""
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r1 = await ac.post(
                "/api/submit",
                data={"filename": "same.md", "body": "first"},
                follow_redirects=False,
            )
            assert r1.status_code == 303
            tid1 = r1.headers["location"].split("/threads/")[1]

            r2 = await ac.post(
                "/api/submit",
                data={"filename": "same.md", "body": "second"},
                follow_redirects=False,
            )
            assert r2.status_code == 303
            tid2 = r2.headers["location"].split("/threads/")[1]

    # Verify two distinct threads were created
    assert tid1 != tid2
    threads_root = cfg.queue.root / "threads"
    assert (threads_root / tid1 / "turns" / "001" / "prompt.md").read_text(encoding="utf-8") == "first"
    assert (threads_root / tid2 / "turns" / "001" / "prompt.md").read_text(encoding="utf-8") == "second"


# ---------- /config ----------

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


# ---------- presets ----------

@pytest.mark.asyncio
async def test_activity_lists_presets_from_docs_examples(cfg, tmp_path: Path):
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
            r = await ac.get("/activity")
            assert r.status_code == 200
            assert "First example" in r.text
            assert "Second example" in r.text
            assert "alpha-task" in r.text       # the id appears as option value
            assert "preset-picker" in r.text    # the <select> id
            assert "presets-data" in r.text     # the JSON data block
            assert "README — should be filtered" not in r.text
            assert "Meta — should be filtered" not in r.text


@pytest.mark.asyncio
async def test_activity_omits_preset_picker_when_no_examples(cfg, config_path):
    """No docs/examples folder near the config -> dropdown not rendered."""
    app = create_app(cfg, client_factory=lambda url: _FakeClient(), config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/activity")
            assert r.status_code == 200
            assert "preset-picker" not in r.text


# ---------- /api/threads/{id}/continue (Task 14) ----------

@pytest.mark.asyncio
async def test_continue_appends_turn(cfg, config_path):
    from llama_agents.thread.meta import read_meta
    threads_root = cfg.queue.root / "threads"
    threads_root.mkdir(parents=True)
    store = ThreadStore(threads_root)
    tid = store.create_thread(title="t")
    (store.turn_dir(tid, 1) / "prompt.md").write_text("first", encoding="utf-8")
    set_status(store.turn_dir(tid, 1), "done")

    app = create_app(cfg, client_factory=lambda url: _FakeClient(),
                     config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(f"/api/threads/{tid}/continue",
                              data={"body": "follow-up"})
            assert r.status_code in (303, 200)
    assert (store.turn_dir(tid, 2) / "prompt.md").read_text(encoding="utf-8") == "follow-up"
    assert (store.turn_dir(tid, 2) / "status").read_text(encoding="utf-8").strip() == "queued"


@pytest.mark.asyncio
async def test_continue_refuses_when_prior_running(cfg, config_path):
    threads_root = cfg.queue.root / "threads"
    threads_root.mkdir(parents=True)
    store = ThreadStore(threads_root)
    tid = store.create_thread(title="t")
    (store.turn_dir(tid, 1) / "prompt.md").write_text("first", encoding="utf-8")
    set_status(store.turn_dir(tid, 1), "processing")

    app = create_app(cfg, client_factory=lambda url: _FakeClient(),
                     config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(f"/api/threads/{tid}/continue",
                              data={"body": "x"})
            assert r.status_code == 409
    # turn 2 was not created
    assert not (store.turn_dir(tid, 2)).exists()


@pytest.mark.asyncio
async def test_patch_thread_updates_title(cfg, config_path):
    from llama_agents.thread.meta import read_meta
    threads_root = cfg.queue.root / "threads"
    threads_root.mkdir(parents=True)
    store = ThreadStore(threads_root)
    tid = store.create_thread(title="original")

    app = create_app(cfg, client_factory=lambda url: _FakeClient(),
                     config_path=config_path)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.patch(f"/api/threads/{tid}",
                               json={"title": "renamed"})
            assert r.status_code == 200
    assert read_meta(threads_root, tid).title == "renamed"


# ---------- I1: queue root resolution ----------

@pytest.mark.asyncio
async def test_thread_routes_use_resolved_queue_root(tmp_path: Path):
    """If queue.root is relative, the web routes should resolve it against
    sandbox.allowed_dirs[0], matching what the worker and CLI do."""
    cfg_rel = Config(
        llama=LlamaConfig(auto_spawn=False),
        sandbox=SandboxConfig(allowed_dirs=[tmp_path]),
        queue=QueueConfig(
            enabled=False,
            root=Path(".llama_agents/queue"),  # relative — should resolve
        ),
    )
    config_path_rel = tmp_path / "config.toml"
    config_path_rel.write_text("[llama]\nauto_spawn = false\n", encoding="utf-8")
    app = create_app(cfg_rel, client_factory=lambda url: _FakeClient(),
                     config_path=config_path_rel)

    # Pre-create a thread at the resolved path
    resolved_threads = tmp_path / ".llama_agents" / "queue" / "threads"
    resolved_threads.mkdir(parents=True)
    store = ThreadStore(resolved_threads)
    store.create_thread(title="findable")

    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/threads")
            assert r.status_code == 200
            assert "findable" in r.text


# ---------- internal helpers ----------

def test_event_style_includes_reviewer_verdict():
    from llama_agents.web.routes import _EVENT_STYLE
    assert "ReviewerVerdict" in _EVENT_STYLE
    color, summary_key = _EVENT_STYLE["ReviewerVerdict"]
    assert color == "teal"
    assert summary_key == "accepted"
