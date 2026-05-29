from pathlib import Path

import pytest
from typer.testing import CliRunner

from llama_agents.cli import app


@pytest.fixture
def cfg_file(tmp_path: Path) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(
        f'[llama]\nauto_spawn = false\n'
        f'[sandbox]\nallowed_dirs = ["{tmp_path.as_posix()}"]\n'
        f'[queue]\nenabled = true\nroot = "{(tmp_path / "q").as_posix()}"\n',
        encoding="utf-8",
    )
    return p


def test_chat_background_creates_queued_thread(cfg_file, tmp_path, monkeypatch):
    """`chat --background` writes a thread with status=queued and exits 0."""
    runner = CliRunner()
    result = runner.invoke(app, [
        "chat", "--config", str(cfg_file), "--background", "hello world",
    ])
    assert result.exit_code == 0, result.stdout
    threads_root = tmp_path / "q" / "threads"
    threads = [p for p in threads_root.iterdir() if p.is_dir()]
    assert len(threads) == 1
    turn1 = threads[0] / "turns" / "001"
    assert (turn1 / "prompt.md").read_text(encoding="utf-8") == "hello world"
    assert (turn1 / "status").read_text(encoding="utf-8").strip() == "queued"
    # Thread id printed on stdout
    assert threads[0].name in result.stdout


def test_chat_thread_continue_refuses_when_prior_running(cfg_file, tmp_path):
    from llama_agents.thread.store import ThreadStore
    from llama_agents.thread.status import set_status

    threads_root = tmp_path / "q" / "threads"
    threads_root.mkdir(parents=True)
    store = ThreadStore(threads_root)
    tid = store.create_thread(title="t")
    (store.turn_dir(tid, 1) / "prompt.md").write_text("first", encoding="utf-8")
    set_status(store.turn_dir(tid, 1), "processing")

    runner = CliRunner()
    result = runner.invoke(app, [
        "chat", "--config", str(cfg_file),
        "--thread", tid[:8], "--background", "follow-up",
    ])
    assert result.exit_code == 3
    assert "active turn" in result.stdout.lower() or "active turn" in result.stderr.lower()


def test_threads_list_shows_threads(cfg_file, tmp_path):
    from llama_agents.thread.store import ThreadStore
    threads_root = tmp_path / "q" / "threads"
    threads_root.mkdir(parents=True)
    store = ThreadStore(threads_root)
    a = store.create_thread(title="alpha")
    b = store.create_thread(title="beta")

    runner = CliRunner()
    result = runner.invoke(app, ["threads", "--config", str(cfg_file), "list"])
    assert result.exit_code == 0
    assert a[:8] in result.stdout
    assert b[:8] in result.stdout
    assert "alpha" in result.stdout
    assert "beta" in result.stdout


def test_threads_show_renders_turns(cfg_file, tmp_path):
    from llama_agents.thread.store import ThreadStore
    from llama_agents.thread.status import set_status
    threads_root = tmp_path / "q" / "threads"
    threads_root.mkdir(parents=True)
    store = ThreadStore(threads_root)
    tid = store.create_thread(title="t")
    (store.turn_dir(tid, 1) / "prompt.md").write_text("ask X", encoding="utf-8")
    (store.turn_dir(tid, 1) / "result.md").write_text("answer Y", encoding="utf-8")
    set_status(store.turn_dir(tid, 1), "done")

    runner = CliRunner()
    result = runner.invoke(app, [
        "threads", "--config", str(cfg_file), "show", tid[:8],
    ])
    assert result.exit_code == 0
    assert "ask X" in result.stdout
    assert "answer Y" in result.stdout
