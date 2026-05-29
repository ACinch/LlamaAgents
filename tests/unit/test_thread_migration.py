from pathlib import Path

from llama_agents.thread.migration import migrate_legacy_queue_dirs
from llama_agents.thread.meta import read_meta
from llama_agents.thread.status import read_status


def test_no_legacy_dirs_returns_zero(tmp_path: Path):
    assert migrate_legacy_queue_dirs(tmp_path) == 0


def test_migrate_inbox_creates_thread(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "task-1.md").write_text("the prompt body", encoding="utf-8")
    assert migrate_legacy_queue_dirs(tmp_path) == 1
    threads_root = tmp_path / "threads"
    threads = [p for p in threads_root.iterdir() if p.is_dir()]
    assert len(threads) == 1
    tid = threads[0].name
    meta = read_meta(threads_root, tid)
    assert meta.title.startswith("the prompt body"[:60])
    assert meta.current_turn == 1
    turn1 = threads_root / tid / "turns" / "001"
    assert (turn1 / "prompt.md").read_text(encoding="utf-8") == "the prompt body"
    assert read_status(turn1) == "queued"
    # source file removed
    assert not (inbox / "task-1.md").exists()


def test_migrate_done_with_sidecars(tmp_path: Path):
    done = tmp_path / "done"
    done.mkdir()
    (done / "x.md").write_text("FINAL ANSWER", encoding="utf-8")
    (done / "x.prompt.md").write_text("ORIGINAL PROMPT", encoding="utf-8")
    (done / "x.events.jsonl").write_text('{"type":"Done"}\n', encoding="utf-8")
    n = migrate_legacy_queue_dirs(tmp_path)
    assert n == 1
    threads_root = tmp_path / "threads"
    tid = [p.name for p in threads_root.iterdir() if p.is_dir()][0]
    turn1 = threads_root / tid / "turns" / "001"
    assert read_status(turn1) == "done"
    assert (turn1 / "result.md").read_text(encoding="utf-8") == "FINAL ANSWER"
    assert (turn1 / "prompt.md").read_text(encoding="utf-8") == "ORIGINAL PROMPT"
    assert "Done" in (turn1 / "events.jsonl").read_text(encoding="utf-8")


def test_migrate_failed_with_error_txt(tmp_path: Path):
    failed = tmp_path / "failed"
    failed.mkdir()
    (failed / "b.md").write_text("(no final answer)", encoding="utf-8")
    (failed / "b.prompt.md").write_text("trigger", encoding="utf-8")
    (failed / "b.events.jsonl").write_text("", encoding="utf-8")
    (failed / "b.error.txt").write_text("attempts: 1\n", encoding="utf-8")
    migrate_legacy_queue_dirs(tmp_path)
    threads_root = tmp_path / "threads"
    tid = [p.name for p in threads_root.iterdir() if p.is_dir()][0]
    turn1 = threads_root / tid / "turns" / "001"
    assert read_status(turn1) == "failed"
    assert (turn1 / "error.txt").read_text(encoding="utf-8") == "attempts: 1\n"


def test_migrate_is_idempotent(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "t.md").write_text("p", encoding="utf-8")
    n1 = migrate_legacy_queue_dirs(tmp_path)
    n2 = migrate_legacy_queue_dirs(tmp_path)
    assert n1 == 1
    assert n2 == 0


def test_migrate_processing_treated_as_queued(tmp_path: Path):
    """Processing files from a prior crash become queued so the worker
    picks them up again."""
    proc = tmp_path / "processing"
    proc.mkdir()
    (proc / "p.md").write_text("stuck job", encoding="utf-8")
    migrate_legacy_queue_dirs(tmp_path)
    threads_root = tmp_path / "threads"
    tid = [p.name for p in threads_root.iterdir() if p.is_dir()][0]
    turn1 = threads_root / tid / "turns" / "001"
    assert read_status(turn1) == "queued"


def test_migrate_ignores_sidecar_only_files(tmp_path: Path):
    """A .prompt.md or .events.jsonl without a matching .md is not migrated
    on its own."""
    done = tmp_path / "done"
    done.mkdir()
    (done / "orphan.prompt.md").write_text("strange", encoding="utf-8")
    n = migrate_legacy_queue_dirs(tmp_path)
    assert n == 0
