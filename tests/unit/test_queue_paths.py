from pathlib import Path

import pytest

from llama_agents.queue.paths import (
    ensure_dirs,
    move_to_processing,
    move_to_terminal,
    sweep_processing_to_inbox,
)


def test_ensure_dirs_creates_four_subdirs(tmp_path: Path):
    ensure_dirs(tmp_path)
    for name in ("inbox", "processing", "done", "failed"):
        assert (tmp_path / name).is_dir()
    # idempotent
    ensure_dirs(tmp_path)


def test_move_to_processing_locks_file(tmp_path: Path):
    ensure_dirs(tmp_path)
    job = tmp_path / "inbox" / "a.md"
    job.write_text("hello")
    moved = move_to_processing(tmp_path, job)
    assert moved == tmp_path / "processing" / "a.md"
    assert moved.exists()
    assert not job.exists()


def test_move_to_processing_returns_none_on_race(tmp_path: Path):
    ensure_dirs(tmp_path)
    job = tmp_path / "inbox" / "a.md"
    job.write_text("hello")
    # Pre-create the destination to simulate a competing worker.
    (tmp_path / "processing" / "a.md").write_text("other")
    assert move_to_processing(tmp_path, job) is None
    # Original file is left untouched in inbox.
    assert job.exists()


def test_move_to_terminal_renames_into_done(tmp_path: Path):
    ensure_dirs(tmp_path)
    src = tmp_path / "processing" / "a.md"
    src.write_text("body")
    dst = move_to_terminal(tmp_path, src, status="done")
    assert dst == tmp_path / "done" / "a.md"
    assert dst.read_text() == "body"
    assert not src.exists()


def test_move_to_terminal_appends_suffix_on_collision(tmp_path: Path):
    ensure_dirs(tmp_path)
    (tmp_path / "done" / "a.md").write_text("old")
    src = tmp_path / "processing" / "a.md"
    src.write_text("new")
    dst = move_to_terminal(tmp_path, src, status="done")
    assert dst.name == "a.1.md"
    assert dst.read_text() == "new"
    # the pre-existing file is untouched
    assert (tmp_path / "done" / "a.md").read_text() == "old"


def test_sweep_processing_to_inbox_requeues_everything(tmp_path: Path):
    ensure_dirs(tmp_path)
    (tmp_path / "processing" / "a.md").write_text("1")
    (tmp_path / "processing" / "b.txt").write_text("2")
    moved = sweep_processing_to_inbox(tmp_path)
    assert sorted(p.name for p in moved) == ["a.md", "b.txt"]
    assert (tmp_path / "inbox" / "a.md").exists()
    assert (tmp_path / "inbox" / "b.txt").exists()
    assert not any((tmp_path / "processing").iterdir())


def test_sweep_processing_is_idempotent(tmp_path: Path):
    ensure_dirs(tmp_path)
    sweep_processing_to_inbox(tmp_path)  # nothing to do
    assert (tmp_path / "processing").is_dir()
