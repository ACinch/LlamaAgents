import time
from pathlib import Path

import pytest

from llama_agents.thread.meta import read_meta
from llama_agents.thread.status import set_status
from llama_agents.thread.store import ThreadStore


def test_create_thread_writes_meta_and_turn1_folder(tmp_path: Path):
    store = ThreadStore(tmp_path)
    tid = store.create_thread(title="hello")
    assert (tmp_path / tid / "meta.json").is_file()
    assert (tmp_path / tid / "turns" / "001").is_dir()
    meta = read_meta(tmp_path, tid)
    assert meta.title == "hello"
    assert meta.current_turn == 1
    assert meta.parent_thread_id is None


def test_create_thread_records_parent(tmp_path: Path):
    store = ThreadStore(tmp_path)
    parent = store.create_thread(title="parent")
    child = store.create_thread(title="child", parent_thread_id=parent,
                                parent_turn_idx=2)
    cm = read_meta(tmp_path, child)
    assert cm.parent_thread_id == parent
    assert cm.parent_turn_idx == 2


def test_list_threads_empty(tmp_path: Path):
    assert ThreadStore(tmp_path).list_threads() == []


def test_list_threads_sorted_by_updated_at_desc(tmp_path: Path):
    store = ThreadStore(tmp_path)
    a = store.create_thread(title="first")
    time.sleep(0.01)
    b = store.create_thread(title="second")
    time.sleep(0.01)
    c = store.create_thread(title="third")
    metas = store.list_threads()
    assert [m.id for m in metas] == [c, b, a]


def test_list_threads_respects_limit(tmp_path: Path):
    store = ThreadStore(tmp_path)
    for _ in range(5):
        store.create_thread(title="t")
        time.sleep(0.005)
    assert len(store.list_threads(limit=3)) == 3
    assert len(store.list_threads(limit=None)) == 5


def test_turn_dir_returns_zero_padded(tmp_path: Path):
    store = ThreadStore(tmp_path)
    tid = store.create_thread(title="t")
    assert store.turn_dir(tid, 1).name == "001"
    assert store.turn_dir(tid, 17).name == "017"
    assert store.turn_dir(tid, 9999).name == "9999"


def test_next_turn_dir_increments_current_turn(tmp_path: Path):
    store = ThreadStore(tmp_path)
    tid = store.create_thread(title="t")
    # turn 1 already exists from create_thread; next is 2
    d, idx = store.next_turn_dir(tid)
    assert idx == 2
    assert d.name == "002"
    assert d.is_dir()
    assert read_meta(tmp_path, tid).current_turn == 2


def test_next_queued_turn_orders_by_mtime(tmp_path: Path):
    store = ThreadStore(tmp_path)
    a = store.create_thread(title="a")
    set_status(store.turn_dir(a, 1), "queued")
    time.sleep(0.02)
    b = store.create_thread(title="b")
    set_status(store.turn_dir(b, 1), "queued")
    result = store.next_queued_turn()
    assert result is not None
    tid, idx = result
    assert tid == a  # older mtime
    assert idx == 1


def test_next_queued_turn_returns_none_when_nothing_queued(tmp_path: Path):
    store = ThreadStore(tmp_path)
    tid = store.create_thread(title="x")
    set_status(store.turn_dir(tid, 1), "done")
    assert store.next_queued_turn() is None
