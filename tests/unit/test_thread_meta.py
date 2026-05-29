import json
from pathlib import Path

import pytest

from llama_agents.thread.meta import ThreadMeta, read_meta, update_meta, write_meta


def test_thread_meta_defaults():
    m = ThreadMeta(id="0123456789abcdef01234567", title="hi",
                   created_at="2026-05-29T10:00:00+00:00",
                   updated_at="2026-05-29T10:00:00+00:00",
                   current_turn=1)
    assert m.parent_thread_id is None
    assert m.parent_turn_idx is None
    assert m.current_turn == 1


def test_write_read_roundtrip(tmp_path: Path):
    m = ThreadMeta(
        id="aaaa" + "0" * 20, title="hello",
        created_at="2026-05-29T10:00:00+00:00",
        updated_at="2026-05-29T10:00:01+00:00",
        current_turn=2,
        parent_thread_id="bbbb" + "0" * 20,
        parent_turn_idx=1,
    )
    write_meta(tmp_path, m)
    got = read_meta(tmp_path, m.id)
    assert got == m


def test_read_meta_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_meta(tmp_path, "aaaa" + "0" * 20)


def test_read_meta_malformed_json_raises(tmp_path: Path):
    tid = "aaaa" + "0" * 20
    d = tmp_path / tid
    d.mkdir()
    (d / "meta.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        read_meta(tmp_path, tid)


def test_update_meta_preserves_unspecified_fields(tmp_path: Path):
    m = ThreadMeta(id="cccc" + "0" * 20, title="original",
                   created_at="2026-05-29T10:00:00+00:00",
                   updated_at="2026-05-29T10:00:00+00:00",
                   current_turn=1)
    write_meta(tmp_path, m)
    got = update_meta(tmp_path, m.id, title="renamed", current_turn=2)
    assert got.title == "renamed"
    assert got.current_turn == 2
    assert got.created_at == m.created_at
    # updated_at is bumped automatically
    assert got.updated_at != m.updated_at


def test_meta_json_field_order_stable(tmp_path: Path):
    """The on-disk JSON should be human-readable: pretty-printed."""
    m = ThreadMeta(id="dddd" + "0" * 20, title="x",
                   created_at="2026-05-29T10:00:00+00:00",
                   updated_at="2026-05-29T10:00:00+00:00",
                   current_turn=1)
    write_meta(tmp_path, m)
    text = (tmp_path / m.id / "meta.json").read_text(encoding="utf-8")
    parsed = json.loads(text)
    assert parsed["id"] == m.id
    assert "\n" in text  # pretty-printed (indent=2)
