from pathlib import Path

import pytest

from llama_agents.thread.ids import (
    AmbiguousPrefix,
    UnknownPrefix,
    mint_thread_id,
    resolve_prefix,
    validate_thread_id,
)


def test_mint_thread_id_returns_24_hex():
    a = mint_thread_id()
    assert len(a) == 24
    assert all(c in "0123456789abcdef" for c in a)


def test_mint_thread_id_is_unique():
    ids = {mint_thread_id() for _ in range(100)}
    assert len(ids) == 100


def test_validate_accepts_24_hex():
    assert validate_thread_id("0123456789abcdef01234567") is True


def test_validate_rejects_wrong_length():
    assert validate_thread_id("0123") is False
    assert validate_thread_id("0" * 32) is False


def test_validate_rejects_non_hex():
    assert validate_thread_id("g" + "0" * 23) is False


def test_resolve_prefix_unique(tmp_path: Path):
    (tmp_path / "8c9f2bd6e041a3b5708141d9").mkdir()
    (tmp_path / "4e1a72fd0000000000000000").mkdir()
    assert resolve_prefix(tmp_path, "8c9f") == "8c9f2bd6e041a3b5708141d9"


def test_resolve_prefix_full_id_passthrough(tmp_path: Path):
    full = "8c9f2bd6e041a3b5708141d9"
    (tmp_path / full).mkdir()
    assert resolve_prefix(tmp_path, full) == full


def test_resolve_prefix_ambiguous_raises(tmp_path: Path):
    (tmp_path / "8c9f000000000000000000aa").mkdir()
    (tmp_path / "8c9f000000000000000000bb").mkdir()
    with pytest.raises(AmbiguousPrefix) as ei:
        resolve_prefix(tmp_path, "8c9f")
    assert "8c9f000000000000000000aa" in str(ei.value)
    assert "8c9f000000000000000000bb" in str(ei.value)


def test_resolve_prefix_unknown_raises(tmp_path: Path):
    (tmp_path / "8c9f000000000000000000aa").mkdir()
    with pytest.raises(UnknownPrefix):
        resolve_prefix(tmp_path, "ffff")


def test_resolve_prefix_too_short_raises(tmp_path: Path):
    (tmp_path / "8c9f000000000000000000aa").mkdir()
    with pytest.raises(ValueError, match="at least 4"):
        resolve_prefix(tmp_path, "8c")


def test_resolve_prefix_root_missing_raises_unknown(tmp_path: Path):
    with pytest.raises(UnknownPrefix):
        resolve_prefix(tmp_path / "nonexistent", "abcd")
