from pathlib import Path

import pytest

from llama_agents.thread.status import (
    claim_for_processing,
    read_status,
    revert_processing_on_startup,
    set_status,
)


def test_read_status_returns_none_when_missing(tmp_path: Path):
    assert read_status(tmp_path) is None


def test_set_status_creates_file(tmp_path: Path):
    set_status(tmp_path, "queued")
    assert read_status(tmp_path) == "queued"


def test_set_status_overwrites_atomically(tmp_path: Path):
    set_status(tmp_path, "queued")
    set_status(tmp_path, "processing")
    set_status(tmp_path, "done")
    assert read_status(tmp_path) == "done"


def test_set_status_rejects_invalid_value(tmp_path: Path):
    with pytest.raises(ValueError, match="invalid status"):
        set_status(tmp_path, "running")


def test_claim_for_processing_flips_queued_to_processing(tmp_path: Path):
    set_status(tmp_path, "queued")
    assert claim_for_processing(tmp_path) is True
    assert read_status(tmp_path) == "processing"


def test_claim_for_processing_refuses_when_not_queued(tmp_path: Path):
    set_status(tmp_path, "processing")
    assert claim_for_processing(tmp_path) is False
    assert read_status(tmp_path) == "processing"


def test_claim_for_processing_refuses_when_missing(tmp_path: Path):
    assert claim_for_processing(tmp_path) is False


def test_revert_processing_on_startup_finds_and_resets(tmp_path: Path):
    # Build a fake threads tree: two threads, three turns total, two in
    # 'processing' from a prior crash.
    t1 = tmp_path / ("aaaa" + "0" * 20) / "turns" / "001"
    t2 = tmp_path / ("aaaa" + "0" * 20) / "turns" / "002"
    t3 = tmp_path / ("bbbb" + "0" * 20) / "turns" / "001"
    for d in (t1, t2, t3):
        d.mkdir(parents=True)
    set_status(t1, "processing")
    set_status(t2, "done")
    set_status(t3, "processing")
    n = revert_processing_on_startup(tmp_path)
    assert n == 2
    assert read_status(t1) == "queued"
    assert read_status(t2) == "done"
    assert read_status(t3) == "queued"


def test_revert_processing_on_startup_with_empty_root_returns_zero(tmp_path: Path):
    empty = tmp_path / "nope"
    assert revert_processing_on_startup(empty) == 0
