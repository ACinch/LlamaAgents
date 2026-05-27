from __future__ import annotations

import pytest

from llama_agents.install import RecordedPrompter


def test_recorded_prompter_ask_returns_seeded_answer():
    p = RecordedPrompter(answers=["alice", "bob"])
    assert p.ask("name?") == "alice"
    assert p.ask("again?") == "bob"
    assert p.prompts_seen == ["name?", "again?"]


def test_recorded_prompter_ask_returns_default_when_empty():
    p = RecordedPrompter(answers=[""])
    assert p.ask("name?", default="alice") == "alice"


def test_recorded_prompter_confirm_yes_no_parsing():
    p = RecordedPrompter(answers=["y", "n", "", "yes", "no"])
    assert p.confirm("a?") is True
    assert p.confirm("b?") is False
    assert p.confirm("c?", default=True) is True
    assert p.confirm("d?") is True
    assert p.confirm("e?") is False


def test_recorded_prompter_choose_returns_index():
    p = RecordedPrompter(answers=["2"])
    assert p.choose("pick", ["a", "b", "c"]) == 1  # 1-indexed input -> 0-indexed return


def test_recorded_prompter_choose_default_index_on_empty():
    p = RecordedPrompter(answers=[""])
    assert p.choose("pick", ["a", "b", "c"], default_index=1) == 1


def test_recorded_prompter_info_and_warn_collected():
    p = RecordedPrompter(answers=[])
    p.info("hello")
    p.warn("careful")
    assert p.messages == [("info", "hello"), ("warn", "careful")]


def test_recorded_prompter_raises_when_out_of_answers():
    p = RecordedPrompter(answers=[])
    with pytest.raises(RuntimeError, match="ran out of scripted answers"):
        p.ask("name?")


from llama_agents.install import CATALOGUE, ModelSpec, recommend_tier, tier_defaults


def test_catalogue_has_three_tiers():
    tiers = {m.tier for m in CATALOGUE}
    assert tiers == {"L", "M", "S"}


def test_catalogue_filenames_unique():
    names = [m.hf_filename for m in CATALOGUE]
    assert len(names) == len(set(names))


def test_recommend_tier_thresholds():
    assert recommend_tier(None) == "unknown"
    assert recommend_tier(0.0) == "unknown"
    assert recommend_tier(7.99) == "unknown"
    assert recommend_tier(8.0) == "S"
    assert recommend_tier(13.99) == "S"
    assert recommend_tier(14.0) == "M"
    assert recommend_tier(23.99) == "M"
    assert recommend_tier(24.0) == "L"
    assert recommend_tier(48.0) == "L"


def test_tier_defaults_match_spec():
    assert tier_defaults("L") == (65536, 2)
    assert tier_defaults("M") == (32768, 2)
    assert tier_defaults("S") == (8192, 1)
    assert tier_defaults("unknown") == (8192, 1)


import subprocess

from llama_agents.install import detect_vram_gb


def test_detect_vram_returns_gb_from_nvidia_smi(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="24576\n", stderr=""
        )
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert detect_vram_gb() == pytest.approx(24.0, abs=0.01)


def test_detect_vram_picks_first_gpu(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="16384\n8192\n", stderr=""
        )
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert detect_vram_gb() == pytest.approx(16.0, abs=0.01)


def test_detect_vram_returns_none_when_no_nvidia_smi(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert detect_vram_gb() is None


def test_detect_vram_returns_none_on_subprocess_error(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.SubprocessError("boom")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert detect_vram_gb() is None


def test_detect_vram_returns_none_on_garbage_output(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="N/A\n", stderr=""
        )
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert detect_vram_gb() is None


import shutil
from pathlib import Path

from llama_agents.install import locate_llama_server


def test_locate_finds_llama_cpp_build_sibling(tmp_path: Path):
    repo = tmp_path / "llama-agents"
    repo.mkdir()
    target = tmp_path / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\0")
    assert locate_llama_server(repo) == target


def test_locate_falls_back_to_llamacpp_bin_sibling(tmp_path: Path):
    repo = tmp_path / "llama-agents"
    repo.mkdir()
    target = tmp_path / "llamacpp-bin" / "llama-server.exe"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\0")
    assert locate_llama_server(repo) == target


def test_locate_uses_path_lookup_when_no_sibling(tmp_path: Path, monkeypatch):
    repo = tmp_path / "llama-agents"
    repo.mkdir()
    fake_path_exe = tmp_path / "fake-on-path" / "llama-server"
    fake_path_exe.parent.mkdir()
    fake_path_exe.write_bytes(b"\0")
    def fake_which(name):
        if name in ("llama-server.exe", "llama-server"):
            return str(fake_path_exe)
        return None
    monkeypatch.setattr(shutil, "which", fake_which)
    assert locate_llama_server(repo) == fake_path_exe


def test_locate_returns_none_when_nothing_found(tmp_path: Path, monkeypatch):
    repo = tmp_path / "llama-agents"
    repo.mkdir()
    monkeypatch.setattr(shutil, "which", lambda *_: None)
    assert locate_llama_server(repo) is None


def test_locate_prefers_build_release_over_llamacpp_bin(tmp_path: Path):
    repo = tmp_path / "llama-agents"
    repo.mkdir()
    a = tmp_path / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe"
    b = tmp_path / "llamacpp-bin" / "llama-server.exe"
    a.parent.mkdir(parents=True)
    a.write_bytes(b"\0")
    b.parent.mkdir(parents=True)
    b.write_bytes(b"\0")
    assert locate_llama_server(repo) == a


from llama_agents.install import find_existing_model, model_search_dirs


def test_model_search_dirs_lists_three_locations(tmp_path: Path):
    repo = tmp_path / "llama-agents"
    repo.mkdir()
    dirs = model_search_dirs(repo)
    assert dirs[0] == repo / "GGUF"
    assert dirs[1] == repo.parent / "GGUF"
    # third entry is ~/GGUF — just check the name; home varies per machine
    assert dirs[2].name == "GGUF"


def test_find_existing_model_returns_first_hit(tmp_path: Path, monkeypatch):
    repo = tmp_path / "llama-agents"
    repo.mkdir()
    target = repo / "GGUF" / "Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"
    target.parent.mkdir()
    target.write_bytes(b"\0")
    # Force the third (home) entry to be tmp-path-anchored so the test stays sandboxed.
    monkeypatch.setattr(
        "llama_agents.install.model_search_dirs",
        lambda r: [r / "GGUF", r.parent / "GGUF", tmp_path / "fake_home" / "GGUF"],
    )
    spec = CATALOGUE[0]
    assert find_existing_model(spec, repo) == target


def test_find_existing_model_returns_none_when_absent(tmp_path: Path, monkeypatch):
    repo = tmp_path / "llama-agents"
    repo.mkdir()
    monkeypatch.setattr(
        "llama_agents.install.model_search_dirs",
        lambda r: [r / "GGUF", r.parent / "GGUF", tmp_path / "fake_home" / "GGUF"],
    )
    assert find_existing_model(CATALOGUE[0], repo) is None


def test_find_existing_model_skips_missing_dirs(tmp_path: Path, monkeypatch):
    """If a search dir doesn't exist, we should not crash."""
    repo = tmp_path / "llama-agents"
    repo.mkdir()
    monkeypatch.setattr(
        "llama_agents.install.model_search_dirs",
        lambda r: [r / "absent1", r / "absent2", tmp_path / "absent3"],
    )
    assert find_existing_model(CATALOGUE[0], repo) is None
