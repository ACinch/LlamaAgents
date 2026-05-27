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


from llama_agents.install import collect_allowed_dirs


def test_collect_allowed_dirs_seeds_with_repo_root(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    p = RecordedPrompter(answers=[""])  # immediately finish
    dirs = collect_allowed_dirs(repo, p)
    assert dirs == [repo.resolve()]


def test_collect_allowed_dirs_appends_user_inputs(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()
    p = RecordedPrompter(answers=[str(extra), ""])
    dirs = collect_allowed_dirs(repo, p)
    assert dirs == [repo.resolve(), extra.resolve()]


def test_collect_allowed_dirs_dedupes(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    p = RecordedPrompter(answers=[str(repo), ""])  # try to add repo again
    dirs = collect_allowed_dirs(repo, p)
    assert dirs == [repo.resolve()]


def test_collect_allowed_dirs_rejects_nonexistent_and_reprompts(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    p = RecordedPrompter(answers=[str(tmp_path / "nope"), str(real), ""])
    dirs = collect_allowed_dirs(repo, p)
    assert dirs == [repo.resolve(), real.resolve()]
    assert any("does not exist" in m[1] for m in p.messages)


def test_collect_allowed_dirs_rejects_files_and_reprompts(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    afile = tmp_path / "afile.txt"
    afile.write_text("x")
    real = tmp_path / "real"
    real.mkdir()
    p = RecordedPrompter(answers=[str(afile), str(real), ""])
    dirs = collect_allowed_dirs(repo, p)
    assert dirs == [repo.resolve(), real.resolve()]
    assert any("not a directory" in m[1] for m in p.messages)


from llama_agents.install import existing_config_action, write_config


def test_existing_config_action_absent_returns_write(tmp_path: Path):
    p = RecordedPrompter(answers=[])
    assert existing_config_action(tmp_path / "missing.toml", p, force=False) == "write"


def test_existing_config_action_force_returns_write(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("old", encoding="utf-8")
    p = RecordedPrompter(answers=[])
    assert existing_config_action(cfg, p, force=True) == "write"


def test_existing_config_action_prompts_and_writes_on_yes(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("old", encoding="utf-8")
    p = RecordedPrompter(answers=["y"])
    assert existing_config_action(cfg, p, force=False) == "write"


def test_existing_config_action_cancels_on_no(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("old", encoding="utf-8")
    p = RecordedPrompter(answers=["n"])
    assert existing_config_action(cfg, p, force=False) == "cancel"


def test_write_config_writes_when_absent(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    write_config(cfg, "[llama]\n", backup_existing=False)
    assert cfg.read_text(encoding="utf-8") == "[llama]\n"


def test_write_config_backs_up_existing(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("OLD", encoding="utf-8")
    write_config(cfg, "NEW", backup_existing=True)
    assert cfg.read_text(encoding="utf-8") == "NEW"
    backups = list(tmp_path.glob("config.toml.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "OLD"


def test_write_config_creates_parent_dir(tmp_path: Path):
    cfg = tmp_path / "sub" / "config.toml"
    write_config(cfg, "x", backup_existing=False)
    assert cfg.is_file()


import tomllib

from llama_agents.install import render_config_toml


def _sample_values(tmp_path: Path) -> dict:
    return {
        "server_bin": tmp_path / "llama-server.exe",
        "model_path": tmp_path / "GGUF" / "model.gguf",
        "model_label": "model.gguf",
        "ctx_size": 32768,
        "n_parallel": 2,
        "allowed_dirs": [tmp_path / "a", tmp_path / "b"],
    }


def test_render_config_toml_reparses_into_valid_config(tmp_path: Path):
    from llama_agents.config import Config

    text = render_config_toml(_sample_values(tmp_path))
    data = tomllib.loads(text)
    # Must round-trip through pydantic — proves we produced a valid Config.
    cfg = Config.model_validate(data)
    assert cfg.llama.ctx_size == 32768
    assert cfg.llama.n_parallel == 2
    assert cfg.queue.enabled is False
    assert cfg.memory.enabled is True
    assert cfg.llama.auto_spawn is True


def test_render_config_toml_includes_chosen_paths(tmp_path: Path):
    text = render_config_toml(_sample_values(tmp_path))
    assert str(tmp_path / "llama-server.exe").replace("\\", "/") in text \
        or str(tmp_path / "llama-server.exe") in text
    assert "model.gguf" in text
    assert str(tmp_path / "a").replace("\\", "/") in text \
        or str(tmp_path / "a") in text


def test_render_config_toml_shell_allowlist_is_git_only(tmp_path: Path):
    text = render_config_toml(_sample_values(tmp_path))
    data = tomllib.loads(text)
    assert data["sandbox"]["shell_allowlist"] == ["git"]


def test_render_config_toml_marks_queue_disabled(tmp_path: Path):
    text = render_config_toml(_sample_values(tmp_path))
    data = tomllib.loads(text)
    assert data["queue"]["enabled"] is False


import hashlib
import zipfile
from io import BytesIO

from llama_agents.install import (
    LLAMA_CPP_RELEASE_SHA256,
    LLAMA_CPP_RELEASE_URL,
    download_llama_cpp,
)


def _make_zip_with_server(payload: bytes = b"FAKE_EXE") -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("llama-server.exe", payload)
        zf.writestr("ggml.dll", b"FAKE_DLL")
    return buf.getvalue()


def test_download_llama_cpp_writes_server_bin(tmp_path: Path, monkeypatch):
    blob = _make_zip_with_server()
    captured_url = []

    def fake_urlopen(url, timeout=None):
        captured_url.append(url)
        from io import BytesIO
        return BytesIO(blob)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "llama_agents.install.LLAMA_CPP_RELEASE_SHA256",
        hashlib.sha256(blob).hexdigest(),
    )
    result = download_llama_cpp(tmp_path)
    assert result == tmp_path / "llama-server.exe"
    assert result.is_file()
    assert result.read_bytes() == b"FAKE_EXE"
    assert (tmp_path / "ggml.dll").is_file()
    assert captured_url == [LLAMA_CPP_RELEASE_URL]


def test_download_llama_cpp_raises_on_sha_mismatch(tmp_path: Path, monkeypatch):
    blob = _make_zip_with_server()
    from io import BytesIO
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda url, timeout=None: BytesIO(blob)
    )
    monkeypatch.setattr(
        "llama_agents.install.LLAMA_CPP_RELEASE_SHA256",
        "0" * 64,  # deliberately wrong
    )
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        download_llama_cpp(tmp_path)


import sys
import types

from llama_agents.install import download_model


def test_download_model_calls_hf_hub_download(tmp_path: Path, monkeypatch):
    captured = {}

    def fake_hf_download(*, repo_id, filename, local_dir, **kwargs):
        captured["repo_id"] = repo_id
        captured["filename"] = filename
        captured["local_dir"] = local_dir
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\0")
        return str(out)

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.hf_hub_download = fake_hf_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    spec = CATALOGUE[0]
    result = download_model(spec, tmp_path / "GGUF")
    assert result == tmp_path / "GGUF" / spec.hf_filename
    assert captured["repo_id"] == spec.hf_repo
    assert captured["filename"] == spec.hf_filename


def test_download_model_raises_clear_error_when_module_missing(tmp_path: Path, monkeypatch):
    # Force the import to fail.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(RuntimeError, match="huggingface_hub is not installed"):
        download_model(CATALOGUE[0], tmp_path / "GGUF")
