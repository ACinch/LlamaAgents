from pathlib import Path

from typer.testing import CliRunner

from llama_agents.cli import app


def test_init_command_is_registered():
    runner = CliRunner()
    result = runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0
    assert "init" in result.stdout.lower() or "wizard" in result.stdout.lower()


def test_init_command_writes_config_with_scripted_answers(tmp_path: Path, monkeypatch):
    """End-to-end through the CLI: monkeypatch the prompter + helpers to a
    fully-scripted state, run the command, assert config.toml is created."""
    import llama_agents.install as install_mod
    from llama_agents.install import CATALOGUE, RecordedPrompter, WizardResult

    # Stage a llama-server sibling and a model file under tmp_path so the
    # wizard's auto-detection succeeds without network.
    server = tmp_path / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe"
    server.parent.mkdir(parents=True)
    server.write_bytes(b"\0")
    repo = tmp_path / "llama-agents"
    repo.mkdir()
    spec = CATALOGUE[0]
    model_file = repo / "GGUF" / spec.hf_filename
    model_file.parent.mkdir()
    model_file.write_bytes(b"\0")

    monkeypatch.chdir(repo)
    monkeypatch.setattr(install_mod, "detect_vram_gb", lambda: 24.0)
    monkeypatch.setattr(
        install_mod, "model_search_dirs",
        lambda r: [r / "GGUF", r.parent / "GGUF", tmp_path / "fake_home" / "GGUF"],
    )
    scripted = RecordedPrompter(answers=["y", "", "y", "", "y"])
    monkeypatch.setattr(install_mod, "RichPrompter", lambda: scripted)

    runner = CliRunner()
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.stdout
    assert (repo / "config.toml").is_file()
