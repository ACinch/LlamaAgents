import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from llama_agents.config import Config, load_config


SAMPLE = """
[llama]
server_url = "http://127.0.0.1:8080"
model = "qwen3-coder-30b"
auto_spawn = true
kill_on_exit = false
server_bin = "D:/repos/LLM/llamacpp-bin/llama-server.exe"
model_path = "D:/repos/LLM/GGUF/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"
ngl = 999
ctx_size = 32768
startup_timeout_seconds = 60

[agent]
max_iterations = 20
max_concurrent_agents = 5
token_budget_pct = 0.8

[sandbox]
allowed_dirs = ["D:/repos/LLM/llama-agents"]
shell_allowlist = ["git", "pytest"]

[http]
host = "127.0.0.1"
port = 9000

[[mcp_servers]]
name = "rag"
command = "node"
args = ["D:/repos/LLM/rag/dist/index.js"]
"""


def test_load_config_parses_full_example(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text(SAMPLE)
    cfg = load_config(p)
    assert isinstance(cfg, Config)
    assert cfg.llama.server_url == "http://127.0.0.1:8080"
    assert cfg.agent.max_concurrent_agents == 5
    assert cfg.sandbox.shell_allowlist == ["git", "pytest"]
    assert len(cfg.mcp_servers) == 1
    assert cfg.mcp_servers[0].name == "rag"


def test_load_config_rejects_bad_token_budget(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text(SAMPLE.replace("token_budget_pct = 0.8", "token_budget_pct = 1.5"))
    with pytest.raises(ValidationError):
        load_config(p)


def test_load_config_normalizes_allowed_dirs(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text(SAMPLE)
    cfg = load_config(p)
    # Paths are absolute and use forward slashes internally on all platforms.
    assert all(d.is_absolute() for d in cfg.sandbox.allowed_dirs)


def test_memory_config_defaults():
    from llama_agents.config import Config

    cfg = Config.model_validate({})
    assert cfg.memory.enabled is True
    assert cfg.memory.root == Path(".llama_agents/memory")
    assert cfg.memory.embedding_model == "BAAI/bge-small-en-v1.5"
    assert cfg.memory.chunk_size == 1500
    assert cfg.memory.chunk_overlap == 150
    assert cfg.memory.plan_recall_k == 3
    assert cfg.memory.plan_recall_threshold == 0.5
    assert cfg.memory.subagent_inline_threshold_chars == 2000
    assert cfg.memory.subagent_summary_max_tokens == 400
    assert cfg.memory.evict_threshold_pct == 70
    assert cfg.memory.evict_tool_result_min_chars == 4000
    assert cfg.memory.scratch_retention_hours == 24


def test_memory_config_disabled_toml(tmp_path):
    from llama_agents.config import load_config

    p = tmp_path / "c.toml"
    p.write_text("[memory]\nenabled = false\n")
    cfg = load_config(p)
    assert cfg.memory.enabled is False


def test_queue_config_defaults():
    from llama_agents.config import Config

    cfg = Config.model_validate({})
    assert cfg.queue.enabled is False
    assert str(cfg.queue.root) in (
        ".llama_agents/queue", ".llama_agents\\queue"
    )
    assert cfg.queue.poll_interval_seconds == 2.0
    assert cfg.queue.max_concurrent == 1
    assert cfg.queue.max_retries == 2
    assert cfg.queue.retry_backoff_seconds == 5.0
    assert cfg.queue.max_iterations == 20
    assert cfg.queue.drain_timeout_seconds == 30.0
    assert cfg.queue.accepted_extensions == [".md", ".txt"]


def test_queue_config_from_toml(tmp_path):
    from llama_agents.config import load_config

    p = tmp_path / "c.toml"
    p.write_text(
        "[queue]\n"
        "enabled = true\n"
        "max_concurrent = 3\n"
        "accepted_extensions = [\".md\"]\n"
    )
    cfg = load_config(p)
    assert cfg.queue.enabled is True
    assert cfg.queue.max_concurrent == 3
    assert cfg.queue.accepted_extensions == [".md"]
