from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class LlamaConfig(BaseModel):
    server_url: str = "http://127.0.0.1:8080"
    model: str = "qwen3-coder-30b"
    auto_spawn: bool = True
    kill_on_exit: bool = False
    server_bin: Path | None = None
    model_path: Path | None = None
    ngl: int = 999
    ctx_size: int = 65536
    n_parallel: int = 2
    startup_timeout_seconds: int = 60


class AgentConfig(BaseModel):
    max_iterations: int = Field(default=20, ge=1)
    max_concurrent_agents: int = Field(default=5, ge=1)
    token_budget_pct: float = Field(default=0.8, gt=0.0, le=1.0)


class SandboxConfig(BaseModel):
    allowed_dirs: list[Path] = Field(default_factory=list)
    shell_allowlist: list[str] = Field(default_factory=list)

    @field_validator("allowed_dirs")
    @classmethod
    def _normalize(cls, v: list[Path]) -> list[Path]:
        return [Path(p).resolve() for p in v]


class HttpConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 9000


class MemoryConfig(BaseModel):
    enabled: bool = True
    root: Path = Field(default=Path(".llama_agents/memory"))
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = Field(default=1500, ge=200)
    chunk_overlap: int = Field(default=150, ge=0)
    plan_recall_k: int = Field(default=3, ge=0)
    plan_recall_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    subagent_inline_threshold_chars: int = Field(default=2000, ge=0)
    subagent_summary_max_tokens: int = Field(default=400, ge=64)
    evict_threshold_pct: int = Field(default=70, ge=10, le=99)
    evict_tool_result_min_chars: int = Field(default=4000, ge=200)
    scratch_retention_hours: int = Field(default=24, ge=-1)


class QueueConfig(BaseModel):
    enabled: bool = False
    root: Path = Field(default=Path(".llama_agents/queue"))
    poll_interval_seconds: float = Field(default=2.0, ge=0.1)
    max_concurrent: int = Field(default=1, ge=1)
    max_retries: int = Field(default=2, ge=0)
    retry_backoff_seconds: float = Field(default=5.0, ge=0.0)
    max_iterations: int = Field(default=20, ge=1)
    drain_timeout_seconds: float = Field(default=30.0, ge=0.0)
    accepted_extensions: list[str] = Field(
        default_factory=lambda: [".md", ".txt"]
    )


class McpServerConfig(BaseModel):
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class Config(BaseModel):
    llama: LlamaConfig = Field(default_factory=LlamaConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    http: HttpConfig = Field(default_factory=HttpConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)


def load_config(path: str | Path) -> Config:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return Config.model_validate(data)
