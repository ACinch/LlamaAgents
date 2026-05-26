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
    ctx_size: int = 32768
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
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)


def load_config(path: str | Path) -> Config:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return Config.model_validate(data)
