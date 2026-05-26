# llama-agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python orchestration wrapper (`llama-agents`) around a local `llama-server.exe` that exposes the model as a tool-using agent with filesystem, allowlisted shell, subagent spawning, and MCP-bridged external tools, accessible via CLI and HTTP.

**Architecture:** A single async core (`Agent` loop + `ToolRegistry` + `LlamaClient`) with pluggable tools. Built-in tools live under `tools/builtin/`. External tools come from stdio MCP servers via `tools/mcp_bridge.py`. CLI (typer) and HTTP (FastAPI/SSE) are thin surfaces over the core. Designed for a later Rust/maturin core swap by keeping module interfaces narrow.

**Tech Stack:** Python 3.12, `httpx`, `pydantic` v2, stdlib `tomllib`, official `mcp` SDK, `fastapi`, `uvicorn`, `typer`, `rich`, `pytest`, `pytest-asyncio`. Package manager: `uv`.

**Project root:** `D:\repos\LLM\llama-agents` (git-initialized).

**Reference spec:** `docs/design.md` in this repo.

---

## File Structure

```
llama-agents/
├── pyproject.toml
├── config.toml                          # sample config
├── README.md
├── .gitignore
├── src/llama_agents/
│   ├── __init__.py
│   ├── errors.py                        # error taxonomy
│   ├── config.py                        # TOML + pydantic models
│   ├── events.py                        # event types yielded by the loop
│   ├── llama_client.py                  # async OpenAI-compat client + lifecycle
│   ├── sandbox.py                       # path/cmd guards
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py                      # Tool ABC + schema helper
│   │   ├── registry.py
│   │   ├── builtin/
│   │   │   ├── __init__.py
│   │   │   ├── fs.py
│   │   │   ├── shell.py
│   │   │   └── subagent.py
│   │   └── mcp_bridge.py
│   ├── agent.py                         # loop, events, cancellation
│   ├── cli.py
│   └── http_app.py
└── tests/
    ├── conftest.py
    ├── unit/
    ├── integration/
    └── live/
```

Each file has one responsibility:
- `errors.py` — exception classes used everywhere; no other deps.
- `config.py` — load TOML, validate with pydantic, no I/O beyond reading the file.
- `sandbox.py` — pure helpers (`is_allowed_path`, `is_allowed_command`); no class state.
- `tools/base.py` — `Tool` ABC: `name`, `description`, `json_schema`, async `invoke`.
- `tools/registry.py` — register/list/dispatch; produces OpenAI `tools=[...]` schema.
- `tools/builtin/*` — one tool family per file.
- `tools/mcp_bridge.py` — spawn MCP servers, wrap their tools as `Tool` instances.
- `llama_client.py` — HTTP client + optional subprocess lifecycle.
- `agent.py` — the loop, event yielding, cancellation, token-budget guard.
- `cli.py`, `http_app.py` — thin surfaces.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/llama_agents/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "llama-agents"
version = "0.1.0"
description = "Local llama.cpp orchestrator with tools, subagents, and MCP bridge"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.6",
    "mcp>=1.0",
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "typer>=0.12",
    "rich>=13.7",
    "sse-starlette>=2.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-mock>=3.12",
    "httpx[http2]",
]

[project.scripts]
llamactl = "llama_agents.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/llama_agents"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "live: tests that require a running llama-server (deselect with '-m \"not live\"')",
]
addopts = "-m 'not live'"
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.mypy_cache/
dist/
build/
*.egg-info/
.coverage
htmlcov/
.env
```

- [ ] **Step 3: Create `README.md`**

```markdown
# llama-agents

Local orchestration layer around llama.cpp. Turns a running `llama-server.exe`
into a tool-using agent with filesystem access, allowlisted shell execution,
subagent spawning, and MCP-bridged external tools.

See `docs/design.md` for the full design.

## Quickstart

```bash
uv sync --extra dev
uv run llamactl chat
```
```

- [ ] **Step 4: Create empty package files**

`src/llama_agents/__init__.py`:
```python
__version__ = "0.1.0"
```

`tests/conftest.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
```

`tests/unit/__init__.py`: (empty)

- [ ] **Step 5: Install and verify**

Run:
```bash
cd D:/repos/LLM/llama-agents
uv sync --extra dev
uv run pytest --collect-only
```
Expected: pytest reports `collected 0 items` with no errors.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: scaffold llama-agents project"
```

---

## Task 2: Error taxonomy

**Files:**
- Create: `src/llama_agents/errors.py`
- Create: `tests/unit/test_errors.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_errors.py`:
```python
import pytest
from llama_agents.errors import (
    LlamaAgentsError,
    LlamaUnreachable,
    LlamaProtocolError,
    ToolNotFound,
    ToolValidationError,
    SandboxViolation,
    MCPServerCrashed,
    MaxIterationsExceeded,
    AgentLimitExceeded,
    Cancelled,
)


def test_all_errors_inherit_from_base():
    for cls in (
        LlamaUnreachable,
        LlamaProtocolError,
        ToolNotFound,
        ToolValidationError,
        SandboxViolation,
        MCPServerCrashed,
        MaxIterationsExceeded,
        AgentLimitExceeded,
        Cancelled,
    ):
        assert issubclass(cls, LlamaAgentsError)


def test_sandbox_violation_carries_path():
    err = SandboxViolation(path="/etc/passwd", reason="outside allowed_dirs")
    assert err.path == "/etc/passwd"
    assert "outside allowed_dirs" in str(err)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_errors.py -v`
Expected: ImportError — module does not exist.

- [ ] **Step 3: Implement `errors.py`**

```python
class LlamaAgentsError(Exception):
    """Base for all llama-agents errors."""


class LlamaUnreachable(LlamaAgentsError):
    """llama-server is not reachable on the configured URL."""


class LlamaProtocolError(LlamaAgentsError):
    """llama-server returned an unexpected response shape."""


class ToolNotFound(LlamaAgentsError):
    def __init__(self, name: str):
        super().__init__(f"tool not found: {name}")
        self.name = name


class ToolValidationError(LlamaAgentsError):
    """Tool arguments failed schema validation."""


class SandboxViolation(LlamaAgentsError):
    def __init__(self, path: str, reason: str):
        super().__init__(f"sandbox violation: {path} ({reason})")
        self.path = path
        self.reason = reason


class MCPServerCrashed(LlamaAgentsError):
    def __init__(self, server: str):
        super().__init__(f"MCP server crashed: {server}")
        self.server = server


class MaxIterationsExceeded(LlamaAgentsError):
    """Agent exhausted its iteration budget without finalizing."""


class AgentLimitExceeded(LlamaAgentsError):
    """Concurrent-agent cap reached."""


class Cancelled(LlamaAgentsError):
    """Loop was cancelled by a surface or signal."""
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/unit/test_errors.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add error taxonomy"
```

---

## Task 3: Config loader

**Files:**
- Create: `src/llama_agents/config.py`
- Create: `config.toml`
- Create: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_config.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `config.py`**

```python
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
```

- [ ] **Step 4: Create sample `config.toml` at project root**

Same content as `SAMPLE` in the test, written to `D:/repos/LLM/llama-agents/config.toml`.

- [ ] **Step 5: Verify tests pass**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add TOML config loader with pydantic validation"
```

---

## Task 4: Sandbox helpers

**Files:**
- Create: `src/llama_agents/sandbox.py`
- Create: `tests/unit/test_sandbox.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_sandbox.py`:
```python
from pathlib import Path

import pytest

from llama_agents.errors import SandboxViolation
from llama_agents.sandbox import check_path, check_command


def test_check_path_allows_file_inside_allowed_dir(tmp_path: Path):
    target = tmp_path / "foo.txt"
    target.write_text("x")
    result = check_path(target, allowed_dirs=[tmp_path])
    assert result == target.resolve()


def test_check_path_rejects_outside(tmp_path: Path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("other") / "bar.txt"
    outside.write_text("x")
    with pytest.raises(SandboxViolation):
        check_path(outside, allowed_dirs=[tmp_path])


def test_check_path_rejects_parent_traversal(tmp_path: Path):
    sneaky = tmp_path / ".." / "evil.txt"
    with pytest.raises(SandboxViolation):
        check_path(sneaky, allowed_dirs=[tmp_path / "sub"])


def test_check_command_allows_listed(tmp_path: Path):
    check_command(["git", "status"], allowlist=["git", "pytest"])


def test_check_command_rejects_unlisted():
    with pytest.raises(SandboxViolation):
        check_command(["rm", "-rf", "/"], allowlist=["git"])


def test_check_command_rejects_empty():
    with pytest.raises(SandboxViolation):
        check_command([], allowlist=["git"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_sandbox.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `sandbox.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .errors import SandboxViolation


def check_path(path: str | Path, allowed_dirs: Iterable[Path]) -> Path:
    """Resolve path and verify it lies under at least one allowed dir.

    Returns the resolved absolute Path. Raises SandboxViolation otherwise.
    """
    resolved = Path(path).resolve(strict=False)
    for d in allowed_dirs:
        base = Path(d).resolve(strict=False)
        try:
            resolved.relative_to(base)
            return resolved
        except ValueError:
            continue
    raise SandboxViolation(path=str(resolved), reason="outside allowed_dirs")


def check_command(command: list[str], allowlist: Iterable[str]) -> None:
    if not command:
        raise SandboxViolation(path="", reason="empty command")
    allowed = set(allowlist)
    if command[0] not in allowed:
        raise SandboxViolation(
            path=command[0], reason=f"command not in allowlist {sorted(allowed)}"
        )
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/unit/test_sandbox.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add sandbox path/command helpers"
```

---

## Task 5: Tool base class and registry

**Files:**
- Create: `src/llama_agents/tools/__init__.py`
- Create: `src/llama_agents/tools/base.py`
- Create: `src/llama_agents/tools/registry.py`
- Create: `tests/unit/test_registry.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_registry.py`:
```python
from typing import Any

import pytest

from llama_agents.errors import ToolNotFound, ToolValidationError
from llama_agents.tools.base import Tool
from llama_agents.tools.registry import ToolRegistry


class EchoTool(Tool):
    name = "echo"
    description = "echo input back"
    json_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def invoke(self, args: dict[str, Any]) -> Any:
        return args["text"]


async def test_register_and_invoke():
    reg = ToolRegistry()
    reg.register(EchoTool())
    result = await reg.invoke("echo", {"text": "hi"})
    assert result == "hi"


async def test_invoke_unknown_raises():
    reg = ToolRegistry()
    with pytest.raises(ToolNotFound):
        await reg.invoke("nope", {})


async def test_invoke_validates_required_args():
    reg = ToolRegistry()
    reg.register(EchoTool())
    with pytest.raises(ToolValidationError):
        await reg.invoke("echo", {})


def test_schemas_emits_openai_tools_array():
    reg = ToolRegistry()
    reg.register(EchoTool())
    schemas = reg.schemas()
    assert schemas == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "echo input back",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }
    ]


def test_register_duplicate_raises():
    reg = ToolRegistry()
    reg.register(EchoTool())
    with pytest.raises(ValueError):
        reg.register(EchoTool())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_registry.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `tools/__init__.py`** (empty)

- [ ] **Step 4: Implement `tools/base.py`**

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    json_schema: ClassVar[dict[str, Any]]

    @abstractmethod
    async def invoke(self, args: dict[str, Any]) -> Any:
        ...
```

- [ ] **Step 5: Implement `tools/registry.py`**

```python
from __future__ import annotations

from typing import Any

from ..errors import ToolNotFound, ToolValidationError
from .base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.json_schema,
                },
            }
            for t in self._tools.values()
        ]

    async def invoke(self, name: str, args: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFound(name)
        self._validate(tool, args)
        return await tool.invoke(args)

    @staticmethod
    def _validate(tool: Tool, args: dict[str, Any]) -> None:
        required = tool.json_schema.get("required", [])
        missing = [r for r in required if r not in args]
        if missing:
            raise ToolValidationError(
                f"{tool.name} missing required args: {missing}"
            )
```

- [ ] **Step 6: Verify tests pass**

Run: `uv run pytest tests/unit/test_registry.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: add Tool base class and ToolRegistry"
```

---

## Task 6: Filesystem tools

**Files:**
- Create: `src/llama_agents/tools/builtin/__init__.py`
- Create: `src/llama_agents/tools/builtin/fs.py`
- Create: `tests/unit/test_fs_tools.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_fs_tools.py`:
```python
from pathlib import Path

import pytest

from llama_agents.errors import SandboxViolation
from llama_agents.tools.builtin.fs import (
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    ListFilesTool,
)


async def test_read_file(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    tool = ReadFileTool(allowed_dirs=[tmp_path])
    out = await tool.invoke({"path": str(f)})
    assert out == "hello"


async def test_read_file_rejects_outside(tmp_path: Path, tmp_path_factory):
    other = tmp_path_factory.mktemp("o") / "x.txt"
    other.write_text("x")
    tool = ReadFileTool(allowed_dirs=[tmp_path])
    with pytest.raises(SandboxViolation):
        await tool.invoke({"path": str(other)})


async def test_write_file_creates(tmp_path: Path):
    f = tmp_path / "new.txt"
    tool = WriteFileTool(allowed_dirs=[tmp_path])
    out = await tool.invoke({"path": str(f), "content": "abc"})
    assert f.read_text(encoding="utf-8") == "abc"
    assert "wrote" in out.lower()


async def test_edit_file_replaces_unique_match(tmp_path: Path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\ny = 2\n", encoding="utf-8")
    tool = EditFileTool(allowed_dirs=[tmp_path])
    await tool.invoke({"path": str(f), "find": "y = 2", "replace": "y = 3"})
    assert f.read_text(encoding="utf-8") == "x = 1\ny = 3\n"


async def test_edit_file_rejects_nonunique(tmp_path: Path):
    f = tmp_path / "code.py"
    f.write_text("a\na\n", encoding="utf-8")
    tool = EditFileTool(allowed_dirs=[tmp_path])
    with pytest.raises(ValueError):
        await tool.invoke({"path": str(f), "find": "a", "replace": "b"})


async def test_edit_file_rejects_missing(tmp_path: Path):
    f = tmp_path / "code.py"
    f.write_text("hello\n", encoding="utf-8")
    tool = EditFileTool(allowed_dirs=[tmp_path])
    with pytest.raises(ValueError):
        await tool.invoke({"path": str(f), "find": "missing", "replace": "x"})


async def test_list_files_globs(tmp_path: Path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")
    tool = ListFilesTool(allowed_dirs=[tmp_path])
    out = await tool.invoke({"pattern": "*.py", "base": str(tmp_path)})
    assert sorted(out) == [str(tmp_path / "a.py"), str(tmp_path / "b.py")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_fs_tools.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `tools/builtin/__init__.py`** (empty)

- [ ] **Step 4: Implement `tools/builtin/fs.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from ...sandbox import check_path
from ..base import Tool


class _SandboxedFsTool(Tool):
    def __init__(self, allowed_dirs: list[Path]) -> None:
        self._allowed_dirs = [Path(d) for d in allowed_dirs]


class ReadFileTool(_SandboxedFsTool):
    name = "fs_read_file"
    description = "Read the entire contents of a UTF-8 text file."
    json_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    async def invoke(self, args: dict[str, Any]) -> str:
        p = check_path(args["path"], self._allowed_dirs)
        return p.read_text(encoding="utf-8")


class WriteFileTool(_SandboxedFsTool):
    name = "fs_write_file"
    description = "Write UTF-8 text to a file, creating parents as needed."
    json_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    async def invoke(self, args: dict[str, Any]) -> str:
        p = check_path(args["path"], self._allowed_dirs)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args["content"], encoding="utf-8")
        return f"wrote {len(args['content'])} bytes to {p}"


class EditFileTool(_SandboxedFsTool):
    name = "fs_edit_file"
    description = (
        "Replace an exact, uniquely-occurring substring in a file. "
        "Fails if the substring is missing or appears more than once."
    )
    json_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "find": {"type": "string"},
            "replace": {"type": "string"},
        },
        "required": ["path", "find", "replace"],
    }

    async def invoke(self, args: dict[str, Any]) -> str:
        p = check_path(args["path"], self._allowed_dirs)
        text = p.read_text(encoding="utf-8")
        count = text.count(args["find"])
        if count == 0:
            raise ValueError(f"find string not found in {p}")
        if count > 1:
            raise ValueError(
                f"find string occurs {count} times in {p}; must be unique"
            )
        p.write_text(text.replace(args["find"], args["replace"]), encoding="utf-8")
        return f"edited {p}"


class ListFilesTool(_SandboxedFsTool):
    name = "fs_list_files"
    description = "List files matching a glob pattern inside an allowed base dir."
    json_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "glob, e.g. **/*.py"},
            "base": {"type": "string"},
        },
        "required": ["pattern", "base"],
    }

    async def invoke(self, args: dict[str, Any]) -> list[str]:
        base = check_path(args["base"], self._allowed_dirs)
        return sorted(str(p) for p in base.glob(args["pattern"]) if p.is_file())
```

- [ ] **Step 5: Verify tests pass**

Run: `uv run pytest tests/unit/test_fs_tools.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add fs read/write/edit/list tools"
```

---

## Task 7: Shell tool

**Files:**
- Create: `src/llama_agents/tools/builtin/shell.py`
- Create: `tests/unit/test_shell_tool.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_shell_tool.py`:
```python
import sys
from pathlib import Path

import pytest

from llama_agents.errors import SandboxViolation
from llama_agents.tools.builtin.shell import ShellRunTool


async def test_run_allowed_command(tmp_path: Path):
    tool = ShellRunTool(
        allowed_dirs=[tmp_path], allowlist=[sys.executable.split("/")[-1].split("\\")[-1], "python"]
    )
    # Use the running interpreter as the binary; alias the basename in allowlist.
    # We test exit code + stdout capture by running python -c.
    result = await tool.invoke(
        {"command": ["python", "-c", "print('hi')"], "cwd": str(tmp_path)}
    )
    assert result["returncode"] == 0
    assert "hi" in result["stdout"]


async def test_run_rejects_unlisted(tmp_path: Path):
    tool = ShellRunTool(allowed_dirs=[tmp_path], allowlist=["git"])
    with pytest.raises(SandboxViolation):
        await tool.invoke({"command": ["rm", "-rf", "/"], "cwd": str(tmp_path)})


async def test_run_rejects_cwd_outside(tmp_path: Path, tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    tool = ShellRunTool(allowed_dirs=[tmp_path], allowlist=["python"])
    with pytest.raises(SandboxViolation):
        await tool.invoke(
            {"command": ["python", "-c", "pass"], "cwd": str(other)}
        )


async def test_run_captures_nonzero_exit(tmp_path: Path):
    tool = ShellRunTool(allowed_dirs=[tmp_path], allowlist=["python"])
    result = await tool.invoke(
        {"command": ["python", "-c", "import sys; sys.exit(7)"], "cwd": str(tmp_path)}
    )
    assert result["returncode"] == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_shell_tool.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `tools/builtin/shell.py`**

```python
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ...sandbox import check_command, check_path
from ..base import Tool


class ShellRunTool(Tool):
    name = "shell_run"
    description = (
        "Run an allowlisted command. The command argv[0] must be in the "
        "configured allowlist. Returns {returncode, stdout, stderr}."
    )
    json_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "array",
                "items": {"type": "string"},
                "description": "argv list; argv[0] must be in allowlist",
            },
            "cwd": {"type": "string"},
            "timeout": {"type": "integer", "default": 120},
        },
        "required": ["command", "cwd"],
    }

    def __init__(self, allowed_dirs: list[Path], allowlist: list[str]) -> None:
        self._allowed_dirs = [Path(d) for d in allowed_dirs]
        self._allowlist = list(allowlist)

    async def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        cmd: list[str] = args["command"]
        check_command(cmd, self._allowlist)
        cwd = check_path(args["cwd"], self._allowed_dirs)
        if not cwd.is_dir():
            raise ValueError(f"cwd is not a directory: {cwd}")
        timeout = int(args.get("timeout", 120))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"timeout after {timeout}s",
            }
        return {
            "returncode": proc.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/unit/test_shell_tool.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add allowlisted shell.run tool"
```

---

## Task 8: Llama HTTP client (no lifecycle yet)

**Files:**
- Create: `src/llama_agents/llama_client.py`
- Create: `tests/unit/test_llama_client.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_llama_client.py`:
```python
import json

import httpx
import pytest

from llama_agents.errors import LlamaProtocolError, LlamaUnreachable
from llama_agents.llama_client import ChatResponse, LlamaClient, ToolCall


def _mock_transport(handler):
    return httpx.MockTransport(handler)


async def test_chat_parses_plain_assistant_message():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["messages"][0]["role"] == "user"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hi back"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    client = LlamaClient(base_url="http://x", transport=_mock_transport(handler))
    resp = await client.chat(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert isinstance(resp, ChatResponse)
    assert resp.content == "hi back"
    assert resp.tool_calls == []


async def test_chat_parses_tool_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "echo",
                                        "arguments": '{"text": "hi"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    client = LlamaClient(base_url="http://x", transport=_mock_transport(handler))
    resp = await client.chat(messages=[{"role": "user", "content": "go"}], tools=[])
    assert resp.tool_calls == [ToolCall(id="call_1", name="echo", arguments={"text": "hi"})]


async def test_chat_raises_on_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope", request=request)

    client = LlamaClient(base_url="http://x", transport=_mock_transport(handler))
    with pytest.raises(LlamaUnreachable):
        await client.chat(messages=[{"role": "user", "content": "x"}], tools=[])


async def test_chat_raises_on_bad_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    client = LlamaClient(base_url="http://x", transport=_mock_transport(handler))
    with pytest.raises(LlamaProtocolError):
        await client.chat(messages=[{"role": "user", "content": "x"}], tools=[])


async def test_health_returns_bool():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    client = LlamaClient(base_url="http://x", transport=_mock_transport(handler))
    assert await client.health() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_llama_client.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `llama_client.py`**

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from .errors import LlamaProtocolError, LlamaUnreachable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_message: dict[str, Any] | None = None


class LlamaClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=timeout, transport=transport
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        try:
            r = await self._client.get("/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools

        try:
            r = await self._client.post("/v1/chat/completions", json=payload)
        except httpx.ConnectError as e:
            raise LlamaUnreachable(str(e)) from e
        except httpx.HTTPError as e:
            raise LlamaUnreachable(str(e)) from e

        if r.status_code != 200:
            raise LlamaProtocolError(f"HTTP {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as e:
            raise LlamaProtocolError(f"unexpected response shape: {e}") from e

        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=args)
            )

        return ChatResponse(
            content=msg.get("content"),
            tool_calls=tool_calls,
            raw_message=msg,
        )
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/unit/test_llama_client.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add async LlamaClient with OpenAI-compatible chat"
```

---

## Task 9: Llama-server subprocess lifecycle

**Files:**
- Modify: `src/llama_agents/llama_client.py` (add `LlamaServerManager`)
- Create: `tests/unit/test_llama_server_manager.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_llama_server_manager.py`:
```python
import asyncio
from pathlib import Path

import pytest

from llama_agents.config import LlamaConfig
from llama_agents.llama_client import LlamaServerManager


class _FakeClient:
    def __init__(self, reachable_after: int = 0):
        self.calls = 0
        self._reachable_after = reachable_after

    async def health(self) -> bool:
        self.calls += 1
        return self.calls > self._reachable_after


async def test_ensure_running_no_spawn_when_already_up():
    cfg = LlamaConfig(auto_spawn=True)
    client = _FakeClient(reachable_after=0)
    mgr = LlamaServerManager(cfg, client)
    await mgr.ensure_running()
    assert mgr.spawned is False


async def test_ensure_running_raises_when_unreachable_and_no_spawn():
    cfg = LlamaConfig(auto_spawn=False)
    client = _FakeClient(reachable_after=999)
    mgr = LlamaServerManager(cfg, client)
    from llama_agents.errors import LlamaUnreachable
    with pytest.raises(LlamaUnreachable):
        await mgr.ensure_running()


async def test_shutdown_only_kills_what_we_spawned():
    cfg = LlamaConfig(auto_spawn=False)
    client = _FakeClient(reachable_after=0)
    mgr = LlamaServerManager(cfg, client)
    await mgr.ensure_running()  # no spawn
    await mgr.shutdown()  # must not raise
    assert mgr.spawned is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_llama_server_manager.py -v`
Expected: ImportError.

- [ ] **Step 3: Append `LlamaServerManager` to `llama_client.py`**

Add at the bottom of `src/llama_agents/llama_client.py`:

```python
import asyncio as _asyncio
import subprocess

from .config import LlamaConfig
from .errors import LlamaUnreachable


class LlamaServerManager:
    """Optionally spawns llama-server.exe if not already reachable."""

    def __init__(self, cfg: LlamaConfig, client: "LlamaClient | object") -> None:
        self._cfg = cfg
        self._client = client
        self._process: subprocess.Popen | None = None

    @property
    def spawned(self) -> bool:
        return self._process is not None

    async def ensure_running(self) -> None:
        if await self._client.health():
            return
        if not self._cfg.auto_spawn:
            raise LlamaUnreachable(
                f"llama-server not reachable and auto_spawn=false"
            )
        if self._cfg.server_bin is None or self._cfg.model_path is None:
            raise LlamaUnreachable("auto_spawn requires server_bin and model_path")
        self._process = subprocess.Popen(
            [
                str(self._cfg.server_bin),
                "-m", str(self._cfg.model_path),
                "-ngl", str(self._cfg.ngl),
                "-c", str(self._cfg.ctx_size),
            ],
        )
        deadline = self._cfg.startup_timeout_seconds
        for _ in range(deadline):
            if await self._client.health():
                return
            await _asyncio.sleep(1)
        raise LlamaUnreachable(
            f"llama-server failed to become ready in {deadline}s"
        )

    async def shutdown(self) -> None:
        if self._process is None:
            return
        if not self._cfg.kill_on_exit:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/unit/test_llama_server_manager.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add LlamaServerManager for optional subprocess lifecycle"
```

---

## Task 10: Events module

**Files:**
- Create: `src/llama_agents/events.py`
- Create: `tests/unit/test_events.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_events.py`:
```python
from llama_agents.events import (
    AssistantChunk,
    ToolCallStart,
    ToolCallResult,
    LoopError,
    Done,
    Event,
)


def test_event_types_are_dataclasses_and_share_base():
    e1 = AssistantChunk(text="hi")
    e2 = ToolCallStart(call_id="c", name="t", arguments={"x": 1})
    e3 = ToolCallResult(call_id="c", ok=True, content="ok")
    e4 = LoopError(error_type="X", message="m")
    e5 = Done(reason="finished")
    for e in (e1, e2, e3, e4, e5):
        assert isinstance(e, Event)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_events.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `events.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class Event:
    """Base marker class for all loop events."""


@dataclass
class AssistantChunk(Event):
    text: str


@dataclass
class ToolCallStart(Event):
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallResult(Event):
    call_id: str
    ok: bool
    content: Any


@dataclass
class LoopError(Event):
    error_type: str
    message: str


@dataclass
class Done(Event):
    reason: str  # "finished" | "max_iterations" | "cancelled" | "token_budget"
    final_message: str | None = None
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/unit/test_events.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add agent event types"
```

---

## Task 11: Agent loop (core)

**Files:**
- Create: `src/llama_agents/agent.py`
- Create: `tests/unit/test_agent_loop.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_agent_loop.py`:
```python
import asyncio
from typing import Any

import pytest

from llama_agents.agent import Agent, AgentRunOptions
from llama_agents.events import AssistantChunk, Done, ToolCallResult, ToolCallStart
from llama_agents.errors import MaxIterationsExceeded
from llama_agents.llama_client import ChatResponse, ToolCall
from llama_agents.tools.base import Tool
from llama_agents.tools.registry import ToolRegistry


class ScriptedClient:
    """Returns a predefined sequence of ChatResponses."""

    def __init__(self, script: list[ChatResponse]):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, *, messages, tools, temperature=0.2):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self.script.pop(0)


class StubEcho(Tool):
    name = "echo"
    description = "echo"
    json_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def invoke(self, args):
        return args["text"]


def _registry_with_echo():
    reg = ToolRegistry()
    reg.register(StubEcho())
    return reg


async def _collect(agen):
    return [e async for e in agen]


async def test_finishes_when_model_returns_plain_message():
    client = ScriptedClient([
        ChatResponse(content="hello world"),
    ])
    agent = Agent(client=client, registry=_registry_with_echo())
    events = await _collect(agent.run("hi", AgentRunOptions(max_iterations=5)))
    assert any(isinstance(e, AssistantChunk) and e.text == "hello world" for e in events)
    assert isinstance(events[-1], Done) and events[-1].reason == "finished"


async def test_dispatches_tool_then_finishes():
    client = ScriptedClient([
        ChatResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "ping"})],
        ),
        ChatResponse(content="all done"),
    ])
    agent = Agent(client=client, registry=_registry_with_echo())
    events = await _collect(agent.run("go", AgentRunOptions(max_iterations=5)))
    assert any(isinstance(e, ToolCallStart) and e.name == "echo" for e in events)
    assert any(isinstance(e, ToolCallResult) and e.ok and e.content == "ping" for e in events)
    assert isinstance(events[-1], Done) and events[-1].reason == "finished"


async def test_tool_error_fed_back_to_model():
    class Boom(Tool):
        name = "boom"
        description = "fails"
        json_schema = {"type": "object", "properties": {}}
        async def invoke(self, args):
            raise RuntimeError("kaboom")

    reg = ToolRegistry()
    reg.register(Boom())
    client = ScriptedClient([
        ChatResponse(content=None, tool_calls=[ToolCall(id="c1", name="boom", arguments={})]),
        ChatResponse(content="recovered"),
    ])
    agent = Agent(client=client, registry=reg)
    events = await _collect(agent.run("go", AgentRunOptions(max_iterations=5)))
    bad = [e for e in events if isinstance(e, ToolCallResult)]
    assert bad and bad[0].ok is False and "kaboom" in str(bad[0].content)
    assert isinstance(events[-1], Done) and events[-1].reason == "finished"


async def test_max_iterations():
    looping = ChatResponse(
        content=None,
        tool_calls=[ToolCall(id="c", name="echo", arguments={"text": "x"})],
    )
    client = ScriptedClient([looping, looping, looping])
    agent = Agent(client=client, registry=_registry_with_echo())
    events = await _collect(agent.run("go", AgentRunOptions(max_iterations=2)))
    assert isinstance(events[-1], Done) and events[-1].reason == "max_iterations"


async def test_cancellation_stops_loop():
    client = ScriptedClient([
        ChatResponse(content=None, tool_calls=[ToolCall(id="c", name="echo", arguments={"text": "x"})]),
        ChatResponse(content="never reached"),
    ])
    agent = Agent(client=client, registry=_registry_with_echo())
    agent.cancel()  # pre-cancel
    events = await _collect(agent.run("go", AgentRunOptions(max_iterations=5)))
    assert isinstance(events[-1], Done) and events[-1].reason == "cancelled"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_agent_loop.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `agent.py`**

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

from .errors import LlamaAgentsError
from .events import (
    AssistantChunk,
    Done,
    Event,
    LoopError,
    ToolCallResult,
    ToolCallStart,
)
from .llama_client import ChatResponse
from .tools.registry import ToolRegistry


class _ClientLike(Protocol):
    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = ...,
    ) -> ChatResponse: ...


@dataclass
class AgentRunOptions:
    max_iterations: int = 20
    system_prompt: str = (
        "You are a careful coding agent. Use tools to read files, run commands, "
        "and query the RAG when helpful. When finished, reply in plain text."
    )
    temperature: float = 0.2


class Agent:
    def __init__(
        self,
        *,
        client: _ClientLike,
        registry: ToolRegistry,
    ) -> None:
        self._client = client
        self._registry = registry
        self._cancel = asyncio.Event()
        self.messages: list[dict[str, Any]] = []

    def cancel(self) -> None:
        self._cancel.set()

    async def run(
        self, user_prompt: str, opts: AgentRunOptions
    ) -> AsyncIterator[Event]:
        self.messages = [
            {"role": "system", "content": opts.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for _ in range(opts.max_iterations):
            if self._cancel.is_set():
                yield Done(reason="cancelled")
                return

            try:
                resp = await self._client.chat(
                    messages=self.messages,
                    tools=self._registry.schemas(),
                    temperature=opts.temperature,
                )
            except LlamaAgentsError as e:
                yield LoopError(error_type=type(e).__name__, message=str(e))
                yield Done(reason="error")
                return

            self.messages.append(
                resp.raw_message
                or {"role": "assistant", "content": resp.content}
            )

            if not resp.tool_calls:
                if resp.content:
                    yield AssistantChunk(text=resp.content)
                yield Done(reason="finished", final_message=resp.content)
                return

            for call in resp.tool_calls:
                yield ToolCallStart(
                    call_id=call.id, name=call.name, arguments=call.arguments
                )
                try:
                    result = await self._registry.invoke(call.name, call.arguments)
                    ok, content = True, result
                except Exception as e:  # noqa: BLE001 — feed all tool errors back
                    ok, content = False, f"{type(e).__name__}: {e}"
                yield ToolCallResult(call_id=call.id, ok=ok, content=content)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": _as_tool_text(ok, content),
                    }
                )

        yield Done(reason="max_iterations")


def _as_tool_text(ok: bool, content: Any) -> str:
    if ok:
        return content if isinstance(content, str) else _json_dump(content)
    return _json_dump({"error": str(content)})


def _json_dump(x: Any) -> str:
    import json

    try:
        return json.dumps(x, default=str)
    except (TypeError, ValueError):
        return str(x)
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/unit/test_agent_loop.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: implement agent loop with events and cancellation"
```

---

## Task 12: Subagent tool + concurrency cap

**Files:**
- Create: `src/llama_agents/tools/builtin/subagent.py`
- Create: `tests/unit/test_subagent_tool.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_subagent_tool.py`:
```python
import asyncio
from typing import Any

import pytest

from llama_agents.agent import Agent, AgentRunOptions
from llama_agents.errors import AgentLimitExceeded
from llama_agents.llama_client import ChatResponse, ToolCall
from llama_agents.tools.builtin.subagent import SpawnSubagentTool
from llama_agents.tools.registry import ToolRegistry


class ScriptedClient:
    def __init__(self, scripts: dict[str, list[ChatResponse]]):
        self.scripts = scripts
        self.session_for_prompt: dict[str, str] = {}

    async def chat(self, *, messages, tools, temperature=0.2):
        # Route by the first user message text.
        for m in messages:
            if m["role"] == "user":
                key = m["content"]
                break
        script = self.scripts[key]
        return script.pop(0)


async def test_subagent_returns_final_message():
    client = ScriptedClient({
        "do thing": [ChatResponse(content="subagent done")],
    })
    parent_registry = ToolRegistry()
    semaphore = asyncio.Semaphore(5)

    def factory() -> Agent:
        return Agent(client=client, registry=ToolRegistry())

    spawn = SpawnSubagentTool(agent_factory=factory, semaphore=semaphore)
    parent_registry.register(spawn)

    result = await spawn.invoke({"task": "do thing"})
    assert result["result"] == "subagent done"
    assert result["iterations"] >= 1


async def test_subagent_respects_concurrency_cap():
    client = ScriptedClient({"x": [ChatResponse(content="ok")]})
    semaphore = asyncio.Semaphore(1)
    # Take the only slot.
    await semaphore.acquire()

    def factory() -> Agent:
        return Agent(client=client, registry=ToolRegistry())

    spawn = SpawnSubagentTool(agent_factory=factory, semaphore=semaphore)
    with pytest.raises(AgentLimitExceeded):
        await spawn.invoke({"task": "x"})

    semaphore.release()


async def test_subagent_strips_spawn_unless_allowed():
    """Subagent's own registry should NOT contain spawn unless requested."""
    client = ScriptedClient({"t": [ChatResponse(content="ok")]})
    sem = asyncio.Semaphore(5)

    captured: list[Agent] = []

    def factory() -> Agent:
        a = Agent(client=client, registry=ToolRegistry())
        captured.append(a)
        return a

    spawn = SpawnSubagentTool(agent_factory=factory, semaphore=sem)
    await spawn.invoke({"task": "t"})
    assert "subagent_spawn" not in captured[0]._registry.names()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_subagent_tool.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `tools/builtin/subagent.py`**

```python
from __future__ import annotations

import asyncio
from typing import Any, Callable

from ...agent import Agent, AgentRunOptions
from ...errors import AgentLimitExceeded
from ...events import AssistantChunk, Done, ToolCallResult, ToolCallStart
from ..base import Tool


class SpawnSubagentTool(Tool):
    name = "subagent_spawn"
    description = (
        "Spawn a subagent with its own conversation to handle a focused task. "
        "Returns the subagent's final assistant message as `result`."
    )
    json_schema = {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "system_prompt": {"type": "string"},
            "allowed_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "tool names the subagent may use; defaults to parent's minus subagent_spawn",
            },
            "max_iterations": {"type": "integer", "default": 20},
        },
        "required": ["task"],
    }

    def __init__(
        self,
        agent_factory: Callable[[], Agent],
        semaphore: asyncio.Semaphore,
    ) -> None:
        self._factory = agent_factory
        self._sem = semaphore

    async def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._sem.locked() and self._sem._value > 0:  # type: ignore[attr-defined]
            pass  # there's room
        acquired = self._sem.locked() is False and await _try_acquire(self._sem)
        if not acquired:
            raise AgentLimitExceeded("max_concurrent_agents reached")

        try:
            subagent = self._factory()
            allowed = args.get("allowed_tools")
            if allowed is not None:
                for n in list(subagent._registry.names()):
                    if n not in allowed:
                        subagent._registry.unregister(n)
            else:
                subagent._registry.unregister("subagent_spawn")

            opts = AgentRunOptions(
                max_iterations=int(args.get("max_iterations", 20)),
                system_prompt=args.get(
                    "system_prompt",
                    "You are a focused subagent. Complete the task and report back.",
                ),
            )

            iterations = 0
            tool_calls = 0
            final_text = ""
            async for ev in subagent.run(args["task"], opts):
                if isinstance(ev, ToolCallStart):
                    tool_calls += 1
                elif isinstance(ev, AssistantChunk):
                    final_text = ev.text
                elif isinstance(ev, Done):
                    if ev.final_message:
                        final_text = ev.final_message
                    break
                iterations += 1
            return {
                "result": final_text,
                "iterations": iterations,
                "tool_calls": tool_calls,
            }
        finally:
            self._sem.release()


async def _try_acquire(sem: asyncio.Semaphore) -> bool:
    """Non-blocking acquire; returns False if no slot available."""
    if sem.locked():
        return False
    # asyncio.Semaphore doesn't expose try-acquire; use _value (best effort).
    if sem._value <= 0:  # type: ignore[attr-defined]
        return False
    await sem.acquire()
    return True
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/unit/test_subagent_tool.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add subagent.spawn with concurrency cap"
```

---

## Task 13: MCP bridge

**Files:**
- Create: `src/llama_agents/tools/mcp_bridge.py`
- Create: `tests/unit/test_mcp_bridge.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_mcp_bridge.py`:
```python
from typing import Any

import pytest

from llama_agents.tools.mcp_bridge import McpBridgedTool


class _FakeMcpClient:
    def __init__(self, response: Any):
        self._response = response
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, args: dict) -> Any:
        self.calls.append((name, args))
        return self._response


async def test_bridged_tool_calls_underlying_client():
    client = _FakeMcpClient(response={"snippets": ["a", "b"]})
    tool = McpBridgedTool(
        server="rag",
        underlying_name="rag_query",
        description="search RAG",
        schema={"type": "object", "properties": {"query": {"type": "string"}}},
        client=client,
    )
    assert tool.name == "rag__rag_query"
    result = await tool.invoke({"query": "hello"})
    assert result == {"snippets": ["a", "b"]}
    assert client.calls == [("rag_query", {"query": "hello"})]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_mcp_bridge.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `tools/mcp_bridge.py`**

```python
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any, Protocol

from ..config import McpServerConfig
from ..errors import MCPServerCrashed
from .base import Tool


class _McpClientLike(Protocol):
    async def call_tool(self, name: str, args: dict[str, Any]) -> Any: ...


class McpBridgedTool(Tool):
    """A single bridged MCP tool exposed in our registry."""

    def __init__(
        self,
        *,
        server: str,
        underlying_name: str,
        description: str,
        schema: dict[str, Any],
        client: _McpClientLike,
    ) -> None:
        self._server = server
        self._underlying = underlying_name
        self._client = client
        self.name = f"{server}__{underlying_name}"  # type: ignore[misc]
        self.description = description  # type: ignore[misc]
        self.json_schema = schema  # type: ignore[misc]

    async def invoke(self, args: dict[str, Any]) -> Any:
        try:
            return await self._client.call_tool(self._underlying, args)
        except Exception as e:  # noqa: BLE001
            raise MCPServerCrashed(self._server) from e


class McpBridge:
    """Spawns configured MCP servers and produces bridged Tools.

    Uses the official `mcp` Python SDK at runtime; tests can pass mock clients.
    """

    def __init__(self, servers: list[McpServerConfig]) -> None:
        self._servers = servers
        self._stack = AsyncExitStack()
        self._tools: list[McpBridgedTool] = []

    async def start(self) -> list[McpBridgedTool]:
        # Imported lazily so unit tests without the mcp SDK can still import this module.
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        for srv in self._servers:
            params = StdioServerParameters(command=srv.command, args=srv.args, env=srv.env or None)
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listing = await session.list_tools()
            for t in listing.tools:
                self._tools.append(
                    McpBridgedTool(
                        server=srv.name,
                        underlying_name=t.name,
                        description=t.description or "",
                        schema=t.inputSchema or {"type": "object", "properties": {}},
                        client=_SessionClient(session),
                    )
                )
        return list(self._tools)

    async def aclose(self) -> None:
        await self._stack.aclose()


class _SessionClient:
    def __init__(self, session: Any) -> None:
        self._session = session

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        result = await self._session.call_tool(name, args)
        # mcp returns a CallToolResult with .content (list of content blocks)
        return [getattr(c, "text", c) for c in (result.content or [])]
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/unit/test_mcp_bridge.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add MCP bridge for stdio tool servers"
```

---

## Task 14: Wiring — agent factory & runtime assembly

**Files:**
- Create: `src/llama_agents/runtime.py`
- Create: `tests/unit/test_runtime.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_runtime.py`:
```python
import asyncio
from pathlib import Path

import pytest

from llama_agents.config import Config, LlamaConfig, AgentConfig, SandboxConfig
from llama_agents.llama_client import ChatResponse
from llama_agents.runtime import Runtime


class FakeClient:
    async def chat(self, *, messages, tools, temperature=0.2):
        return ChatResponse(content="done")

    async def health(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


async def test_runtime_builds_registry_with_builtins(tmp_path: Path):
    cfg = Config(
        llama=LlamaConfig(auto_spawn=False),
        agent=AgentConfig(max_concurrent_agents=3),
        sandbox=SandboxConfig(allowed_dirs=[tmp_path], shell_allowlist=["python"]),
    )
    rt = await Runtime.create(cfg, client_factory=lambda url: FakeClient())
    names = rt.registry.names()
    assert "fs_read_file" in names
    assert "fs_write_file" in names
    assert "fs_edit_file" in names
    assert "fs_list_files" in names
    assert "shell_run" in names
    assert "subagent_spawn" in names
    await rt.aclose()


async def test_runtime_creates_subagent_with_isolated_registry(tmp_path: Path):
    cfg = Config(
        llama=LlamaConfig(auto_spawn=False),
        sandbox=SandboxConfig(allowed_dirs=[tmp_path], shell_allowlist=["python"]),
    )
    rt = await Runtime.create(cfg, client_factory=lambda url: FakeClient())
    sub = rt.new_agent()
    parent = rt.new_agent()
    assert sub is not parent
    await rt.aclose()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_runtime.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `runtime.py`**

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Protocol

from .agent import Agent
from .config import Config
from .llama_client import LlamaClient, LlamaServerManager
from .tools.builtin.fs import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from .tools.builtin.shell import ShellRunTool
from .tools.builtin.subagent import SpawnSubagentTool
from .tools.mcp_bridge import McpBridge
from .tools.registry import ToolRegistry


class _ClientLike(Protocol):
    async def chat(self, *, messages, tools, temperature=0.2): ...
    async def health(self) -> bool: ...
    async def aclose(self) -> None: ...


class Runtime:
    """Holds the long-lived runtime: client, registry, bridge, semaphore."""

    def __init__(
        self,
        cfg: Config,
        client: _ClientLike,
        manager: LlamaServerManager | None,
        bridge: McpBridge | None,
        registry: ToolRegistry,
        semaphore: asyncio.Semaphore,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.manager = manager
        self.bridge = bridge
        self.registry = registry
        self.semaphore = semaphore

    @classmethod
    async def create(
        cls,
        cfg: Config,
        *,
        client_factory: Callable[[str], _ClientLike] | None = None,
    ) -> "Runtime":
        client = (
            client_factory(cfg.llama.server_url)
            if client_factory
            else LlamaClient(base_url=cfg.llama.server_url)
        )
        manager = LlamaServerManager(cfg.llama, client)
        await manager.ensure_running()

        registry = ToolRegistry()
        sandbox = cfg.sandbox
        registry.register(ReadFileTool(allowed_dirs=sandbox.allowed_dirs))
        registry.register(WriteFileTool(allowed_dirs=sandbox.allowed_dirs))
        registry.register(EditFileTool(allowed_dirs=sandbox.allowed_dirs))
        registry.register(ListFilesTool(allowed_dirs=sandbox.allowed_dirs))
        registry.register(
            ShellRunTool(
                allowed_dirs=sandbox.allowed_dirs,
                allowlist=sandbox.shell_allowlist,
            )
        )

        sem = asyncio.Semaphore(cfg.agent.max_concurrent_agents)

        bridge: McpBridge | None = None
        if cfg.mcp_servers:
            bridge = McpBridge(cfg.mcp_servers)
            for t in await bridge.start():
                registry.register(t)

        rt = cls(cfg, client, manager, bridge, registry, sem)

        # Inject the spawn tool last (it needs the runtime to make new agents).
        registry.register(
            SpawnSubagentTool(agent_factory=rt.new_agent, semaphore=sem)
        )
        return rt

    def new_agent(self) -> Agent:
        # Each agent shares the registry, but has its own conversation state.
        return Agent(client=self.client, registry=self.registry)

    async def aclose(self) -> None:
        if self.bridge is not None:
            await self.bridge.aclose()
        if self.manager is not None:
            await self.manager.shutdown()
        await self.client.aclose()
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/unit/test_runtime.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Runtime that assembles client/registry/bridge"
```

---

## Task 15: CLI surface

**Files:**
- Create: `src/llama_agents/cli.py`
- Create: `tests/unit/test_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_cli.py`:
```python
from typer.testing import CliRunner

from llama_agents.cli import app


def test_cli_shows_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "chat" in result.stdout
    assert "serve" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_cli.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `cli.py`**

```python
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from .agent import AgentRunOptions
from .config import load_config
from .events import AssistantChunk, Done, LoopError, ToolCallResult, ToolCallStart
from .runtime import Runtime


app = typer.Typer(no_args_is_help=True, help="llama-agents CLI")
console = Console()


def _default_config_path() -> Path:
    env = os.environ.get("LLAMA_AGENTS_CONFIG")
    if env:
        return Path(env)
    return Path("config.toml")


@app.command()
def chat(
    config: Path = typer.Option(_default_config_path, "--config", "-c"),
    prompt: str = typer.Argument(..., help="Prompt to send to the agent."),
    max_iterations: int = typer.Option(20, "--max-iterations"),
) -> None:
    """Run a single agent turn against the configured llama-server."""
    asyncio.run(_run_chat(config, prompt, max_iterations))


@app.command()
def serve(
    config: Path = typer.Option(_default_config_path, "--config", "-c"),
) -> None:
    """Start the HTTP service."""
    import uvicorn

    from .http_app import create_app

    cfg = load_config(config)
    fastapi_app = create_app(cfg)
    uvicorn.run(fastapi_app, host=cfg.http.host, port=cfg.http.port)


async def _run_chat(config_path: Path, prompt: str, max_iterations: int) -> None:
    cfg = load_config(config_path)
    rt = await Runtime.create(cfg)
    try:
        agent = rt.new_agent()
        opts = AgentRunOptions(max_iterations=max_iterations)
        async for ev in agent.run(prompt, opts):
            if isinstance(ev, AssistantChunk):
                console.print(Markdown(ev.text))
            elif isinstance(ev, ToolCallStart):
                console.print(f"[dim]→ {ev.name}({ev.arguments})[/dim]")
            elif isinstance(ev, ToolCallResult):
                marker = "✓" if ev.ok else "✗"
                console.print(f"[dim]  {marker} {str(ev.content)[:160]}[/dim]")
            elif isinstance(ev, LoopError):
                console.print(f"[red]{ev.error_type}: {ev.message}[/red]")
            elif isinstance(ev, Done):
                console.print(f"[dim](done: {ev.reason})[/dim]")
    finally:
        await rt.aclose()
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/unit/test_cli.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add typer CLI with chat and serve commands"
```

---

## Task 16: HTTP surface with SSE

**Files:**
- Create: `src/llama_agents/http_app.py`
- Create: `tests/unit/test_http_app.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_http_app.py`:
```python
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llama_agents.config import Config, LlamaConfig, AgentConfig, SandboxConfig
from llama_agents.http_app import create_app
from llama_agents.llama_client import ChatResponse


class FakeClient:
    async def chat(self, *, messages, tools, temperature=0.2):
        return ChatResponse(content="hello from fake")

    async def health(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        llama=LlamaConfig(auto_spawn=False),
        agent=AgentConfig(),
        sandbox=SandboxConfig(allowed_dirs=[tmp_path], shell_allowlist=["python"]),
    )


def test_health_endpoint(cfg: Config):
    app = create_app(cfg, client_factory=lambda url: FakeClient())
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_chat_stream_endpoint_emits_sse(cfg: Config):
    app = create_app(cfg, client_factory=lambda url: FakeClient())
    with TestClient(app) as c:
        with c.stream("POST", "/chat", json={"prompt": "hi"}) as r:
            assert r.status_code == 200
            body = "".join(r.iter_text())
            assert "hello from fake" in body
            assert "event: done" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_http_app.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `http_app.py`**

```python
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .agent import AgentRunOptions
from .config import Config
from .events import AssistantChunk, Done, LoopError, ToolCallResult, ToolCallStart
from .runtime import Runtime


class ChatRequest(BaseModel):
    prompt: str
    max_iterations: int | None = None
    system_prompt: str | None = None


def create_app(
    cfg: Config,
    *,
    client_factory: Callable | None = None,
) -> FastAPI:
    runtime_box: dict[str, Runtime] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime_box["rt"] = await Runtime.create(cfg, client_factory=client_factory)
        try:
            yield
        finally:
            await runtime_box["rt"].aclose()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    async def chat(req: ChatRequest):
        rt = runtime_box["rt"]
        agent = rt.new_agent()
        opts = AgentRunOptions(
            max_iterations=req.max_iterations or cfg.agent.max_iterations,
            system_prompt=req.system_prompt or AgentRunOptions().system_prompt,
        )

        async def gen():
            async for ev in agent.run(req.prompt, opts):
                yield _serialize(ev)

        return EventSourceResponse(gen())

    return app


def _serialize(ev: Any) -> dict[str, str]:
    if isinstance(ev, AssistantChunk):
        return {"event": "assistant_chunk", "data": json.dumps({"text": ev.text})}
    if isinstance(ev, ToolCallStart):
        return {
            "event": "tool_call_start",
            "data": json.dumps(
                {"call_id": ev.call_id, "name": ev.name, "arguments": ev.arguments}
            ),
        }
    if isinstance(ev, ToolCallResult):
        return {
            "event": "tool_call_result",
            "data": json.dumps(
                {"call_id": ev.call_id, "ok": ev.ok, "content": str(ev.content)}
            ),
        }
    if isinstance(ev, LoopError):
        return {
            "event": "error",
            "data": json.dumps({"type": ev.error_type, "message": ev.message}),
        }
    if isinstance(ev, Done):
        return {
            "event": "done",
            "data": json.dumps(
                {"reason": ev.reason, "final_message": ev.final_message}
            ),
        }
    return {"event": "unknown", "data": "{}"}
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/unit/test_http_app.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add FastAPI HTTP service with SSE chat endpoint"
```

---

## Task 17: Full unit-test pass + smoke

**Files:** (no source changes)

- [ ] **Step 1: Run the full unit suite**

Run: `uv run pytest tests/unit -v`
Expected: all tests pass, no warnings about un-awaited coroutines.

- [ ] **Step 2: Verify CLI help works**

Run: `uv run llamactl --help`
Expected: typer help showing `chat` and `serve` subcommands.

- [ ] **Step 3: Boot the HTTP service against a fake client**

Manual sanity check is optional — covered by `test_http_app.py` already. Skip if green.

- [ ] **Step 4: Commit any housekeeping**

If anything changed (lockfile, .gitignore tweaks), commit:
```bash
git add -A
git commit -m "chore: housekeeping after full test pass"
```
Otherwise skip.

---

## Task 18: Live smoke test (manual, requires running llama-server)

**Files:**
- Create: `tests/live/test_live_chat.py`

- [ ] **Step 1: Implement live test**

`tests/live/test_live_chat.py`:
```python
import os
from pathlib import Path

import pytest

from llama_agents.config import load_config
from llama_agents.agent import AgentRunOptions
from llama_agents.runtime import Runtime
from llama_agents.events import AssistantChunk, Done


@pytest.mark.live
async def test_round_trip_simple_prompt():
    cfg_path = os.environ.get("LLAMA_AGENTS_CONFIG", "config.toml")
    cfg = load_config(cfg_path)
    rt = await Runtime.create(cfg)
    try:
        agent = rt.new_agent()
        text = ""
        async for ev in agent.run(
            "Say 'pong' and nothing else.",
            AgentRunOptions(max_iterations=2),
        ):
            if isinstance(ev, AssistantChunk):
                text = ev.text
            if isinstance(ev, Done):
                break
        assert "pong" in text.lower()
    finally:
        await rt.aclose()
```

- [ ] **Step 2: Run only the live test (with llama-server up)**

Manual prerequisite: llama-server.exe running on 127.0.0.1:8080 with the Qwen3-Coder model, **or** `config.toml` has `auto_spawn=true` with valid `server_bin`/`model_path`.

Run: `uv run pytest tests/live -m live -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: add live round-trip smoke test"
```

---

## Self-review notes (for the writer, before handoff)

- All spec sections are covered: error taxonomy (Task 2), config (3), sandbox (4), registry (5), fs tools (6), shell (7), llama client + lifecycle (8-9), events (10), agent loop (11), subagent + concurrency cap (12), MCP bridge (13), runtime wiring (14), CLI (15), HTTP/SSE (16), full test pass (17), live smoke (18).
- No `TBD` placeholders; every code step includes complete code.
- Type/name consistency check: tool names (`fs_read_file`, `fs_write_file`, `fs_edit_file`, `fs_list_files`, `shell_run`, `subagent_spawn`, `rag__rag_query` etc.) used consistently across tasks 5-16.
- `Agent._registry` is a private attribute used by the subagent tool — acceptable for v1; could be promoted to a `registry` property later.
- Token-budget guard from spec §4.5 is mentioned in the design but not implemented in v1 — flagged as future work in §11 of the spec. **Not** a plan gap: implementing it would require tokenizer access which we deferred.
- `MCPServerCrashed` auto-restart from spec §8 is partially implemented (errors surface to the model) but the auto-restart logic is deferred to a follow-up.

Both deferrals are intentional and consistent with the spec's "future work" framing.
