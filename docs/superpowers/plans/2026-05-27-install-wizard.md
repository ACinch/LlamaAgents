# `llamactl init` Install Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an interactive `llamactl init` wizard that detects the user's llama-server binary, recommends and optionally downloads a GGUF model sized to their GPU's VRAM, collects allowed_dirs, and writes a working `config.toml`.

**Architecture:** Single new module `src/llama_agents/install.py` containing pure data (`ModelSpec`, `CATALOGUE`, tier rules), single-responsibility I/O helpers (VRAM detection, binary discovery, file search, downloads, config rendering), a `Prompter` Protocol with `RichPrompter` (prod) and `RecordedPrompter` (test) implementations, and a `run_install_wizard` orchestrator. A thin `init` subcommand is added to `cli.py`.

**Tech Stack:** Python 3.12+, pydantic, typer, rich.prompt, stdlib subprocess/urllib/zipfile/hashlib, lazy-imported `huggingface_hub` (optional dep), pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-05-27-install-wizard-design.md`

---

## File structure (locked in by this plan)

**New:**
- `src/llama_agents/install.py` — all wizard logic
- `tests/unit/test_install.py` — full coverage
- `docs/install.md` — user-facing setup guide

**Modified:**
- `src/llama_agents/cli.py` — add `init` subcommand
- `pyproject.toml` — add `huggingface_hub` as an optional extra under `[project.optional-dependencies] install`
- `README.md` — replace "edit config.toml" instructions with `llamactl init`
- `CLAUDE.md` — add `install.py` row to module map

---

## Conventions for this plan

- **Always run from the repo root** (`D:\repos\llm\llama-agents`).
- **Always use `uv run pytest ...`** to invoke tests. On Windows, `uv` lives at `%USERPROFILE%\AppData\Roaming\Python\Python314\Scripts\uv.exe` if it's not on PATH.
- **Commit after each task** with the verbatim message in each task's final step (conventional commits).
- **Branch:** work on `master` (project convention).
- **TDD:** every behavioural task writes a failing test first, then the minimal code to pass.
- All paths in tests use `tmp_path` (the pytest fixture) — never the real filesystem outside it.
- Mock the network. No test makes a real HTTP request.

---

## Task 1: Prompter protocol + test double + Rich adapter

**Files:**
- Create: `src/llama_agents/install.py`
- Create: `tests/unit/test_install.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_install.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
$env:Path = "$env:USERPROFILE\AppData\Roaming\Python\Python314\Scripts;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; uv run pytest tests/unit/test_install.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement the Prompter Protocol, RecordedPrompter, and RichPrompter**

Create `src/llama_agents/install.py`:

```python
from __future__ import annotations

from typing import Protocol


class Prompter(Protocol):
    """Interaction surface for the install wizard.

    Production uses RichPrompter (terminal). Tests use RecordedPrompter
    (scripted answers, recorded prompts).
    """

    def ask(self, question: str, *, default: str | None = None) -> str: ...
    def confirm(self, question: str, *, default: bool = True) -> bool: ...
    def choose(
        self, question: str, options: list[str], *, default_index: int = 0
    ) -> int: ...
    def info(self, message: str) -> None: ...
    def warn(self, message: str) -> None: ...


class RecordedPrompter:
    """Test double: returns scripted answers; records prompts and messages."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.prompts_seen: list[str] = []
        self.messages: list[tuple[str, str]] = []

    def _next(self) -> str:
        if not self._answers:
            raise RuntimeError("RecordedPrompter ran out of scripted answers")
        return self._answers.pop(0)

    def ask(self, question: str, *, default: str | None = None) -> str:
        self.prompts_seen.append(question)
        raw = self._next()
        if not raw and default is not None:
            return default
        return raw

    def confirm(self, question: str, *, default: bool = True) -> bool:
        self.prompts_seen.append(question)
        raw = self._next().strip().lower()
        if not raw:
            return default
        return raw in ("y", "yes", "true", "1")

    def choose(
        self, question: str, options: list[str], *, default_index: int = 0
    ) -> int:
        self.prompts_seen.append(question)
        raw = self._next().strip()
        if not raw:
            return default_index
        idx = int(raw) - 1  # 1-indexed input
        if idx < 0 or idx >= len(options):
            raise ValueError(f"choose: index {idx + 1} out of range 1..{len(options)}")
        return idx

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def warn(self, message: str) -> None:
        self.messages.append(("warn", message))


class RichPrompter:
    """Production adapter using rich.prompt."""

    def ask(self, question: str, *, default: str | None = None) -> str:
        from rich.prompt import Prompt
        return Prompt.ask(question, default=default or "")

    def confirm(self, question: str, *, default: bool = True) -> bool:
        from rich.prompt import Confirm
        return Confirm.ask(question, default=default)

    def choose(
        self, question: str, options: list[str], *, default_index: int = 0
    ) -> int:
        from rich.prompt import IntPrompt
        lines = [f"  [{i + 1}] {opt}" for i, opt in enumerate(options)]
        print("\n".join(lines))
        choice = IntPrompt.ask(question, default=default_index + 1)
        return choice - 1

    def info(self, message: str) -> None:
        from rich.console import Console
        Console().print(message)

    def warn(self, message: str) -> None:
        from rich.console import Console
        Console().print(f"[yellow]{message}[/yellow]")
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_install.py -v
```

Expected: all green (7 tests).

- [ ] **Step 5: Commit**

```
git add src/llama_agents/install.py tests/unit/test_install.py
git commit -m "feat(install): Prompter protocol with RecordedPrompter + RichPrompter"
```

---

## Task 2: ModelSpec catalogue + tier rules

**Files:**
- Modify: `src/llama_agents/install.py`
- Modify: `tests/unit/test_install.py`

- [ ] **Step 1: Append failing tests**

Add to `tests/unit/test_install.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

```
uv run pytest tests/unit/test_install.py -v
```

- [ ] **Step 3: Add the catalogue and tier helpers**

In `src/llama_agents/install.py`, after the `RichPrompter` class, append:

```python
from dataclasses import dataclass
from typing import Literal

Tier = Literal["L", "M", "S", "unknown"]


@dataclass(frozen=True)
class ModelSpec:
    tier: Literal["L", "M", "S"]
    label: str
    hf_repo: str
    hf_filename: str
    size_gb: float


CATALOGUE: list[ModelSpec] = [
    ModelSpec(
        tier="L",
        label="Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL",
        hf_repo="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        hf_filename="Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf",
        size_gb=17.6,
    ),
    ModelSpec(
        tier="M",
        label="DeepSeek-R1-Distill-Qwen-14B-Q6_K_L",
        hf_repo="bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        hf_filename="DeepSeek-R1-Distill-Qwen-14B-Q6_K_L.gguf",
        size_gb=12.0,
    ),
    ModelSpec(
        tier="S",
        label="Llama-3.2-3B-Instruct-Q5_K_M",
        hf_repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        hf_filename="Llama-3.2-3B-Instruct-Q5_K_M.gguf",
        size_gb=2.4,
    ),
]


def recommend_tier(vram_gb: float | None) -> Tier:
    if vram_gb is None:
        return "unknown"
    if vram_gb >= 24.0:
        return "L"
    if vram_gb >= 14.0:
        return "M"
    if vram_gb >= 8.0:
        return "S"
    return "unknown"


def tier_defaults(tier: str) -> tuple[int, int]:
    """(ctx_size, n_parallel) tuned to each VRAM tier."""
    return {
        "L": (65536, 2),
        "M": (32768, 2),
        "S": (8192, 1),
        "unknown": (8192, 1),
    }[tier]
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_install.py -v
```

Expected: 11 tests pass.

- [ ] **Step 5: Commit**

```
git add src/llama_agents/install.py tests/unit/test_install.py
git commit -m "feat(install): ModelSpec catalogue with VRAM-tiered recommendations"
```

---

## Task 3: VRAM detection via `nvidia-smi`

**Files:**
- Modify: `src/llama_agents/install.py`
- Modify: `tests/unit/test_install.py`

- [ ] **Step 1: Append failing tests**

Add to `tests/unit/test_install.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

- [ ] **Step 3: Implement `detect_vram_gb`**

Append to `src/llama_agents/install.py`:

```python
import subprocess


def detect_vram_gb() -> float | None:
    """Return the first GPU's total VRAM in GB, or None if undetectable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5.0, check=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        mib = int(lines[0])
    except ValueError:
        return None
    return mib / 1024.0
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_install.py -v
```

Expected: 16 tests pass.

- [ ] **Step 5: Commit**

```
git add src/llama_agents/install.py tests/unit/test_install.py
git commit -m "feat(install): detect VRAM via nvidia-smi"
```

---

## Task 4: `locate_llama_server` discovery

**Files:**
- Modify: `src/llama_agents/install.py`
- Modify: `tests/unit/test_install.py`

- [ ] **Step 1: Append failing tests**

Add to `tests/unit/test_install.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

- [ ] **Step 3: Implement `locate_llama_server`**

Append to `src/llama_agents/install.py`:

```python
import shutil
from pathlib import Path


def locate_llama_server(repo_root: Path) -> Path | None:
    """Probe known locations for llama-server (.exe). Returns first hit or None."""
    parent = repo_root.parent
    candidates = [
        parent / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
        parent / "llamacpp-bin" / "llama-server.exe",
    ]
    for c in candidates:
        if c.is_file():
            return c
    for name in ("llama-server.exe", "llama-server"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_install.py -v
```

Expected: 21 tests pass.

- [ ] **Step 5: Commit**

```
git add src/llama_agents/install.py tests/unit/test_install.py
git commit -m "feat(install): locate llama-server via siblings + PATH"
```

---

## Task 5: `find_existing_model` across known dirs

**Files:**
- Modify: `src/llama_agents/install.py`
- Modify: `tests/unit/test_install.py`

- [ ] **Step 1: Append failing tests**

Add to `tests/unit/test_install.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

- [ ] **Step 3: Implement the helpers**

Append to `src/llama_agents/install.py`:

```python
def model_search_dirs(repo_root: Path) -> list[Path]:
    return [
        repo_root / "GGUF",
        repo_root.parent / "GGUF",
        Path.home() / "GGUF",
    ]


def find_existing_model(spec: ModelSpec, repo_root: Path) -> Path | None:
    for d in model_search_dirs(repo_root):
        candidate = d / spec.hf_filename
        if candidate.is_file():
            return candidate
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_install.py -v
```

Expected: 25 tests pass.

- [ ] **Step 5: Commit**

```
git add src/llama_agents/install.py tests/unit/test_install.py
git commit -m "feat(install): find existing GGUF in known model dirs"
```

---

## Task 6: `collect_allowed_dirs`

**Files:**
- Modify: `src/llama_agents/install.py`
- Modify: `tests/unit/test_install.py`

- [ ] **Step 1: Append failing tests**

Add to `tests/unit/test_install.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

- [ ] **Step 3: Implement `collect_allowed_dirs`**

Append to `src/llama_agents/install.py`:

```python
def collect_allowed_dirs(repo_root: Path, prompter: Prompter) -> list[Path]:
    """Build allowed_dirs list: seed with repo_root, loop for additions."""
    result: list[Path] = [repo_root.resolve()]
    while True:
        raw = prompter.ask(
            "Add another path? (empty to finish)", default=""
        )
        if not raw:
            return result
        p = Path(raw)
        if not p.exists():
            prompter.warn(f"{p} does not exist; try again.")
            continue
        if not p.is_dir():
            prompter.warn(f"{p} is not a directory; try again.")
            continue
        resolved = p.resolve()
        if resolved in result:
            prompter.info(f"{resolved} already in list; skipping.")
            continue
        result.append(resolved)
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_install.py -v
```

Expected: 30 tests pass.

- [ ] **Step 5: Commit**

```
git add src/llama_agents/install.py tests/unit/test_install.py
git commit -m "feat(install): interactive allowed_dirs collector with validation"
```

---

## Task 7: `existing_config_action` + `write_config`

**Files:**
- Modify: `src/llama_agents/install.py`
- Modify: `tests/unit/test_install.py`

- [ ] **Step 1: Append failing tests**

Add to `tests/unit/test_install.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

- [ ] **Step 3: Implement the helpers**

Append to `src/llama_agents/install.py`:

```python
import time
from typing import Literal as _Literal

ExistingAction = _Literal["write", "cancel"]


def existing_config_action(
    path: Path, prompter: Prompter, *, force: bool
) -> ExistingAction:
    if not path.exists():
        return "write"
    if force:
        return "write"
    overwrite = prompter.confirm(
        f"{path.name} already exists. Overwrite (existing will be backed up)?",
        default=False,
    )
    return "write" if overwrite else "cancel"


def write_config(path: Path, content: str, *, backup_existing: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup_existing and path.exists():
        backup = path.with_name(f"{path.name}.bak.{int(time.time())}")
        path.replace(backup)
    path.write_text(content, encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_install.py -v
```

Expected: 37 tests pass.

- [ ] **Step 5: Commit**

```
git add src/llama_agents/install.py tests/unit/test_install.py
git commit -m "feat(install): existing-config overwrite prompt + backup-on-write"
```

---

## Task 8: `render_config_toml` (golden + reparse)

**Files:**
- Modify: `src/llama_agents/install.py`
- Modify: `tests/unit/test_install.py`

- [ ] **Step 1: Append failing tests**

Add to `tests/unit/test_install.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

- [ ] **Step 3: Implement `render_config_toml`**

Append to `src/llama_agents/install.py`:

```python
def _toml_str(p: Path) -> str:
    """Forward-slash a path and wrap it as a TOML basic string."""
    return '"' + str(p).replace("\\", "/") + '"'


def render_config_toml(values: dict) -> str:
    """Build a config.toml from wizard outputs.

    Required keys in `values`:
      server_bin (Path), model_path (Path), model_label (str),
      ctx_size (int), n_parallel (int), allowed_dirs (list[Path]).
    """
    allowed = ",".join(_toml_str(p) for p in values["allowed_dirs"])
    return (
        "[llama]\n"
        'server_url = "http://127.0.0.1:8080"\n'
        f'model = "{values["model_label"]}"\n'
        "auto_spawn = true\n"
        "kill_on_exit = true\n"
        f"server_bin = {_toml_str(values['server_bin'])}\n"
        f"model_path = {_toml_str(values['model_path'])}\n"
        "ngl = 999\n"
        f"ctx_size = {values['ctx_size']}\n"
        f"n_parallel = {values['n_parallel']}\n"
        "startup_timeout_seconds = 300\n"
        "\n"
        "[agent]\n"
        "max_iterations = 20\n"
        "max_concurrent_agents = 5\n"
        "token_budget_pct = 0.8\n"
        "\n"
        "[sandbox]\n"
        f"allowed_dirs = [{allowed}]\n"
        'shell_allowlist = ["git"]\n'
        "\n"
        "[http]\n"
        'host = "127.0.0.1"\n'
        "port = 9000\n"
        "\n"
        "[memory]\n"
        "enabled = true\n"
        '# root = ".llama_agents/memory"\n'
        '# embedding_model = "BAAI/bge-small-en-v1.5"\n'
        "# chunk_size = 1500\n"
        "# chunk_overlap = 150\n"
        "# plan_recall_k = 3\n"
        "# plan_recall_threshold = 0.5\n"
        "# subagent_inline_threshold_chars = 2000\n"
        "# subagent_summary_max_tokens = 400\n"
        "# evict_threshold_pct = 70\n"
        "# evict_tool_result_min_chars = 4000\n"
        "# scratch_retention_hours = 24\n"
        "\n"
        "[queue]\n"
        "enabled = false\n"
        '# root = ".llama_agents/queue"\n'
        "# poll_interval_seconds = 2.0\n"
        "# max_concurrent = 1\n"
        "# max_retries = 2\n"
        "# retry_backoff_seconds = 5.0\n"
        "# max_iterations = 20\n"
        "# drain_timeout_seconds = 30.0\n"
        '# accepted_extensions = [".md", ".txt"]\n'
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_install.py -v
```

Expected: 41 tests pass.

- [ ] **Step 5: Commit**

```
git add src/llama_agents/install.py tests/unit/test_install.py
git commit -m "feat(install): render config.toml that round-trips through pydantic"
```

---

## Task 9: `download_llama_cpp` (pinned release zip)

**Files:**
- Modify: `src/llama_agents/install.py`
- Modify: `tests/unit/test_install.py`

The URL and sha256 are constants in `install.py`. The current pin used by this plan is the latest stable Windows CUDA-12.4 build at time of writing: tag `b9370`, asset `llama-b9370-bin-win-cuda-12.4-x64.zip`. The implementer must download the file once locally, compute its sha256 (`sha256sum llama-b9370-bin-win-cuda-12.4-x64.zip` or `(Get-FileHash llama-b9370-bin-win-cuda-12.4-x64.zip -Algorithm SHA256).Hash.ToLower()`), and paste that constant into the source. The tests use a mocked download path, so the constant's literal value is not exercised by CI.

- [ ] **Step 1: Append failing tests**

Add to `tests/unit/test_install.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

- [ ] **Step 3: Implement `download_llama_cpp`**

Append to `src/llama_agents/install.py`:

```python
import hashlib
import urllib.request
import zipfile
from io import BytesIO

LLAMA_CPP_RELEASE_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    "b9370/llama-b9370-bin-win-cuda-12.4-x64.zip"
)
# Placeholder — implementer: replace with real sha256 of the URL above.
LLAMA_CPP_RELEASE_SHA256 = "REPLACE_ME_WITH_REAL_SHA256_LOWERCASE_HEX"


def download_llama_cpp(dest_dir: Path) -> Path:
    """Download the pinned llama.cpp Windows CUDA release into dest_dir.

    Returns the path to the extracted llama-server.exe. Verifies sha256
    before extracting; raises RuntimeError on mismatch.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(LLAMA_CPP_RELEASE_URL, timeout=120) as resp:
        blob = resp.read()
    actual = hashlib.sha256(blob).hexdigest()
    if actual != LLAMA_CPP_RELEASE_SHA256:
        raise RuntimeError(
            f"sha256 mismatch for llama.cpp release: "
            f"expected {LLAMA_CPP_RELEASE_SHA256}, got {actual}"
        )
    with zipfile.ZipFile(BytesIO(blob)) as zf:
        zf.extractall(dest_dir)
    server = dest_dir / "llama-server.exe"
    if not server.is_file():
        raise RuntimeError(
            f"downloaded zip did not contain llama-server.exe at {server}"
        )
    return server
```

- [ ] **Step 4: Pin the real sha256**

Compute the sha256 of the real zip and replace `REPLACE_ME_WITH_REAL_SHA256_LOWERCASE_HEX` in `install.py`. PowerShell:

```
$tmp = "$env:TEMP\llama-b9370.zip"
Invoke-WebRequest -Uri "https://github.com/ggml-org/llama.cpp/releases/download/b9370/llama-b9370-bin-win-cuda-12.4-x64.zip" -OutFile $tmp
(Get-FileHash $tmp -Algorithm SHA256).Hash.ToLower()
Remove-Item $tmp
```

Paste the printed value into `LLAMA_CPP_RELEASE_SHA256`.

- [ ] **Step 5: Run tests to verify they pass**

```
uv run pytest tests/unit/test_install.py -v
```

Expected: 43 tests pass.

- [ ] **Step 6: Commit**

```
git add src/llama_agents/install.py tests/unit/test_install.py
git commit -m "feat(install): download pinned llama.cpp Windows CUDA release"
```

---

## Task 10: `download_model` via lazy `huggingface_hub`

**Files:**
- Modify: `src/llama_agents/install.py`
- Modify: `pyproject.toml`
- Modify: `tests/unit/test_install.py`

- [ ] **Step 1: Append failing tests**

Add to `tests/unit/test_install.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

- [ ] **Step 3: Implement `download_model`**

Append to `src/llama_agents/install.py`:

```python
def download_model(spec: ModelSpec, dest_dir: Path) -> Path:
    """Download the given GGUF into dest_dir using huggingface_hub.

    Lazy-imports the library so the install module loads cleanly even
    when the optional extra is not installed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is not installed. Run:\n"
            "  uv add huggingface_hub\n"
            f"Then re-run `llamactl init`. (Or download manually from\n"
            f"  https://huggingface.co/{spec.hf_repo}/blob/main/{spec.hf_filename}\n"
            f"and put it at {dest_dir / spec.hf_filename}.)"
        ) from e

    dest_dir.mkdir(parents=True, exist_ok=True)
    path_str = hf_hub_download(
        repo_id=spec.hf_repo,
        filename=spec.hf_filename,
        local_dir=str(dest_dir),
    )
    return Path(path_str)
```

- [ ] **Step 4: Add optional dependency**

Edit `pyproject.toml`. Find the `[project]` section. If it has an `[project.optional-dependencies]` block, add an `install` extra; otherwise create the block:

```toml
[project.optional-dependencies]
install = ["huggingface_hub>=0.24"]
```

- [ ] **Step 5: Run tests to verify they pass**

```
uv run pytest tests/unit/test_install.py -v
```

Expected: 45 tests pass.

- [ ] **Step 6: Commit**

```
git add src/llama_agents/install.py tests/unit/test_install.py pyproject.toml
git commit -m "feat(install): download GGUF via lazy huggingface_hub import"
```

---

## Task 11: `run_install_wizard` orchestrator

**Files:**
- Modify: `src/llama_agents/install.py`
- Modify: `tests/unit/test_install.py`

The orchestrator wires together every helper from Tasks 1–10. It is the only function that needs an end-to-end test.

- [ ] **Step 1: Append a failing end-to-end test**

Add to `tests/unit/test_install.py`:

```python
import tomllib

from llama_agents.install import WizardResult, run_install_wizard


def test_wizard_end_to_end_writes_valid_config(tmp_path: Path, monkeypatch):
    from llama_agents.config import Config

    # Pre-stage: pretend llama-server.exe is at a sibling path.
    server = tmp_path / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe"
    server.parent.mkdir(parents=True)
    server.write_bytes(b"\0")

    # Pre-stage: pretend the L-tier model is already on disk.
    repo = tmp_path / "llama-agents"
    repo.mkdir()
    spec = CATALOGUE[0]
    model_file = repo / "GGUF" / spec.hf_filename
    model_file.parent.mkdir()
    model_file.write_bytes(b"\0")

    # Stub VRAM detection to return 24 GB -> recommends L-tier.
    monkeypatch.setattr("llama_agents.install.detect_vram_gb", lambda: 24.0)
    # Stub model_search_dirs so the home-dir entry stays in the sandbox.
    monkeypatch.setattr(
        "llama_agents.install.model_search_dirs",
        lambda r: [r / "GGUF", r.parent / "GGUF", tmp_path / "fake_home" / "GGUF"],
    )

    # Scripted answers, in order:
    #   1. (no overwrite prompt — config absent)
    #   2. confirm found llama-server binary [Y] -> "y"
    #   3. choose model — default index 0 (L tier) -> ""  (Enter)
    #   4. use existing model file? [Y] -> "y"
    #   5. allowed_dirs: empty -> finish
    #   6. accept tier defaults? [Y] -> "y"
    p = RecordedPrompter(answers=["y", "", "y", "", "y"])

    result = run_install_wizard(repo_root=repo, prompter=p, force=False)
    assert isinstance(result, WizardResult)
    assert result.config_path == repo / "config.toml"
    assert result.server_bin == server
    assert result.model_path == model_file
    assert result.allowed_dirs == [repo.resolve()]
    assert result.tier == "L"

    # Final assertion: the written config parses cleanly.
    data = tomllib.loads(result.config_path.read_text(encoding="utf-8"))
    cfg = Config.model_validate(data)
    assert cfg.llama.ctx_size == 65536
    assert cfg.llama.n_parallel == 2
    assert Path(cfg.llama.server_bin) == server
    assert Path(cfg.llama.model_path) == model_file


def test_wizard_cancels_when_user_declines_overwrite(tmp_path: Path, monkeypatch):
    repo = tmp_path / "llama-agents"
    repo.mkdir()
    existing = repo / "config.toml"
    existing.write_text("# old", encoding="utf-8")

    p = RecordedPrompter(answers=["n"])
    result = run_install_wizard(repo_root=repo, prompter=p, force=False)
    assert result is None
    # Existing file untouched.
    assert existing.read_text(encoding="utf-8") == "# old"
```

- [ ] **Step 2: Run tests — expect ImportError on `run_install_wizard` / `WizardResult`**

- [ ] **Step 3: Implement the orchestrator and result type**

Append to `src/llama_agents/install.py`:

```python
@dataclass
class WizardResult:
    config_path: Path
    server_bin: Path
    model_path: Path
    allowed_dirs: list[Path]
    vram_gb: float | None
    tier: Tier


def _present_models(
    recommended: Tier, prompter: Prompter
) -> ModelSpec | None:
    """Show the catalogue, return chosen spec or None for 'user picks file'."""
    options: list[str] = []
    default_index = 0
    for i, m in enumerate(CATALOGUE):
        tag = " ← recommended" if m.tier == recommended else ""
        flag = ""
        if recommended == "S" and m.tier in ("L", "M"):
            flag = "  (may not fit)"
        elif recommended == "M" and m.tier == "L":
            flag = "  (may not fit)"
        options.append(f"{m.label}  (~{m.size_gb} GB){tag}{flag}")
        if m.tier == recommended:
            default_index = i
    options.append("Use a local file I'll specify")
    idx = prompter.choose(
        "Choose model", options, default_index=default_index
    )
    if idx == len(CATALOGUE):
        return None
    return CATALOGUE[idx]


def _resolve_server_bin(repo_root: Path, prompter: Prompter) -> Path | None:
    """Find or ask for llama-server.exe. Returns None on cancel."""
    found = locate_llama_server(repo_root)
    if found is not None:
        if prompter.confirm(
            f"Found llama-server at {found} — use it?", default=True
        ):
            return found
    # Not found or user declined: offer alternatives.
    options = [
        "Download a pinned llama.cpp Windows CUDA release (~250 MB)",
        "Enter a path manually",
        "Cancel and install it yourself",
    ]
    idx = prompter.choose("How would you like to proceed?", options, default_index=0)
    if idx == 0:
        try:
            return download_llama_cpp(repo_root / "llamacpp-bin")
        except Exception as e:
            prompter.warn(f"download failed: {e}")
            return None
    if idx == 1:
        raw = prompter.ask("Path to llama-server.exe", default="")
        p = Path(raw)
        if not p.is_file():
            prompter.warn(f"{p} is not a file; cancelling.")
            return None
        return p
    return None


def _resolve_model_path(
    spec: ModelSpec | None, repo_root: Path, prompter: Prompter
) -> Path | None:
    if spec is None:
        raw = prompter.ask("Path to .gguf model", default="")
        p = Path(raw)
        if not p.is_file():
            prompter.warn(f"{p} is not a file; cancelling.")
            return None
        return p
    existing = find_existing_model(spec, repo_root)
    if existing is not None:
        if prompter.confirm(
            f"Found existing model at {existing} — use it?", default=True
        ):
            return existing
    if not prompter.confirm(
        f"Download {spec.hf_filename} (~{spec.size_gb} GB) into ./GGUF/?",
        default=True,
    ):
        return None
    try:
        return download_model(spec, repo_root / "GGUF")
    except Exception as e:
        prompter.warn(f"download failed: {e}")
        return None


def run_install_wizard(
    *,
    repo_root: Path,
    prompter: Prompter,
    force: bool = False,
) -> WizardResult | None:
    """Drive the full install wizard. Returns None if user cancels."""
    config_path = repo_root / "config.toml"
    action = existing_config_action(config_path, prompter, force=force)
    if action == "cancel":
        prompter.info("Setup cancelled. No changes made.")
        return None

    server_bin = _resolve_server_bin(repo_root, prompter)
    if server_bin is None:
        prompter.info("Setup cancelled (no llama-server).")
        return None

    vram = detect_vram_gb()
    tier = recommend_tier(vram)
    if vram is not None:
        prompter.info(f"Detected GPU with {vram:.1f} GB VRAM (tier {tier}).")
    else:
        prompter.warn("Could not auto-detect VRAM — recommendations skipped.")

    spec = _present_models(tier, prompter)
    model_path = _resolve_model_path(spec, repo_root, prompter)
    if model_path is None:
        prompter.info("Setup cancelled (no model).")
        return None

    allowed = collect_allowed_dirs(repo_root, prompter)

    ctx_size, n_parallel = tier_defaults(tier)
    if not prompter.confirm(
        f"Context window: ctx_size={ctx_size}, n_parallel={n_parallel} — use these?",
        default=True,
    ):
        raw_ctx = prompter.ask("ctx_size", default=str(ctx_size))
        raw_np = prompter.ask("n_parallel", default=str(n_parallel))
        ctx_size = int(raw_ctx)
        n_parallel = int(raw_np)

    text = render_config_toml({
        "server_bin": server_bin,
        "model_path": model_path,
        "model_label": spec.hf_filename if spec else model_path.name,
        "ctx_size": ctx_size,
        "n_parallel": n_parallel,
        "allowed_dirs": allowed,
    })
    write_config(config_path, text, backup_existing=config_path.exists())

    prompter.info(f"Wrote {config_path}")
    prompter.info("Start the server with:  uv run llamactl serve")
    prompter.info('Or run a single task:  uv run llamactl chat "your task"')

    return WizardResult(
        config_path=config_path,
        server_bin=server_bin,
        model_path=model_path,
        allowed_dirs=allowed,
        vram_gb=vram,
        tier=tier,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_install.py -v
```

Expected: 47 tests pass.

- [ ] **Step 5: Commit**

```
git add src/llama_agents/install.py tests/unit/test_install.py
git commit -m "feat(install): run_install_wizard orchestrator wires every step"
```

---

## Task 12: `llamactl init` CLI subcommand

**Files:**
- Modify: `src/llama_agents/cli.py`
- Create: `tests/unit/test_cli_init.py`

- [ ] **Step 1: Write failing test for the CLI integration**

Create `tests/unit/test_cli_init.py`:

```python
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
```

- [ ] **Step 2: Run the test — expect failure (no `init` command yet)**

```
uv run pytest tests/unit/test_cli_init.py -v
```

- [ ] **Step 3: Add the `init` command to `cli.py`**

Open `src/llama_agents/cli.py` and add at the bottom (before any `if __name__ == "__main__"`):

```python
@app.command()
def init(
    force: bool = typer.Option(
        False, "--force",
        help="Overwrite existing config.toml without prompting (backs up first).",
    ),
) -> None:
    """Interactive first-run setup: detects llama-server, picks a model, writes config.toml."""
    from .install import RichPrompter, run_install_wizard
    result = run_install_wizard(
        repo_root=Path.cwd(),
        prompter=RichPrompter(),
        force=force,
    )
    if result is None:
        raise typer.Exit(code=1)
```

- [ ] **Step 4: Run the test — expect pass**

```
uv run pytest tests/unit/test_cli_init.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Run the full unit suite to confirm no regression**

```
uv run pytest tests/unit -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git add src/llama_agents/cli.py tests/unit/test_cli_init.py
git commit -m "feat(cli): add llamactl init subcommand"
```

---

## Task 13: User guide + module-map update + README pointer

**Files:**
- Create: `docs/install.md`
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Write the user guide**

Create `docs/install.md`:

```markdown
# Install Wizard

After cloning the repo, run:

```
uv run llamactl init
```

The wizard:

1. Asks whether to overwrite an existing `config.toml` (backs it up to
   `config.toml.bak.<timestamp>` if you say yes).
2. Locates `llama-server.exe` in three places:
   `../llama.cpp/build/bin/Release/`, `../llamacpp-bin/`, then your
   `PATH`. If none are found, offers to download a pinned Windows CUDA
   release of llama.cpp into `./llamacpp-bin/`.
3. Detects your GPU's VRAM via `nvidia-smi` and recommends a GGUF
   model:
   - ≥ 24 GB → Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL
   - 14–23 GB → DeepSeek-R1-Distill-Qwen-14B-Q6_K_L
   - 8–13 GB → Llama-3.2-3B-Instruct-Q5_K_M
   - unknown / < 8 GB → no recommendation; pick manually
4. Scans `./GGUF/`, `../GGUF/`, and `~/GGUF/` for the chosen file. If
   absent, offers to download it from HuggingFace into `./GGUF/`.
   (Requires `huggingface_hub` — install with
   `uv add huggingface_hub` if the wizard says it's missing.)
5. Collects `allowed_dirs`. The repo root is added automatically; you
   can add more paths in a loop.
6. Picks `ctx_size` and `n_parallel` for your VRAM tier
   (24 GB → 65536/2, 14 GB → 32768/2, 8 GB → 8192/1). Accept the
   defaults or override.
7. Writes `config.toml` and prints the next commands.

## Flags

- `--force` — overwrite an existing `config.toml` without prompting
  (still backs up first).

## When to re-run

Re-run any time you change GPU, add a new model, or move the repo. The
wizard is deterministic given the same answers, so you can re-run it
to regenerate `config.toml` after editing helpers.

## Edge cases

- **No NVIDIA GPU / no `nvidia-smi`** → wizard skips VRAM detection
  and the tier becomes "unknown". You can pick any model from the
  full list.
- **No `huggingface_hub` installed** → the model-download prompt
  errors with a clear instruction. Install with
  `uv add huggingface_hub` or download the GGUF manually into
  `./GGUF/`.
- **Pinned llama.cpp release moved** → if the auto-download fails
  (404 or sha mismatch), the wizard tells you, and you can supply a
  path manually.
```

- [ ] **Step 2: Update `CLAUDE.md` module map**

Open `D:\repos\llm\llama-agents\CLAUDE.md`. In the "Module map" table (the same one Task 12 of the queue plan added rows to), append a new row at the end:

```
| `install.py` | `llamactl init` wizard: VRAM detect, model pick, config render. |
```

- [ ] **Step 3: Update `README.md`**

Open `README.md`. Find any current setup/installation section that says "edit `config.toml` to…" and add at the top of that section:

```markdown
## First-time setup

After installing dependencies, run:

```
uv run llamactl init
```

This interactive wizard detects your llama-server binary, recommends
a GGUF model sized to your GPU, downloads it if needed, and writes a
`config.toml`. See `docs/install.md` for details.
```

If no setup section exists, add the block above just after the intro paragraph.

- [ ] **Step 4: Run the full unit suite (regression check, no behavioural change)**

```
uv run pytest tests/unit -q
```

Expected: all green.

- [ ] **Step 5: Commit**

```
git add docs/install.md "D:\repos\llm\llama-agents\CLAUDE.md" README.md
git commit -m "docs(install): user guide, README pointer, module map row"
```

---

## Done criteria

After Task 13 the following should be true:

- `uv run pytest tests/unit -q` is green (~47 new install tests plus all pre-existing).
- `uv run llamactl init --help` shows the new subcommand.
- On a fresh checkout, `uv run llamactl init` walks through the wizard and produces a `config.toml` that loads cleanly under `Config.model_validate(...)`.
- `docs/install.md` explains the wizard for end-users; `CLAUDE.md` module map lists `install.py`; `README.md` directs new users to `llamactl init`.
- The auto-download of `llama.cpp` uses a pinned URL with a pinned sha256 in `install.py` constants.
