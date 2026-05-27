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


import shutil
import subprocess
from pathlib import Path


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


def model_search_dirs(repo_root: Path) -> list[Path]:
    """Return the canonical search order for GGUF model files."""
    return [
        repo_root / "GGUF",
        repo_root.parent / "GGUF",
        Path.home() / "GGUF",
    ]


def find_existing_model(spec: ModelSpec, repo_root: Path) -> Path | None:
    """Search for a model matching spec in known directories. Returns first hit or None."""
    for d in model_search_dirs(repo_root):
        candidate = d / spec.hf_filename
        if candidate.is_file():
            return candidate
    return None
