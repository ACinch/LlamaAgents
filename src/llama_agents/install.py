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
import time
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


ExistingAction = Literal["write", "cancel"]


def existing_config_action(
    path: Path, prompter: Prompter, *, force: bool
) -> ExistingAction:
    """Decide what to do with an existing config file.

    Returns "write" if we should overwrite (missing, forced, or user confirmed).
    Returns "cancel" if user declined to overwrite.
    """
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
    """Write config file, optionally backing up an existing one.

    Creates parent directories if needed. If backup_existing is True and the
    file exists, the old file is moved to config.toml.bak.{timestamp}.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup_existing and path.exists():
        backup = path.with_name(f"{path.name}.bak.{int(time.time())}")
        path.replace(backup)
    path.write_text(content, encoding="utf-8")


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


import hashlib
import urllib.request
import zipfile
from io import BytesIO

LLAMA_CPP_RELEASE_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    "b9370/llama-b9370-bin-win-cuda-12.4-x64.zip"
)
LLAMA_CPP_RELEASE_SHA256 = "01f960b644114955fbca4788aa3028d10165f6c43185c61176e8c4a54c58544b"


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
