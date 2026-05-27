# `llamactl init` Install Wizard — Design

**Status:** Draft
**Date:** 2026-05-27
**Owner:** Maarten

## Goal

After `git clone`, a fresh user should run `uv run llamactl init` and end
up with a working `config.toml`: a valid `llama-server.exe`, a downloaded
GGUF model sized to their GPU, and a sensible `allowed_dirs` /
`shell_allowlist` / context-window setup. The wizard is interactive but
auto-detects whatever it can so the user mostly hits Enter.

## Non-goals

- Headless / CI-friendly mode. (A `--non-interactive` flag exists to
  fail fast if any prompt would be required, but the wizard does not
  try to produce a "best-effort" config silently.)
- Installing Python, `uv`, CUDA drivers, or NVIDIA toolkits. The
  wizard assumes those are present.
- Managing multiple model profiles, swappable configs, or
  per-project overrides. One `config.toml` per checkout.
- Modifying an existing valid config in-place. If the user already
  has `config.toml`, the wizard either bails or backs up and rewrites
  — never merges.

## User experience

```
$ uv run llamactl init
🔧 llama-agents setup wizard

✓ Found llama-server.exe at ..\llama.cpp\build\bin\Release\llama-server.exe
✓ Detected NVIDIA GPU with 24 GB VRAM

Recommended model for your GPU:
  [1] Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL   (~17.6 GB) ← recommended
  [2] DeepSeek-R1-Distill-Qwen-14B-Q6_K_L       (~12 GB)
  [3] Llama-3.2-3B-Instruct-Q5_K_M              (~2.4 GB)
  [4] Use a local file I'll specify

Choose model [1]:
✓ Found existing file at .\GGUF\Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf

Allowed directories (the agent can read/write files here):
  default: D:\repos\llm\llama-agents
Add another path? (empty to finish):
> C:\Users\maarten\projects
Add another path? (empty to finish):
>

Context window settings for your tier (24 GB):
  ctx_size = 65536   n_parallel = 2
Use these? [Y/n]:

Writing config.toml... done.

Start the server with:  uv run llamactl serve
Or run a single task:   uv run llamactl chat "your task"
```

## Architecture

### New module: `src/llama_agents/install.py`

Pure functions plus one orchestrator. No `input()` calls — prompting
goes through a `Prompter` Protocol so unit tests can inject scripted
answers.

```python
class Prompter(Protocol):
    def ask(self, question: str, *, default: str | None = None) -> str: ...
    def confirm(self, question: str, *, default: bool = True) -> bool: ...
    def choose(self, question: str, options: list[str], *,
               default_index: int = 0) -> int: ...
    def info(self, message: str) -> None: ...
    def warn(self, message: str) -> None: ...

@dataclass
class WizardResult:
    config_path: Path
    server_bin: Path
    model_path: Path
    allowed_dirs: list[Path]
    vram_gb: float | None
    tier: Literal["L", "M", "S", "unknown"]

def run_install_wizard(
    *,
    repo_root: Path,
    prompter: Prompter,
    force: bool = False,
    non_interactive: bool = False,
) -> WizardResult: ...
```

The orchestrator calls these single-responsibility helpers, all
testable in isolation:

- `existing_config_action(path, prompter, force) -> Literal["write", "cancel"]`
- `locate_llama_server(repo_root) -> Path | None` — probes 3 paths, returns first hit.
- `download_llama_cpp(dest_dir, prompter) -> Path` — downloads pinned release zip, sha256-checks, extracts.
- `detect_vram_gb() -> float | None` — runs `nvidia-smi`; returns `None` on any failure.
- `recommend_tier(vram_gb: float | None) -> Literal["L", "M", "S", "unknown"]`
- `present_models(tier, prompter) -> ModelSpec` — shows the 3-item list with the recommended one pre-selected; returns the chosen spec or a sentinel for "user-supplied path".
- `find_existing_model(spec, search_dirs) -> Path | None` — scans `./GGUF/`, `../GGUF/`, `~/GGUF/`.
- `download_model(spec, dest_dir, prompter) -> Path` — uses `huggingface_hub.hf_hub_download` (lazy import).
- `collect_allowed_dirs(repo_root, prompter) -> list[Path]`
- `tier_defaults(tier) -> tuple[int, int]` — `(ctx_size, n_parallel)`.
- `render_config_toml(values: dict) -> str` — string-templates the TOML.
- `write_config(path, content, backup_existing: bool) -> None`

### `llamactl init` CLI command

In `src/llama_agents/cli.py`:

```python
@app.command()
def init(
    force: bool = typer.Option(False, "--force",
        help="Overwrite existing config.toml without prompting (backs up first)."),
    non_interactive: bool = typer.Option(False, "--non-interactive",
        help="Fail fast if any prompt would be required."),
) -> None:
    """Interactive first-run setup: detects llama-server, picks a model, writes config.toml."""
    from .install import RichPrompter, run_install_wizard
    repo_root = Path.cwd()
    prompter = RichPrompter()
    result = run_install_wizard(
        repo_root=repo_root, prompter=prompter,
        force=force, non_interactive=non_interactive,
    )
    console.print(f"\n[green]✓ wrote {result.config_path}[/green]")
```

`RichPrompter` is a thin adapter in `install.py` that uses the
`rich.prompt` family (`Prompt.ask`, `Confirm.ask`, `IntPrompt.ask`) so
the wizard inherits the project's existing visual style.

### Model catalogue

Hardcoded in `install.py`:

```python
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
```

### Tier rules

```python
def recommend_tier(vram_gb: float | None) -> Literal["L", "M", "S", "unknown"]:
    if vram_gb is None:        return "unknown"
    if vram_gb >= 24.0:        return "L"
    if vram_gb >= 14.0:        return "M"
    if vram_gb >= 8.0:         return "S"
    return "unknown"   # below 8 GB: no auto-recommendation

def tier_defaults(tier: str) -> tuple[int, int]:
    """(ctx_size, n_parallel) suited to each VRAM tier."""
    return {
        "L":       (65536, 2),    # 24 GB+: full 64k context, 2 slots
        "M":       (32768, 2),    # 14-24 GB: 32k context, 2 slots
        "S":       (8192,  1),    # 8-14 GB: 8k context, 1 slot
        "unknown": (8192,  1),    # safest possible fallback
    }[tier]
```

`present_models` always shows all 3 entries from `CATALOGUE` plus an
"I'll specify a local file" option. The recommended-for-this-tier
entry is pre-selected and gets the `← recommended` annotation. Entries
that exceed the detected VRAM tier are shown with a `(may not fit)`
flag but remain selectable.

### llama-server discovery and download

Search order (first hit wins):

1. `<repo_root>/../llama.cpp/build/bin/Release/llama-server.exe`
2. `<repo_root>/../llamacpp-bin/llama-server.exe`
3. `shutil.which("llama-server.exe")` (and `"llama-server"` on POSIX)

If none found, prompt:

```
No llama-server binary found.
[1] Download a pinned llama.cpp Windows CUDA release (~250 MB)
[2] Enter a path manually
[3] Cancel and install it yourself
```

If `[1]`: download a fixed URL — the version is a constant in
`install.py` (e.g. `LLAMA_CPP_RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b6800/llama-b6800-bin-win-cuda-12.4-x64.zip"`),
verify sha256 against an inline constant, extract to
`<repo_root>/llamacpp-bin/`. Set `server_bin` to the extracted
`llama-server.exe`.

On non-Windows, the wizard skips the download offer and only
suggests options `[2]` and `[3]`.

### VRAM detection

```python
def detect_vram_gb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5.0, check=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    first = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    try:
        return int(first) / 1024.0   # nvidia-smi reports MiB
    except ValueError:
        return None
```

When `None`, the wizard prints "could not detect VRAM" and asks the
user to enter a number in GB (or pick "skip" → tier becomes
`"unknown"`).

### Model search dirs

```python
def model_search_dirs(repo_root: Path) -> list[Path]:
    return [
        repo_root / "GGUF",
        repo_root.parent / "GGUF",
        Path.home() / "GGUF",
    ]
```

Match is by exact `hf_filename`. Symlinks are followed. If found,
prompt "use existing at `<path>`? (Y/n)". If the user declines, fall
through to the download prompt.

### Download path for models

`huggingface_hub.hf_hub_download` is imported lazily inside
`download_model`. If the import fails, surface a clear message:

```
huggingface_hub is not installed. Run:
  uv add huggingface_hub
Then re-run `llamactl init`. (Or download manually from:
  https://huggingface.co/<repo>/blob/main/<filename>
and put it at ./GGUF/<filename>.)
```

The download target is always `<repo_root>/GGUF/<filename>`. Existing
file at that path triggers the "use existing" branch above, so a
half-completed download from a previous attempt is detected.

### allowed_dirs collection

Seed with `[repo_root]`. Loop:

```
Add another path? (empty to finish): <user input>
```

Each entry is validated:
- Must exist on disk → if not, warn and re-prompt.
- Must be a directory → if not, warn and re-prompt.
- Resolve to absolute via `Path(p).resolve()`.

Dedupe before writing.

### shell_allowlist

Hardcoded to `["git"]`. Rationale: the only built-in command our
examples rely on. The user can edit `config.toml` to add `pytest`,
`node`, etc. No prompt — keeps the wizard short and the default safe.

### config.toml rendering

A single string template, populated from `WizardResult`. The
`[memory]` and `[queue]` blocks are written with `enabled = true` and
`enabled = false` respectively, all other knobs as commented defaults
— same shape as the current Task 11 commit produced for `[queue]`.

The wizard never preserves any user comments from a pre-existing
config; if overwrite is chosen, the previous file is backed up to
`config.toml.bak.<unix_timestamp>` first.

### Overwrite behaviour

| State of `config.toml` | `--force` | Behaviour |
|---|---|---|
| absent | n/a | write directly |
| present | no | prompt "overwrite? backs up to .bak / cancel" |
| present | yes | back up + overwrite, no prompt |

`--non-interactive` mode: any path that would prompt is a hard error
exit `2` with a message indicating which step needed input.

## Errors

- `nvidia-smi` not on PATH → soft error, prompt for VRAM.
- llama.cpp release download fails (HTTP error, sha256 mismatch) →
  abort that branch, fall back to "enter a path manually".
- HuggingFace download fails (network, gated repo, disk full) → print
  the underlying error, ask user whether to retry or pick a different
  model.
- User cancels mid-wizard (Ctrl+C / explicit cancel choice) → exit
  cleanly without writing anything.

## Testing

### Unit tests (`tests/unit/test_install.py`)

All run against a `RecordedPrompter` test double that returns
pre-seeded answers and records the prompts seen, so the test reads
like a transcript.

1. **`tier`** — table-driven: `recommend_tier(None) == "unknown"`,
   `recommend_tier(24.0) == "L"`, boundary at 23.9 == "M", etc. Same
   for `tier_defaults` for each tier.
2. **`locate_llama_server`** — `tmp_path` with stubbed sibling layouts;
   verify each probe site is checked in order.
3. **`detect_vram_gb`** — monkeypatch `subprocess.run` to return
   scripted stdout / raise `FileNotFoundError`.
4. **`find_existing_model`** — create fake `.gguf` files in each
   candidate dir; assert correct hit / `None`.
5. **`present_models`** — verify recommended model is pre-selected;
   verify "may not fit" annotations appear correctly per tier;
   verify the "specify local file" path returns the right sentinel.
6. **`collect_allowed_dirs`** — feed sequence of inputs ending in
   empty; assert resolved, deduped, includes repo_root.
7. **`render_config_toml`** — golden-file test: assemble a known
   `WizardResult`, render, assert exact output. Re-parse with
   `tomllib` to confirm it loads back into a valid `Config`.
8. **`write_config`** — existing-file behaviour: writes `.bak.<ts>`
   then overwrites; absent-file behaviour: writes directly.
9. **`run_install_wizard`** end-to-end with `RecordedPrompter`,
   stubbed `subprocess.run`, and pre-seeded fake binaries / GGUFs.
   Asserts the resulting `config.toml` parses into a valid `Config`
   and has the expected field values.

### No live test

The wizard touches the network (downloads) and external tools
(`nvidia-smi`); a live test would be brittle and provide little
value over the unit suite. A manual smoke-test step in the user
guide replaces it.

## Files changed / added

**New:**
- `src/llama_agents/install.py`
- `tests/unit/test_install.py`
- `docs/install.md` — user-facing setup guide pointing at `llamactl init`.

**Modified:**
- `src/llama_agents/cli.py` — add `init` subcommand.
- `pyproject.toml` — add `huggingface_hub` as an optional dep
  (`[project.optional-dependencies] install = ["huggingface_hub>=0.24"]`).
  The wizard's lazy import will detect its absence and instruct the
  user to install it.
- `README.md` — replace any current "edit config.toml" instructions
  with `uv run llamactl init`.
- `CLAUDE.md` — add `install.py` to the module map.

## Open questions

None at design time. Two things deliberately deferred:

- **POSIX llama.cpp auto-download.** Wizard is Windows-first; on
  Linux/macOS it skips the download offer. A pre-built tarball URL
  matrix per OS is a small follow-up if needed.
- **Multi-GPU systems.** `detect_vram_gb` reads only GPU 0. Sufficient
  for the target workstation; multi-GPU sums / picks are not modelled.
