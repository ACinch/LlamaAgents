# llama-agents

Local orchestration layer around llama.cpp. Turns a running
`llama-server.exe` into a tool-using agent with filesystem access,
allowlisted shell execution, subagent spawning, and MCP-bridged external
tools — driven through a CLI or an HTTP/SSE service.

> If you are Claude Code (or another AI assistant) reading this repo,
> open [`CLAUDE.md`](CLAUDE.md) first. It has the architecture map and
> the non-obvious invariants you need before editing.

## Highlights

- **Plan + self-review baked into every orchestrator turn.** Before the
  main tool loop runs, the agent produces a numbered plan and a strict
  reviewer either accepts it or rejects with feedback and the planner
  iterates. Subagents skip planning to avoid recursion. Toggle via
  `AgentRunOptions.skip_planning`.
- **Subagent fan-out** with a global concurrency cap, isolated
  per-agent tool registry, and tool-allowlist scoping.
- **MCP stdio bridge** — drop a server into `config.toml` and its tools
  appear in the agent's toolset as `<server>__<tool>`.
- **Sandboxed filesystem + allowlisted shell**, configurable per
  deployment.
- **Bounded reasoning** — per-turn cap on Qwen3/DeepSeek-R1 `<think>`
  blocks so the model can't fill the context window in a single
  monologue.

## Quickstart

## First-time setup

After installing dependencies, run:

```
uv run llamactl init
```

This interactive wizard detects your llama-server binary, recommends
a GGUF model sized to your GPU, downloads it if needed, and writes a
`config.toml`. See `docs/install.md` for details.

```powershell
# install (uv lives at %USERPROFILE%\AppData\Roaming\Python\Python314\Scripts\uv.exe on Windows)
uv sync --extra dev

# unit tests
uv run pytest tests/unit -q

# one-shot agent run (needs llama-server reachable or auto_spawn=true in config.toml)
$env:PYTHONIOENCODING = "utf-8"
uv run llamactl chat "Summarize what this repo does."

# HTTP service
uv run llamactl serve
```

## Web UI

When `llamactl serve` is running, open `http://127.0.0.1:9000/` in a
browser. The UI lets you submit jobs to the queue (file upload or
pasted text), watch them progress, click into a finished job to see
its event timeline, and view the active `config.toml`. See
`docs/web.md` for details.

## Configuration

Edit `config.toml`. The defaults target an RTX 4090 / 64 GB / Gen4 NVMe
workstation:

| Setting | Default | Why |
|---------|---------|-----|
| `llama.ctx_size` | 65536 | 64k per slot; ~6 GiB KV cache at parallel=2 |
| `llama.n_parallel` | 2 | Two concurrent slots → can run 2 subagents in parallel |
| `llama.ngl` | 999 | Offload all layers to GPU |
| `agent.max_iterations` | 20 | Main tool-loop turns (planning is separate) |
| `agent.max_concurrent_agents` | 5 | Subagent semaphore cap |
| `sandbox.allowed_dirs` | (list) | The only paths fs tools may touch |
| `sandbox.shell_allowlist` | (list) | argv[0] values shell_run will run |

If you change `ctx_size` or `n_parallel`, do the KV math first. On a
4090 with this 35B-MoE Q2_K_P model (KV ≈ 47 KiB/token/slot, ~10 GiB
VRAM headroom after weights):

| n_parallel | max ctx that fits | recommended |
|-----------:|------------------:|------------:|
| 4 | ~49k | 32768 |
| **2** | **~98k** | **65536 (current default)** |
| 1 | ~196k | 131072 |

## Examples

Three end-to-end examples in [`docs/examples/`](docs/examples/) with
matching runnable scripts in [`examples/`](examples/):

1. **Story from a folder of `.txt` files** — multi-pass writer with
   style analysis, outline, section-by-section drafting via subagents,
   editorial review, revision, polish.
2. **Security architect code review** — uses `git ls-files` to honor
   `.gitignore`, fans out per-domain reviews via subagents, writes the
   collated report to disk.
3. **Marketing suggestions from RAG** — grounds copy in actual RAG
   index content, refuses to invent unsupported claims.

## Memory

llama-agents has a built-in local RAG-backed memory system:

- accepted plans are stored across runs and the planner/reviewer get top-k
  similar past plans injected into their prompts;
- large subagent outputs are written to memory and returned as a summary +
  handle, keeping the orchestrator's context tight;
- old tool results are evicted to memory when the context window fills.

Storage lives at `.llama_agents/memory/` (SQLite + per-run markdown files).
Embeddings use `fastembed` (BAAI/bge-small-en-v1.5, ONNX). See
[docs/memory.md](docs/memory.md) for details.

## Tools out of the box

| Tool | Description |
|------|-------------|
| `fs_read_file` | Read a UTF-8 file inside `allowed_dirs`. |
| `fs_write_file` | Write a UTF-8 file inside `allowed_dirs`. |
| `fs_edit_file` | Replace a uniquely-occurring substring in a file. |
| `fs_list_files` | Glob inside `allowed_dirs` (raw — does NOT honor `.gitignore`). |
| `shell_run` | Run an allowlisted command (`argv[0]` only is checked). |
| `subagent_spawn` | Run a focused subagent with a restricted toolset. |
| `<server>__<tool>` | Each MCP server's tools, auto-prefixed. |

## Project layout

```
llama-agents/
├── src/llama_agents/        # the package
├── tests/unit/              # fast hermetic tests
├── tests/live/              # tests needing a running llama-server
├── examples/                # runnable scripts
├── docs/
│   ├── design.md            # original spec
│   └── examples/            # one walkthrough per example
├── config.toml              # default configuration
├── .mcp.json                # MCP server config for Claude Code sessions
└── CLAUDE.md                # required reading for AI assistants
```

## See also

- [`CLAUDE.md`](CLAUDE.md) — architecture, invariants, dev workflow.
- [`docs/design.md`](docs/design.md) — original design spec.
- [`docs/examples/README.md`](docs/examples/README.md) — example index.
