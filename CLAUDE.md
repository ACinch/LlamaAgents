# CLAUDE.md — llama-agents

> Guidance for Claude Code (and other AI assistants) working inside this
> repository. Read this file end-to-end before touching the code or
> proposing changes.

## What this project is

`llama-agents` is a Python orchestration layer around a local
`llama-server.exe` (from llama.cpp). It turns a quantized LLM into a
tool-using **agent** that can:

- read/write files (sandboxed)
- run allowlisted shell commands
- spawn **subagents** with their own conversation, restricted toolset,
  and concurrency cap
- talk to **stdio MCP servers** (the bridge exposes their tools as
  first-class tools)
- be driven through a CLI (`llamactl`) or an HTTP service
  (FastAPI + SSE)

The core is a single async loop with pluggable tools. CLI and HTTP are
thin surfaces. The intent is that the Python core can later be swapped
for a Rust/maturin implementation without changing public interfaces.

## Hardware target

Authored for a single workstation: RTX 4090 (24 GB), 64 GB RAM, Gen4
NVMe. The defaults in `config.toml` are tuned for this profile:

- `ctx_size = 65536` — 64k context per slot.
- `n_parallel = 2` — two concurrent slots. With ~47 KiB/token KV cache,
  this fills ~6 GiB VRAM for KV cache, leaving headroom on a 4090.
- `reasoning_budget_tokens = 8000` (AgentRunOptions default) — caps
  per-turn `<think>` blocks so Qwen3/DeepSeek-R1-style models don't fill
  the context window in a single reasoning monologue.

If you change `ctx_size` or `n_parallel`, recompute the KV math before
shipping — see the table in `README.md`.

## Architecture in one screen

```
                        ┌────────────────────────┐
                        │       CLI surface      │   ← src/llama_agents/cli.py
                        │  HTTP surface (SSE)    │   ← src/llama_agents/http_app.py
                        └────────────┬───────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │        Runtime         │   ← src/llama_agents/runtime.py
                        │  (assembles all parts) │
                        └────────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
   ┌─────────────────┐    ┌─────────────────┐    ┌────────────────────┐
   │   LlamaClient   │    │      Agent      │    │   ToolRegistry     │
   │  +  Manager     │    │   loop + plan   │    │   (per-agent copy) │
   └────────┬────────┘    └────────┬────────┘    └─────────┬──────────┘
            │                      │                       │
            ▼                      │                       ▼
   ┌─────────────────┐             │           ┌──────────────────────┐
   │  llama-server   │             │           │  Built-in tools      │
   │    (subproc)    │             │           │  fs_* shell_run      │
   └─────────────────┘             │           │  subagent_spawn      │
                                   │           │  + MCP-bridged tools │
                                   ▼           └──────────────────────┘
                          tool calls × N
                          (incl. subagent_spawn)
```

### Module map (one responsibility per file)

| File | Job |
|------|-----|
| `errors.py` | Exception taxonomy. No other deps. |
| `config.py` | TOML + pydantic models. Reads file, no other I/O. |
| `sandbox.py` | Pure helpers `check_path`, `check_command`. |
| `events.py` | Dataclass event types yielded by `Agent.run()`. |
| `llama_client.py` | `LlamaClient` (HTTP) + `LlamaServerManager` (subproc). |
| `tools/base.py` | `Tool` ABC + JSON-schema contract. |
| `tools/registry.py` | Register / list / dispatch. Emits OpenAI tools array. |
| `tools/builtin/fs.py` | `fs_read_file`, `fs_write_file`, `fs_edit_file`, `fs_list_files`. |
| `tools/builtin/shell.py` | `shell_run` (allowlisted argv[0]). |
| `tools/builtin/subagent.py` | `subagent_spawn` with concurrency cap. |
| `tools/mcp_bridge.py` | Wraps stdio MCP server tools as `Tool` instances. Names get a `<server>__` prefix. |
| `agent.py` | The loop. Cancellation. **Planning + self-review phase**. |
| `runtime.py` | Factory: builds client, registry, bridge, agent factory. |
| `cli.py` | `llamactl chat` / `llamactl serve`. |
| `http_app.py` | FastAPI + SSE chat endpoint. |
| `queue/paths.py` | Atomic move + sweep helpers for queue folders. |
| `queue/worker.py` | `JobQueueWorker`: polls inbox/, runs jobs, writes outputs. |
| `install.py` | `llamactl init` wizard: VRAM detect, model pick, config render. |

### Agent.run() shape

Every call to `agent.run(prompt, opts)`:

1. **Planning phase** (only when `subagent_spawn` is in the registry and
   `opts.skip_planning is False`):
   - Calls the model as a planner → `PlanProposed(attempt, plan)` event.
   - Calls the model as a reviewer → `PlanReviewed(attempt, accepted,
     feedback)` event.
   - If REJECT, feeds reviewer's feedback back to planner. Loops up to
     `opts.max_planning_iterations` (default 3).
   - On ACCEPT (or budget exhaustion) emits `PlanAccepted(plan,
     attempts)` and the accepted plan becomes the preamble to the user
     prompt in the main loop.
   - Subagents skip planning because their registry doesn't contain
     `subagent_spawn` — this avoids unbounded recursion.

2. **Main tool loop** — standard OpenAI-compatible chat:
   - Send messages + tool schemas to llama-server.
   - On `tool_calls`, dispatch through registry, append result, loop.
   - On plain content, emit `AssistantChunk` + `Done(reason='finished')`.
   - Bounded by `opts.max_iterations` (default 20).

Events surfaced to callers:
`PlanProposed`, `PlanReviewed`, `PlanAccepted`, `ToolCallStart`,
`ToolCallResult`, `AssistantChunk`, `LoopError`, `Done`.

### Why the registry is cloned per agent

`Runtime.new_agent()` returns an Agent whose registry is a **deep-ish
clone** of the runtime's. When a subagent unregisters tools (e.g.
removes `subagent_spawn` to prevent recursion), the parent's registry
is unaffected. Don't undo this — the prior bug (`c2a0665`) was a real
incident.

## How to talk to llama-server

Endpoint: `POST /v1/chat/completions` (OpenAI-compatible). Request body
includes optional `reasoning_budget_tokens` (int) — see `LlamaClient.chat`.
Default in `AgentRunOptions` is 8000.

For Qwen3-style models on long structured prompts, the failure mode is
the model entering its `<think>` block and never leaving until the
context fills. The reasoning budget is the load-bearing mitigation.
Don't set it to `None` ("server default = unlimited") unless you've
proven the model is well-behaved on your workload.

## How tools work

A tool is a class with `name`, `description`, `json_schema`, and an
async `invoke(args)`. The registry serializes them into the OpenAI
`tools=[...]` array. JSON schema validation is currently shallow
(required-field presence only) — don't add complex validation here;
the model gets the schema and usually conforms.

To add a built-in tool:

1. New class in `src/llama_agents/tools/builtin/<area>.py`.
2. Register it in `runtime.py` alongside the existing built-ins.
3. Test in `tests/unit/test_<area>_tool.py` with an in-process invoke.

To add an MCP server, just append a `[[mcp_servers]]` block to
`config.toml`. The bridge auto-prefixes the tool names with
`<server-name>__` to avoid collisions.

## MCP configuration in this repo

This repo ships an `.mcp.json` pointing at the local RAG server
(`D:\repos\LLM\rag\dist\index.js`). That server is unrelated to
llama-agents itself; it's a Node MCP server that exposes:

- `rag_query` — vector search over indexed projects.
- `rag_ingest` — index new content.
- `rag_delete`, `rag_list_projects`, `rag_status`.

When bridged into llama-agents, these become `rag__rag_query`,
`rag__rag_ingest`, etc. The agent uses `rag__rag_query` in the
marketing example to ground suggestions in stored content.

If you're a Claude Code session running from this repo, the `.mcp.json`
already configures the RAG server for your use too — query it directly
to see what content is indexed.

## Sandboxing — the load-bearing invariant

- Filesystem tools resolve and verify every path against
  `config.sandbox.allowed_dirs` before touching disk.
- `shell_run` only executes commands whose `argv[0]` is in
  `config.sandbox.shell_allowlist`. **Arguments are NOT validated** —
  this is an intentional limitation called out in the
  `_security-review.md` for follow-up. Don't widen the allowlist
  carelessly.
- Subagents inherit the parent's sandbox by default but can be given a
  narrower `allowed_tools` whitelist via the `subagent_spawn` args.

## Dev workflow

```powershell
# install
uv sync --extra dev

# unit tests (fast, hermetic)
uv run pytest tests/unit -q

# live test (requires llama-server to be reachable or auto_spawn)
uv run pytest tests/live -m live -v

# one-shot agent run
$env:PYTHONIOENCODING = "utf-8"
uv run llamactl chat "your task here"

# HTTP service
uv run llamactl serve
```

`uv` may not be on PATH on Windows. It lives at
`%USERPROFILE%\AppData\Roaming\Python\Python314\Scripts\uv.exe`.
Prepend that to `$env:Path` before invoking, or call by full path.

`PYTHONIOENCODING=utf-8` is needed on Windows when stdout is captured
to a pipe/file, because the rich-formatted status glyphs (→, ✓, ✗)
crash cp1252. `cli.py` reconfigures stdout to utf-8 on import as a
belt-and-braces measure.

## Examples

See `docs/examples/` for walkthroughs and `examples/` for runnable
scripts.

| # | Pattern | Tools exercised |
|---|---------|-----------------|
| 1 | Story from `.txt` inspirations | fs_*, subagent_spawn (multi-pass: style → outline → drafting → editorial → polish) |
| 2 | Security architect code review | shell_run (`git ls-files`), subagent_spawn (per-domain), fs_write_file |
| 3 | Marketing suggestions from RAG | rag__rag_query × N |

Example 2 is the canonical demonstration of the orchestrator pattern —
auto-planning fires, the model produces a plan, the reviewer accepts,
and then subagents fan out per domain.

## Known limitations / future work

- **No token-budget guard.** The `token_budget_pct` config field exists
  but isn't enforced because we don't have a tokenizer locally. Adding
  this would let us pre-empt context overflow instead of bouncing off
  HTTP 400 from the server.
- **MCP server auto-restart.** `MCPServerCrashed` errors surface to the
  model but the bridge doesn't restart the subprocess.
- **`shell_run` argv validation.** As above — `argv[0]` only.
- **No recursion depth limit on subagents.** Subagents can't directly
  spawn more subagents (their registry has `subagent_spawn` removed),
  but if a tool indirectly triggers re-entry into the spawn semaphore,
  there's no depth tracking. Flagged in the security review.
- **RAG memory:** implemented in phase 2 — accepted plans persist
  across runs; large subagent outputs and overflow tool results are
  offloaded to a local SQLite + fastembed store; `memory_recall` retrieves
  them. See `docs/memory.md`.
- **Reviewer can confirm bad plans.** Self-review by the same model is
  cheap but prone to confirmation bias. A reviewer-subagent variant is
  on the table.
- **Queue worker has no priority / cron / control API.** Re-queuing is
  a manual file move; status is "look at the folders". A future
  `/queue/*` HTTP surface can be built later if needed.

## When you're editing this codebase

Read these in order before any non-trivial change:

1. `docs/design.md` — the original spec the implementation tracks.
2. `src/llama_agents/agent.py` — the loop is the heart.
3. `src/llama_agents/tools/builtin/subagent.py` — registry isolation is
   subtle.
4. `tests/unit/test_agent_loop.py` — the contract the loop is held to.

Commit style: conventional commits (`feat:`, `fix:`, `chore:`, `test:`,
`docs:`). One concern per commit. Don't bundle unrelated changes.

When you add behavior to the loop or change the event sequence, add a
test in `test_agent_loop.py` and update the **Agent.run() shape**
section of this file.
