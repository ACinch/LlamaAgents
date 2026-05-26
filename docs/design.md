# llama-agents — Design

**Date:** 2026-05-25
**Project root:** `D:\repos\LLM\llama-agents`
**Status:** Spec — awaiting implementation plan

## 1. Purpose

`llama-agents` (working package name `llama_agents`, working CLI name `llamactl`) is a Python
orchestration layer around a locally running `llama-server.exe` (llama.cpp). It turns the
underlying model into a tool-using agent with:

- **Built-in tools**: filesystem (read/write/edit/list), allowlisted shell execution, and
  subagent spawning.
- **MCP-bridged tools**: any stdio MCP server listed in config (RAG to start) is auto-spawned
  and its tools are exposed to the agent.
- **Two surfaces today, one tomorrow**: a CLI/REPL and an HTTP service, both thin wrappers
  over the same async core. A WebUI is planned to sit on top of the HTTP service.

The orchestration loop is ReAct-style: prompt → model decides whether to call a tool →
orchestrator dispatches → result returns to model → loop until the model produces a final
assistant message or `max_iterations` is hit.

## 2. Non-goals

- Not a general MCP server. (We are an MCP *client* that bridges remote tools.)
- Not a model server. (llama-server.exe is the model server; we just talk to it.)
- Not a multi-tenant service. v1 is single-user, local.
- Not a replacement for the existing `D:\repos\LLM` Ollama-fronted RAG MCP server — we
  consume it.

## 3. High-level architecture

```
              ┌──────────────────────────────────────────────┐
              │                llamactl core                 │
              │                                              │
   CLI ──────►│  ┌──────────────┐    ┌────────────────────┐  │
              │  │ Agent loop   │◄──►│   Tool registry    │  │
   HTTP ─────►│  │ (ReAct via   │    │ ┌────────────────┐ │  │
  (+WebUI     │  │  function    │    │ │ Built-in tools │ │  │
   later)     │  │  calling)    │    │ │ - fs.*         │ │  │
              │  └──────┬───────┘    │ │ - shell.run    │ │  │
              │         │            │ │ - subagent.*   │ │  │
              │         ▼            │ └────────────────┘ │  │
              │  ┌──────────────┐    │ ┌────────────────┐ │  │
              │  │ Llama client │    │ │ MCP bridge     │─┼──┼──► spawned MCP servers
              │  │ (OpenAI API) │    │ │ - rag__*       │ │  │      (RAG today, more later)
              │  └──────┬───────┘    │ │ - …            │ │  │
              │         │            │ └────────────────┘ │  │
              │         ▼            └────────────────────┘  │
              └─────────│────────────────────────────────────┘
                        ▼
                 llama-server.exe  (single instance, OpenAI-compatible /v1/chat/completions)
```

### 3.1 Modules

1. **`llama_client`** — async wrapper over llama-server's `/v1/chat/completions`. Handles
   the OpenAI `tools=[...]` parameter, streaming, retries, and (optionally) lifecycle of
   the llama-server subprocess. Knows nothing about agents.
2. **`tool_registry`** — registers tools (built-in or MCP-bridged). Produces the
   OpenAI-format tool schema list, dispatches `tool_call` results back to invocations.
3. **`tools.builtin`** — `fs.*`, `shell.run`, `subagent.*`. Each tool is a class with a
   JSON schema and an async `invoke()`.
4. **`tools.mcp_bridge`** — uses the official Python `mcp` SDK to spawn stdio MCP servers
   from config, lists their tools, and exposes them in the registry under a `<name>__<tool>`
   prefix (e.g., `rag__rag_query`).
5. **`agent`** — the loop. Owns conversation state, emits an event stream, supports
   cancellation. Surfaces consume the event stream.

### 3.2 Future modularity for a Rust core

The eventual plan is to reimplement the performance-sensitive pieces (loop, tool registry,
llama client) in Rust, exposed to Python via a `maturin`-built extension — same pattern as
RGPdb. The Python layer designs around clean interfaces between modules so that swap is
mechanical:

- `llama_client`, `tool_registry`, and `agent` are pure-Python classes with narrow,
  typed interfaces.
- Tools and surfaces only depend on those interfaces, not on internals.
- No global state; no `import`-time side effects.

## 4. Agent loop

```python
messages = [system_prompt, user_prompt]
for i in range(max_iterations):
    response = await llama_client.chat(messages, tools=registry.schemas())
    messages.append(response.assistant_message)

    if not response.tool_calls:
        return response.content  # done

    for call in response.tool_calls:
        result = await registry.invoke(call.name, call.args)
        messages.append(tool_result_message(call.id, result))

raise MaxIterationsExceeded
```

### 4.1 Streaming & events

The loop yields events (`assistant_chunk`, `tool_call_start`, `tool_call_result`,
`error`, `done`). CLI prints them live; HTTP forwards as Server-Sent Events; future
WebUI consumes the same SSE stream. The internal API is uniform — surfaces only differ
in rendering.

### 4.2 Tool errors are recoverable

A tool failure (file not found, shell non-zero exit, RAG timeout, MCP crash) becomes a
`tool` message with `{"error": "..."}` content. The model observes it and adapts. Only
orchestration-level errors (llama-server unreachable, max iterations, cancellation)
bubble out of the loop.

### 4.3 Cancellation

Long loops must be killable from the CLI (Ctrl-C) or HTTP (client disconnect). The loop
checks an `asyncio.Event` between iterations and during tool invocation; in-flight HTTP
requests and subprocesses receive a cancel signal.

### 4.4 Persistence

Each `Agent` instance is one conversation. CLI holds one per session. HTTP service
keeps them in an in-memory `dict[session_id, Agent]`. No database in v1.

### 4.5 Token budget

The loop tracks an approximate running token count. When usage exceeds
`agent.token_budget_pct` (default 0.8) of the model's context window, the loop emits a
warning event and refuses new tool calls (forces the model to produce a final answer).
Smarter compaction is future work.

## 5. Tool catalog

### 5.1 Built-in

**Filesystem (sandboxed to `sandbox.allowed_dirs`):**

- `fs.read_file(path)`
- `fs.write_file(path, content)`
- `fs.edit_file(path, find, replace)` — exact-string replacement, fails if `find`
  is missing or non-unique
- `fs.list_files(glob_pattern, base?)` — glob inside allowed dirs

**Shell:**

- `shell.run(command: list[str], cwd?: str, timeout?: int)` — first token must be in
  `sandbox.shell_allowlist`; never `shell=True`; cwd must be inside an allowed dir.

**Subagents:**

- `subagent.spawn(task, system_prompt?, allowed_tools?, max_iterations=20)`
  → `{result, iterations, tool_calls}`

### 5.2 MCP-bridged

Each `[[mcp_servers]]` block in config produces tools named `<name>__<tool>`. The bridge
calls `tools/list` once at startup to discover tool schemas, and `tools/call` on
invocation. The RAG server gives us:

- `rag__rag_query(query, project?, limit=10, threshold=0.2)`
- `rag__rag_ingest(path, project?, recursive=true, patterns?, respectGitignore=true)`
- `rag__rag_delete(path? | project?)`
- `rag__rag_list_projects()`
- `rag__rag_status()`

## 6. Subagent semantics (v1)

- **Synchronous to the parent.** Parent's tool call blocks until subagent finishes or
  errors. No parallel subagents in v1.
- **Same llama-server, fresh conversation.** Subagent has its own messages list, its
  own system prompt, its own (subset of) tools. Shares the underlying server.
- **No nested spawning by default.** Subagents can't spawn subagents unless the parent
  explicitly passes `allowed_tools` that includes `subagent.spawn`. Prevents runaway
  trees.
- **Sandbox is inherited.** Allowed dirs and shell allowlist come from the parent
  config; a subagent cannot escape sandboxing the parent had.
- **Result is a string.** The subagent's final assistant message becomes the parent's
  tool result. No structured handoff in v1.
- **Concurrency cap.** `agent.max_concurrent_agents` (default 5) bounds the total
  active agent count (parent + descendants) via a semaphore. In v1 this caps recursion
  depth; once parallel spawning lands, it also caps parallelism. Hitting the cap
  raises `AgentLimitExceeded`, which the loop surfaces to the model as a tool error.

### 6.1 Designed-for-later

- **Parallel subagents** via `asyncio.gather` — interface is already async.
- **llama.cpp slot-based shared KV cache** for context-preserving subagents — the
  `llama_client` will accept a `slot_id` parameter; the registry will track slot
  assignments. Signatures should make room for this.
- **Per-subagent model override** — `model` field on spawn for cases with multiple
  llama-server processes.

## 7. Configuration

Single TOML file at `D:\repos\LLM\llama-agents\config.toml`. Path overridable via
`--config <path>` or `LLAMA_AGENTS_CONFIG` env var.

```toml
[llama]
server_url    = "http://127.0.0.1:8080"
model         = "qwen3-coder-30b"        # informational
auto_spawn    = true                      # spawn if /health unreachable
kill_on_exit  = false                     # only kills if we spawned it
server_bin    = "D:/repos/LLM/llamacpp-bin/llama-server.exe"
model_path    = "D:/repos/LLM/GGUF/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"
ngl           = 999
ctx_size      = 32768
startup_timeout_seconds = 60

[agent]
max_iterations         = 20
max_concurrent_agents  = 5
token_budget_pct       = 0.8

[sandbox]
allowed_dirs    = ["D:/repos/LLM/llama-agents", "D:/repos/ACinch/ac-refresh"]
shell_allowlist = ["git", "cargo", "pnpm", "pytest", "python", "node"]

[http]
host = "127.0.0.1"
port = 9000

[[mcp_servers]]
name    = "rag"
command = "node"
args    = ["D:/repos/LLM/rag/dist/index.js"]
```

### 7.1 llama-server lifecycle

On startup, the orchestrator probes `<server_url>/health`. If unreachable and
`llama.auto_spawn = true`, it spawns `server_bin` with the configured arguments and
waits up to `startup_timeout_seconds` for readiness (looking for "HTTP server listening"
on stderr, with a fallback `/health` poll). On shutdown, the orchestrator kills the
subprocess only if `kill_on_exit = true` **and** the orchestrator was the one that
spawned it. A server that was already running when the orchestrator started is never
killed.

### 7.2 Sandbox enforcement

- **Filesystem**: every `path` argument is converted to an absolute path via
  `Path(p).resolve(strict=False)` (resolves symlinks). The result must satisfy
  `is_relative_to(d)` for some `d` in `allowed_dirs`. Otherwise → `SandboxViolation`.
- **Shell**: the first element of the `command` list must be in `shell_allowlist`.
  `subprocess.run(..., shell=False)` always. `cwd`, if supplied, is checked the same
  way as filesystem paths.
- **MCP servers**: trusted as user-configured code; their output is treated like any
  other tool output, but they are not themselves sandboxed.

## 8. Error taxonomy

| Error                       | Source           | Handling                                                |
|-----------------------------|------------------|---------------------------------------------------------|
| `LlamaUnreachable`          | `llama_client`   | bubble — surface returns 503 / CLI exits non-zero       |
| `LlamaProtocolError`        | `llama_client`   | one automatic retry, then bubble                        |
| `ToolNotFound`              | `tool_registry`  | fed back to model as a tool error                       |
| `ToolValidationError`       | tool invoke      | fed back to model as a tool error                       |
| `SandboxViolation`          | tool invoke      | fed back to model as a tool error                       |
| `MCPServerCrashed`          | `mcp_bridge`     | one auto-restart; if it crashes again, bridged tools    |
|                             |                  | become unavailable for the session                      |
| `MaxIterationsExceeded`     | `agent`          | bubble — surface returns partial transcript             |
| `AgentLimitExceeded`        | `subagent.spawn` | fed back to model as a tool error                       |
| `Cancelled`                 | `agent`          | bubble — surface acks cleanly                           |

## 9. Testing

- **Unit**: each tool tested without llama-server. Sandbox checks tested with
  adversarial paths (`..`, symlinks, absolute paths outside allowed dirs, mixed
  separators on Windows).
- **Integration with a mock llama**: a fake `llama_client` returns scripted
  `tool_calls`. End-to-end coverage of loop termination, recoverable tool errors,
  max-iterations cap, cancellation, subagent recursion, and the concurrency cap.
- **Live smoke tests**: hit a real llama-server with the actual Qwen3-Coder model.
  Marked `@pytest.mark.live`, skipped by default in CI.

## 10. Project layout

```
D:\repos\LLM\llama-agents\
├── pyproject.toml                  # uv-managed, Python 3.12+
├── config.toml                     # sample/default config
├── README.md
├── src/llama_agents/
│   ├── __init__.py
│   ├── config.py                   # TOML loader + pydantic dataclasses
│   ├── llama_client.py             # async OpenAI-compat client + lifecycle
│   ├── tool_registry.py
│   ├── agent.py                    # loop + events + cancellation
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── builtin/
│   │   │   ├── __init__.py
│   │   │   ├── fs.py
│   │   │   ├── shell.py
│   │   │   └── subagent.py
│   │   └── mcp_bridge.py
│   ├── cli.py                      # typer entrypoint
│   └── http_app.py                 # FastAPI app (SSE for streaming)
└── tests/
    ├── unit/
    ├── integration/
    └── live/
```

### 10.1 Dependencies

`httpx`, `pydantic`, stdlib `tomllib` (Python 3.11+), `mcp` (official Python SDK),
`fastapi`, `uvicorn`, `typer`, `rich`, `pytest`, `pytest-asyncio`.

## 11. Future surfaces & extensions

- **WebUI** on top of the HTTP service — consumes the same SSE event stream the CLI
  uses. The HTTP layer is designed for this from day one (CORS configurable, SSE
  endpoints, session IDs in URLs).
- **Rust core via maturin** — replaces `llama_client`, `tool_registry`, and `agent`
  with a Python-importable Rust extension. Pure-Python tool implementations and
  surfaces are unaffected.
- **Parallel + slot-based subagents** — flips synchronous spawn into
  `asyncio.gather`, threads `slot_id` through the llama client.
- **Persistent sessions** — back the HTTP session store with SQLite when in-memory
  is no longer sufficient.
