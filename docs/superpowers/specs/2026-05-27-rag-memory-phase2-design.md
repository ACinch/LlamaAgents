# RAG-Backed Memory Layer (Phase 2) — Design

**Status:** approved, ready for plan
**Date:** 2026-05-27
**Scope:** `src/llama_agents/` core + tools + agent loop

## Goals

Extend the planning/review cycle and the subagent return path with a local,
portable RAG-backed memory layer. Three concrete capabilities:

1. **Subagent → memory → parent.** A subagent's full final output is stored
   in per-run scratch. The orchestrator receives a short summary plus a
   `memory_handle`, not the full text. Long outputs never inflate the
   parent's context.
2. **Plan memory across runs.** Every accepted plan is appended to a
   persistent `plans` project. The planner and reviewer both get top-k
   similar past plans injected into their system prompt automatically.
3. **Overflow eviction.** When the parent's estimated message size crosses
   a configurable fraction of `ctx_size`, the loop evicts the oldest large
   tool results to scratch and replaces them with handle stubs. The model
   recalls on demand via a new `memory_recall` tool.

## Non-goals

- Code/codebase indexing (the existing RAG MCP server still does that if a
  user wants it).
- Cross-run learning beyond the plans project.
- Capturing run *outcomes* (success/failure) into plan records. The schema
  supports it via `metadata_json`; analysis is future RL work.
- Reviewer-subagent variant.
- Auto-restart of any subprocess.
- A tokenizer-accurate budget guard. We approximate with characters and
  refine using llama-server's reported `usage.prompt_tokens` opportunistically.
- Auto-deletion or compaction of the persistent plans project.
- Sharing the memory directory across machines.

## Portability stance

The previous approach considered piggybacking on the local RAG MCP server.
Rejected: that server is user-specific infrastructure. Phase 2 must work
out of the box for anyone who clones the repo. Decision: in-process
embeddings via `fastembed` (ONNX runtime, CPU, ~30 MB model), SQLite for
storage, brute-force cosine in numpy. No external services required.

## Architecture

```
                ┌──────────────────────────────────────────┐
                │              Runtime                     │
                │   builds MemoryStore, ToolRegistry,      │
                │   Agent factory                          │
                └───────────────┬──────────────────────────┘
                                │
                ┌───────────────┼─────────────────┐
                ▼               ▼                 ▼
      ┌──────────────────┐ ┌─────────────┐ ┌──────────────┐
      │   MemoryStore    │ │   Agent     │ │ ToolRegistry │
      │ (singleton per   │ │ plan+review │ │ (per agent;  │
      │  Runtime)        │ │ + main loop │ │  cloned)     │
      └────────┬─────────┘ └──────┬──────┘ └──────┬───────┘
               │                  │               │
     ┌─────────┼──────────┐       │               │
     ▼         ▼          ▼       │               ▼
  ┌──────┐ ┌──────┐ ┌──────────┐  │   ┌──────────────────────┐
  │Chunk │ │Embed │ │ SQLite   │  │   │ memory_recall (new)  │
  │ er   │ │ der  │ │ + numpy  │  │   │ subagent_spawn (mod) │
  └──────┘ └──────┘ └──────────┘  │   └──────────────────────┘
                                  ▼
                          (plan retrieval injects into
                           planner/reviewer system prompts)
```

### New module: `src/llama_agents/memory/`

| File | Responsibility |
|------|----------------|
| `__init__.py` | re-exports `MemoryStore`, `RecalledChunk`, `BlobMeta` |
| `store.py` | `MemoryStore` class — public interface, lifecycle |
| `chunker.py` | markdown-header chunking with overflow split (port of the RAG project's strategy, markdown only — no code-aware path) |
| `embedder.py` | thin wrapper around `fastembed.TextEmbedding`; lazy load; batch embed |
| `db.py` | SQLite schema, DAO, cosine search via numpy |

### `MemoryStore` public surface

```python
@dataclass
class RecalledChunk:
    blob_id: str
    chunk_idx: int
    text: str
    score: float
    title: str
    kind: str

@dataclass
class BlobMeta:
    id: str
    scope: str         # 'run' | 'plans'
    run_id: str | None
    kind: str
    title: str
    file_path: str
    created_at: str

class MemoryStore:
    def __init__(self, root: Path, *, embedder: Embedder | None = None,
                 retention_hours: int = 24) -> None: ...

    def start_run(self, run_id: str) -> None: ...
    async def end_run(self, run_id: str) -> None: ...

    async def store_plan(self, *, task: str, plan: str,
                         accepted_attempt: int,
                         run_id: str | None = None) -> str: ...

    async def store_blob(self, *, kind: str, title: str, body: str,
                         scope: Literal["run", "plans"] = "run",
                         run_id: str | None = None,
                         metadata: dict | None = None) -> str: ...

    async def recall(self, query: str, *,
                     scope: Literal["run", "plans", "all"] = "all",
                     run_id: str | None = None,
                     handle: str | None = None,
                     k: int = 5,
                     min_score: float | None = None) -> list[RecalledChunk]: ...

    async def list_handles(self, *, scope: Literal["run", "plans"],
                           run_id: str | None = None) -> list[BlobMeta]: ...

    async def gc_expired(self) -> int:
        """Remove run scratch older than retention_hours. Returns count."""

    async def close(self) -> None: ...
```

The interface is what the rest of the system depends on. The fastembed +
SQLite implementation lives behind it. When phase 3 wants a different
backend (or the Rust rewrite swaps in something else) only the
implementation changes.

### Inert variant

When `memory.enabled = false` (or fastembed import fails), `Runtime`
constructs `_InertMemoryStore`. Every method returns empty/no-op:
`recall` returns `[]`, writes return a stub id and discard the body. Call
sites in `agent.py` and `subagent.py` never branch on
`memory.enabled` — they always call the same interface.

### Storage layout

Under the first `allowed_dirs` entry:

```
.llama_agents/memory/
  index.sqlite              # single file; WAL mode; tables: blobs, chunks
  runs/<run_id>/
    <blob_id>.md            # raw blob text — readable via fs_read_file
  plans/
    <blob_id>.md            # persistent across runs
```

The raw markdown files exist for two reasons:
- The model can `fs_read_file` a blob in full when chunk-level recall is
  insufficient.
- Debugging and inspection are trivial — no special tooling needed.

### SQLite schema

```sql
PRAGMA journal_mode = WAL;

CREATE TABLE blobs (
  id            TEXT PRIMARY KEY,    -- ulid
  scope         TEXT NOT NULL,       -- 'run' | 'plans'
  run_id        TEXT,                -- nullable for plans
  kind          TEXT NOT NULL,       -- 'plan' | 'subagent_output' | 'evicted_tool' | 'user'
  title         TEXT NOT NULL,
  file_path     TEXT NOT NULL,
  metadata_json TEXT,
  created_at    TEXT NOT NULL
);
CREATE INDEX idx_blobs_scope_run ON blobs(scope, run_id);

CREATE TABLE chunks (
  id        TEXT PRIMARY KEY,
  blob_id   TEXT NOT NULL REFERENCES blobs(id) ON DELETE CASCADE,
  chunk_idx INTEGER NOT NULL,
  text      TEXT NOT NULL,
  embedding BLOB NOT NULL             -- float32 array, 384 dims (BGE-small)
);
CREATE INDEX idx_chunks_blob ON chunks(blob_id);
```

Cosine search loads the in-scope embeddings (filtered in SQL by
`scope`/`run_id`/`blob_id`) into a numpy matrix, normalizes the query
vector once, computes scores with a single matmul, returns top-k. At
hundreds–low thousands of chunks this is sub-millisecond and avoids any
vector-index dependency.

## Data flow

### Subagent spawn — new return shape

`SpawnSubagentTool.invoke`:

1. Run the subagent loop as today; collect `final_text`,
   `iterations`, `tool_calls`.
2. If `len(final_text) <= subagent_inline_threshold_chars` (default 2000):
   return as today: `{result, iterations, tool_calls}`. No memory write.
3. Otherwise:
   - `blob_id = await memory.store_blob(kind="subagent_output",
     scope="run", run_id=..., title=f"subagent: {task[:60]}",
     body=final_text, metadata={"task": task, "iterations": ...,
     "tool_calls": ...})`.
   - One-shot summarizer call:
     - `messages = [{role: system, content: SUMMARIZER_SYS},
       {role: user, content: f"TASK:\n{task}\n\nOUTPUT:\n{final_text[:8000]}"}]`
     - `tools=[], temperature=0.0, reasoning_budget_tokens=0`.
   - Return `{summary, memory_handle: blob_id, result_bytes,
     iterations, tool_calls}`.

The summarizer input is truncated at 8000 characters to keep the
summarization itself cheap and bounded.

The orchestrator sees the summary inline. To dig into specifics it calls
`memory_recall(handle=blob_id, query=...)`.

### Plan retrieval and storage

Inside `Agent._plan_and_review`:

- **Before the first planner call:**

  ```python
  prior = await self._memory.recall(
      query=user_prompt, scope="plans",
      k=opts.plan_recall_k, min_score=opts.plan_recall_threshold,
  )
  if prior:
      injected = "\n\nPRIOR ACCEPTED PLANS FOR SIMILAR TASKS:\n" + \
                 "\n---\n".join(c.text for c in prior)
      planner_system_effective = planner_system + injected
      reviewer_system_effective = reviewer_system + injected
  ```

  Both planner and reviewer get the same injected context. The reviewer
  becomes able to flag "this plan repeats an approach that was previously
  accepted but then ran out of iterations".

- **On `PlanAccepted`:** fire-and-forget `MemoryStore.store_plan(...)`.
  Failures are logged to stderr but never block the loop. Stored as:

  ```markdown
  # Plan for: <first 80 chars of task>

  ## Task
  <user_prompt>

  ## Accepted on attempt N

  ## Plan
  <plan body>
  ```

### Overflow eviction in the main loop

After each tool result is appended to `self.messages`, the loop calls
`self._maybe_evict()`:

```
estimated_tokens = _estimate_chars(self.messages) / EST_CHARS_PER_TOKEN
budget = ctx_size * (evict_threshold_pct / 100)
if estimated_tokens < budget: return

candidates = [
    m for m in self.messages[:-4]            # protect last 4 turns
    if m["role"] == "tool"
       and len(m["content"]) > evict_tool_result_min_chars
]

for msg in candidates:                       # oldest first by iteration order
    body = msg["content"]
    blob_id = await memory.store_blob(
        kind="evicted_tool", scope="run", run_id=self._run_id,
        title=f"tool result @ turn {self._turn_idx(msg)}",
        body=body, metadata={"tool_call_id": msg.get("tool_call_id")},
    )
    msg["content"] = (
        f"[evicted to memory — use memory_recall("
        f"handle=\"{blob_id}\", query=...) to retrieve. "
        f"Original size: {len(body)} chars.]"
    )
    yield MemoryEvicted(blob_id=blob_id, turn=..., bytes_freed=len(body)-len(msg["content"]))
    estimated_tokens -= (len(body) - len(msg["content"])) / EST_CHARS_PER_TOKEN
    if estimated_tokens < ctx_size * 0.5:
        break
```

`EST_CHARS_PER_TOKEN` defaults to 3.5 (conservative). When the previous
chat response contained `usage.prompt_tokens` and we know the char count
that produced it, we update the running ratio to be more accurate. No
tokenizer dependency added.

Eviction never removes a message — only rewrites the `content` of tool
messages. `tool_call_id` references stay intact, OpenAI API stays happy.

### `memory_recall` tool

```python
class MemoryRecallTool(Tool):
    name = "memory_recall"
    description = (
        "Retrieve previously-stored content from this run's scratch memory "
        "and past plans. Use this when you see [evicted to memory ...] in "
        "earlier tool results, or to look up a subagent's full output via "
        "its memory_handle."
    )
    json_schema = {
        "type": "object",
        "properties": {
            "query":  {"type": "string"},
            "handle": {"type": "string", "description":
                       "Optional. If provided, restricts results to chunks "
                       "from this blob_id."},
            "k":      {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }
    async def invoke(self, args):
        chunks = await self._store.recall(
            query=args["query"], handle=args.get("handle"),
            scope="all", run_id=self._run_id, k=args.get("k", 5),
        )
        return {"chunks": [
            {"text": c.text, "blob_id": c.blob_id, "chunk_idx": c.chunk_idx,
             "score": c.score, "title": c.title, "kind": c.kind}
            for c in chunks
        ]}
```

Added to the default registry so both orchestrators and subagents have it.
Subagents need it because parents will pass `memory_handle` values in
subtask descriptions ("summarize what's in memory_handle=01J...").

## Events

Two new event types in `events.py`:

```python
@dataclass
class MemoryStored(Event):
    blob_id: str
    kind: str
    scope: str
    bytes_: int

@dataclass
class MemoryEvicted(Event):
    blob_id: str
    turn: int
    bytes_freed: int
```

The CLI renders them as dim status lines:
- `◦ stored plan → mem:01J...`
- `◦ evicted tool result → -8.2 KB`

HTTP service forwards them on the SSE stream alongside existing events.

## Config

Additions to `config.toml`:

```toml
[memory]
enabled = true
root = ".llama_agents/memory"
embedding_model = "BAAI/bge-small-en-v1.5"
chunk_size = 1500
chunk_overlap = 150
plan_recall_k = 3
plan_recall_threshold = 0.5
subagent_inline_threshold_chars = 2000
subagent_summary_max_tokens = 400
evict_threshold_pct = 70
evict_tool_result_min_chars = 4000
scratch_retention_hours = 24       # 0 = delete on run end; -1 = keep forever
```

`scratch_retention_hours` is the knob that enables the future RL hook —
set to `-1` to keep all per-run scratch indefinitely for offline analysis.

## Lifecycle

- `Runtime.new_agent()` generates a `run_id` (ulid) for each top-level
  call to `Agent.run()` on an orchestrator. Subagents inherit their
  parent's `run_id` so they share scratch.
- `MemoryStore.start_run(run_id)` is called at run start; creates
  `runs/<run_id>/` lazily on first write.
- `MemoryStore.end_run(run_id)` is called when the orchestrator's run
  completes (success, error, cancel). With default retention, this is a
  no-op; the directory is GC'd later. With `retention=0`, scratch is
  deleted immediately. With `retention=-1`, never.
- `llamactl memory gc` triggers `MemoryStore.gc_expired()` for one-shot
  cleanup; also runs automatically on `Runtime` startup.

## Error handling

| Failure | Behavior |
|---------|----------|
| `fastembed` import fails | `Runtime` falls back to `_InertMemoryStore`, logs a warning to stderr. Loop runs normally. |
| Embedder model download fails (offline first run) | `MemoryStoreError` on first write; bubbles up to a single startup-time error message. Setting `enabled = false` avoids it. |
| `embed()` fails mid-run | Write paths catch and log; reads return `[]`. Loop continues. |
| `recall` returns empty | Tool returns `{"chunks": []}`. Model proceeds. Not an error. |
| Plan write fails | Logged, no impact on loop. |
| Eviction write fails | No content rewrite happens; the loop continues. Next chat call may hit HTTP 400 (same as today; no regression). |
| Concurrent subagent writes | SQLite WAL + per-store `asyncio.Lock` around writes. Reads unlocked. |
| Stale `memory_handle` reference | `memory_recall` returns `[]`. Model treats it as "no data". |

## Sandboxing

The memory root is **inside** the sandbox (under the first
`allowed_dirs` entry). The agent can `fs_read_file` raw blob markdown
when chunk-level recall isn't enough. The directory is added to the
project's default `.gitignore` to prevent accidental commits of run
scratch.

`memory_recall` reads do not go through filesystem tools — they read
SQLite directly via `MemoryStore`. They are not restricted by the
sandbox path checks because they only return content the agent
previously authored. Raw-file reads do go through `fs_read_file` and
honor the sandbox.

## Testing

| Test file | Scope |
|-----------|-------|
| `tests/unit/test_memory_chunker.py` | Markdown header chunking, small/large sections, overlap correctness, edge cases (no headers, only headers) |
| `tests/unit/test_memory_store.py` | `store_blob` + `recall` round-trip with a deterministic fake embedder (hash-based vectors). No fastembed import. |
| `tests/unit/test_memory_eviction.py` | Fake `_ClientLike` returning canned tool calls; assert tool message contents get rewritten and `MemoryEvicted` events fire in expected order |
| `tests/unit/test_plan_retrieval.py` | Stub `MemoryStore.recall`; assert injected text appears in planner/reviewer system prompts |
| `tests/unit/test_subagent_summary_return.py` | Mock client + memory store; large `final_text` → assert handle + summary return; small `final_text` → assert today's shape |
| `tests/unit/test_memory_inert.py` | `enabled=false` build of `Runtime`; assert no exceptions, empty recalls |
| `tests/live/test_memory_e2e.py` (marked `live`) | Real fastembed, real llama-server; small run that triggers all three paths (subagent overflow, plan retrieval, eviction) |

The fake embedder used in unit tests produces deterministic vectors from
a hash of the input text, so cosine results are predictable. This avoids
the 30 MB fastembed model download in CI and makes the tests fast and
hermetic.

## Dependency additions

In `pyproject.toml`:

- `fastembed >= 0.4` — runtime
- `numpy` — pinned to a minimum; already transitive via fastembed but
  declare explicitly because `db.py` uses it directly

No new dev dependencies.

## Backwards compatibility

With `memory.enabled = false`:
- `Runtime` builds `_InertMemoryStore`
- All call sites continue to call the same `MemoryStore` interface
- All recalls return `[]`; all writes return stub ids and are discarded
- No event types are emitted

The agent loop's external shape (events, options, return values) is
unchanged except for the new event types and the new `memory_handle`
field on subagent return for large outputs. CLI and HTTP surfaces gain
new event renderings but no breaking schema changes.

## Out-of-scope follow-ups (future phases)

- Capture run outcome (success/failure/cancel + final assistant text)
  into the plan record; use it to bias retrieval and inform the reviewer.
- A reviewer-subagent variant: have a fresh-context subagent review the
  proposed plan instead of the same-model self-review.
- Reinforcement-learning extension: with `scratch_retention_hours = -1`,
  collect run traces and use them to fine-tune planning or to score plan
  retrieval. Hook is already in place; analysis tooling is not.
- Plan project compaction: once persistent plans grow past several
  thousand entries, add deduplication or summarization to keep retrieval
  quality high.
- Multi-modal blobs (images, binary tool results).
- Cross-machine sync of the memory directory.

## Affected files

New:
- `src/llama_agents/memory/__init__.py`
- `src/llama_agents/memory/store.py`
- `src/llama_agents/memory/chunker.py`
- `src/llama_agents/memory/embedder.py`
- `src/llama_agents/memory/db.py`
- `src/llama_agents/tools/builtin/memory.py` (`MemoryRecallTool`)
- `tests/unit/test_memory_chunker.py`
- `tests/unit/test_memory_store.py`
- `tests/unit/test_memory_eviction.py`
- `tests/unit/test_plan_retrieval.py`
- `tests/unit/test_subagent_summary_return.py`
- `tests/unit/test_memory_inert.py`
- `tests/live/test_memory_e2e.py`
- `docs/memory.md`

Modified:
- `src/llama_agents/agent.py` — plan retrieval injection, `_maybe_evict`,
  `run_id` plumbing
- `src/llama_agents/runtime.py` — build `MemoryStore`, pass to tools,
  register `MemoryRecallTool`
- `src/llama_agents/config.py` — `[memory]` section, pydantic model
- `src/llama_agents/events.py` — `MemoryStored`, `MemoryEvicted`
- `src/llama_agents/tools/builtin/subagent.py` — summary + handle return
  path, summarizer call
- `src/llama_agents/cli.py` — render new events
- `src/llama_agents/http_app.py` — forward new events on SSE
- `pyproject.toml` — fastembed, numpy
- `config.toml` — `[memory]` defaults
- `README.md` — new "Memory" section
- `CLAUDE.md` — strike "No RAG memory" limitation; reference this design
- `.gitignore` — `.llama_agents/memory/`
