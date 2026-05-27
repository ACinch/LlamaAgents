# Memory layer

llama-agents ships with a local RAG-backed memory system that:

- stores accepted plans across runs, and retrieves similar past plans
  for the planner and reviewer;
- offloads large subagent outputs to memory, returning a summary plus a
  handle the orchestrator can recall on demand;
- evicts old tool results from the context window when it fills, with
  the model retrieving on demand.

## On-disk layout

`.llama_agents/memory/` under the first allowed_dirs entry:

- `index.sqlite` — blob and chunk metadata, embeddings
- `runs/<run_id>/<blob_id>.md` — per-run scratch (subagent output, evicted tool results)
- `plans/<blob_id>.md` — persistent across runs

## Embeddings

We use `fastembed` with `BAAI/bge-small-en-v1.5` (384-dim ONNX, ~30 MB).
First import downloads to `~/.cache/fastembed/` and takes 10-30 s on a
fresh machine. Subsequent runs load instantly.

To disable the memory layer entirely:

```toml
[memory]
enabled = false
```

## Tools

The agent gets one new built-in tool:

- **memory_recall(query, handle?, k?)** — search this run's scratch +
  persistent plans. When `handle` is set, restricts to chunks of that
  blob.

## Lifecycle

- A `run_id` is generated for every top-level `Agent.run()` call.
  Subagents inherit it; the whole tree shares scratch.
- After a run, scratch is retained for `scratch_retention_hours` hours
  (default 24, `0` = delete immediately, `-1` = keep forever).
- Persistent plans are never auto-deleted.

## Tuning

- `evict_threshold_pct` — when estimated context use crosses this %, old
  large tool results are evicted to memory. Default 70.
- `evict_tool_result_min_chars` — never evicts tiny results.
- `plan_recall_k` / `plan_recall_threshold` — how many past plans to
  inject and how similar they must be.
- `subagent_inline_threshold_chars` — subagent outputs below this stay
  inline; above this go to memory.
