# Multi-Turn Threads + Prompt History — Design

**Status:** Draft
**Date:** 2026-05-29
**Owner:** Maarten

## Goal

Replace the existing single-shot inbox/done/failed queue model with a
thread-centric model where every submission belongs to a durable
**thread**. Threads support:

- **Follow-up turns** — continuing a conversation with prior message
  history (including tool calls + results) hydrated into the agent's
  context.
- **Rerun-as-fork** — reproducing a past turn (with or without an
  edit) as the opening of a new thread that inherits the parent's
  RAG memory via ancestor traversal.
- **Browseable history** — a UI page and CLI subcommand that list
  threads, drill into a thread's turns, and surface every prior
  prompt + answer + event log.

The migration is one-way: there is no "single-shot" mode anymore.
One-turn threads still feel one-shot to the user and are auto-cleaned
the same way scratch memory is.

## Non-goals

- Thread search, full-text filtering, tag/label management. The list
  view sorts by `updated_at` and that's it for v1.
- Thread archive / soft-delete / prune UI. Threads can be deleted by
  `rm -rf threads/<id>/`; no command surface exposes this.
- Cross-thread merge ("squash turns 3-5 of thread A as a single
  context for thread B"). Out of scope.
- Editing arbitrary historical turn contents in place — the immutable
  "rerun forks, original preserved" model is the answer to that need.
- Authentication or any kind of per-user separation. Threads are a
  flat global namespace under the queue root.
- Server-side message-history compaction. We trust the existing memory
  eviction layer; threads with very long histories will hit the same
  eviction path as long single-turn runs do.

## User experience

### CLI

```
$ uv run llamactl chat "What does this repo do?"
[plan + run + tool events stream as today]
Thread: 8c9f2bd6e041a3b5708141d9c0e2

$ uv run llamactl chat --thread 8c9f "Pick a single example and walk through it."
[hydrates prior messages, runs the follow-up turn]
Thread: 8c9f2bd6e041a3b5708141d9c0e2 (turn 2)
```

```
$ uv run llamactl threads list
ID        Title                                              Turns  Updated   Status
8c9f2bd6  What does this repo do?                            2      30s ago   done
4e1a72fd  Story from txt files                               1      4m ago    done
6b3098ef  Security review of llama-agents                    1      12m ago   failed

$ uv run llamactl threads show 8c9f
[full thread rendered turn by turn]

$ uv run llamactl threads rerun 8c9f 1 --edit "What does this repo do, in one paragraph?"
Forked → thread 2d4c81ae...  (running turn 1)
```

### Web UI

Top nav: **Activity** / **Threads** / **Config**.

- `/activity` — the existing 4-panel dashboard, but each row is
  `<thread title> — turn N` instead of a free-floating filename. Two
  submit forms at the top still create new threads; the redirect now
  lands on `/threads/<new-id>`.
- `/threads` — vertical list of threads, newest activity first. Each
  row: title (60 chars of opening prompt), turn count, age, status
  dot for the latest turn, and a one-line excerpt of the latest
  assistant answer.
- `/threads/<id>` — full thread view. Inline-editable title at the
  top, then a vertical stack of turn blocks (prompt + collapsible
  events + final answer or error). Each turn carries a **Rerun**
  button. Below the last turn, a **Continue** textarea — disabled
  while any turn in the thread is `queued` or `processing`.

## Architecture

### Storage layout

Under `<queue_root>/threads/<thread_id>/`:

```
threads/<thread_id>/
├── meta.json              # ThreadMeta, see below
├── messages.jsonl         # running OpenAI-shaped conversation
└── turns/
    ├── 001/
    │   ├── prompt.md      # the user's submission for this turn
    │   ├── status         # single-word: queued|processing|done|failed
    │   ├── result.md      # present when status ∈ {done, failed}
    │   ├── events.jsonl   # full event stream
    │   └── error.txt      # present only when failed
    ├── 002/ ...
```

`<thread_id>` is a 24-character lowercase hex string (uuid4 first 24
chars; matches the existing run_id mint).

`meta.json` shape:

```json
{
    "id": "8c9f2bd6e041a3b5708141d9c0e2",
    "title": "What does this repo do?",
    "created_at": "2026-05-29T10:00:00+00:00",
    "updated_at": "2026-05-29T10:01:14+00:00",
    "current_turn": 2,
    "parent_thread_id": null,
    "parent_turn_idx": null
}
```

`messages.jsonl` holds one OpenAI chat message per line — the same
shape `Agent.messages` uses internally (`{"role": "user"|"assistant"|
"tool", "content": ..., "tool_calls": [...]}` etc.). On a follow-up
turn, every prior line is read and used to seed
`agent.messages` before the new user prompt is appended. On turn
completion, the new turn's messages are atomically appended (write to
`messages.jsonl.tmp`, then `os.replace`).

### Status state machine

```
queued → processing → done
                   ↘ failed
```

Transitions are atomic via the existing `os.replace` pattern: write
the new status to `status.tmp`, then replace `status`. Crash recovery:
on worker startup, any `status == processing` is rolled back to
`queued`. (Worker is the only writer of `processing`.)

The web's **Continue** button refuses (409) when the latest turn's
status is `queued` or `processing`. Rerun is always allowed (it forks,
doesn't mutate the source).

### Worker pickup

The queue worker's `_pick_one` is rewritten to scan
`threads/*/turns/*/status` instead of `inbox/`. Algorithm:

1. List all `threads/<id>/turns/<n>/status` paths.
2. Filter to those whose content is `queued`.
3. Sort by mtime ascending.
4. For each, attempt the atomic `queued → processing` claim. First
   one to win is returned.

This is O(threads × turns) per poll. At expected scale (≤ 1000
threads × ≤ 10 turns) it's a sub-millisecond directory walk per
2-second poll; if it ever matters, we add an `index.json` next to
`threads/` — not for v1.

The rest of the worker (the actual agent run, the JobResult writer,
the retry loop) is unchanged in behavior; the I/O destinations move
from `done/`/`failed/` to `turns/NNN/`.

### Hydration

Before running a turn whose number > 1:

1. Read `messages.jsonl`. Each line is a chat message.
2. Walk the ancestor chain (`meta.parent_thread_id` → parent →
   grandparent …, capped at depth 32) and collect each ancestor's
   `messages.jsonl` contents up through `parent_turn_idx` rows. (The
   parent's later turns are NOT inherited — only the lineage up to
   the fork point.)
3. The agent's `Agent.__init__` accepts a new optional
   `prior_messages: list[dict]` parameter. When non-empty, the agent
   skips its standard `messages = [system, user]` initialization and
   instead builds `messages = [system, ...prior_messages, {role:
   "user", content: turn_prompt}]`. The system prompt is still
   sourced from `opts.system_prompt`.
4. After the turn finishes, only the NEW messages produced in this
   turn (everything appended to `self.messages` past the seeded
   prefix length) are written to the thread's `messages.jsonl`. We
   never duplicate inherited messages on disk.

### Rerun-as-fork

When the user reruns turn N of thread A (optionally with an edited
prompt):

1. Mint a new thread id B.
2. Create `threads/B/` with `meta.json` populated:
   - `parent_thread_id = A`
   - `parent_turn_idx = N - 1`  (the fork point is *just before* the
     reran turn — turn 1 inherits no parent messages even from a
     rerun)
3. `messages.jsonl` of B starts empty. Hydration walks the ancestor
   chain at run time; we don't copy data on disk.
4. `turns/001/prompt.md` is the (possibly edited) prompt; `status =
   queued`.
5. The new thread shows up in the list immediately; once the worker
   picks it up, the turn runs with the hydrated history.

The original thread A is untouched. Multiple reruns of the same turn
each fork independently.

### Memory layer changes

The existing `run_id`-scoped memory becomes thread-scoped. Concretely:

- The `blobs.run_id` column is renamed to `blobs.thread_id` via a
  one-shot SQLite migration gated on `pragma user_version`.
- Every `MemoryStore` write that previously took `run_id=X` now takes
  `thread_id=X`. The semantics are: when this thread is the active
  one, this blob is recallable.
- `MemoryStore.recall(scope, run_id=...)` becomes
  `MemoryStore.recall(scope, thread_ids: list[str] = [])`. Single-
  thread callers pass `[my_id]`. Forked-thread callers pass
  `[my_id, *ancestor_chain]`.
- `plans` scope is unchanged — they're cross-thread by design.

The `_ACTIVE_RUN_ID` contextvar in `agent.py` becomes
`_ACTIVE_THREAD_ID`. The `memory_recall` tool reads it plus the
ancestor chain (cached on the active thread at run start) and passes
the full chain into `recall`.

#### Schema migration

`memory/db.py` gains a small migration on `__init__`:

```python
ver = self._conn.execute("PRAGMA user_version").fetchone()[0]
if ver == 0:
    # rename run_id -> thread_id; rebuild affected indices
    self._conn.executescript("""
        ALTER TABLE blobs RENAME COLUMN run_id TO thread_id;
        DROP INDEX IF EXISTS idx_blobs_scope_run;
        CREATE INDEX IF NOT EXISTS idx_blobs_scope_thread
            ON blobs(scope, thread_id);
        PRAGMA user_version = 1;
    """)
```

Wrapped in a transaction; if it raises, the prior db file is
preserved and the runtime surfaces a `LlamaAgentsError("memory
schema migration failed: <reason>; restore your backup or delete
.llama_agents/memory/index.sqlite to start fresh")`. (We may add a
narrower `MemoryMigrationError` subclass — non-blocking; the base
class makes the failure path explicit.)

The `runs/` subdirectory next to `index.sqlite` (where blob bodies
live as markdown files) is migrated lazily: the `_purge_thread`
helper looks under both `runs/<id>/` and `threads/<id>/` and removes
whichever exists. Future writes always go under `threads/<id>/`.

### Auto-migration of legacy queue artifacts

On `JobQueueWorker.__init__`, after `ensure_dirs` and before the
processing-sweep, call `migrate_legacy_queue_dirs(cfg.root)`. That
helper:

1. Looks for `inbox/`, `processing/`, `done/`, `failed/` siblings of
   `threads/`.
2. For each `<name>.md` file (excluding `.prompt.md` and
   `.events.jsonl` side-cars):
   - Mint a thread id.
   - Create `threads/<new-id>/turns/001/`.
   - Move/copy: `prompt.md`, `result.md`, `events.jsonl`, `error.txt`
     as available.
   - Write `meta.json` with `title = <first 60 chars of prompt body>`,
     `current_turn = 1`, `created_at = <file mtime>`, parent fields
     null.
   - Set `status` to the legacy folder name (queued for `inbox`,
     processing for `processing`, done for `done`, failed for
     `failed`).
3. After successful migration of all files in a folder, remove that
   folder. If any file in a folder fails to migrate, log a warning
   and leave the folder in place — next startup will retry just the
   leftovers.
4. The migration is idempotent: re-running it on an already-migrated
   tree finds nothing to do and returns immediately.

Logged at INFO level: `migrated N legacy queue files into M threads`.

### Worker `_finalize` change

Today `_finalize` calls `move_to_terminal(root, processing_path,
status="done")` which moves an `inbox/x.md` (via `processing/`) into
`done/x.md`, then writes side-cars. After the change, it instead
writes directly into the turn folder:

```python
turn_dir = thread_store.turn_dir(thread_id, turn_idx)
turn_dir.joinpath("result.md").write_text(result.final_text, ...)
turn_dir.joinpath("events.jsonl").write_text(...)
if not result.success:
    turn_dir.joinpath("error.txt").write_text(...)
thread_store.append_messages(thread_id, new_messages)
thread_store.set_status(thread_id, turn_idx, "done" or "failed")
```

The atomic `status` rewrite is the visible "this turn is done"
signal. Side-car files are written *before* the status flip so
readers polling the thread page never see a `done` turn with a
missing result.md.

### Surfaces — HTTP routes

Added in `web/routes.py`:

| Path | Method | Returns | Purpose |
|---|---|---|---|
| `/` | GET | 302 → `/activity` | back-compat redirect |
| `/activity` | GET | HTML | 4-panel turn dashboard |
| `/threads` | GET | HTML | thread list |
| `/threads/{id}` | GET | HTML | thread detail page |
| `/api/threads/{id}` | PATCH | JSON | update title |
| `/api/threads/{id}/continue` | POST | 303 → `/threads/{id}` | append turn |
| `/api/threads/{id}/rerun/{turn}` | POST | 303 → `/threads/{new}` | fork |
| `/api/threads/{id}/fragment` | GET | HTML fragment | HTMX poll target |
| `/api/jobs/{status}` | GET | HTML fragment | unchanged shape; now scans turns |

The `/api/submit` endpoint is updated to write a new thread (with one
turn, status `queued`) instead of an inbox file. The web UI's
"Submit" buttons keep their semantics from the user's perspective —
the redirect now lands on `/threads/<new-id>`.

### Surfaces — CLI

`src/llama_agents/cli.py`:

- `chat` gains `--thread / -t` and `--background` flags.
  - Without `--background`: runs the agent in-process (no worker
    involvement) and writes the turn folder directly with status
    flipping `queued → processing → done` synchronously.
  - With `--background`: writes the turn folder with status
    `queued`, prints the thread id, exits 0. The worker picks it up.
- `--thread <id>` accepts a unique prefix (≥ 4 chars) or full id.
  Refuses (exit 2) if no thread matches or the prefix is ambiguous.
  Refuses (exit 3) if the thread's latest turn is currently
  `queued`/`processing`.
- New `threads` subcommand group:
  - `threads list [--limit N]` — printed table.
  - `threads show <id> [--full]` — render every turn. With `--full`,
    every event in the timeline.
  - `threads rerun <id> <turn> [--edit "..."]` — fork. Without
    `--edit`, reuses the original turn's prompt verbatim. Runs
    in-process like `chat`. Prints the new thread id.

### Module layout

New package `src/llama_agents/thread/`:

| File | Responsibility |
|---|---|
| `ids.py` | `mint_thread_id() -> str`, `validate_thread_id(s) -> bool`, `resolve_prefix(root, prefix) -> str` (raises on ambiguous / unknown). |
| `meta.py` | `ThreadMeta` dataclass; `read_meta(root, id) -> ThreadMeta`, `write_meta(root, meta) -> None`, `update_meta(root, id, **fields) -> ThreadMeta`. |
| `store.py` | `ThreadStore`: `create_thread(title, parent_thread_id, parent_turn_idx) -> str`, `list_threads(limit) -> list[ThreadMeta]`, `next_turn_dir(thread_id) -> (Path, int)`, `turn_dir(thread_id, n) -> Path`, `append_messages(thread_id, msgs) -> None`, `read_messages(thread_id) -> list[dict]`, `ancestor_chain(thread_id) -> list[str]`, `_purge_thread(thread_id) -> None`. |
| `status.py` | `read_status(turn_dir) -> str`, `set_status(turn_dir, new) -> None` (atomic via `.tmp` + replace), `claim_for_processing(turn_dir) -> bool`, `revert_processing_on_startup(root) -> int`. |
| `migration.py` | `migrate_legacy_queue_dirs(root) -> int` (count of files migrated). Idempotent; logs at INFO. |

### Modules modified

| Path | Change |
|---|---|
| `src/llama_agents/queue/worker.py` | rewrite `_pick_one` to use `thread_store.next_queued_turn`; rewrite `_finalize` to write into turn folders; startup hook calls `migrate_legacy_queue_dirs` + `revert_processing_on_startup`. |
| `src/llama_agents/queue/paths.py` | most of its move-between-folders logic relocates to `thread/status.py`. The file shrinks to just `ensure_dirs` (still useful for the threads/ root). |
| `src/llama_agents/agent.py` | `Agent.__init__` gains `thread_id: str \| None`, `prior_messages: list[dict] \| None`. The loop builds `self.messages` from prior_messages when present. `_ACTIVE_RUN_ID` → `_ACTIVE_THREAD_ID`. |
| `src/llama_agents/memory/db.py` | schema migration; `recall` takes `thread_ids: list[str]`; column rename. |
| `src/llama_agents/memory/store.py` | `recall(scope, thread_ids: list[str])`. Every other method that previously took `run_id` keyword now takes `thread_id`. |
| `src/llama_agents/tools/builtin/memory.py` | `MemoryRecallTool` reads `_ACTIVE_THREAD_ID` plus `ThreadStore.ancestor_chain` and passes the chain into `recall`. |
| `src/llama_agents/tools/builtin/subagent.py` | propagates parent thread id same way it propagates run_id today. |
| `src/llama_agents/runtime.py` | constructs `ThreadStore`; exposes `Runtime.thread_store`; `new_agent` accepts optional `thread_id` + `prior_messages`. |
| `src/llama_agents/cli.py` | `chat` flag additions; `threads` subcommand group. |
| `src/llama_agents/web/routes.py` | route additions per the HTTP table above; `_api_submit` writes a thread, not an inbox file; `_list_jobs` becomes `_list_turns` querying the thread store. |
| `src/llama_agents/web/templates/` | `dashboard.html` → `activity.html` (rename + content swap); new `threads.html`, `thread.html`; `base.html` nav update; `job.html` + `_partials/job_*.html` deleted. |
| `CLAUDE.md` | module map gains four `thread/*.py` rows; struck "no multi-turn" limitation; `web/routes.py` row updated. |
| `docs/web.md`, `docs/install.md` | brief updates pointing at the new nav. |
| `docs/threads.md` (new) | user guide. |

### What gets deleted

- `src/llama_agents/web/templates/job.html`
- `src/llama_agents/web/templates/_partials/job_list.html`
- `src/llama_agents/web/templates/_partials/job_row.html`
- The old single-job HTTP route handlers (`/jobs/{status}/{name}`)
  collapse into the new thread detail page.

## Errors and observability

- Migration failure (schema or legacy queue): the runtime refuses to
  start with a clear error and a pointer to the failing step.
  Filesystem migration leaves source files in place so the user can
  intervene; SQL migration is transactional so the db file is either
  fully migrated or fully untouched.
- Continue while a turn is running: HTTP returns 409 with a small
  JSON body `{"error": "thread has an active turn", "turn": N}`;
  CLI exits 3 with the same text.
- Prefix ambiguous: CLI exits 2 with the list of matching ids
  (shortened to the prefix + next 4 chars for easy disambiguation).
  Web routes use full ids; this doesn't apply.
- Ancestor chain cycle (shouldn't happen but defense): detected by
  depth cap; emits a `LoopError` if a thread runs while its chain is
  malformed.
- Worker logs every status transition at INFO. Migrations log a
  one-line summary at INFO.

## Testing

Tests live in `tests/unit/test_thread_*.py` plus updates to the
existing queue / agent / memory / web suites.

### New test files

- `test_thread_ids.py` — id minting (uniqueness, length, charset);
  prefix resolution (unique / ambiguous / unknown distinguishable).
- `test_thread_meta.py` — read/write round-trip; update preserves
  unspecified fields; missing file raises; malformed JSON raises.
- `test_thread_store.py` — create_thread populates the right paths;
  list_threads sorts by updated_at desc; next_queued_turn ordering
  by mtime; ancestor_chain (linear, single, capped, cyclic guard).
- `test_thread_status.py` — every transition; atomic claim race
  (two callers, first wins); revert_processing_on_startup count.
- `test_thread_migration.py` — staged legacy `inbox/`, `done/`,
  `failed/` with all side-car shapes → migrated tree matches expected
  structure; idempotent on re-run; partial-failure leaves source
  intact.

### Updated test files

- `tests/unit/test_queue_worker.py` — the existing happy-path /
  retry / drain / sweep tests are rewritten to stage thread folders
  instead of inbox files. Worker now pulls from
  `threads/*/turns/*/status==queued`. ~12 tests updated; structure
  preserved.
- `tests/unit/test_agent_loop.py` — new tests:
  - `test_agent_hydrates_prior_messages` — `prior_messages=[...]`
    is seeded into `self.messages` between the system prompt and the
    new user message.
  - `test_memory_recall_uses_ancestor_chain` — staged
    parent→child thread with a blob written under the parent; the
    child's `recall` returns the parent's blob.
- `tests/unit/test_memory_db.py` + `test_memory_store.py` —
  - `test_schema_migration_renames_run_id_to_thread_id` — write
    rows under the old schema, open the store with new code, verify
    `pragma user_version == 1` and recall still returns the rows.
  - `test_recall_thread_ids_list_semantics` — single id, multiple
    ids, empty list (returns nothing), unknown id (returns nothing).
- `tests/unit/test_web_routes.py` — ~12 new tests:
  - GET `/` redirects to `/activity`.
  - `/activity` 200; rendered HTML mentions all 4 buckets + at
    least one staged turn from a thread.
  - `/threads` 200; lists staged thread.
  - `/threads/{id}` 200; renders title + turn block.
  - `/api/threads/{id}/continue` POST 303; writes
    `turns/NNN/prompt.md` + `status=queued`; returns 409 if prior
    turn is still running.
  - `/api/threads/{id}/rerun/{turn}` POST 303; new thread folder
    exists with `meta.parent_thread_id == old_id` and
    `meta.parent_turn_idx == turn - 1`.
  - `/api/threads/{id}` PATCH title updates meta.json.
  - existing `test_dashboard_*` tests adapted (the dashboard moved
    to `/activity`).
- `tests/unit/test_cli_threads.py` (new) — Typer `CliRunner` against
  the new subcommand group. `chat --thread`, `chat --background`,
  `threads list`, `threads show`, `threads rerun --edit`. ~6 tests.
- `tests/unit/test_http_app.py`, `test_install.py`, `test_cli_init.py`
  — no changes required (install flow doesn't touch the thread
  surface; the lifespan startup just runs the new sweeps).

### Live test

`tests/live/test_thread_e2e.py` (marked `live`): boot a Runtime,
submit an opening prompt via `chat --background`, poll the thread's
turn 1 for `status == done`, submit a follow-up via the same flow,
assert the follow-up's response shows the agent had access to the
opening turn's outcome (we check that the agent's reply does not
include "I have no prior context" sentinel phrases — soft check, but
useful as a smoke).

## Files changed / added

**New:**
- `src/llama_agents/thread/__init__.py`
- `src/llama_agents/thread/ids.py`
- `src/llama_agents/thread/meta.py`
- `src/llama_agents/thread/store.py`
- `src/llama_agents/thread/status.py`
- `src/llama_agents/thread/migration.py`
- `src/llama_agents/web/templates/threads.html`
- `src/llama_agents/web/templates/thread.html`
- `src/llama_agents/web/templates/activity.html` (rename + content swap of dashboard.html)
- `tests/unit/test_thread_ids.py`
- `tests/unit/test_thread_meta.py`
- `tests/unit/test_thread_store.py`
- `tests/unit/test_thread_status.py`
- `tests/unit/test_thread_migration.py`
- `tests/unit/test_cli_threads.py`
- `tests/live/test_thread_e2e.py`
- `docs/threads.md`

**Modified:**
- `src/llama_agents/queue/worker.py`
- `src/llama_agents/queue/paths.py`
- `src/llama_agents/agent.py`
- `src/llama_agents/memory/db.py`
- `src/llama_agents/memory/store.py`
- `src/llama_agents/tools/builtin/memory.py`
- `src/llama_agents/tools/builtin/subagent.py`
- `src/llama_agents/runtime.py`
- `src/llama_agents/cli.py`
- `src/llama_agents/web/routes.py`
- `src/llama_agents/web/templates/base.html`
- `tests/unit/test_queue_worker.py`
- `tests/unit/test_agent_loop.py`
- `tests/unit/test_memory_db.py`
- `tests/unit/test_memory_store.py`
- `tests/unit/test_web_routes.py`
- `CLAUDE.md`
- `docs/web.md`
- `docs/install.md`

**Deleted:**
- `src/llama_agents/web/templates/job.html`
- `src/llama_agents/web/templates/_partials/job_list.html`
- `src/llama_agents/web/templates/_partials/job_row.html`
- `src/llama_agents/web/templates/dashboard.html` (renamed to `activity.html`)

## Open questions

None at design time. Two deliberate deferrals:

- **Thread search / filtering.** Once you have 100s of threads, the
  list view will want a search box. The schema doesn't preclude
  adding it (just iterate meta files); deferred for v1.
- **Thread archive / soft-delete UI.** Threads can be removed by
  `rm -rf threads/<id>/`. No command-surface for it yet; if usage
  patterns demand it, a `llamactl threads delete <id>` command is a
  small follow-up.
