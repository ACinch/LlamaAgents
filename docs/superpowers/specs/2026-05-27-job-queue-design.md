# Job Queue Worker — Design

**Status:** Draft
**Date:** 2026-05-27
**Owner:** Maarten

## Goal

Let users drop task files into an `inbox/` folder and have the running
`llamactl serve` process pick them up, run them through the agent loop,
and write the results to a `done/` folder. Failures land in `failed/`.
Crashes are recoverable. Retries are bounded and only fire on
infrastructure errors.

## Non-goals

- Distributed queueing (this is single-process, single-host).
- Priorities, scheduling, cron-like time triggers.
- A control-plane API for inspecting/cancelling jobs (the filesystem
  *is* the API — move a file, see a result).
- Job specs richer than "task file" (no JSON job descriptors with
  per-job model overrides, allowed-tool lists, etc.). Jobs are plain
  markdown or text task descriptions, same shape as the existing
  `execute_task` pattern.

## User experience

```
.llama_agents/queue/
├── inbox/        ← user drops <jobname>.md or .txt here
├── processing/   ← worker moves the file here while running it
├── done/         ← <jobname>.md (final answer) + <jobname>.events.jsonl
└── failed/       ← same two files + <jobname>.error.txt
```

The worker runs inside `llamactl serve` as a background asyncio task,
gated by `cfg.queue.enabled`. There is no separate `llamactl queue`
subcommand.

## Architecture

### New module: `src/llama_agents/queue/`

```
src/llama_agents/queue/
├── __init__.py
├── paths.py    — pure helpers: ensure_dirs, move_atomic, sweep_processing_to_inbox
└── worker.py   — JobQueueWorker class
```

### `JobQueueWorker`

```python
class JobQueueWorker:
    def __init__(self, runtime: Runtime, cfg: QueueConfig) -> None: ...
    async def run(self) -> None:
        """Main loop. Returns when self._stop is set."""
    async def drain(self, timeout: float) -> None:
        """Stop accepting new jobs; await in-flight up to `timeout`s."""
```

Internals (all `async`, all pure on their inputs except for FS I/O):

- `_pick_one() -> Path | None` — list `inbox/` sorted by mtime, attempt
  atomic rename of the oldest candidate into `processing/`. Return the
  new path on success; `None` if nothing available or the rename lost
  a race. Files whose rename raises `PermissionError` (still being
  written by the user) are skipped and retried next tick.
- `_process(path: Path, attempt: int) -> JobResult` — read the file,
  call `runtime.new_agent().run(prompt, opts)`, stream events into an
  in-memory list, collect the final assistant text. Returns a
  `JobResult` dataclass with `success: bool`, `final_text: str`,
  `events: list[dict]`, `error: LoopError | None`.
- `_classify_failure(result) -> Literal["retry", "fail"]` — see
  retry policy below.
- `_finalize(path, result, attempt) -> None` — write outputs to
  `done/` or `failed/` and remove the file from `processing/`.

### Lifecycle integration

`http_app.py:create_app()` already has a FastAPI lifespan that builds
the `Runtime`. Extend it:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    rt = await Runtime.create(cfg)
    worker_task: asyncio.Task | None = None
    if cfg.queue.enabled:
        worker = JobQueueWorker(rt, cfg.queue)
        worker_task = asyncio.create_task(worker.run())
    try:
        app.state.rt = rt
        yield
    finally:
        if worker_task is not None:
            await worker.drain(cfg.queue.drain_timeout_seconds)
        await rt.aclose()
```

On startup the worker first calls
`sweep_processing_to_inbox(cfg.queue.root)` — anything left in
`processing/` from a prior crash gets requeued. This is safe because
the worker is the only writer of `processing/`.

### Concurrency model

A single asyncio loop. `JobQueueWorker.run()`:

1. `await asyncio.sleep(poll_interval_seconds)` between polls.
2. Each tick: while the in-flight set has fewer than
   `cfg.queue.max_concurrent` entries, pick one job from `inbox/` and
   spawn it as `asyncio.create_task(self._run_job(path))`. Stop picking
   when `_pick_one()` returns `None`.
3. Done callbacks remove the task from the in-flight set.

The bound is implemented as a simple counter + set, not a semaphore,
because we need to be able to check "is there room?" without blocking
on `acquire()` — we want to keep polling other folders even when full.

### Atomic move semantics

`paths.move_atomic(src, dst)` uses `os.replace(src, dst)`, which is
atomic on both NTFS and POSIX when source and destination are on the
same volume (they always are here — same queue root). If the destination
already exists, `os.replace` overwrites it; we treat that as a name
collision in `done/`/`failed/` and append a numeric suffix
(`<name>.1.md`, `<name>.2.md`, …) before calling `os.replace`.

For `inbox/ → processing/` we don't want collisions to clobber, so we
use `os.rename` and catch `FileExistsError` (POSIX) /
`PermissionError` (Windows when target exists) to detect a lost race
and return `None` from `_pick_one()`.

### Output shape (per the user's choice)

For job `foo.md`:

**On success →** `done/foo.md` + `done/foo.events.jsonl`
- `foo.md`: the final `AssistantChunk` text(s) concatenated. If no
  assistant text was produced, write `[no final answer]`.
- `foo.events.jsonl`: one JSON object per line, one line per event.
  Each line: `{"type": "<EventClass>", "ts": "<iso>", ...fields}`. The
  serializer uses `dataclasses.asdict` plus a `type` tag.

**On failure →** `failed/foo.md` + `failed/foo.events.jsonl`
+ `failed/foo.error.txt`
- The `.error.txt` captures the terminal `LoopError`'s `error_type` and
  `message`, plus the attempt count.

### Retry policy

`_classify_failure(result)` returns `"retry"` iff **all** of:

1. `result.success is False`.
2. The terminating event is a `LoopError` (not just `Done` with
   `reason="max_iterations"`).
3. `loop_error.error_type` is in the infrastructure allowlist:
   `{"LlamaUnreachable"}` (the sole subclass of `LlamaAgentsError`
   that wraps transient HTTP/connect failures; see `errors.py` and
   `llama_client.py`). Other LoopErrors — e.g. `LlamaProtocolError`
   for unexpected response shape — are deterministic and not retried.
4. `attempt < cfg.queue.max_retries`.

Otherwise returns `"fail"`. Retries wait
`retry_backoff_seconds * (2 ** attempt)` before re-running the same
file (which has stayed in `processing/` throughout — no folder churn
during retries). On terminal failure, the file is moved to `failed/`.

A run that exits with `Done(reason="max_iterations")` and at least one
`AssistantChunk` is treated as **success** — the user gets whatever
the agent produced. Reason: max-iterations is a logical outcome, not
infrastructure, and forcing a retry would produce the same result.

### Shutdown drain

`drain(timeout)`:

1. Sets `self._stop`, so `run()` exits its polling loop after the
   current sleep.
2. `await asyncio.wait(self._in_flight, timeout=timeout)`.
3. Any tasks still running after `timeout` are `.cancel()`'d. Their
   files are left in `processing/` — the next startup sweep returns
   them to `inbox/`.

## Config

New block in `src/llama_agents/config.py`:

```python
class QueueConfig(BaseModel):
    enabled: bool = False
    root: str = ".llama_agents/queue"
    poll_interval_seconds: float = Field(default=2.0, ge=0.1)
    max_concurrent: int = Field(default=1, ge=1)
    max_retries: int = Field(default=2, ge=0)
    retry_backoff_seconds: float = Field(default=5.0, ge=0.0)
    max_iterations: int = Field(default=20, ge=1)
    drain_timeout_seconds: float = Field(default=30.0, ge=0.0)
    accepted_extensions: list[str] = Field(default_factory=lambda: [".md", ".txt"])
```

Relative `root` resolves against `sandbox.allowed_dirs[0]`, matching
the rule already used by `memory.root`. Files in `inbox/` whose suffix
is not in `accepted_extensions` are ignored (let the user park notes
in the folder without them being picked up).

## Per-job `run_id` and memory

Each job calls `runtime.new_agent().run(prompt, opts)`. `Agent.run`
already mints a fresh `run_id`, calls `memory.start_run(run_id)`, and
`memory.end_run(run_id)` in the `finally` block. This means:

- **Run-scoped memory** (evicted tool outputs, large subagent results)
  is isolated per job and garbage-collected by the existing
  `scratch_retention_hours` policy.
- **Plan memory** (accepted plans) persists across jobs — that is the
  intended cross-job benefit. A job that solves a similar task to a
  previous one will see the prior plan retrieved during planning.

The `events.jsonl` captures `MemoryStored`/`MemoryEvicted` events,
so a reader can reconstruct what was offloaded.

## Errors and observability

- The worker logs every state transition to stderr via the existing
  `logging` setup: pickup, completion, retry, failure, drain.
- It does **not** emit events on the HTTP SSE stream — those are for
  request-scoped chats. The filesystem is the queue's interface.
- A future `/queue/status` HTTP endpoint is **out of scope** for this
  spec but the worker exposes a `snapshot()` method returning
  `{in_flight: int, last_pickup_at, last_finish_at}` to make adding
  one cheap later.

## Testing

### Unit tests

`tests/unit/test_queue_paths.py`
- `move_atomic` overwrites only when explicitly allowed.
- `sweep_processing_to_inbox` requeues every file and is idempotent.
- `ensure_dirs` creates the four subdirs and is idempotent.

`tests/unit/test_queue_worker.py` — uses a fake `Runtime` whose
`new_agent()` returns an `Agent` driven by a scripted `_ClientLike`
(same harness as `tests/unit/test_agent_loop.py`). Cases:

1. **Happy path:** drop a file in inbox → after one tick it appears in
   `processing/` → after the run completes it lands in `done/` with the
   two output files. `events.jsonl` contains a `Done` event.
2. **Infra retry:** scripted client raises `LlamaUnreachable` on
   attempt 1, succeeds on attempt 2. File stays in `processing/`
   between attempts; final landing is `done/`.
3. **Terminal fail:** scripted client raises a non-infra error. File
   lands in `failed/` after one attempt; `error.txt` present.
4. **Max-iterations is success:** scripted client never calls
   `task_complete`; loop exits with `Done(reason="max_iterations")`.
   File lands in `done/`, no retry.
5. **Concurrency cap:** drop three files with `max_concurrent=2`. At
   any instant the in-flight set size is ≤ 2.
6. **Drain:** start a job that hangs (scripted client awaits an event
   the test never sets), then call `drain(timeout=0.1)`. Task is
   cancelled, file remains in `processing/`. A second `sweep` puts it
   back in `inbox/`.
7. **Sweep on startup:** prime `processing/` with a file before
   starting the worker. After one tick it's been picked up again.
8. **Ignored extensions:** drop `foo.tmp` next to `bar.md`; only `bar`
   is picked up.

### Live test

`tests/live/test_queue_e2e.py` (marked `live`): boot a full Runtime
against a real llama-server, drop a trivial markdown task in inbox,
poll `done/` for up to 60s, assert both output files exist and the
final answer is non-empty.

## Files changed/added

**New:**
- `src/llama_agents/queue/__init__.py`
- `src/llama_agents/queue/paths.py`
- `src/llama_agents/queue/worker.py`
- `tests/unit/test_queue_paths.py`
- `tests/unit/test_queue_worker.py`
- `tests/live/test_queue_e2e.py`
- `docs/queue.md`

**Modified:**
- `src/llama_agents/config.py` — add `QueueConfig`, wire into `Config`.
- `src/llama_agents/http_app.py` — start/drain worker in lifespan.
- `config.toml` — `[queue]` block with defaults (enabled = false).
- `.gitignore` — `.llama_agents/queue/`.
- `README.md` — Queue section.
- `CLAUDE.md` — module map row for `queue/`, brief note in
  "Known limitations" section that this exists.

## Open questions

None at design time. Two things deliberately deferred:

- **Watchdog-based discovery.** Polling at 2s is fine for this use
  case; we can add `watchdog` later behind a `discovery = "watch"`
  config option if latency becomes a complaint.
- **HTTP `/queue/*` endpoints.** `worker.snapshot()` exists so a status
  endpoint is a small follow-up if wanted.
