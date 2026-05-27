# Job Queue

A filesystem-backed job queue runs inside `llamactl serve`. Drop a task
file in the inbox; the worker picks it up, runs it through the agent,
and writes the result to the done folder.

## Layout

Under `cfg.queue.root` (default `.llama_agents/queue/`, relative to
your first `sandbox.allowed_dirs` entry):

- `inbox/`      drop `<name>.md` or `<name>.txt` here
- `processing/` worker moves the file here while running it
- `done/`       on success: `<name>.md` (final answer) + `<name>.events.jsonl`
- `failed/`     on failure: the same two files + `<name>.error.txt`

## Enabling the worker

In `config.toml`:

```toml
[queue]
enabled = true
max_concurrent = 2
max_retries = 2
```

Then start the server as usual:

```
uv run llamactl serve
```

The worker runs as a background task in the same process. Stopping the
server triggers a drain — in-flight jobs get
`cfg.queue.drain_timeout_seconds` to finish; anything still running is
cancelled and left in `processing/` for the next startup to sweep back
into `inbox/`.

## Failure handling

- A run that yields a final assistant message (or exhausts
  `max_iterations` cleanly) → `done/`.
- A run that raises `LlamaUnreachable` (server down, connection refused)
  → retried up to `max_retries` times with exponential backoff
  (`retry_backoff_seconds * 2^attempt`). If retries exhaust, → `failed/`.
- Any other `LoopError` (`LlamaProtocolError`, etc.) → straight to
  `failed/`. These are deterministic and won't fix themselves on retry.

## Re-queueing a failed job

Just move the file back from `failed/<name>.md` into `inbox/`. The
worker picks it up on the next poll.

## Inspecting an events log

`<name>.events.jsonl` contains one event per line — `ToolCallStart`,
`ToolCallResult`, `PlanProposed`, `MemoryEvicted`, `Done`, etc. — in
the order they were emitted. Each line is a JSON object with `type`,
`ts`, and the event's dataclass fields.

```
type Get-Content done\foo.events.jsonl | ConvertFrom-Json
```
