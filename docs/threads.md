# Threads

Every submission to llama-agents lives in a **thread**. A thread has
an id, a title, and one or more **turns**. A one-shot prompt is just
a thread with a single turn.

## Layout

Under `<queue_root>/threads/<thread_id>/`:

- `meta.json` — title, timestamps, current_turn, optional parent link
- `messages.jsonl` — running conversation, one OpenAI-shaped message per line
- `turns/<NNN>/`
  - `prompt.md` — the user's submission for this turn
  - `status` — `queued` | `processing` | `done` | `failed`
  - `result.md`, `events.jsonl`, `error.txt` as appropriate

## Continuing a thread

Web: click any thread, scroll to the **Continue** form at the bottom,
type a follow-up, hit Send.

CLI: `uv run llamactl chat --thread <id-or-prefix> "follow-up"`.

The agent's prior conversation (messages + tool calls + tool results)
is hydrated as the context for the new turn.

## Rerunning a turn

Web: each turn block has a **⟳ Rerun** button. Optionally edit the
prompt, hit Submit. The original thread is preserved; you get a new
thread that's a fork at the rerun point.

CLI: `uv run llamactl threads rerun <id-or-prefix> <turn> [--edit "..."]`.
Without `--edit`, the original prompt is reused verbatim — useful for
retrying after a transient error.

## How rerun-as-fork inherits memory

When thread B is a fork of thread A, B's agent can recall everything
A stored in scratch memory. The memory layer walks the parent chain
(`meta.parent_thread_id`) up to a depth of 32 and includes all
ancestor thread ids in the recall query.

## Thread ids and prefixes

Thread ids are 24 lowercase hex characters. Most commands accept a
prefix of at least 4 characters as long as it uniquely matches one
thread. The web URLs always use the full id.

## Migration from the old queue model

The first time you start `llamactl serve` (or run any other entry
point) after upgrading, any files left in the old
`inbox/`/`processing/`/`done/`/`failed/` folders are automatically
migrated into single-turn threads. The migration is idempotent and
safe to re-run; partial failures leave the source files in place for
inspection.
