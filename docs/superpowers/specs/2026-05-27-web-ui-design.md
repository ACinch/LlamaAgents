# Web UI — Design

**Status:** Draft
**Date:** 2026-05-27
**Owner:** Maarten

## Goal

Add a lightweight web UI for the job queue so the user can:
- See jobs in inbox/processing/done/failed at a glance, auto-refreshing.
- Submit new jobs by uploading a `.md`/`.txt` file or pasting prompt
  text directly.
- Click a job to see its prompt, the event log, and (on failure) the
  error message.
- View the current `config.toml` read-only.

The UI runs inside the existing `llamactl serve` FastAPI process. No
new subcommand, no second port, no build step.

## Non-goals

- Editing `config.toml` from the UI. View-only.
- Re-queueing failed jobs via a button. User moves files in the
  filesystem; can be added later if desired.
- Cancelling in-flight jobs from the UI. Same reason — possible
  future, not v1.
- Authentication or remote access. UI is bound to the same
  `cfg.http.host` (default `127.0.0.1`) as the rest of the server.
- Searching/filtering jobs, pagination beyond a hardcoded cap.
- Live websocket or SSE updates for the queue surface (HTMX polling
  every 2s is sufficient for the scale).

## User experience

### Dashboard (`/`)

Top region (full width): nav bar with two links — **Dashboard** /
**Config**.

Below the nav: two side-by-side submit forms.
- **Upload file** — `<input type="file" accept=".md,.txt">` + Submit.
- **Paste prompt** — `<input name="filename" placeholder="task-<unix>.md">`
  + `<textarea name="body">` + Submit.

Both POST to `/api/submit`; both redirect back to `/` on success.

Main region: 2×2 grid of panels.
```
┌──────────────────┬──────────────────┐
│  Inbox (3)       │  Processing (1)  │
│  • foo.md   2s   │  • bar.md   12s  │
│  • baz.md  47s   │                  │
├──────────────────┼──────────────────┤
│  Done (12)       │  Failed (1)      │
│  • old.md  3m    │  • boom.md 1h    │
│  • …             │                  │
└──────────────────┴──────────────────┘
```

Each panel header shows a count. Each `<ul>` polls
`/api/jobs/<status>` every 2 seconds via
`hx-trigger="load, every 2s" hx-swap="innerHTML"`. Done/failed lists
show newest 50 (by mtime desc). Inbox/processing show all.

Each row is a link to `/jobs/<status>/<name>`.

### Job detail (`/jobs/{status}/{name}`)

`status` ∈ `{inbox, processing, done, failed}`. `name` is the
filename (`foo.md`).

Three stacked sections:

1. **Prompt** — the original task text.
   - For `inbox` / `processing`: read `<status>/<name>` directly.
   - For `done` / `failed`: read `<status>/<name>.prompt.md` (new
     side-car file, see "Worker change" below). If absent
     (legacy job), fall back to a placeholder string.
2. **Events timeline** — read `<status>/<name>.events.jsonl` if it
   exists; one row per event with timestamp, type badge (color-coded
   per type), and key fields. Raw JSON of each line collapsed behind
   a `<details>` toggle. Inbox/processing have no events file yet,
   so this section shows "(no events recorded)".
3. **Footer** — for `failed/`, show `<name>.error.txt` content in a
   styled callout. "Back to dashboard" link.

### Config view (`/config`)

Single `<pre>` block containing the verbatim contents of the
`config.toml` file at the path used to launch the server. Read at
request time so live edits surface without a restart. A "Copy to
clipboard" button next to the heading. No syntax highlighting.

## Architecture

### New module: `src/llama_agents/web/`

```
src/llama_agents/web/
├── __init__.py            — empty
├── routes.py              — register_routes(app, cfg, runtime_accessor)
├── templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── job.html
│   ├── config.html
│   └── _partials/
│       ├── job_list.html
│       └── job_row.html
└── static/
    ├── style.css
    └── htmx.min.js        — vendored, version pinned
```

`routes.py` is one file. Total surface is ~6 endpoints and ~150 lines
of handlers + helpers. No need to split.

### Public entry point

```python
def register_routes(
    app: FastAPI,
    cfg: Config,
    config_path: Path,
) -> None: ...
```

Called from `src/llama_agents/http_app.py:create_app()` after the
existing chat route is registered. `config_path` is the path the
server was launched with (so `/config` can read it back). No
dependency on a runtime accessor — the UI doesn't talk to the agent
loop; it only reads/writes files under `cfg.queue.root` and reads
`config_path`.

`http_app.py` change:

```python
from .web.routes import register_routes
...
# After the existing routes:
register_routes(app, cfg, config_path=cfg_path)
```

This means `create_app` gains a `config_path: Path` parameter. The
CLI's `serve` command already has the path in scope (it loaded the
config); pass it through. Other callers (tests) can pass any path
they have a file at.

### Route table

| Path | Method | Returns | Purpose |
|---|---|---|---|
| `/` | GET | `dashboard.html` | main page |
| `/jobs/{status}/{name}` | GET | `job.html` | job detail |
| `/config` | GET | `config.html` | read-only config |
| `/api/jobs/{status}` | GET | `_partials/job_list.html` | HTMX polled list |
| `/api/submit` | POST | 303 → `/` | submit new job |
| `/static/*` | GET | static files | css, htmx.min.js |

### `/api/submit` semantics

Accepts either:
- `multipart/form-data` with field `file` (a `.md` or `.txt` upload).
  Filename derived from the upload's filename header.
- `application/x-www-form-urlencoded` with fields `filename` and
  `body`. If `filename` is empty, default to `task-<unix-ts>.md`.

Validation (returns 400 with a flash message on failure):
- Filename must match `^[A-Za-z0-9._-]+$`. No path traversal, no
  spaces, no Unicode shenanigans. Reject anything else.
- Extension must be in `cfg.queue.accepted_extensions`
  (`.md`/`.txt` by default).
- If the file already exists in `inbox/`, reject (don't overwrite).

On success: write the file to `<queue_root>/inbox/<name>` via the
same `os.replace` atomicity story used by the worker. The worker's
existing 2-second poll picks it up. Return `303 See Other` →
`Location: /` so the form submission doesn't replay on refresh.

### Listing filter for the side-cars

`done/` and `failed/` contain three files per job:
`<name>.md`, `<name>.events.jsonl`, `<name>.prompt.md`. The dashboard
list and the `/api/jobs/<status>` partial show only files whose
suffix is `.md` AND whose stem does not end in `.prompt`. Concretely:

```python
def _list_jobs(dir_: Path) -> list[Path]:
    if not dir_.is_dir():
        return []
    return sorted(
        (p for p in dir_.iterdir()
         if p.is_file()
         and p.suffix == ".md"
         and not p.stem.endswith(".prompt")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
```

Inbox/processing also use this helper. Inbox files won't have
`.prompt.md` siblings, but the filter is harmless there.

### Worker change: `<name>.prompt.md` side-car

`src/llama_agents/queue/worker.py:_finalize` currently writes the
final answer to `<status>/<name>.md` and the events to
`<status>/<name>.events.jsonl`. Add a third write: copy the original
prompt text to `<status>/<name>.prompt.md`. Implementation: in
`_invoke_agent` we already `path.read_text(...)`, capture that
string and thread it through to `_finalize`.

Plan-level change is minimal:
```python
# In _invoke_agent:
prompt = path.read_text(encoding="utf-8")
...
return JobResult(..., prompt_text=prompt, ...)

# In _finalize:
prompt_path = dst.with_suffix(".prompt.md")
prompt_path.write_text(result.prompt_text, encoding="utf-8")
```

Add `prompt_text: str` to the `JobResult` dataclass.

This costs nothing at runtime (one extra small file write) and
unlocks the entire job-detail UX.

### Event rendering

Each line of `<name>.events.jsonl` is JSON: `{type, ts, ...fields}`.
The template renders them in document order. Type-to-style mapping:

| Type | Badge color | Key fields shown |
|---|---|---|
| `PlanProposed` | gray | `attempt` |
| `PlanReviewed` | gray | `attempt`, `accepted` |
| `PlanAccepted` | green | `attempts` |
| `ToolCallStart` | blue | `name`, `arguments` (truncated 200ch) |
| `ToolCallResult` | blue (ok) / orange (!ok) | `ok`, `content` (truncated 400ch) |
| `AssistantChunk` | violet | `text` (truncated 400ch, expandable) |
| `MemoryStored` | teal | `kind`, `scope`, `bytes_` |
| `MemoryEvicted` | teal | `turn`, `bytes_freed` |
| `LoopError` | red | `error_type`, `message` |
| `Done` | green | `reason`, `final_message` (truncated) |

Each row has a `<details>` element whose `<summary>` is the badge +
key fields, and whose body is the raw JSON pretty-printed.

### Vendored htmx

`htmx.min.js` is checked into `src/llama_agents/web/static/`. The
version pin is recorded in `docs/web.md`. Rationale: zero
network/CDN dependency, the UI works on an air-gapped machine, no
SRI churn. ~50KB cost in the wheel.

### Templates: Jinja2 setup

`fastapi.templating.Jinja2Templates`, pointed at the
`src/llama_agents/web/templates/` directory. Auto-escape on by
default. Two custom Jinja filters:
- `fmt_ts(iso_string)` — render `"2026-05-27T10:00:00+00:00"` as
  `"2026-05-27 10:00:00 UTC"`.
- `age(mtime: float)` — relative time: `"2s"`, `"3m"`, `"1h"`,
  `"2d"`. Used in the row template.

Both filters live in `routes.py` and are registered when
`register_routes` is called.

## Config

No new config block. The UI uses values already in scope:
- `cfg.queue.root` — for listing and writing files.
- `cfg.http.host`/`cfg.http.port` — already drive the server bind.
- `config_path` — the path used to launch the server; passed into
  `register_routes`.

## Dependencies

Two new runtime deps in `pyproject.toml`:
- `jinja2>=3.1` — templating.
- `python-multipart>=0.0.9` — needed by FastAPI to parse multipart
  uploads (currently only used by the chat endpoint via JSON, so
  this is a true new dep).

No client-side build tooling. No `node_modules`. No CDN at runtime.

## Errors and observability

- `/api/submit` validation failures: 400 with a small flash message
  rendered on the dashboard via an HTMX swap into a banner div.
- `/jobs/{status}/{name}` for a non-existent job: 404.
- `/jobs/{invalid_status}/...`: 404.
- `/config` when `config_path` is unreadable: 500 with the actual
  error (helpful in dev; not a security concern given localhost).
- All routes log to the existing FastAPI logger at INFO level
  (request method/path) and ERROR for handler exceptions.

## Testing

`tests/unit/test_web_routes.py` uses the same ASGI harness as
`tests/unit/test_http_app.py`:

```python
from httpx import ASGITransport, AsyncClient
from asgi_lifespan import LifespanManager
```

Test list:
1. `GET /` → 200; body contains "Inbox", "Processing", "Done", "Failed".
2. `GET /api/jobs/inbox` with one staged `foo.md` → 200, body
   contains "foo.md".
3. `GET /api/jobs/processing` with empty dir → 200, empty `<ul>`.
4. `POST /api/submit` multipart with `t.md` → 303 to `/`; file
   exists in `inbox/t.md` with expected content.
5. `POST /api/submit` form with `filename=t.md&body=hello` → 303;
   `inbox/t.md` content == `hello`.
6. `POST /api/submit` form with empty filename → file landed at
   `inbox/task-<digits>.md`.
7. `POST /api/submit` with extension `.exe` → 400.
8. `POST /api/submit` with `filename="../escape.md"` → 400.
9. `POST /api/submit` with a name that already exists in inbox → 400.
10. `GET /jobs/inbox/foo.md` → 200; body contains the file content;
    events section says "(no events recorded)".
11. `GET /jobs/done/foo.md` with all three side-cars staged → 200;
    body contains prompt text, events count, final answer.
12. `GET /jobs/failed/foo.md` with `foo.error.txt` staged → 200;
    body contains the error text.
13. `GET /jobs/done/missing.md` → 404.
14. `GET /jobs/elsewhere/foo.md` → 404.
15. `GET /config` → 200; body contains the literal `[llama]` from
    the config.toml passed to `create_app`.

Worker test addition in `tests/unit/test_queue_worker.py`:

16. `test_finalize_writes_prompt_sidecar_to_done` — extends the
    happy-path setup; asserts `done/foo.prompt.md` exists with the
    original prompt body.

No live test. The UI is fully covered by ASGI-level tests.

## Files changed / added

**New:**
- `src/llama_agents/web/__init__.py`
- `src/llama_agents/web/routes.py`
- `src/llama_agents/web/templates/base.html`
- `src/llama_agents/web/templates/dashboard.html`
- `src/llama_agents/web/templates/job.html`
- `src/llama_agents/web/templates/config.html`
- `src/llama_agents/web/templates/_partials/job_list.html`
- `src/llama_agents/web/templates/_partials/job_row.html`
- `src/llama_agents/web/static/style.css`
- `src/llama_agents/web/static/htmx.min.js` (vendored)
- `tests/unit/test_web_routes.py`
- `docs/web.md`

**Modified:**
- `src/llama_agents/http_app.py` — gain `config_path` parameter; call
  `register_routes`.
- `src/llama_agents/cli.py` — `serve` passes the config path through
  to `create_app`.
- `src/llama_agents/queue/worker.py` — `_finalize` writes
  `<name>.prompt.md`; `JobResult` gains `prompt_text` field.
- `tests/unit/test_queue_worker.py` — assert prompt side-car.
- `tests/unit/test_http_app.py` — pass `config_path` in tests.
- `pyproject.toml` — add `jinja2`, `python-multipart`.
- `CLAUDE.md` — add `web/routes.py` to module map.
- `README.md` — note the UI URL.

## Open questions

None at design time. Two deliberate deferrals:

- **Re-queue button.** Future "POST /api/jobs/failed/{name}/requeue"
  that moves the file back to `inbox/` and removes the side-cars.
  Not v1.
- **Syntax highlighting on `/config`.** Future Prism.js add. Not v1.
