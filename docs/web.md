# Web UI

The web UI runs inside `llamactl serve` whenever a config file is
passed (the CLI always does this). Open
`http://<cfg.http.host>:<cfg.http.port>/` in a browser — default
`http://127.0.0.1:9000/`.

## Pages

- `/activity` — dashboard. Activity panel polling every 2 seconds.
  Submit form at the top: file upload (`.md`/`.txt`) and paste-prompt-as-text.
- `/threads` — list recent threads with turn counts; click to view a thread's
  full conversation history.
- `/jobs/{status}/{name}` — job detail. Shows the original prompt,
  the event timeline (collapsible), the final answer (for `done/`),
  and the error (for `failed/`).
- `/config` — read-only view of the `config.toml` file the server
  was launched with. Re-read on every request; edits show up
  immediately on refresh.

## Threads

Every job belongs to a thread. The Threads page (`/threads`) lists
recent threads; click one to see all turns, continue the conversation,
or rerun (fork) any past turn. See [`threads.md`](threads.md) for the
full model.

## Submitting a job

Two paths, both create a new thread:

Web: use the submit form on `/activity`.

CLI: `uv run llamactl chat "task description"` creates a single-turn thread.

Previously (old queue model): submissions wrote files into `<queue_root>/inbox/`:

1. **File upload** — pick a `.md` or `.txt` file from your computer.
2. **Paste prompt** — type a prompt or task description directly into
   the form.

## Auto-refresh

Each panel uses HTMX (`hx-trigger="load, every 2s"`) to refresh
only its own contents. The full page never reloads; submissions
redirect back to `/` (303) so a browser refresh won't re-submit.

## Static assets

`htmx.min.js` (v2.0.4) is vendored under
`src/llama_agents/web/static/`. No CDN; the UI works on an
air-gapped machine.

## Limitations

- No syntax highlighting on `/config`.
- No auth. Bind to `127.0.0.1` (the default) and don't expose to
  untrusted networks.
