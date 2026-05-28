# Web UI

The web UI runs inside `llamactl serve` whenever a config file is
passed (the CLI always does this). Open
`http://<cfg.http.host>:<cfg.http.port>/` in a browser — default
`http://127.0.0.1:9000/`.

## Pages

- `/` — dashboard. Four panels (Inbox / Processing / Done / Failed)
  polling every 2 seconds. Two submit forms at the top: file upload
  (`.md`/`.txt`) and paste-prompt-as-text.
- `/jobs/{status}/{name}` — job detail. Shows the original prompt,
  the event timeline (collapsible), the final answer (for `done/`),
  and the error (for `failed/`).
- `/config` — read-only view of the `config.toml` file the server
  was launched with. Re-read on every request; edits show up
  immediately on refresh.

## Submitting a job

Two paths, both write a file into `<queue_root>/inbox/`:

1. **File upload** — pick a `.md` or `.txt` file. The filename is
   kept as-is (after safety validation).
2. **Paste prompt** — type a filename (or leave blank to get
   `task-<unix-ts>.md`) and the prompt body. The body is written
   verbatim.

Validation: the filename must match `^[A-Za-z0-9._-]+$` (no spaces,
no slashes, no Unicode), the extension must be in
`cfg.queue.accepted_extensions`, and a duplicate name in `inbox/`
is rejected.

## Auto-refresh

Each panel uses HTMX (`hx-trigger="load, every 2s"`) to refresh
only its own contents. The full page never reloads; submissions
redirect back to `/` (303) so a browser refresh won't re-submit.

## Static assets

`htmx.min.js` (v2.0.4) is vendored under
`src/llama_agents/web/static/`. No CDN; the UI works on an
air-gapped machine.

## Limitations

- No re-queue button — to retry a failed job, move
  `failed/<name>.md` back to `inbox/` (and the side-cars to keep
  history, or delete them).
- No syntax highlighting on `/config`.
- No auth. Bind to `127.0.0.1` (the default) and don't expose to
  untrusted networks.
