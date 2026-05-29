from __future__ import annotations

import json as _json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import escape as _esc

from ..config import Config
from ..thread.ids import validate_thread_id
from ..thread.meta import read_meta, update_meta
from ..thread.status import read_status, set_status
from ..thread.store import ThreadStore

_WEB_DIR = Path(__file__).parent


def _highlight_toml(text: str) -> str:
    """Return HTML with span classes for a small TOML syntax subset.

    Recognises comments, section headers, keys, strings, numbers, and
    bools. Anything else is passed through escaped. Intentionally
    line-by-line and regex-based — no full TOML parse.
    """
    out_lines: list[str] = []
    section_re = re.compile(r"^(\s*)(\[\[?[^\]]+\]?\])(\s*)$")
    kv_re = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_.-]*)(\s*=\s*)(.*?)(\s*)$")
    for raw in text.splitlines():
        line = raw
        # comments (whole-line)
        if line.lstrip().startswith("#"):
            indent_len = len(line) - len(line.lstrip())
            out_lines.append(
                line[:indent_len]
                + f'<span class="tk-comment">{_esc(line[indent_len:])}</span>'
            )
            continue
        m = section_re.match(line)
        if m:
            indent, sect, tail = m.groups()
            out_lines.append(
                f"{indent}<span class=\"tk-section\">{_esc(sect)}</span>{tail}"
            )
            continue
        m = kv_re.match(line)
        if m:
            indent, key, eq, val, tail = m.groups()
            out_lines.append(
                f"{indent}<span class=\"tk-key\">{_esc(key)}</span>"
                f"<span class=\"tk-punct\">{_esc(eq)}</span>"
                f"{_highlight_toml_value(val)}{tail}"
            )
            continue
        out_lines.append(str(_esc(line)))
    return "\n".join(out_lines)


_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _highlight_toml_value(val: str) -> str:
    """Highlight a single TOML value or a comma-separated list of them."""
    stripped = val.strip()
    if not stripped:
        return ""
    # Trailing inline comment?
    inline_comment = ""
    if "#" in stripped:
        # only treat as comment if it's outside quotes — cheap heuristic:
        # find the first # that is not inside a "...":
        in_quote = False
        for i, ch in enumerate(stripped):
            if ch == '"':
                in_quote = not in_quote
            elif ch == "#" and not in_quote:
                inline_comment = (
                    f' <span class="tk-comment">{_esc(stripped[i:])}</span>'
                )
                stripped = stripped[:i].rstrip()
                break
    # Array
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1]
        parts = [_highlight_toml_value(p.strip()) for p in _split_top_commas(inner)]
        return (
            '<span class="tk-punct">[</span>'
            + '<span class="tk-punct">, </span>'.join(parts)
            + '<span class="tk-punct">]</span>'
            + inline_comment
        )
    # String
    if stripped.startswith('"') and stripped.endswith('"'):
        return f'<span class="tk-string">{_esc(stripped)}</span>' + inline_comment
    # Bool
    if stripped in ("true", "false"):
        return f'<span class="tk-bool">{stripped}</span>' + inline_comment
    # Number
    if _NUMBER_RE.match(stripped):
        return f'<span class="tk-number">{stripped}</span>' + inline_comment
    return str(_esc(stripped)) + inline_comment


def _split_top_commas(s: str) -> list[str]:
    """Split on commas that are not inside quotes or brackets."""
    out: list[str] = []
    depth = 0
    in_quote = False
    buf = []
    for ch in s:
        if ch == '"':
            in_quote = not in_quote
        elif not in_quote and ch in "[{":
            depth += 1
        elif not in_quote and ch in "]}":
            depth -= 1
        elif not in_quote and ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def _fmt_ts(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _age(mtime: float) -> str:
    now = datetime.now(timezone.utc).timestamp()
    delta = max(0.0, now - mtime)
    if delta < 60:
        return f"{int(delta)}s"
    if delta < 3600:
        return f"{int(delta // 60)}m"
    if delta < 86400:
        return f"{int(delta // 3600)}h"
    return f"{int(delta // 86400)}d"


_VALID_STATUSES = ("queued", "processing", "done", "failed")

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_name(name: str, accepted_exts: list[str]) -> str | None:
    """Return the validated name, or None if it fails validation."""
    if not name:
        return None
    if not _SAFE_NAME.match(name):
        return None
    suffix = Path(name).suffix.lower()
    if suffix not in accepted_exts:
        return None
    return name


def _list_turns(
    thread_store: ThreadStore, status: str, *, limit: int | None = None
) -> list[dict]:
    """Find turns across all threads matching the given status."""
    if status not in _VALID_STATUSES:
        raise HTTPException(status_code=404, detail="unknown status")
    rows: list[dict] = []
    root = thread_store.root
    if not root.is_dir():
        return []
    for thread_dir in root.iterdir():
        if not thread_dir.is_dir():
            continue
        turns_dir = thread_dir / "turns"
        if not turns_dir.is_dir():
            continue
        try:
            meta = read_meta(root, thread_dir.name)
        except (FileNotFoundError, ValueError):
            continue
        for turn_dir in turns_dir.iterdir():
            if not turn_dir.is_dir() or not turn_dir.name.isdigit():
                continue
            if read_status(turn_dir) != status:
                continue
            try:
                mtime = (turn_dir / "status").stat().st_mtime
            except FileNotFoundError:
                continue
            rows.append({
                "thread_id": thread_dir.name,
                "turn_idx": int(turn_dir.name),
                "title": meta.title,
                "mtime": mtime,
            })
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    if limit is not None:
        rows = rows[:limit]
    return rows


_EVENT_STYLE: dict[str, tuple[str, str | None]] = {
    "PlanProposed":     ("gray",   "attempt"),
    "PlanReviewed":     ("gray",   "accepted"),
    "PlanAccepted":     ("green",  "attempts"),
    "ReviewerVerdict":  ("teal",   "accepted"),
    "ToolCallStart":    ("blue",   "name"),
    "ToolCallResult":   ("blue",   "ok"),
    "AssistantChunk":   ("violet", None),
    "MemoryStored":     ("teal",   "kind"),
    "MemoryEvicted":    ("teal",   "bytes_freed"),
    "LoopError":        ("red",    "error_type"),
    "Done":             ("green",  "reason"),
}


def _decorate_event(raw_line: str) -> dict | None:
    try:
        ev = _json.loads(raw_line)
    except _json.JSONDecodeError:
        return None
    t = ev.get("type", "?")
    color, summary_key = _EVENT_STYLE.get(t, ("gray", None))
    summary_val = ev.get(summary_key) if summary_key else None
    if isinstance(summary_val, str) and len(summary_val) > 80:
        summary_val = summary_val[:80] + "…"
    # Compute the raw view BEFORE attaching internal keys, so the
    # collapsed-details view shows only the event's real fields.
    raw = _json.dumps(ev, indent=2, ensure_ascii=False)
    ev["_color"] = color
    ev["_summary"] = "" if summary_val is None else f"{summary_key}={summary_val}"
    ev["_raw"] = raw
    return ev


def _read_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        decorated = _decorate_event(line)
        if decorated is not None:
            out.append(decorated)
    return out


def _load_presets(repo_root: Path) -> list[dict]:
    """Read docs/examples/*.md as paste-prompt presets.

    Each preset gets {id, label, body}. Skips README.md and files whose
    name starts with '_' (meta/notes). Label is the first '# ' heading,
    falling back to the filename stem.
    """
    examples_dir = repo_root / "docs" / "examples"
    if not examples_dir.is_dir():
        return []
    out: list[dict] = []
    for p in sorted(examples_dir.glob("*.md")):
        if p.name.lower() == "readme.md" or p.name.startswith("_"):
            continue
        try:
            body = p.read_text(encoding="utf-8")
        except OSError:
            continue
        label = p.stem
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                label = stripped[2:].strip()
                break
        out.append({"id": p.stem, "label": label, "body": body})
    return out


def register_routes(
    app: FastAPI, cfg: Config, *, config_path: Path
) -> None:
    """Mount the web UI routes onto an existing FastAPI app."""
    templates = Jinja2Templates(directory=str(_WEB_DIR / "templates"))
    templates.env.filters["fmt_ts"] = _fmt_ts
    templates.env.filters["age"] = _age

    repo_root = config_path.parent

    threads_root = Path(cfg.queue.root) / "threads"
    thread_store = ThreadStore(threads_root)

    app.mount(
        "/static",
        StaticFiles(directory=str(_WEB_DIR / "static")),
        name="static",
    )

    @app.get("/", response_class=HTMLResponse)
    async def root_redirect():
        return RedirectResponse(url="/activity", status_code=302)

    @app.get("/activity", response_class=HTMLResponse)
    async def activity(request: Request):
        return templates.TemplateResponse(
            request, "activity.html",
            {"presets": _load_presets(repo_root), "active": "activity"},
        )

    @app.get("/threads", response_class=HTMLResponse)
    async def threads_index(request: Request):
        return templates.TemplateResponse(
            request, "threads.html",
            {"threads": thread_store.list_threads(limit=200),
             "active": "threads"},
        )

    @app.get("/threads/{thread_id}", response_class=HTMLResponse)
    async def thread_detail(request: Request, thread_id: str):
        if not validate_thread_id(thread_id):
            raise HTTPException(status_code=404, detail="invalid thread id")
        try:
            meta = read_meta(threads_root, thread_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="thread not found")
        turns = []
        for n in range(1, meta.current_turn + 1):
            td = thread_store.turn_dir(thread_id, n)
            if not td.is_dir():
                continue
            prompt = (td / "prompt.md").read_text(encoding="utf-8") \
                if (td / "prompt.md").is_file() else ""
            result = (td / "result.md").read_text(encoding="utf-8") \
                if (td / "result.md").is_file() else ""
            error = (td / "error.txt").read_text(encoding="utf-8") \
                if (td / "error.txt").is_file() else ""
            events = _read_events(td / "events.jsonl")
            turns.append({
                "idx": n, "status": read_status(td) or "unknown",
                "prompt": prompt, "result": result,
                "error": error, "events": events,
            })
        latest_status = turns[-1]["status"] if turns else ""
        can_continue = latest_status in ("done", "failed")
        return templates.TemplateResponse(
            request, "thread.html",
            {"thread": meta, "turns": turns,
             "can_continue": can_continue, "active": "threads"},
        )

    @app.post("/api/threads/{thread_id}/continue")
    async def continue_thread(thread_id: str, body: str = Form(...)):
        if not validate_thread_id(thread_id):
            raise HTTPException(status_code=404, detail="invalid thread id")
        try:
            meta = read_meta(threads_root, thread_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="thread not found")
        latest_status = read_status(thread_store.turn_dir(thread_id, meta.current_turn))
        if latest_status in ("queued", "processing"):
            return PlainTextResponse(
                f"thread has an active turn (turn {meta.current_turn})",
                status_code=409,
            )
        new_dir, new_idx = thread_store.next_turn_dir(thread_id)
        (new_dir / "prompt.md").write_text(body, encoding="utf-8")
        set_status(new_dir, "queued")
        return RedirectResponse(
            url=f"/threads/{thread_id}#turn-{new_idx}", status_code=303,
        )

    @app.post("/api/threads/{thread_id}/rerun/{turn_idx}")
    async def rerun_turn(
        thread_id: str, turn_idx: int,
        body: str | None = Form(None),
    ):
        if not validate_thread_id(thread_id):
            raise HTTPException(status_code=404, detail="invalid thread id")
        try:
            meta = read_meta(threads_root, thread_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="thread not found")
        if turn_idx < 1 or turn_idx > meta.current_turn:
            raise HTTPException(status_code=404, detail="turn not found")
        # Read original prompt for fallback
        orig_path = thread_store.turn_dir(thread_id, turn_idx) / "prompt.md"
        original_prompt = orig_path.read_text(encoding="utf-8") if orig_path.is_file() else ""

        new_prompt = (body or "").strip() or original_prompt
        if not new_prompt:
            raise HTTPException(status_code=400, detail="rerun prompt is empty")

        # Fork: parent_turn_idx = turn_idx - 1 (fork point is BEFORE the
        # reran turn — so turn 1 inherits no parent messages)
        new_tid = thread_store.create_thread(
            title=new_prompt.strip().splitlines()[0][:60] or meta.title,
            parent_thread_id=thread_id,
            parent_turn_idx=turn_idx - 1,
        )
        (thread_store.turn_dir(new_tid, 1) / "prompt.md").write_text(
            new_prompt, encoding="utf-8",
        )
        set_status(thread_store.turn_dir(new_tid, 1), "queued")
        return RedirectResponse(
            url=f"/threads/{new_tid}#turn-1", status_code=303,
        )

    @app.patch("/api/threads/{thread_id}")
    async def patch_thread(thread_id: str, payload: dict = Body(...)):
        if not validate_thread_id(thread_id):
            raise HTTPException(status_code=404, detail="invalid thread id")
        if "title" not in payload:
            raise HTTPException(status_code=400, detail="title required")
        title = str(payload["title"]).strip()
        if not title:
            raise HTTPException(status_code=400, detail="title cannot be empty")
        try:
            update_meta(threads_root, thread_id, title=title)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="thread not found")
        return {"ok": True, "title": title}

    @app.get("/api/jobs/{status}", response_class=HTMLResponse)
    async def jobs_partial(request: Request, status: str):
        limit = 50 if status in ("done", "failed") else None
        rows = _list_turns(thread_store, status, limit=limit)
        return templates.TemplateResponse(
            request, "_partials/turn_list.html",
            {"status": status, "rows": rows},
        )

    @app.post("/api/submit")
    async def submit(
        request: Request,
        file: UploadFile | None = File(None),
        filename: str | None = Form(None),
        body: str | None = Form(None),
    ):
        accepted = list(cfg.queue.accepted_extensions)
        if file is not None and file.filename:
            name = _validate_name(file.filename, accepted)
            if name is None:
                return PlainTextResponse(
                    f"invalid filename or extension: {file.filename!r}",
                    status_code=400,
                )
            content_bytes = await file.read()
            content = content_bytes.decode("utf-8", errors="replace")
        else:
            raw_name = (filename or "").strip() or f"task-{int(time.time())}.md"
            name = _validate_name(raw_name, accepted)
            if name is None:
                return PlainTextResponse(
                    f"invalid filename or extension: {raw_name!r}",
                    status_code=400,
                )
            content = body or ""

        inbox = Path(cfg.queue.root) / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        target = inbox / name
        if target.exists():
            return PlainTextResponse(
                f"{name} already exists in inbox; rename and retry",
                status_code=400,
            )
        tmp = inbox / f".{name}.partial"
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, target)
        return RedirectResponse(url="/activity", status_code=303)

    @app.get("/config", response_class=HTMLResponse)
    async def config_view(request: Request):
        try:
            content = config_path.read_text(encoding="utf-8")
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"cannot read config: {e}")
        return templates.TemplateResponse(
            request, "config.html",
            {
                "path": str(config_path),
                "content": content,
                "highlighted": _highlight_toml(content),
                "active": "config",
            },
        )
