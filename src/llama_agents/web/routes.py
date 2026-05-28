from __future__ import annotations

import json as _json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import Config

_WEB_DIR = Path(__file__).parent


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


_VALID_STATUSES = ("inbox", "processing", "done", "failed")


@dataclass
class _JobEntry:
    name: str
    mtime: float


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


def _list_jobs(root: Path, status: str, *, limit: int | None = None) -> list[_JobEntry]:
    if status not in _VALID_STATUSES:
        raise HTTPException(status_code=404, detail="unknown status")
    dir_ = root / status
    if not dir_.is_dir():
        return []
    rows: list[_JobEntry] = []
    for p in dir_.iterdir():
        if not p.is_file():
            continue
        if p.suffix != ".md" or p.stem.endswith(".prompt"):
            continue
        try:
            mtime = p.stat().st_mtime
        except FileNotFoundError:
            # Worker moved/removed the file between iterdir and stat;
            # skip rather than 500.
            continue
        rows.append(_JobEntry(name=p.name, mtime=mtime))
    rows.sort(key=lambda r: r.mtime, reverse=True)
    if limit is not None:
        rows = rows[:limit]
    return rows


_EVENT_STYLE: dict[str, tuple[str, str | None]] = {
    "PlanProposed":     ("gray",   "attempt"),
    "PlanReviewed":     ("gray",   "accepted"),
    "PlanAccepted":     ("green",  "attempts"),
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


def register_routes(
    app: FastAPI, cfg: Config, *, config_path: Path
) -> None:
    """Mount the web UI routes onto an existing FastAPI app."""
    templates = Jinja2Templates(directory=str(_WEB_DIR / "templates"))
    templates.env.filters["fmt_ts"] = _fmt_ts
    templates.env.filters["age"] = _age

    app.mount(
        "/static",
        StaticFiles(directory=str(_WEB_DIR / "static")),
        name="static",
    )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        return templates.TemplateResponse(request, "dashboard.html", {})

    @app.get("/api/jobs/{status}", response_class=HTMLResponse)
    async def jobs_partial(request: Request, status: str):
        limit = 50 if status in ("done", "failed") else None
        rows = _list_jobs(Path(cfg.queue.root), status, limit=limit)
        return templates.TemplateResponse(
            request, "_partials/job_list.html",
            {"status": status, "jobs": rows},
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
        return RedirectResponse(url="/", status_code=303)

    @app.get("/jobs/{status}/{name}", response_class=HTMLResponse)
    async def job_detail(request: Request, status: str, name: str):
        if status not in _VALID_STATUSES:
            raise HTTPException(status_code=404, detail="unknown status")
        if not _SAFE_NAME.match(name) or not name.endswith(".md"):
            raise HTTPException(status_code=404, detail="invalid name")
        root = Path(cfg.queue.root)
        main = root / status / name
        if not main.is_file():
            raise HTTPException(status_code=404, detail="job not found")

        if status in ("inbox", "processing"):
            prompt_text = main.read_text(encoding="utf-8")
            result_text = ""
        else:
            prompt_sidecar = main.with_suffix(".prompt.md")
            prompt_text = (
                prompt_sidecar.read_text(encoding="utf-8")
                if prompt_sidecar.is_file() else ""
            )
            result_text = main.read_text(encoding="utf-8")

        events = _read_events(main.with_suffix(".events.jsonl"))
        error_text = ""
        if status == "failed":
            err = main.with_suffix(".error.txt")
            if err.is_file():
                error_text = err.read_text(encoding="utf-8")

        return templates.TemplateResponse(
            request, "job.html",
            {
                "status": status,
                "name": name,
                "prompt": prompt_text,
                "events": events,
                "result": result_text,
                "error": error_text,
            },
        )

    @app.get("/config", response_class=HTMLResponse)
    async def config_view(request: Request):
        try:
            content = config_path.read_text(encoding="utf-8")
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"cannot read config: {e}")
        return templates.TemplateResponse(
            request, "config.html",
            {"path": str(config_path), "content": content},
        )
