from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
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


def _list_jobs(root: Path, status: str, *, limit: int | None = None) -> list[_JobEntry]:
    if status not in _VALID_STATUSES:
        raise HTTPException(status_code=404, detail="unknown status")
    dir_ = root / status
    if not dir_.is_dir():
        return []
    rows = [
        _JobEntry(name=p.name, mtime=p.stat().st_mtime)
        for p in dir_.iterdir()
        if p.is_file()
        and p.suffix == ".md"
        and not p.stem.endswith(".prompt")
    ]
    rows.sort(key=lambda r: r.mtime, reverse=True)
    if limit is not None:
        rows = rows[:limit]
    return rows


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
