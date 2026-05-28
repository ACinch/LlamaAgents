from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
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
