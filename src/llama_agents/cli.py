from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

from .agent import AgentRunOptions
from .config import load_config
from .events import AssistantChunk, Done, LoopError, MemoryEvicted, MemoryStored, ToolCallResult, ToolCallStart
from .runtime import Runtime


app = typer.Typer(no_args_is_help=True, help="llama-agents CLI")
console = Console()


def _default_config_path() -> Path:
    env = os.environ.get("LLAMA_AGENTS_CONFIG")
    if env:
        return Path(env)
    return Path("config.toml")


def _render_event(ev: object) -> None:
    """Render a single agent event to the console / stderr."""
    if isinstance(ev, AssistantChunk):
        console.print(Markdown(ev.text))
    elif isinstance(ev, ToolCallStart):
        console.print(f"[dim]→ {ev.name}({ev.arguments})[/dim]")
    elif isinstance(ev, ToolCallResult):
        marker = "✓" if ev.ok else "✗"
        console.print(f"[dim]  {marker} {str(ev.content)[:160]}[/dim]")
    elif isinstance(ev, LoopError):
        console.print(f"[red]{ev.error_type}: {ev.message}[/red]")
    elif isinstance(ev, Done):
        console.print(f"[dim](done: {ev.reason})[/dim]")
    elif isinstance(ev, MemoryStored):
        print(f"  ◦ stored {ev.kind} → mem:{ev.blob_id[:8]} ({ev.bytes_} B)", file=sys.stderr)
    elif isinstance(ev, MemoryEvicted):
        kb = ev.bytes_freed / 1024
        print(f"  ◦ evicted tool result → -{kb:.1f} KB (mem:{ev.blob_id[:8]})", file=sys.stderr)


@app.command()
def chat(
    config: Path = typer.Option(_default_config_path, "--config", "-c"),
    prompt: str = typer.Argument(..., help="Prompt to send to the agent."),
    max_iterations: int = typer.Option(20, "--max-iterations"),
) -> None:
    """Run a single agent turn against the configured llama-server."""
    asyncio.run(_run_chat(config, prompt, max_iterations))


@app.command()
def serve(
    config: Path = typer.Option(_default_config_path, "--config", "-c"),
) -> None:
    """Start the HTTP service."""
    import uvicorn

    from .http_app import create_app

    cfg = load_config(config)
    fastapi_app = create_app(cfg)
    uvicorn.run(fastapi_app, host=cfg.http.host, port=cfg.http.port)


async def _run_chat(config_path: Path, prompt: str, max_iterations: int) -> None:
    cfg = load_config(config_path)
    rt = await Runtime.create(cfg)
    try:
        agent = rt.new_agent()
        opts = AgentRunOptions(max_iterations=max_iterations)
        async for ev in agent.run(prompt, opts):
            _render_event(ev)
    finally:
        await rt.aclose()
