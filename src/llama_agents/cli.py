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
from .events import (
    AssistantChunk, Done, LoopError, MemoryEvicted, MemoryStored,
    ReviewerVerdict, ToolCallResult, ToolCallStart,
)
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
    elif isinstance(ev, ReviewerVerdict):
        marker = "✓" if ev.accepted else "✗"
        excerpt = ev.feedback[:80]
        console.print(
            f"[dim]  {marker} reviewer {ev.reviewer_idx}: {excerpt}[/dim]"
        )


@app.command()
def chat(
    config: Path = typer.Option(_default_config_path, "--config", "-c"),
    prompt: str = typer.Argument(..., help="Prompt to send to the agent."),
    max_iterations: int = typer.Option(20, "--max-iterations"),
    thread: str | None = typer.Option(
        None, "--thread", "-t",
        help="Continue an existing thread (full id or unique prefix >=4 chars).",
    ),
    background: bool = typer.Option(
        False, "--background",
        help="Submit the turn with status=queued and exit immediately. "
             "The queue worker (running in `llamactl serve`) picks it up.",
    ),
) -> None:
    """Run a single agent turn against the configured llama-server."""
    from .thread.ids import AmbiguousPrefix, UnknownPrefix, resolve_prefix
    from .thread.meta import read_meta
    from .thread.status import read_status, set_status
    from .thread.store import ThreadStore
    from .runtime import _resolve_queue_root

    cfg = load_config(config)
    queue_root = _resolve_queue_root(cfg)
    threads_root = queue_root / "threads"
    threads_root.mkdir(parents=True, exist_ok=True)
    store = ThreadStore(threads_root)

    # Resolve --thread prefix
    if thread is not None:
        try:
            thread_id = resolve_prefix(threads_root, thread)
        except UnknownPrefix as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=2)
        except AmbiguousPrefix as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=2)
        meta = read_meta(threads_root, thread_id)
        latest_status = read_status(store.turn_dir(thread_id, meta.current_turn))
        if latest_status in ("queued", "processing"):
            typer.echo(
                f"thread {thread_id[:8]} has an active turn ({meta.current_turn})"
            )
            raise typer.Exit(code=3)
        # New turn in existing thread
        turn_dir, turn_idx = store.next_turn_dir(thread_id)
    else:
        # New thread
        thread_id = store.create_thread(
            title=prompt.strip().splitlines()[0][:60] or "untitled",
        )
        turn_dir = store.turn_dir(thread_id, 1)
        turn_idx = 1

    (turn_dir / "prompt.md").write_text(prompt, encoding="utf-8")

    if background:
        set_status(turn_dir, "queued")
        typer.echo(f"Thread: {thread_id} (turn {turn_idx})")
        return

    # In-process run (synchronous)
    set_status(turn_dir, "processing")
    asyncio.run(_run_chat_in_process(cfg, thread_id, turn_idx, turn_dir,
                                     prompt, max_iterations, store))


@app.command()
def serve(
    config: Path = typer.Option(_default_config_path, "--config", "-c"),
) -> None:
    """Start the HTTP service."""
    import uvicorn

    from .http_app import create_app

    cfg = load_config(config)
    fastapi_app = create_app(cfg, config_path=config)
    uvicorn.run(fastapi_app, host=cfg.http.host, port=cfg.http.port)


@app.command()
def init(
    force: bool = typer.Option(
        False, "--force",
        help="Overwrite existing config.toml without prompting (backs up first).",
    ),
) -> None:
    """Interactive first-run setup: detects llama-server, picks a model, writes config.toml."""
    from .install import RichPrompter, run_install_wizard
    result = run_install_wizard(
        repo_root=Path.cwd(),
        prompter=RichPrompter(),
        force=force,
    )
    if result is None:
        raise typer.Exit(code=1)


threads_app = typer.Typer(no_args_is_help=True, help="Manage threads.")
app.add_typer(threads_app, name="threads")


_threads_config: Path | None = None


@threads_app.callback(invoke_without_command=False)
def threads_callback(
    config: Path = typer.Option(_default_config_path, "--config", "-c"),
) -> None:
    """Threads subcommand group."""
    global _threads_config
    _threads_config = config


def _get_threads_config() -> Path:
    """Get config, preferring the one set by callback."""
    if _threads_config is not None:
        return _threads_config
    return _default_config_path()


@threads_app.command("list")
def threads_list(
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """List threads, newest first."""
    from .thread.store import ThreadStore
    from .runtime import _resolve_queue_root

    config = _get_threads_config()
    cfg = load_config(config)
    threads_root = _resolve_queue_root(cfg) / "threads"
    threads_root.mkdir(parents=True, exist_ok=True)
    store = ThreadStore(threads_root)
    metas = store.list_threads(limit=limit)
    if not metas:
        typer.echo("No threads yet.")
        return
    # 4-column table
    typer.echo(f"{'ID':<10}{'Title':<50}{'Turns':>6}  {'Updated'}")
    for m in metas:
        title = (m.title[:47] + "...") if len(m.title) > 50 else m.title
        typer.echo(f"{m.id[:8]:<10}{title:<50}{m.current_turn:>6}  {m.updated_at}")


@threads_app.command("show")
def threads_show(
    thread: str = typer.Argument(...),
    full: bool = typer.Option(False, "--full"),
) -> None:
    """Render every turn in a thread."""
    from .thread.ids import resolve_prefix, AmbiguousPrefix, UnknownPrefix
    from .thread.meta import read_meta
    from .thread.status import read_status
    from .thread.store import ThreadStore
    from .runtime import _resolve_queue_root

    config = _get_threads_config()
    cfg = load_config(config)
    threads_root = _resolve_queue_root(cfg) / "threads"
    store = ThreadStore(threads_root)
    try:
        tid = resolve_prefix(threads_root, thread)
    except (UnknownPrefix, AmbiguousPrefix) as e:
        typer.echo(str(e))
        raise typer.Exit(code=2)
    meta = read_meta(threads_root, tid)
    typer.echo(f"{meta.title} ({tid})")
    for n in range(1, meta.current_turn + 1):
        td = store.turn_dir(tid, n)
        status = read_status(td) or "unknown"
        typer.echo(f"\n── Turn {n} — {status} ──")
        prompt_p = td / "prompt.md"
        if prompt_p.is_file():
            typer.echo("Prompt:")
            typer.echo(prompt_p.read_text(encoding="utf-8"))
        result_p = td / "result.md"
        if result_p.is_file():
            typer.echo("Result:")
            typer.echo(result_p.read_text(encoding="utf-8"))
        error_p = td / "error.txt"
        if error_p.is_file():
            typer.echo(error_p.read_text(encoding="utf-8"))


@threads_app.command("rerun")
def threads_rerun(
    thread: str = typer.Argument(...),
    turn: int = typer.Argument(...),
    edit: str | None = typer.Option(None, "--edit"),
) -> None:
    """Fork a thread by rerunning turn N, optionally with an edited prompt."""
    from .thread.ids import resolve_prefix, AmbiguousPrefix, UnknownPrefix
    from .thread.meta import read_meta
    from .thread.status import set_status
    from .thread.store import ThreadStore
    from .runtime import _resolve_queue_root

    config = _get_threads_config()
    cfg = load_config(config)
    threads_root = _resolve_queue_root(cfg) / "threads"
    store = ThreadStore(threads_root)
    try:
        parent_id = resolve_prefix(threads_root, thread)
    except (UnknownPrefix, AmbiguousPrefix) as e:
        typer.echo(str(e))
        raise typer.Exit(code=2)
    parent_meta = read_meta(threads_root, parent_id)
    if turn < 1 or turn > parent_meta.current_turn:
        typer.echo(f"turn {turn} out of range")
        raise typer.Exit(code=2)

    orig_prompt = (store.turn_dir(parent_id, turn) / "prompt.md").read_text(
        encoding="utf-8",
    )
    new_prompt = (edit or "").strip() or orig_prompt
    new_tid = store.create_thread(
        title=new_prompt.strip().splitlines()[0][:60],
        parent_thread_id=parent_id,
        parent_turn_idx=turn - 1,
    )
    (store.turn_dir(new_tid, 1) / "prompt.md").write_text(new_prompt, encoding="utf-8")
    set_status(store.turn_dir(new_tid, 1), "queued")
    typer.echo(f"Forked → thread {new_tid} (queued; will run via worker or `chat -t {new_tid[:8]}`)")


async def _run_chat_in_process(cfg, thread_id, turn_idx, turn_dir, prompt,
                               max_iterations, store):
    import json as _json
    from .thread.status import set_status as _set_status
    rt = await Runtime.create(cfg)
    try:
        agent = rt.new_agent()
        opts = AgentRunOptions(max_iterations=max_iterations)
        prior = store.read_messages(thread_id)
        events: list[dict] = []
        final_chunks: list[str] = []
        had_error = False
        async for ev in agent.run(prompt, opts, thread_id=thread_id,
                                  prior_messages=prior):
            _render_event(ev)
            from .queue.worker import _serialize_event
            events.append(_serialize_event(ev))
            if isinstance(ev, AssistantChunk):
                final_chunks.append(ev.text)
            elif isinstance(ev, LoopError):
                had_error = True
        result_text = "\n\n".join(final_chunks) or "[no final answer]"
        (turn_dir / "result.md").write_text(result_text, encoding="utf-8")
        (turn_dir / "events.jsonl").write_text(
            "\n".join(_json.dumps(e) for e in events) + "\n", encoding="utf-8",
        )
        # Append new messages to the thread
        tail_start = 1 + len(prior) + 1
        new_msgs = list(agent.messages[tail_start:])
        seed_user = {"role": "user", "content": prompt}
        store.append_messages(thread_id, [seed_user, *new_msgs])
        _set_status(turn_dir, "failed" if had_error else "done")
        console.print(f"\n[dim]Thread: {thread_id}[/dim]")
    finally:
        await rt.aclose()
