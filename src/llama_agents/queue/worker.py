from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ..agent import AgentRunOptions
from ..config import QueueConfig
from ..events import AssistantChunk, Done, LoopError
from .paths import (
    ensure_dirs,
    move_to_processing,
    move_to_terminal,
    sweep_processing_to_inbox,
)

logger = logging.getLogger(__name__)

# Error types (LoopError.error_type strings) that warrant a retry.
INFRA_ERROR_TYPES: frozenset[str] = frozenset({"LlamaUnreachable"})


class _RuntimeLike(Protocol):
    def new_agent(self) -> Any: ...


@dataclasses.dataclass
class JobResult:
    success: bool
    final_text: str
    events: list[dict[str, Any]]
    loop_error: LoopError | None
    prompt_text: str


class JobQueueWorker:
    """Polls inbox/, runs each job through the agent, writes outputs."""

    def __init__(self, runtime: _RuntimeLike, cfg: QueueConfig) -> None:
        self._rt = runtime
        self._cfg = cfg
        self._stop = asyncio.Event()
        self._in_flight: set[asyncio.Task] = set()
        ensure_dirs(cfg.root)
        sweep_processing_to_inbox(cfg.root)

    async def run(self) -> None:
        """Main loop. Returns when `drain` is called."""
        while not self._stop.is_set():
            await self._fill_slots()
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self._cfg.poll_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass

    async def drain(self, timeout: float) -> None:
        """Stop accepting new jobs; wait for in-flight to finish."""
        self._stop.set()
        if not self._in_flight:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._in_flight, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            for t in list(self._in_flight):
                t.cancel()

    async def _fill_slots(self) -> None:
        while (
            not self._stop.is_set()
            and len(self._in_flight) < self._cfg.max_concurrent
        ):
            picked = self._pick_one()
            if picked is None:
                return
            task = asyncio.create_task(self._run_job(picked))
            self._in_flight.add(task)
            task.add_done_callback(self._in_flight.discard)

    def _pick_one(self) -> Path | None:
        inbox = Path(self._cfg.root) / "inbox"
        if not inbox.is_dir():
            return None
        candidates = sorted(
            (p for p in inbox.iterdir()
             if p.is_file() and p.suffix in self._cfg.accepted_extensions),
            key=lambda p: p.stat().st_mtime,
        )
        for src in candidates:
            moved = move_to_processing(self._cfg.root, src)
            if moved is not None:
                logger.info("queue: picked up %s", src.name)
                return moved
        return None

    async def _run_job(self, path: Path) -> None:
        attempt = 0
        while True:
            logger.info("queue: running %s (attempt %d)", path.name, attempt + 1)
            result = await self._invoke_agent(path)
            if self._should_retry(result, attempt):
                delay = self._cfg.retry_backoff_seconds * (2 ** attempt)
                logger.info(
                    "queue: %s infra error %s — retrying in %.1fs",
                    path.name,
                    result.loop_error.error_type if result.loop_error else "?",
                    delay,
                )
                if delay > 0:
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=delay)
                    except asyncio.TimeoutError:
                        pass
                if self._stop.is_set():
                    # Shutting down — leave file in processing/ for sweep.
                    return
                attempt += 1
                continue
            await self._finalize(path, result, attempt=attempt)
            return

    def _should_retry(self, result: JobResult, attempt: int) -> bool:
        if result.success or result.loop_error is None:
            return False
        if attempt >= self._cfg.max_retries:
            return False
        return result.loop_error.error_type in INFRA_ERROR_TYPES

    async def _invoke_agent(self, path: Path) -> JobResult:
        prompt = path.read_text(encoding="utf-8")
        agent = self._rt.new_agent()
        opts = AgentRunOptions(max_iterations=self._cfg.max_iterations)

        events: list[dict[str, Any]] = []
        final_chunks: list[str] = []
        loop_error: LoopError | None = None
        async for ev in agent.run(prompt, opts):
            events.append(_serialize_event(ev))
            if isinstance(ev, AssistantChunk):
                final_chunks.append(ev.text)
            elif isinstance(ev, LoopError):
                loop_error = ev

        success = loop_error is None
        return JobResult(
            success=success,
            final_text="\n\n".join(final_chunks) or "[no final answer]",
            events=events,
            loop_error=loop_error,
            prompt_text=prompt,
        )

    async def _finalize(
        self, path: Path, result: JobResult, attempt: int
    ) -> None:
        status = "done" if result.success else "failed"
        dst = move_to_terminal(self._cfg.root, path, status=status)
        dst.write_text(result.final_text, encoding="utf-8")
        events_path = dst.with_suffix(".events.jsonl")
        events_path.write_text(
            "\n".join(json.dumps(e) for e in result.events) + "\n",
            encoding="utf-8",
        )
        prompt_path = dst.with_suffix(".prompt.md")
        prompt_path.write_text(result.prompt_text, encoding="utf-8")
        if not result.success:
            err_path = dst.with_suffix(".error.txt")
            err = result.loop_error
            err_path.write_text(
                f"attempts: {attempt + 1}\n"
                f"error_type: {err.error_type if err else 'Unknown'}\n"
                f"message: {err.message if err else 'no LoopError captured'}\n",
                encoding="utf-8",
            )
        logger.info("queue: %s -> %s", path.name, status)


def _serialize_event(ev: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": type(ev).__name__,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    try:
        payload.update(dataclasses.asdict(ev))
    except TypeError:
        # Non-dataclass event (shouldn't happen — every Event subclass is one).
        payload["repr"] = repr(ev)
    return payload
