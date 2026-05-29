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
from ..thread.migration import migrate_legacy_queue_dirs
from ..thread.status import (
    claim_for_processing, revert_processing_on_startup, set_status,
)
from ..thread.store import ThreadStore
from .paths import ensure_dirs

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
    new_messages: list[dict] = dataclasses.field(default_factory=list)


class JobQueueWorker:
    """Polls thread folders, runs each queued turn through the agent."""

    def __init__(
        self,
        runtime: _RuntimeLike,
        cfg: QueueConfig,
        *,
        thread_store: ThreadStore,
    ) -> None:
        self._rt = runtime
        self._cfg = cfg
        self._thread_store = thread_store
        self._stop = asyncio.Event()
        self._in_flight: set[asyncio.Task] = set()
        ensure_dirs(cfg.root)
        migrate_legacy_queue_dirs(cfg.root)
        revert_processing_on_startup(cfg.root / "threads")

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
            thread_id, turn_idx, turn_dir = picked
            task = asyncio.create_task(self._run_job(thread_id, turn_idx, turn_dir))
            self._in_flight.add(task)
            task.add_done_callback(self._in_flight.discard)

    def _pick_one(self) -> tuple[str, int, Path] | None:
        """Find the oldest queued turn and atomically claim it.

        Returns (thread_id, turn_idx, turn_dir) on success.
        """
        candidate = self._thread_store.next_queued_turn()
        if candidate is None:
            return None
        thread_id, turn_idx = candidate
        turn_dir = self._thread_store.turn_dir(thread_id, turn_idx)
        if not claim_for_processing(turn_dir):
            # Lost the race; try again next poll
            return None
        logger.info("queue: picked up %s/turn-%d", thread_id, turn_idx)
        return thread_id, turn_idx, turn_dir

    async def _run_job(self, thread_id: str, turn_idx: int,
                       turn_dir: Path) -> None:
        attempt = 0
        while True:
            logger.info(
                "queue: running %s/turn-%d (attempt %d)",
                thread_id, turn_idx, attempt + 1,
            )
            result = await self._invoke_agent(thread_id, turn_idx, turn_dir)
            if self._should_retry(result, attempt):
                delay = self._cfg.retry_backoff_seconds * (2 ** attempt)
                logger.info(
                    "queue: %s/turn-%d infra error %s — retrying in %.1fs",
                    thread_id, turn_idx,
                    result.loop_error.error_type if result.loop_error else "?",
                    delay,
                )
                if delay > 0:
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=delay)
                    except asyncio.TimeoutError:
                        pass
                if self._stop.is_set():
                    return
                attempt += 1
                continue
            await self._finalize(thread_id, turn_idx, turn_dir, result,
                                 attempt=attempt)
            return

    def _should_retry(self, result: JobResult, attempt: int) -> bool:
        if result.success or result.loop_error is None:
            return False
        if attempt >= self._cfg.max_retries:
            return False
        return result.loop_error.error_type in INFRA_ERROR_TYPES

    async def _invoke_agent(self, thread_id: str, turn_idx: int,
                            turn_dir: Path) -> JobResult:
        prompt = (turn_dir / "prompt.md").read_text(encoding="utf-8")
        agent = self._rt.new_agent()
        opts = AgentRunOptions(max_iterations=self._cfg.max_iterations)
        prior = self._thread_store.read_messages(thread_id)

        events: list[dict[str, Any]] = []
        final_chunks: list[str] = []
        loop_error: LoopError | None = None
        async for ev in agent.run(prompt, opts, thread_id=thread_id,
                                  prior_messages=prior):
            events.append(_serialize_event(ev))
            if isinstance(ev, AssistantChunk):
                final_chunks.append(ev.text)
            elif isinstance(ev, LoopError):
                loop_error = ev

        # The agent appends to its own self.messages; capture only the new
        # tail beyond `prior`.
        tail_start = 1 + len(prior) + 1  # [system, *prior, user]
        new_messages = list(agent.messages[tail_start:])
        success = loop_error is None
        return JobResult(
            success=success,
            final_text="\n\n".join(final_chunks) or "[no final answer]",
            events=events,
            loop_error=loop_error,
            prompt_text=prompt,
            new_messages=new_messages,
        )

    async def _finalize(self, thread_id: str, turn_idx: int,
                        turn_dir: Path, result: JobResult,
                        attempt: int) -> None:
        # Write side-cars BEFORE flipping status, so readers polling the
        # turn page never see a done turn with a missing result.md.
        (turn_dir / "result.md").write_text(result.final_text, encoding="utf-8")
        (turn_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in result.events) + "\n",
            encoding="utf-8",
        )
        # Append the new agent messages to the thread's messages.jsonl,
        # including the seed user turn we don't already have on disk.
        seed_user = {"role": "user", "content": result.prompt_text}
        self._thread_store.append_messages(
            thread_id, [seed_user, *result.new_messages]
        )

        status = "done" if result.success else "failed"
        if not result.success:
            err = result.loop_error
            (turn_dir / "error.txt").write_text(
                f"attempts: {attempt + 1}\n"
                f"error_type: {err.error_type if err else 'Unknown'}\n"
                f"message: {err.message if err else 'no LoopError captured'}\n",
                encoding="utf-8",
            )
        set_status(turn_dir, status)
        logger.info("queue: %s/turn-%d -> %s", thread_id, turn_idx, status)


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
