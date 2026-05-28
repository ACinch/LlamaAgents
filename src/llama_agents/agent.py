from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Protocol

from .errors import LlamaAgentsError
from .events import (
    AssistantChunk,
    Done,
    Event,
    LoopError,
    MemoryEvicted,
    MemoryStored,
    PlanAccepted,
    PlanProposed,
    PlanReviewed,
    ReviewerVerdict,
    ToolCallResult,
    ToolCallStart,
)


_ACTIVE_RUN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llama_agents_active_run_id", default=None
)


def get_active_run_id() -> str | None:
    return _ACTIVE_RUN_ID.get()
from .llama_client import ChatResponse
from .memory.store import InertMemoryStore, MemoryStore
from .tools.registry import ToolRegistry


_REVIEWER_SYSTEM_PROMPT = (
    "You are an independent reviewer. You did NOT write the plan you are "
    "about to evaluate. Treat it skeptically.\n"
    "\n"
    "Score the plan against this five-item checklist. For each item write "
    "PASS / FAIL / SKIP followed by one sentence of justification.\n"
    "\n"
    "1. Tool validity — every step names a tool from the provided list.\n"
    "2. Scope match — the plan accomplishes the user's stated goal.\n"
    "3. Context safety — no obvious context-window blowups (broad globs "
    "over directories like .venv or node_modules; chained reads of very "
    "large files).\n"
    "4. Specificity — no vague hand-waves (e.g. \"use shell to figure "
    "out the project\").\n"
    "5. Executability — a worker following the steps verbatim could "
    "actually run them, in order, without filling gaps.\n"
    "\n"
    "After the checklist, on its own final line, output EXACTLY one of:\n"
    "  ACCEPT\n"
    "  REJECT: <one sentence describing the single most important fix>"
)


def _normalize_verdict(raw: str) -> tuple[bool, str]:
    """Parse a reviewer's response into (accepted, feedback).

    Permissive but conservative: a response is ACCEPT only if its
    final non-empty line, stripped, starts with 'ACCEPT' (case-insensitive).
    Anything else is REJECT with the relevant feedback extracted.
    """
    text = (raw or "").rstrip()
    if not text:
        return False, ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False, ""
    last = lines[-1].strip()
    if last.upper().startswith("ACCEPT"):
        return True, ""
    if last.upper().startswith("REJECT"):
        # Strip the "REJECT:" prefix if present.
        if ":" in last:
            return False, last.split(":", 1)[1].strip()
        return False, last
    # Malformed: no recognized verdict marker. Use raw output as feedback,
    # truncated to 500 chars.
    truncated = text[:500]
    return False, truncated


async def _one_reviewer(
    client: "_ClientLike",
    *,
    user_prompt: str,
    plan: str,
    reviewer_system: str,
    temperature: float,
    reasoning_budget_tokens: "int | None",
) -> tuple[bool, str]:
    """Run one reviewer pass. Returns (accepted, feedback)."""
    review_msgs = [
        {"role": "system", "content": reviewer_system},
        {
            "role": "user",
            "content": (
                "TASK:\n" + user_prompt + "\n\nPROPOSED PLAN:\n" + plan
            ),
        },
    ]
    resp = await client.chat(
        messages=review_msgs,
        tools=[],
        temperature=temperature,
        reasoning_budget_tokens=reasoning_budget_tokens,
    )
    return _normalize_verdict(resp.content or "")


class _ClientLike(Protocol):
    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = ...,
        reasoning_budget_tokens: int | None = ...,
    ) -> ChatResponse: ...


@dataclass
class AgentRunOptions:
    max_iterations: int = 20
    system_prompt: str = (
        "You are a careful coding agent. Use tools to read files, run commands, "
        "and query the RAG when helpful. When finished, reply in plain text."
    )
    temperature: float = 0.2
    reasoning_budget_tokens: int | None = 8000
    """Per-turn cap on extended-reasoning tokens (Qwen3/DeepSeek-R1 style
    <think> blocks). Default 8000 fits comfortably inside a 64k context and
    gives the planner room for genuine chain-of-thought without runaway
    monologues. Set to None to defer to the server default (unlimited —
    only safe for well-behaved models). Set to 0 to disable thinking."""
    skip_planning: bool = False
    """If False (default), Agent.run() runs a plan + self-review phase
    before the main tool loop, but ONLY when the registry includes
    subagent_spawn (i.e. only for orchestrator agents — subagents skip it
    to avoid recursion). Set True to suppress unconditionally."""
    max_planning_iterations: int = 3
    """Maximum plan-then-review cycles. After this many rejections the
    most recent plan is used as-is."""
    reviewer_count: int = 3
    """Number of independent reviewer calls per planning attempt. The
    plan is accepted iff a strict majority (>count//2) vote ACCEPT.
    Set to 1 to disable consensus (legacy single-reviewer behavior)."""
    reviewer_temperature: float = 0.5
    """Sampling temperature for reviewer calls. Higher than the planner
    so reviewers have room to disagree with each other."""
    plan_recall_k: int = 3
    plan_recall_threshold: float = 0.5
    evict_threshold_pct: int = 70
    evict_tool_result_min_chars: int = 4000
    ctx_size_for_eviction: int = 65536  # in tokens


class Agent:
    def __init__(
        self,
        *,
        client: _ClientLike,
        registry: ToolRegistry,
        memory: "MemoryStore | InertMemoryStore | None" = None,
    ) -> None:
        self._client = client
        self._registry = registry
        self._memory = memory or InertMemoryStore()
        self._cancel = asyncio.Event()
        self._run_id: str | None = None
        self.messages: list[dict[str, Any]] = []

    def cancel(self) -> None:
        self._cancel.set()

    async def run(
        self, user_prompt: str, opts: AgentRunOptions, *,
        run_id: str | None = None,
    ) -> AsyncIterator[Event]:
        import uuid
        self._run_id = run_id or uuid.uuid4().hex[:24]
        self._memory.start_run(self._run_id)
        token = _ACTIVE_RUN_ID.set(self._run_id)
        try:
            effective_prompt = user_prompt
            if self._should_plan(opts):
                async for ev in self._plan_and_review(user_prompt, opts):
                    yield ev
                    if isinstance(ev, PlanAccepted):
                        effective_prompt = (
                            "APPROVED PLAN (already reviewed — execute it; do not "
                            "re-plan):\n"
                            f"{ev.plan}\n\n"
                            "ORIGINAL TASK:\n"
                            f"{user_prompt}"
                        )

            self.messages = [
                {"role": "system", "content": opts.system_prompt},
                {"role": "user", "content": effective_prompt},
            ]
            for _ in range(opts.max_iterations):
                if self._cancel.is_set():
                    yield Done(reason="cancelled")
                    return

                try:
                    resp = await self._client.chat(
                        messages=self.messages,
                        tools=self._registry.schemas(),
                        temperature=opts.temperature,
                        reasoning_budget_tokens=opts.reasoning_budget_tokens,
                    )
                except LlamaAgentsError as e:
                    yield LoopError(error_type=type(e).__name__, message=str(e))
                    yield Done(reason="error")
                    return

                self.messages.append(
                    resp.raw_message
                    or {"role": "assistant", "content": resp.content}
                )

                if not resp.tool_calls:
                    if resp.content:
                        yield AssistantChunk(text=resp.content)
                    yield Done(reason="finished", final_message=resp.content)
                    return

                for call in resp.tool_calls:
                    yield ToolCallStart(
                        call_id=call.id, name=call.name, arguments=call.arguments
                    )
                    try:
                        result = await self._registry.invoke(call.name, call.arguments)
                        ok, content = True, result
                    except Exception as e:  # noqa: BLE001 — feed all tool errors back
                        ok, content = False, f"{type(e).__name__}: {e}"
                    yield ToolCallResult(call_id=call.id, ok=ok, content=content)
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": _as_tool_text(ok, content),
                        }
                    )

                async for ev in self._maybe_evict(opts):
                    yield ev

            yield Done(reason="max_iterations")
        finally:
            _ACTIVE_RUN_ID.reset(token)
            await self._memory.end_run(self._run_id)

    _EST_CHARS_PER_TOKEN: float = 3.5

    async def _maybe_evict(self, opts: "AgentRunOptions"):
        """Async generator — yields MemoryEvicted events when threshold crossed."""
        budget_tokens = opts.ctx_size_for_eviction * (opts.evict_threshold_pct / 100.0)
        est = sum(len(_msg_str(m)) for m in self.messages) / self._EST_CHARS_PER_TOKEN
        if est < budget_tokens:
            return
        last_preserved = max(0, len(self.messages) - 4)
        for i in range(last_preserved):
            msg = self.messages[i]
            if msg.get("role") != "tool":
                continue
            body = msg.get("content") or ""
            if not isinstance(body, str) or len(body) < opts.evict_tool_result_min_chars:
                continue
            try:
                blob_id = await self._memory.store_blob(
                    kind="evicted_tool", scope="run", run_id=self._run_id,
                    title=f"tool result @ msg {i}",
                    body=body,
                    metadata={"tool_call_id": msg.get("tool_call_id")},
                )
            except Exception as e:  # noqa: BLE001
                import sys
                print(f"[memory] eviction store failed: {e}", file=sys.stderr)
                continue
            if not blob_id:
                # Inert store (or other no-op) returned empty; skip rewrite to
                # avoid destroying the message body irrecoverably.
                continue
            freed = len(body)
            stub = (
                f"[evicted to memory — use memory_recall("
                f"handle=\"{blob_id}\", query=...) to retrieve. "
                f"Original size: {freed} chars.]"
            )
            msg["content"] = stub
            yield MemoryStored(blob_id=blob_id, kind="evicted_tool",
                               scope="run", bytes_=freed)
            yield MemoryEvicted(blob_id=blob_id, turn=i,
                                bytes_freed=freed - len(stub))
            est -= (freed - len(stub)) / self._EST_CHARS_PER_TOKEN
            if est < opts.ctx_size_for_eviction * 0.5:
                break

    def _should_plan(self, opts: AgentRunOptions) -> bool:
        if opts.skip_planning:
            return False
        # Subagents (registries without subagent_spawn) skip planning to
        # prevent unbounded recursion.
        return "subagent_spawn" in self._registry.names()

    async def _plan_and_review(
        self, user_prompt: str, opts: AgentRunOptions
    ) -> AsyncIterator[Event]:
        tool_names = ", ".join(sorted(self._registry.names()))
        planner_system = (
            "You are a planning agent. Produce a concise numbered plan (3-8 "
            "steps) for accomplishing the user's task. Each step must name a "
            "specific tool to call and the arguments it should receive when "
            "the step is executed. Available tools: " + tool_names + ". "
            "Output ONLY the numbered list — no preamble, no explanation."
        )
        reviewer_system = _REVIEWER_SYSTEM_PROMPT

        prior = []
        try:
            prior = await self._memory.recall(
                query=user_prompt, scope="plans",
                k=opts.plan_recall_k,
                min_score=opts.plan_recall_threshold,
            )
        except Exception as e:  # noqa: BLE001
            import sys
            print(f"[memory] plan recall failed: {e}", file=sys.stderr)
        if prior:
            banner = "\n\nPRIOR ACCEPTED PLANS FOR SIMILAR TASKS:\n" + \
                "\n---\n".join(c.text for c in prior)
            planner_system = planner_system + banner
            reviewer_system = reviewer_system + banner

        plan_history: list[dict[str, Any]] = [
            {"role": "system", "content": planner_system},
            {"role": "user", "content": user_prompt},
        ]
        last_plan = ""
        for attempt in range(1, opts.max_planning_iterations + 1):
            if self._cancel.is_set():
                return
            try:
                plan_resp = await self._client.chat(
                    messages=plan_history,
                    tools=[],
                    temperature=opts.temperature,
                    reasoning_budget_tokens=opts.reasoning_budget_tokens,
                )
            except LlamaAgentsError as e:
                yield LoopError(error_type=type(e).__name__, message=str(e))
                return
            last_plan = (plan_resp.content or "").strip()
            yield PlanProposed(attempt=attempt, plan=last_plan)

            tasks = [
                _one_reviewer(
                    self._client,
                    user_prompt=user_prompt,
                    plan=last_plan,
                    reviewer_system=reviewer_system,
                    temperature=opts.reviewer_temperature,
                    reasoning_budget_tokens=opts.reasoning_budget_tokens,
                )
                for _ in range(opts.reviewer_count)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            if self._cancel.is_set():
                return

            # If EVERY reviewer raised — or reviewer_count was misconfigured
            # to 0 so we got no results at all — fail the planning phase.
            if results and all(isinstance(r, Exception) for r in results):
                first = next(r for r in results if isinstance(r, Exception))
                yield LoopError(error_type=type(first).__name__, message=str(first))
                return
            if not results:
                yield LoopError(
                    error_type="ValueError",
                    message=f"reviewer_count must be >= 1, got {opts.reviewer_count}",
                )
                return

            # Normalize exceptions into REJECT votes; otherwise unpack tuples.
            verdicts: list[tuple[bool, str]] = []
            for r in results:
                if isinstance(r, Exception):
                    msg = f"{type(r).__name__}: {r}"[:500]
                    verdicts.append((False, msg))
                else:
                    verdicts.append(r)

            for i, (accepted_i, feedback_i) in enumerate(verdicts):
                yield ReviewerVerdict(
                    attempt=attempt, reviewer_idx=i,
                    accepted=accepted_i, feedback=feedback_i,
                )

            # Strict majority.
            accepts = sum(1 for v, _ in verdicts if v)
            majority_accepted = accepts > opts.reviewer_count // 2

            if majority_accepted:
                feedback = ""
            else:
                # Dedupe rejection reasons (lowercase-strip equality)
                seen: set[str] = set()
                distinct: list[str] = []
                for v, f in verdicts:
                    if v:
                        continue
                    key = f.strip().lower()
                    if key and key not in seen:
                        seen.add(key)
                        distinct.append(f.strip())
                if len(distinct) <= 1:
                    feedback = distinct[0] if distinct else ""
                else:
                    feedback = "Reviewers rejected:\n" + "\n".join(
                        f"- {d}" for d in distinct
                    )

            yield PlanReviewed(attempt=attempt, accepted=majority_accepted,
                               feedback=feedback)
            accepted = majority_accepted
            if accepted:
                blob_id = ""
                try:
                    blob_id = await self._memory.store_plan(
                        task=user_prompt, plan=last_plan,
                        accepted_attempt=attempt, run_id=self._run_id,
                    )
                except Exception as e:  # noqa: BLE001
                    import sys
                    print(f"[memory] plan store failed: {e}", file=sys.stderr)
                if blob_id:
                    yield MemoryStored(blob_id=blob_id, kind="plan",
                                       scope="plans", bytes_=len(last_plan))
                yield PlanAccepted(plan=last_plan, attempts=attempt)
                return
            plan_history.append({"role": "assistant", "content": last_plan})
            plan_history.append(
                {
                    "role": "user",
                    "content": (
                        f"Reviewer rejected: {feedback}\n"
                        "Revise the plan addressing the rejection. Output the "
                        "full revised numbered list only."
                    ),
                }
            )

        # Exhausted retries — accept the last attempt rather than block the loop.
        blob_id = ""
        try:
            blob_id = await self._memory.store_plan(
                task=user_prompt, plan=last_plan,
                accepted_attempt=opts.max_planning_iterations,
                run_id=self._run_id,
            )
        except Exception as e:  # noqa: BLE001
            import sys
            print(f"[memory] plan store failed: {e}", file=sys.stderr)
        if blob_id:
            yield MemoryStored(blob_id=blob_id, kind="plan",
                               scope="plans", bytes_=len(last_plan))
        yield PlanAccepted(plan=last_plan, attempts=opts.max_planning_iterations)


def _msg_str(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    return _json_dump(c)


def _as_tool_text(ok: bool, content: Any) -> str:
    if ok:
        return content if isinstance(content, str) else _json_dump(content)
    return _json_dump({"error": str(content)})


def _json_dump(x: Any) -> str:
    import json

    try:
        return json.dumps(x, default=str)
    except (TypeError, ValueError):
        return str(x)
