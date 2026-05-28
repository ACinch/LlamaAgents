# Reviewer-as-Subagent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current single-call plan reviewer in `Agent._plan_and_review` with an adversarial, checklist-driven reviewer that runs three independent passes in parallel and accepts only on a strict-majority vote — meaningfully reducing the same-model confirmation bias without adding a second model.

**Architecture:** All work happens inside `src/llama_agents/agent.py`. A new `_one_reviewer` private helper performs a single reviewer call (new adversarial system prompt + five-item checklist parsing). `_plan_and_review` launches N copies via `asyncio.gather(..., return_exceptions=True)`, tallies a strict majority, and emits one new `ReviewerVerdict` event per reviewer in addition to the existing `PlanReviewed`. New per-run knobs `reviewer_count` (default 3) and `reviewer_temperature` (default 0.5) on `AgentRunOptions`. No new modules. No second model. No queue/memory plumbing changes.

**Tech Stack:** Python 3.12+, `asyncio.gather`, the project's existing `LlamaClient`, pytest. Surface plumbing touches CLI (`rich`), HTTP (`fastapi`/`sse-starlette`), and the web UI (Jinja2 — already wired for events).

**Spec:** `docs/superpowers/specs/2026-05-28-reviewer-as-subagent-design.md`

---

## File structure (locked in by this plan)

**Modified — no new modules:**

- `src/llama_agents/events.py` — add `ReviewerVerdict` dataclass.
- `src/llama_agents/agent.py` — new `_REVIEWER_SYSTEM_PROMPT` constant, new `_one_reviewer` helper, new `_normalize_verdict` parser, rewritten reviewer section in `_plan_and_review`, two new fields on `AgentRunOptions`.
- `src/llama_agents/cli.py` — render `ReviewerVerdict` events.
- `src/llama_agents/http_app.py` — `_serialize` branch for `ReviewerVerdict`.
- `src/llama_agents/web/routes.py` — `_EVENT_STYLE` entry for `ReviewerVerdict`.
- `tests/unit/test_agent_loop.py` — extend existing planning tests for the 3× reviewer flow; add the 10 new tests in the spec.
- `tests/unit/test_events.py` — small dataclass construction test for `ReviewerVerdict`.
- `CLAUDE.md` — remove the "Reviewer can confirm bad plans" entry from the limitations list; update the Agent.run() shape section to mention the consensus rule + defaults.

---

## Conventions for this plan

- **Always run from the repo root** (`D:\repos\llm\llama-agents`).
- **Always use `uv run pytest ...`** for tests. On Windows, prefix:
  ```
  $env:Path = "$env:USERPROFILE\AppData\Roaming\Python\Python314\Scripts;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; uv run pytest ...
  ```
- **Commit after each task** with the verbatim commit message in the task's final step.
- **Branch:** `main` (project convention since rename).
- **TDD:** every behavioural task writes failing tests first, then minimal code to pass.
- No real network calls — everything is `ScriptedClient` based.

---

## Task 1: Event + AgentRunOptions fields

**Files:**
- Modify: `src/llama_agents/events.py`
- Modify: `src/llama_agents/agent.py`
- Modify: `tests/unit/test_events.py`

- [ ] **Step 1: Append a failing event-construction test**

Add to `tests/unit/test_events.py`:

```python
def test_reviewer_verdict_constructs():
    from llama_agents.events import Event, ReviewerVerdict

    v = ReviewerVerdict(attempt=2, reviewer_idx=1, accepted=False,
                        feedback="step 3 references a tool that does not exist")
    assert isinstance(v, Event)
    assert v.attempt == 2
    assert v.reviewer_idx == 1
    assert v.accepted is False
    assert v.feedback.startswith("step 3")
```

And a test that `AgentRunOptions` exposes the new fields with the right defaults. Append to `tests/unit/test_agent_loop.py`:

```python
def test_agent_run_options_has_reviewer_defaults():
    from llama_agents.agent import AgentRunOptions

    opts = AgentRunOptions()
    assert opts.reviewer_count == 3
    assert opts.reviewer_temperature == 0.5
    # Backwards compat: existing fields untouched
    assert opts.max_planning_iterations == 3
    assert opts.max_iterations == 20
```

- [ ] **Step 2: Run tests — expect ImportError + AttributeError**

```
$env:Path = "$env:USERPROFILE\AppData\Roaming\Python\Python314\Scripts;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; uv run pytest tests/unit/test_events.py::test_reviewer_verdict_constructs tests/unit/test_agent_loop.py::test_agent_run_options_has_reviewer_defaults -v
```

- [ ] **Step 3: Add `ReviewerVerdict` to events.py**

Append to `src/llama_agents/events.py`:

```python
@dataclass
class ReviewerVerdict(Event):
    attempt: int
    reviewer_idx: int   # 0-indexed within an attempt
    accepted: bool
    feedback: str
```

- [ ] **Step 4: Add reviewer fields to `AgentRunOptions`**

In `src/llama_agents/agent.py`, find the `AgentRunOptions` dataclass and add two fields after `max_planning_iterations`:

```python
    reviewer_count: int = 3
    """Number of independent reviewer calls per planning attempt. The
    plan is accepted iff a strict majority (>count//2) vote ACCEPT.
    Set to 1 to disable consensus (legacy single-reviewer behavior)."""
    reviewer_temperature: float = 0.5
    """Sampling temperature for reviewer calls. Higher than the planner
    so reviewers have room to disagree with each other."""
```

- [ ] **Step 5: Run tests to verify they pass**

```
uv run pytest tests/unit/test_events.py tests/unit/test_agent_loop.py -q
```

Expected: all green (existing planning tests still pass because the new fields default to 3/0.5 but the reviewer code path itself is unchanged in this task — gather/consensus arrives in Task 3).

Wait — actually the new default of `reviewer_count=3` doesn't affect anything yet because we haven't touched `_plan_and_review`. Existing planning tests provide 1 reviewer response per attempt and the existing code still consumes exactly 1 per attempt. Good.

- [ ] **Step 6: Commit**

```
git add src/llama_agents/events.py src/llama_agents/agent.py tests/unit/test_events.py tests/unit/test_agent_loop.py
git commit -m "feat(agent): ReviewerVerdict event + reviewer_count/temperature options"
```

---

## Task 2: Adversarial reviewer prompt + verdict parser + `_one_reviewer` helper

**Files:**
- Modify: `src/llama_agents/agent.py`
- Modify: `tests/unit/test_agent_loop.py`

This task replaces the inline reviewer call inside `_plan_and_review` with a `_one_reviewer` helper. Behavior under `reviewer_count=1` (the path effective after this task) is functionally equivalent to today: one ACCEPT/REJECT verdict drives the loop. The prompt becomes the adversarial 5-item checklist version, and the parser becomes "final non-empty line starts with ACCEPT".

- [ ] **Step 1: Write failing parser tests**

Append to `tests/unit/test_agent_loop.py`:

```python
def test_normalize_verdict_accepts_simple_accept():
    from llama_agents.agent import _normalize_verdict
    accepted, feedback = _normalize_verdict("ACCEPT")
    assert accepted is True
    assert feedback == ""


def test_normalize_verdict_accepts_with_checklist_preamble():
    from llama_agents.agent import _normalize_verdict
    raw = (
        "Tool validity: PASS\n"
        "Scope match: PASS\n"
        "Context safety: PASS\n"
        "Specificity: PASS\n"
        "Executability: PASS\n"
        "\n"
        "ACCEPT"
    )
    accepted, feedback = _normalize_verdict(raw)
    assert accepted is True


def test_normalize_verdict_rejects_with_reason():
    from llama_agents.agent import _normalize_verdict
    raw = "Tool validity: FAIL — step 3 names rag_query, no such tool.\nREJECT: step 3 names a non-existent tool"
    accepted, feedback = _normalize_verdict(raw)
    assert accepted is False
    assert "step 3" in feedback


def test_normalize_verdict_handles_lowercase():
    from llama_agents.agent import _normalize_verdict
    accepted, feedback = _normalize_verdict("accept")
    assert accepted is True


def test_normalize_verdict_malformed_counts_as_reject_with_truncated_raw():
    from llama_agents.agent import _normalize_verdict
    raw = "the plan seems okay i guess " * 50  # no ACCEPT/REJECT prefix
    accepted, feedback = _normalize_verdict(raw)
    assert accepted is False
    assert len(feedback) <= 500
    assert "the plan seems okay" in feedback


def test_normalize_verdict_empty_string_rejects():
    from llama_agents.agent import _normalize_verdict
    accepted, feedback = _normalize_verdict("")
    assert accepted is False


def test_normalize_verdict_trailing_whitespace_stripped():
    from llama_agents.agent import _normalize_verdict
    accepted, feedback = _normalize_verdict("ACCEPT\n\n  \n")
    assert accepted is True
```

- [ ] **Step 2: Run tests — expect ImportError**

```
uv run pytest tests/unit/test_agent_loop.py -k normalize_verdict -v
```

- [ ] **Step 3: Add `_REVIEWER_SYSTEM_PROMPT` constant and `_normalize_verdict` to agent.py**

In `src/llama_agents/agent.py`, near the top (after imports, before any class), add:

```python
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
```

- [ ] **Step 4: Run parser tests — expect pass**

```
uv run pytest tests/unit/test_agent_loop.py -k normalize_verdict -v
```

Expected: 7 tests pass.

- [ ] **Step 5: Extract `_one_reviewer` helper and wire it into `_plan_and_review`**

Read `src/llama_agents/agent.py` and find the `_plan_and_review` method. Inside the planning loop, find the block that builds `review_msgs` and calls `self._client.chat` for the review:

```python
            review_msgs = [
                {"role": "system", "content": reviewer_system},
                {
                    "role": "user",
                    "content": (
                        "TASK:\n" + user_prompt + "\n\nPROPOSED PLAN:\n" + last_plan
                    ),
                },
            ]
            try:
                rev_resp = await self._client.chat(
                    messages=review_msgs,
                    tools=[],
                    temperature=0.0,
                    reasoning_budget_tokens=opts.reasoning_budget_tokens,
                )
            except LlamaAgentsError as e:
                yield LoopError(error_type=type(e).__name__, message=str(e))
                return
            verdict = (rev_resp.content or "").strip()
            accepted = verdict.upper().startswith("ACCEPT")
            yield PlanReviewed(attempt=attempt, accepted=accepted, feedback=verdict)
```

Replace it with a call to the new helper. First, near the top of the `Agent` class (or as a module-level function — module-level is cleaner since it doesn't touch self-state), add:

```python
async def _one_reviewer(
    client: _ClientLike,
    *,
    user_prompt: str,
    plan: str,
    reviewer_system: str,
    temperature: float,
    reasoning_budget_tokens: int | None,
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
```

Then, inside `_plan_and_review`, replace the old reviewer block with:

```python
            try:
                accepted, feedback = await _one_reviewer(
                    self._client,
                    user_prompt=user_prompt,
                    plan=last_plan,
                    reviewer_system=reviewer_system,
                    temperature=opts.reviewer_temperature,
                    reasoning_budget_tokens=opts.reasoning_budget_tokens,
                )
            except LlamaAgentsError as e:
                yield LoopError(error_type=type(e).__name__, message=str(e))
                return
            yield ReviewerVerdict(attempt=attempt, reviewer_idx=0,
                                  accepted=accepted, feedback=feedback)
            yield PlanReviewed(attempt=attempt, accepted=accepted, feedback=feedback)
```

Replace the old `reviewer_system = (...)` block at the top of `_plan_and_review` with:

```python
        reviewer_system = _REVIEWER_SYSTEM_PROMPT
```

(The old reviewer prompt is gone; we keep the variable name so the prior-plans-banner concatenation a few lines below still works.)

Add `ReviewerVerdict` to the imports at the top of `agent.py`:

```python
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
```

- [ ] **Step 6: Update the two existing planning tests to expect a `ReviewerVerdict` per attempt**

In `tests/unit/test_agent_loop.py`, the existing `test_planning_runs_when_orchestrator_registry_has_spawn` and `test_planning_iterates_on_reject_then_accepts` tests don't assert on `ReviewerVerdict` but they DO consume the same number of chat calls as today (1 planner + 1 reviewer per attempt). Both should still pass without changes because we haven't changed the call count yet (Task 3 does that). Run them:

```
uv run pytest tests/unit/test_agent_loop.py -v
```

Expected: all green. If anything fails, it's likely because the script for these tests now generates a `ReviewerVerdict` event that wasn't there before — the existing assertions don't filter on it but they may check specific event counts. Re-read failing assertions and fix only if they explicitly count events; the tests as written check by `isinstance` filter so they should be unaffected.

- [ ] **Step 7: Commit**

```
git add src/llama_agents/agent.py tests/unit/test_agent_loop.py
git commit -m "feat(agent): adversarial reviewer prompt + _one_reviewer helper + verdict parser"
```

---

## Task 3: Multi-reviewer parallel consensus

**Files:**
- Modify: `src/llama_agents/agent.py`
- Modify: `tests/unit/test_agent_loop.py`

Now wire `asyncio.gather` of N copies of `_one_reviewer`, strict-majority math, feedback dedup, and per-reviewer `ReviewerVerdict` events. Existing planning tests get extended to provide 3 reviewer responses per attempt (matching the new default).

- [ ] **Step 1: Write failing tests for the multi-vote behaviors**

Append to `tests/unit/test_agent_loop.py`:

```python
async def test_three_accepts_accepts_plan_emits_three_verdicts():
    client = ScriptedClient([
        ChatResponse(content="1. echo hi\n2. done"),  # planner
        ChatResponse(content="ACCEPT"),                 # reviewer 0
        ChatResponse(content="ACCEPT"),                 # reviewer 1
        ChatResponse(content="ACCEPT"),                 # reviewer 2
        ChatResponse(content="all done"),               # main loop final
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(agent.run("orchestrate", AgentRunOptions(max_iterations=3)))
    rvs = [e for e in events if isinstance(e, ReviewerVerdict)]
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(rvs) == 3
    assert all(rv.accepted for rv in rvs)
    assert len(accepted) == 1 and accepted[0].attempts == 1


async def test_majority_accept_wins():
    client = ScriptedClient([
        ChatResponse(content="plan"),                   # planner
        ChatResponse(content="ACCEPT"),                 # reviewer 0
        ChatResponse(content="REJECT: nope"),           # reviewer 1
        ChatResponse(content="ACCEPT"),                 # reviewer 2
        ChatResponse(content="done"),
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(agent.run("orchestrate", AgentRunOptions(max_iterations=3)))
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(accepted) == 1 and accepted[0].attempts == 1


async def test_majority_reject_triggers_retry():
    client = ScriptedClient([
        ChatResponse(content="plan 1"),
        ChatResponse(content="REJECT: a"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="REJECT: b"),
        ChatResponse(content="plan 2"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="done"),
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(agent.run("orchestrate", AgentRunOptions(max_iterations=3)))
    plans = [e for e in events if isinstance(e, PlanProposed)]
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(plans) == 2
    assert len(accepted) == 1 and accepted[0].attempts == 2


async def test_three_rejects_collates_distinct_feedback():
    """Three distinct rejection reasons should appear as a bulleted list
    in the planner's retry user message."""
    client = ScriptedClient([
        ChatResponse(content="plan 1"),
        ChatResponse(content="REJECT: missing tool"),
        ChatResponse(content="REJECT: scope drift"),
        ChatResponse(content="REJECT: vague step 2"),
        ChatResponse(content="plan 2"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="done"),
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    await _collect(agent.run("orchestrate", AgentRunOptions(max_iterations=3)))
    # Inspect the planner's second-attempt user message (the second
    # ScriptedClient call for the planner is at index that needs the
    # full conversation history — easier to verify via the recorded calls).
    # The retry message is appended to plan_history before the second
    # planner call, so it's in client.calls[4]["messages"][-1]["content"].
    retry_user_msg = client.calls[4]["messages"][-1]["content"]
    assert "missing tool" in retry_user_msg
    assert "scope drift" in retry_user_msg
    assert "vague step 2" in retry_user_msg
    # Bulleted format when >1 distinct reasons
    assert retry_user_msg.count("- ") >= 3


async def test_three_rejects_dedupes_identical_feedback():
    """Three reviewers giving identical feedback should produce only one
    bullet in the retry message."""
    client = ScriptedClient([
        ChatResponse(content="plan 1"),
        ChatResponse(content="REJECT: same problem"),
        ChatResponse(content="REJECT: same problem"),
        ChatResponse(content="REJECT: same problem"),
        ChatResponse(content="plan 2"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="done"),
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    await _collect(agent.run("orchestrate", AgentRunOptions(max_iterations=3)))
    retry_user_msg = client.calls[4]["messages"][-1]["content"]
    # Only one occurrence of the reason
    assert retry_user_msg.count("same problem") == 1


async def test_reviewer_count_one_consumes_single_call_per_attempt():
    """Backwards-compat: reviewer_count=1 falls back to single-reviewer flow."""
    client = ScriptedClient([
        ChatResponse(content="plan"),
        ChatResponse(content="ACCEPT"),
        ChatResponse(content="done"),
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(agent.run(
        "orchestrate",
        AgentRunOptions(max_iterations=3, reviewer_count=1),
    ))
    rvs = [e for e in events if isinstance(e, ReviewerVerdict)]
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(rvs) == 1
    assert len(accepted) == 1
```

Update the existing tests' scripts to provide 3 reviewer responses per attempt. Edit `test_planning_runs_when_orchestrator_registry_has_spawn`:

```python
async def test_planning_runs_when_orchestrator_registry_has_spawn():
    client = ScriptedClient([
        ChatResponse(content="1. do X\n2. do Y"),    # planner draft
        ChatResponse(content="ACCEPT"),                # reviewer 0
        ChatResponse(content="ACCEPT"),                # reviewer 1
        ChatResponse(content="ACCEPT"),                # reviewer 2
        ChatResponse(content="all done"),              # main loop final reply
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(agent.run("orchestrate me", AgentRunOptions(max_iterations=3)))
    plans = [e for e in events if isinstance(e, PlanProposed)]
    reviews = [e for e in events if isinstance(e, PlanReviewed)]
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(plans) == 1 and plans[0].attempt == 1
    assert len(reviews) == 1 and reviews[0].accepted is True
    assert len(accepted) == 1 and accepted[0].attempts == 1
```

Edit `test_planning_iterates_on_reject_then_accepts`:

```python
async def test_planning_iterates_on_reject_then_accepts():
    client = ScriptedClient([
        ChatResponse(content="bad plan"),                              # draft 1
        ChatResponse(content="REJECT: step 2 names a tool that does not exist"),
        ChatResponse(content="REJECT: step 2 names a tool that does not exist"),
        ChatResponse(content="REJECT: step 2 names a tool that does not exist"),
        ChatResponse(content="1. echo hi\n2. done"),                   # draft 2
        ChatResponse(content="ACCEPT"),                                # reviewer 0
        ChatResponse(content="ACCEPT"),                                # reviewer 1
        ChatResponse(content="ACCEPT"),                                # reviewer 2
        ChatResponse(content="done"),                                  # main loop final
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(agent.run("orchestrate", AgentRunOptions(max_iterations=3)))
    plans = [e for e in events if isinstance(e, PlanProposed)]
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(plans) == 2
    assert len(accepted) == 1 and accepted[0].attempts == 2
```

Edit `test_planning_gives_up_after_max_iterations_and_uses_last_plan` to provide 3 REJECTs per attempt × 3 attempts:

```python
async def test_planning_gives_up_after_max_iterations_and_uses_last_plan():
    client = ScriptedClient([
        ChatResponse(content="plan A"),
        ChatResponse(content="REJECT: bad"),
        ChatResponse(content="REJECT: bad"),
        ChatResponse(content="REJECT: bad"),
        ChatResponse(content="plan B"),
        ChatResponse(content="REJECT: still bad"),
        ChatResponse(content="REJECT: still bad"),
        ChatResponse(content="REJECT: still bad"),
        ChatResponse(content="plan C"),
        ChatResponse(content="REJECT: worse"),
        ChatResponse(content="REJECT: worse"),
        ChatResponse(content="REJECT: worse"),
        ChatResponse(content="answer"),  # main loop after exhausting retries
    ])
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(
        agent.run(
            "orchestrate",
            AgentRunOptions(max_iterations=2, max_planning_iterations=3),
        )
    )
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(accepted) == 1
    assert accepted[0].plan == "plan C"
    assert accepted[0].attempts == 3
```

- [ ] **Step 2: Run the new + updated tests — expect failures**

```
uv run pytest tests/unit/test_agent_loop.py -v
```

Expected: many failures — the old `_plan_and_review` calls `_one_reviewer` exactly once per attempt; tests that script 3 reviewer responses will leave 2 unconsumed responses each.

- [ ] **Step 3: Replace the single `_one_reviewer` call with a parallel gather + majority + dedup**

In `src/llama_agents/agent.py`, find the block in `_plan_and_review` that does the single-reviewer call (added in Task 2) and replace it with:

```python
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

            # Normalize exceptions into REJECT votes; otherwise unpack tuples.
            verdicts: list[tuple[bool, str]] = []
            for r in results:
                if isinstance(r, Exception):
                    msg = f"{type(r).__name__}: {r}"[:500]
                    verdicts.append((False, msg))
                else:
                    verdicts.append(r)

            # If EVERY reviewer raised, fail the planning phase like today.
            if all(isinstance(r, Exception) for r in results):
                first = next(r for r in results if isinstance(r, Exception))
                yield LoopError(error_type=type(first).__name__, message=str(first))
                return

            for i, (accepted, feedback) in enumerate(verdicts):
                yield ReviewerVerdict(
                    attempt=attempt, reviewer_idx=i,
                    accepted=accepted, feedback=feedback,
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
```

The `accepted` and `feedback` variables are then used by the existing accept/reject branches further down in the loop, which keep the existing memory-store-plan and replan-history logic working as before.

Make sure `asyncio` is already imported at the top of agent.py — it is, since the existing file uses `asyncio.Event`.

Also make sure `ReviewerVerdict` is in the events import (added in Task 2).

- [ ] **Step 4: Run all agent-loop tests**

```
uv run pytest tests/unit/test_agent_loop.py -v
```

Expected: all green.

- [ ] **Step 5: Run the full unit suite**

```
uv run pytest tests/unit -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git add src/llama_agents/agent.py tests/unit/test_agent_loop.py
git commit -m "feat(agent): parallel multi-reviewer consensus with strict majority + dedup"
```

---

## Task 4: Per-reviewer exception resilience

**Files:**
- Modify: `tests/unit/test_agent_loop.py`

The exception-handling code added in Task 3 already covers both single-reviewer-crash and all-reviewers-crash. This task just adds dedicated tests to lock that behaviour in.

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_agent_loop.py`:

```python
class _PartialFailureClient:
    """Returns scripted responses, but raises on a specific call index."""

    def __init__(self, script: list, fail_indices: set[int], exc: Exception):
        self._script = list(script)
        self._fail = fail_indices
        self._exc = exc
        self._i = -1
        self.calls: list[dict[str, Any]] = []

    async def chat(self, *, messages, tools, temperature=0.2, reasoning_budget_tokens=None):
        self._i += 1
        self.calls.append({"messages": list(messages), "tools": tools})
        if self._i in self._fail:
            raise self._exc
        return self._script.pop(0)


async def test_one_reviewer_exception_treated_as_reject_vote():
    """1 reviewer raises + 2 ACCEPT = 2/3 majority, plan accepted."""
    from llama_agents.errors import LlamaUnreachable

    # Script: planner (idx 0), reviewer-0 raises (idx 1), reviewer-1 ACCEPT (idx 2),
    # reviewer-2 ACCEPT (idx 3), main loop final (idx 4)
    script = [
        ChatResponse(content="plan"),  # planner (idx 0)
        ChatResponse(content="ACCEPT"),  # reviewer-1 (idx 2 after failure)
        ChatResponse(content="ACCEPT"),  # reviewer-2 (idx 3)
        ChatResponse(content="done"),    # main loop (idx 4)
    ]
    client = _PartialFailureClient(script, {1}, LlamaUnreachable("conn refused"))
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(agent.run("orchestrate", AgentRunOptions(max_iterations=3)))
    rvs = [e for e in events if isinstance(e, ReviewerVerdict)]
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    # One verdict is False (the exception), two are True
    falses = [r for r in rvs if not r.accepted]
    trues = [r for r in rvs if r.accepted]
    assert len(falses) == 1
    assert "LlamaUnreachable" in falses[0].feedback
    assert len(trues) == 2
    assert len(accepted) == 1


async def test_all_reviewers_exception_emits_loop_error_and_no_plan_accepted():
    from llama_agents.errors import LlamaUnreachable

    # Script: planner (idx 0), all 3 reviewers raise (idx 1,2,3), no further
    script = [ChatResponse(content="plan")]  # planner only
    client = _PartialFailureClient(script, {1, 2, 3}, LlamaUnreachable("server down"))
    agent = Agent(client=client, registry=_orchestrator_registry())
    events = await _collect(agent.run("orchestrate", AgentRunOptions(max_iterations=3)))
    errors = [e for e in events if isinstance(e, LoopError)]
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(errors) == 1
    assert errors[0].error_type == "LlamaUnreachable"
    assert len(accepted) == 0
```

- [ ] **Step 2: Run the new tests — expect pass (logic already in place from Task 3)**

```
uv run pytest tests/unit/test_agent_loop.py -k "exception_treated_as_reject or all_reviewers_exception" -v
```

Expected: both pass. If they don't, re-check the gather block in Task 3.

- [ ] **Step 3: Commit**

```
git add tests/unit/test_agent_loop.py
git commit -m "test(agent): reviewer-exception resilience: per-call + all-fail paths"
```

---

## Task 5: Cancellation mid-review

**Files:**
- Modify: `src/llama_agents/agent.py`
- Modify: `tests/unit/test_agent_loop.py`

If `agent.cancel()` fires while reviewers are in flight, the function should return cleanly without yielding `PlanAccepted`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_agent_loop.py`:

```python
class _SlowReviewerClient:
    """First call returns the planner response immediately; subsequent calls
    sleep so a cancel() can fire while they're in flight."""

    def __init__(self, planner_response, delay: float = 0.5):
        self._planner = planner_response
        self._delay = delay
        self._called_planner = False
        self.calls: list[dict[str, Any]] = []

    async def chat(self, *, messages, tools, temperature=0.2, reasoning_budget_tokens=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self._called_planner:
            self._called_planner = True
            return self._planner
        await asyncio.sleep(self._delay)
        return ChatResponse(content="ACCEPT")


async def test_cancellation_mid_review_returns_without_plan_accepted():
    client = _SlowReviewerClient(
        planner_response=ChatResponse(content="plan"),
        delay=0.3,
    )
    agent = Agent(client=client, registry=_orchestrator_registry())
    # Schedule a cancel while reviewers are sleeping.
    async def _runner():
        await asyncio.sleep(0.1)
        agent.cancel()
    cancel_task = asyncio.create_task(_runner())
    events = await _collect(agent.run("orchestrate", AgentRunOptions(max_iterations=3)))
    await cancel_task
    accepted = [e for e in events if isinstance(e, PlanAccepted)]
    assert len(accepted) == 0
```

- [ ] **Step 2: Run the test — expect failure (no cancel check yet)**

```
uv run pytest tests/unit/test_agent_loop.py::test_cancellation_mid_review_returns_without_plan_accepted -v
```

Without the cancel check, the test will see `PlanAccepted` because all 3 reviewers eventually return ACCEPT.

- [ ] **Step 3: Add cancellation check after `gather`**

In `src/llama_agents/agent.py`, in `_plan_and_review`, immediately after the `results = await asyncio.gather(...)` line, add:

```python
            if self._cancel.is_set():
                return
```

- [ ] **Step 4: Run the test — expect pass**

```
uv run pytest tests/unit/test_agent_loop.py::test_cancellation_mid_review_returns_without_plan_accepted -v
```

- [ ] **Step 5: Run the full agent suite**

```
uv run pytest tests/unit/test_agent_loop.py -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git add src/llama_agents/agent.py tests/unit/test_agent_loop.py
git commit -m "feat(agent): honor cancellation immediately after reviewer gather"
```

---

## Task 6: Surface plumbing — CLI, HTTP, Web

**Files:**
- Modify: `src/llama_agents/cli.py`
- Modify: `src/llama_agents/http_app.py`
- Modify: `src/llama_agents/web/routes.py`
- Modify: `tests/unit/test_cli.py` (or create if absent)
- Modify: `tests/unit/test_http_app.py`
- Modify: `tests/unit/test_web_routes.py`

- [ ] **Step 1: Write failing surface tests**

Add to `tests/unit/test_http_app.py`:

```python
def test_http_serialize_reviewer_verdict():
    from llama_agents.events import ReviewerVerdict
    from llama_agents.http_app import _serialize

    ev = ReviewerVerdict(attempt=2, reviewer_idx=1, accepted=False,
                         feedback="missing tool")
    out = _serialize(ev)
    assert out["event"] == "reviewer_verdict"
    import json as _j
    data = _j.loads(out["data"])
    assert data["attempt"] == 2
    assert data["reviewer_idx"] == 1
    assert data["accepted"] is False
    assert data["feedback"] == "missing tool"
```

Add to `tests/unit/test_web_routes.py` (anywhere after the other helper tests):

```python
def test_event_style_includes_reviewer_verdict():
    from llama_agents.web.routes import _EVENT_STYLE
    assert "ReviewerVerdict" in _EVENT_STYLE
    color, summary_key = _EVENT_STYLE["ReviewerVerdict"]
    assert color == "teal"
    assert summary_key == "accepted"
```

For the CLI test, check if `tests/unit/test_cli.py` exists:

```
$env:Path = "$env:USERPROFILE\AppData\Roaming\Python\Python314\Scripts;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; uv run pytest tests/unit -k cli --collect-only -q
```

If it does, append; if not, create with:

```python
from io import StringIO

from rich.console import Console

from llama_agents.cli import _render_event
from llama_agents.events import ReviewerVerdict


def test_cli_renders_reviewer_verdict(monkeypatch, capsys):
    # _render_event uses the module-level `console` to print. We can't
    # easily capture rich output via capsys (rich writes to a Console),
    # so monkeypatch the module-level console with one that writes to a
    # StringIO buffer.
    import llama_agents.cli as cli_mod
    buf = StringIO()
    cli_mod.console = Console(file=buf, force_terminal=False, no_color=True, width=200)
    _render_event(ReviewerVerdict(attempt=1, reviewer_idx=2, accepted=True,
                                  feedback=""))
    text = buf.getvalue()
    assert "reviewer 2" in text or "Reviewer 2" in text
    assert "✓" in text
```

- [ ] **Step 2: Run the new tests — expect failures**

```
uv run pytest tests/unit/test_http_app.py::test_http_serialize_reviewer_verdict tests/unit/test_web_routes.py::test_event_style_includes_reviewer_verdict tests/unit/test_cli.py -v
```

- [ ] **Step 3: Wire up the CLI**

In `src/llama_agents/cli.py`, find the `_render_event` function and add a branch for `ReviewerVerdict`. Update the import too.

Replace the existing import line:

```python
from .events import AssistantChunk, Done, LoopError, MemoryEvicted, MemoryStored, ToolCallResult, ToolCallStart
```

with:

```python
from .events import (
    AssistantChunk, Done, LoopError, MemoryEvicted, MemoryStored,
    ReviewerVerdict, ToolCallResult, ToolCallStart,
)
```

Inside `_render_event`, add a branch (place it after `ToolCallResult` for grouping):

```python
    elif isinstance(ev, ReviewerVerdict):
        marker = "✓" if ev.accepted else "✗"
        excerpt = ev.feedback[:80]
        console.print(
            f"[dim]  {marker} reviewer {ev.reviewer_idx}: {excerpt}[/dim]"
        )
```

- [ ] **Step 4: Wire up the HTTP serializer**

In `src/llama_agents/http_app.py`, update the import:

```python
from .events import (
    AssistantChunk, Done, LoopError, MemoryEvicted, MemoryStored,
    ReviewerVerdict, ToolCallResult, ToolCallStart,
)
```

In the `_serialize` function (near the bottom of the file), add a branch:

```python
    if isinstance(ev, ReviewerVerdict):
        return {
            "event": "reviewer_verdict",
            "data": json.dumps({
                "attempt": ev.attempt,
                "reviewer_idx": ev.reviewer_idx,
                "accepted": ev.accepted,
                "feedback": ev.feedback,
            }),
        }
```

- [ ] **Step 5: Wire up the web event style**

In `src/llama_agents/web/routes.py`, find the `_EVENT_STYLE` dict and add an entry:

```python
    "ReviewerVerdict":  ("teal",   "accepted"),
```

Place it next to the other plan-related entries (`PlanProposed`, `PlanReviewed`, `PlanAccepted`).

- [ ] **Step 6: Run the surface tests**

```
uv run pytest tests/unit/test_http_app.py::test_http_serialize_reviewer_verdict tests/unit/test_web_routes.py::test_event_style_includes_reviewer_verdict tests/unit/test_cli.py -v
```

Expected: all green.

- [ ] **Step 7: Run the full unit suite**

```
uv run pytest tests/unit -q
```

Expected: all green.

- [ ] **Step 8: Commit**

```
git add src/llama_agents/cli.py src/llama_agents/http_app.py src/llama_agents/web/routes.py tests/unit/test_cli.py tests/unit/test_http_app.py tests/unit/test_web_routes.py
git commit -m "feat(web,cli,http): render ReviewerVerdict events through every surface"
```

---

## Task 7: Documentation update

**Files:**
- Modify: `CLAUDE.md` (path: `D:\repos\llm\llama-agents\CLAUDE.md`)

- [ ] **Step 1: Read the current CLAUDE.md**

Use Read on `D:\repos\llm\llama-agents\CLAUDE.md` to locate two sections:
- "Agent.run() shape" (around the Planning phase description)
- "Known limitations / future work" (contains the "Reviewer can confirm bad plans" bullet)

- [ ] **Step 2: Strike the limitation**

In the "Known limitations / future work" section, remove the bullet that reads roughly:

> - **Reviewer can confirm bad plans.** Self-review by the same model is cheap but prone to confirmation bias. A reviewer-subagent variant is on the table.

- [ ] **Step 3: Update the planning phase description**

In the "Agent.run() shape" section, find the planning-phase description and add a sentence after the existing "PlanReviewed" mention:

> - The reviewer runs as **N parallel passes** (default 3, configurable via `AgentRunOptions.reviewer_count`) with an adversarial system prompt and a 5-item checklist. Verdict is by strict majority (`accepts > N // 2`). Each pass emits a `ReviewerVerdict` event in addition to the single per-attempt `PlanReviewed`.

Also update the event list at the bottom of the same section:

Find:
> `PlanProposed`, `PlanReviewed`, `PlanAccepted`, `ToolCallStart`, ...

Add `ReviewerVerdict` to the list, between `PlanReviewed` and `PlanAccepted`.

- [ ] **Step 4: Regression check**

```
$env:Path = "$env:USERPROFILE\AppData\Roaming\Python\Python314\Scripts;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; uv run pytest tests/unit -q
```

Expected: all green.

- [ ] **Step 5: Commit**

```
git add "D:\repos\llm\llama-agents\CLAUDE.md"
git commit -m "docs(claude): record reviewer consensus design; remove stale limitation"
```

---

## Done criteria

After Task 7:

- `uv run pytest tests/unit -q` is green; the suite includes ~15 new reviewer-related tests.
- An orchestrator agent (one with `subagent_spawn` in its registry) by default runs 3 reviewer passes in parallel per planning attempt; the planner re-iterates unless a strict majority accept.
- A single reviewer crashing no longer kills the planning phase — its vote counts as REJECT and the consensus continues.
- The CLI, the SSE chat stream, and the web UI's job-detail timeline all render per-reviewer verdicts.
- Setting `AgentRunOptions(reviewer_count=1)` reproduces the legacy single-reviewer behavior for users / tests that need it.
- `CLAUDE.md` no longer claims "Reviewer can confirm bad plans" as a known limitation.
