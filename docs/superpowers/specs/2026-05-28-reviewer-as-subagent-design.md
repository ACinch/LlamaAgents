# Reviewer-as-Subagent — Design

**Status:** Draft
**Date:** 2026-05-28
**Owner:** Maarten

## Goal

Reduce the confirmation bias in the plan-review phase by reframing the
reviewer as an adversarial critic that must score the plan against a
fixed checklist, and by running **three independent reviewer passes in
parallel** with a majority verdict. The same model produces the plan
and the reviews, but the reviewer's prompt, temperature, and consensus
mechanism are designed to make a vacuous "looks good" much harder than
in the current single-call review.

## Non-goals

- Adding a second model. The system overhead of running two llama-server
  instances (extra VRAM, extra config, extra download) is prohibitive
  for most users. The improvement here comes purely from prompt shape,
  sampling variance across multiple reviewers, and consensus.
- Letting the reviewer use tools, browse the codebase, or call other
  subagents. It is a one-shot LLM call that returns a structured
  checklist + verdict.
- Promoting the reviewer to a real `Agent` instance with its own run
  loop. The "subagent" framing is conceptual; mechanically each
  reviewer is a single `LlamaClient.chat` call.
- Changing the planner. It keeps the existing system prompt and
  iteration loop.
- Persisting individual reviewer verdicts to memory. Only the final
  accepted plan is stored, same as today.

## Architecture

All work happens inside `_plan_and_review` in
`src/llama_agents/agent.py`. No new module.

### Reviewer prompt

The current single-paragraph reviewer prompt is replaced with one that:

1. Frames the reviewer as an *independent critic* explicitly told it
   did not write the plan and must treat it skeptically.
2. Requires a five-item checklist before the verdict, each scored
   `PASS` / `FAIL` / `SKIP` with a one-sentence justification:
   - **Tool validity** — every step names a tool from the provided list.
   - **Scope match** — the plan accomplishes the user's stated goal.
   - **Context safety** — no obvious context-window blowups
     (broad globs over directories that contain `.venv`,
     `node_modules`, etc.; chained reads of very large files).
   - **Specificity** — no vague hand-waves like "use shell to figure
     out the project."
   - **Executability** — a worker following the steps verbatim could
     actually run them, in order, without filling gaps.
3. Ends with one of:
   - `ACCEPT`
   - `REJECT: <one sentence describing the fix>`

The exact prompt text lives in `agent.py` as a module-level
constant `_REVIEWER_SYSTEM_PROMPT`.

### Multi-reviewer consensus

`_one_reviewer(idx, last_plan, user_prompt, tool_names, opts) -> tuple[bool, str]`
is a private helper that performs one reviewer call and returns
`(accepted, feedback)`. The `_plan_and_review` loop launches
`opts.reviewer_count` (default 3) of these in parallel via:

```python
results = await asyncio.gather(
    *[
        _one_reviewer(i, last_plan, user_prompt, tool_names, opts)
        for i in range(opts.reviewer_count)
    ],
    return_exceptions=True,
)
```

`return_exceptions=True` means a single reviewer crash (e.g.
`LlamaUnreachable`) does not poison the others. An exception is
treated as a REJECT vote whose feedback is
`f"{type(e).__name__}: {e}"` (truncated to 500 chars).

For each result the agent yields a `ReviewerVerdict(attempt=N,
reviewer_idx=i, accepted=bool, feedback=str)` event so callers can
render the per-vote outcome.

### Verdict aggregation

- Count `accepted=True` results. If `accepts > reviewer_count // 2`
  the plan is accepted (a strict majority — half-or-less rejects).
  Table:

  | `reviewer_count` | Accept threshold |
  |---:|---:|
  | 1 | 1 (any accept passes) |
  | 2 | 2 (both must accept; a 1-1 tie rejects) |
  | 3 | 2 |
  | 4 | 3 |
  | 5 | 3 |
- Otherwise REJECT. Concatenate the distinct rejection reasons:
  lowercase-strip-dedupe (so identical complaints don't repeat), then
  render as:

  ```
  Reviewers rejected:
  - <reason 1>
  - <reason 2>
  ```

  If only one distinct reason survives the dedupe, send it through as
  the unbulletted single-line feedback the planner expects today.

- Emit `PlanReviewed(attempt=N, accepted=bool, feedback=str)` once per
  attempt (unchanged from today's contract; just now driven by the
  majority result).

- If **every** reviewer raised an exception (`gather` returned all
  `Exception` instances), the planning phase emits a `LoopError` whose
  `error_type` is the first exception's class name and exits the loop —
  identical failure path to today.

### Verdict parser

A reviewer response is treated as ACCEPT iff, after stripping trailing
whitespace, the **final non-empty line** starts with `ACCEPT`
(case-insensitive). Anything else, including:

- a final line starting with `REJECT:`
- a verdict mid-response with text after it
- truncated response with no verdict at all
- malformed checklist (missing items)

…is parsed as REJECT. The feedback is extracted from the `REJECT:` line
if present; otherwise the entire raw output truncated to 500 chars is
used as the feedback string. This is intentionally conservative — a
broken reviewer response can never accept a bad plan.

### Cancellation

`self._cancel.is_set()` is already checked at the top of each planning
iteration. We add one more check immediately after `gather` returns:
if cancellation fired while reviews were in flight, the function
returns without emitting `PlanAccepted`. The already-collected
`ReviewerVerdict` events still flow (they were yielded as the helpers
completed via streaming — see below).

Actually, since `asyncio.gather` collects results and we yield events
*after* it returns, the cleanest pattern is:

```python
verdict_pairs = await asyncio.gather(..., return_exceptions=True)
if self._cancel.is_set():
    return
for i, vp in enumerate(verdict_pairs):
    accepted, feedback = _normalize_verdict(vp)
    yield ReviewerVerdict(attempt=attempt, reviewer_idx=i,
                          accepted=accepted, feedback=feedback)
# then majority math + PlanReviewed
```

So reviewer events fire as a batch right after all three reviewers
return. That's fine — the parallel reviewer call is fast enough that
streaming individual votes mid-flight adds complexity without
meaningful UX gain.

## Configuration

Two new fields on `AgentRunOptions` (per-run knobs, not in `config.toml`):

```python
reviewer_count: int = 3
reviewer_temperature: float = 0.5
```

- `reviewer_count=1` falls back to the legacy single-call behavior
  (still using the new structured prompt — there's no reason to keep
  the old reviewer prompt around).
- `reviewer_count=3` is the new default.

`max_planning_iterations` stays at 3. Worst case per agent run:
3 planning attempts × 3 reviewers = 9 reviewer calls before falling
through to "accept the last plan anyway."

`llama.cpp`'s `n_parallel=2` (the project default) means at most two
reviewers actually run concurrently; the third queues. Total reviewer
wall-clock is roughly 2× a single review, not 3×.

## New event type

In `src/llama_agents/events.py`:

```python
@dataclass
class ReviewerVerdict(Event):
    attempt: int
    reviewer_idx: int   # 0-indexed within an attempt
    accepted: bool
    feedback: str
```

### Surface plumbing

- **CLI** (`cli.py`) — render `ReviewerVerdict` as a dimmed line
  prefixed with `✓` (accepted) or `✗` (rejected) and the reviewer
  index: `[dim]  ✓ reviewer 2: <feedback excerpt>[/dim]`.
- **HTTP** (`http_app.py`) — `_serialize` adds a branch that emits
  `{"event": "reviewer_verdict", "data": {"attempt", "reviewer_idx",
  "accepted", "feedback"}}`.
- **Web UI** (`web/routes.py`) — `_EVENT_STYLE` adds
  `"ReviewerVerdict": ("teal", "accepted")`. Renders in the job-detail
  timeline with a teal badge.

No queue/memory changes; the queue worker passes events through
unmodified.

## Backwards compatibility

- **Existing tests in `test_agent_loop.py`** — two tests today
  exercise the planning loop:
  `test_planning_happy_path` and
  `test_planning_rejected_then_accepted`. Both pass a scripted client
  that returns one `ChatResponse` per chat call. With the new default
  of `reviewer_count=3` they would consume their script after one
  iteration. Each must be updated to either:
  - extend the script to provide three reviewer responses per attempt
    (preferred — exercises the new path), or
  - construct `AgentRunOptions(reviewer_count=1)` to preserve the old
    single-reviewer flow.

  Plan: convert both to `reviewer_count=3` and extend their scripts.
  Add new tests for the multi-vote-specific behaviors.

- **External callers (`runtime.py`, `cli.py`, `http_app.py`)** —
  none construct `AgentRunOptions` with explicit reviewer settings, so
  they pick up the new defaults transparently. The web/HTTP request
  shapes are unchanged.

- **`AgentRunOptions` is a dataclass** — adding two new fields with
  defaults is non-breaking for keyword-only instantiation, which is
  the pattern in use everywhere.

## Errors and observability

- Single reviewer raises → counted as REJECT, `ReviewerVerdict(accepted=False,
  feedback="ExceptionType: msg")` event emitted.
- All reviewers raise → `LoopError(error_type=<first>, message=<first>)`
  + planning phase exits the same way it does today on a planner
  exception.
- Reviewer returns no verdict / malformed → counted as REJECT with
  raw output as feedback.
- Total review latency visible to the user via the cluster of
  `ReviewerVerdict` events that arrive together; can be inferred from
  surrounding timestamps if needed.

## Testing

Unit tests in `tests/unit/test_agent_loop.py`:

1. **`test_three_accepts_accepts_plan`** — three scripted ACCEPTs,
   plan accepted on attempt 1. Asserts 3 `ReviewerVerdict` events
   (all `accepted=True`) and one `PlanAccepted`.
2. **`test_three_rejects_triggers_retry`** — three REJECTs with
   distinct reasons, second attempt accepts. Asserts the planner's
   second-attempt user message contains a bulleted list of the three
   rejection reasons.
3. **`test_majority_accept_wins`** — 2 ACCEPT + 1 REJECT, plan
   accepted on attempt 1.
4. **`test_majority_reject_loses`** — 1 ACCEPT + 2 REJECT, planner
   retries.
5. **`test_one_reviewer_exception_treated_as_reject_vote`** — flaky
   client raises `LlamaUnreachable` on reviewer 0, returns ACCEPT on
   reviewers 1 + 2 → plan accepted (2/3 majority). One
   `ReviewerVerdict(accepted=False, feedback="LlamaUnreachable: …")`
   event emitted for the failing reviewer.
6. **`test_all_reviewers_exception_emits_loop_error`** — all three
   raise → `LoopError` event, no `PlanAccepted`.
7. **`test_malformed_verdict_counts_as_reject`** — reviewer returns
   "the plan seems okay" (no `ACCEPT`/`REJECT` prefix). Counted as
   REJECT with the raw text (truncated to 500 chars) as feedback.
8. **`test_reviewer_count_one_falls_back_to_single_call`** —
   `AgentRunOptions(reviewer_count=1)` consumes exactly one reviewer
   response per attempt. Asserts only one `ReviewerVerdict` per
   attempt.
9. **`test_dedup_rejection_reasons`** — three REJECTs with identical
   feedback strings → planner's retry message contains the reason
   once, not three times.
10. **`test_cancellation_mid_review`** — set `agent.cancel()` while
    the `gather` is awaited; assert no `PlanAccepted` is emitted and
    the function returns cleanly.

Surface-rendering tests:
11. **`test_cli_renders_reviewer_verdict`** — `cli._render_event`
    against a `ReviewerVerdict` produces a dimmed line containing the
    reviewer index and feedback.
12. **`test_http_serialize_reviewer_verdict`** —
    `http_app._serialize(ReviewerVerdict(...))` returns the expected
    `{"event": "reviewer_verdict", "data": {...}}` shape.
13. **`test_web_event_style_includes_reviewer_verdict`** — the
    `_EVENT_STYLE` map has an entry for `"ReviewerVerdict"` with a
    teal color and `"accepted"` as the summary key.

No live test. The change is fully covered by scripted-client unit
tests.

## Files changed / added

**Modified — no new modules:**

- `src/llama_agents/agent.py` — `_REVIEWER_SYSTEM_PROMPT` constant
  rewrite; new `_one_reviewer` helper; multi-reviewer + consensus
  logic in `_plan_and_review`; new fields on `AgentRunOptions`.
- `src/llama_agents/events.py` — add `ReviewerVerdict` dataclass.
- `src/llama_agents/cli.py` — render `ReviewerVerdict` events.
- `src/llama_agents/http_app.py` — `_serialize` branch.
- `src/llama_agents/web/routes.py` — `_EVENT_STYLE` entry.
- `tests/unit/test_agent_loop.py` — extend planning tests; add the
  10 new tests above.
- `CLAUDE.md` — strike the "Reviewer can confirm bad plans" entry from
  Known limitations; document the new defaults under Agent.run() shape.

## Open questions

None at design time. One deliberate deferral:

- **Per-reviewer different temperatures or different prompts.**
  Currently all three reviewers use the same prompt and the same
  `reviewer_temperature`. A future tweak could give each a slightly
  different persona ("security-minded critic", "skeptic about
  execution", etc.) for better coverage. Out of scope for v1.
