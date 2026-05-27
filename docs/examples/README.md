# Examples

Each example has two forms:

- A **markdown walkthrough** in this folder — explains the use case, the prompt
  to feed the agent, and any `config.toml` tweaks needed (allowed dirs, MCP
  servers, iteration budget).
- A **runnable script** in `../../examples/` — same task, but driven from
  Python via `Runtime` so you can post-process events programmatically.

| # | Example | Markdown | Script |
|---|---------|----------|--------|
| 1 | Story from a folder of `.txt` files | [`story-from-txt-files.md`](story-from-txt-files.md) | `examples/story_from_txt_files.py` |
| 2 | Security-architect code review | [`security-architect-code-review.md`](security-architect-code-review.md) | `examples/security_architect_code_review.py` |
| 3 | Marketing suggestions from RAG | [`marketing-suggestions-from-rag.md`](marketing-suggestions-from-rag.md) | `examples/marketing_suggestions_from_rag.py` |

All examples assume:

- `llama-server.exe` is reachable (or `auto_spawn=true` is configured).
- The RAG MCP server is configured if the example uses it.
- `PYTHONIOENCODING=utf-8` is set on Windows so the CLI's status glyphs don't
  trip cp1252 when stdout is redirected.

## How auto-planning interacts with examples

Every orchestrator-level run (i.e. an Agent whose registry includes
`subagent_spawn`) automatically does a plan + self-review pass before
the main tool loop. You'll see `PlanProposed`, `PlanReviewed`, and
`PlanAccepted` events in the stream before any `ToolCallStart`. If a
plan is rejected the planner iterates (up to
`AgentRunOptions.max_planning_iterations`, default 3).

The example prompts below already describe the expected execution shape
in detail — those prompts effectively brief the planner. The planner's
job is to compress them into a concrete numbered plan; the reviewer's
job is to catch obvious flaws (missing tools, context bombs like
`fs_list_files('**/*.py', repo_root)`, vague steps).

Subagents skip planning to avoid recursion.

To suppress planning for a particular run, pass
`AgentRunOptions(skip_planning=True)`.
