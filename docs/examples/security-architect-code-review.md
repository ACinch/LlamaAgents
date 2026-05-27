# Example 2 — Security-architect code review of a repository

**Goal:** Walk a code repository as if you were a senior security architect on
the team. Catalog the attack surface, look for the usual classes of issue
(input handling, auth, secrets, sandboxing, deserialization, command
execution, dependency hygiene), and produce a written report ranked by
severity.

## Tools exercised

- `shell_run` with `git ls-files` — enumerate tracked source files
  (gitignore-aware, so `.venv` and other junk are excluded automatically).
  Also useful for `git log` / `git show` on recent risky changes.
- `fs_read_file` — read suspicious files in full.
- `subagent_spawn` — fan out per-domain reviews (auth, sandbox, network,
  tooling) so each subagent has a focused, smaller context window.
- `fs_write_file` — save the report.

The fan-out matters here: trying to review a whole repo in one context blows
through the agent's window. Subagents keep each pass scoped.

## Config

The target repo must be inside `sandbox.allowed_dirs`. For reviewing
`llama-agents` itself, the default config already covers it:

```toml
[sandbox]
allowed_dirs = ["D:/repos/LLM/llama-agents"]
shell_allowlist = ["git", "pytest"]   # git is useful for blame/log

[agent]
max_iterations = 30                   # parent needs headroom for fan-out
max_concurrent_agents = 4
```

For an *external* repo (e.g. ACinch web app), add it to `allowed_dirs` —
don't replace the llama-agents entry, append.

## Prompt

```text
Act as a senior security architect reviewing the repository at
D:/repos/LLM/llama-agents.

Plan:
1. Survey: call shell_run EXACTLY ONCE with
   command=['git', 'ls-files', '*.py'] and cwd=<repo path>. Parse the
   stdout — newline-separated paths relative to the repo root — as the
   authoritative file list. This honors .gitignore automatically (no
   .venv, no build artifacts, no untracked clutter). Group the files
   into review domains: (a) sandbox + tools, (b) llama client + server
   lifecycle, (c) MCP bridge, (d) agent loop + subagent spawning,
   (e) CLI + HTTP surfaces.
2. For each domain, spawn a subagent. Give it the domain name, the relevant
   file paths, and the instruction: 'Read these files and report security
   findings in the format
   [severity: HIGH|MED|LOW] <file:line>: <issue> — <impact> — <fix>.
   Limit to findings you can defend by quoting the code.'
3. Wait for all subagents, collate the findings, deduplicate, and rank by
   severity.
4. Write the final report to
   D:/repos/LLM/llama-agents/docs/examples/_security-review.md using
   fs_write_file. The report should have sections: Summary, High-severity
   findings, Medium, Low, Notes / future work.
5. Reply with the path and a 3-bullet executive summary.
```

## Invocation

```powershell
$env:PYTHONIOENCODING = "utf-8"
uv run llamactl chat --max-iterations 30 @'
Act as a senior security architect ...
'@
```

Or use `examples/security_architect_code_review.py` to drive it from Python
(prints each `ToolCallStart` as it happens — useful for long runs).

## Expected event stream

1. `→ shell_run(['git', 'ls-files', '*.py'], cwd=<repo>)` — one call,
   returns the tracked file list.
2. Several `→ subagent_spawn(...)` calls, one per domain, each returning a
   `result` of structured findings.
3. The parent's final `AssistantChunk` is the 3-bullet summary; the full
   report is in `_security-review.md`.
4. `Done(reason=finished)`.

## Notes

- A 30-iteration ceiling fits about 5-6 subagents plus a survey and a
  write. Raise it for larger repos.
- **Why `git ls-files` and not `fs_list_files`:** `fs_list_files` is a raw
  glob — it doesn't honor `.gitignore`. A bare `**/*.py` against a repo
  containing `.venv/` will return tens of thousands of paths and blow the
  model's context. `git ls-files` only returns tracked files, which is
  exactly the surface a security review should cover.
- The model is honest about uncertainty when asked to "defend findings by
  quoting code." Without that line it will pattern-match generically.
- Pair this with a follow-up run that asks subagents to *fix* the HIGH
  findings, gated by `git diff` review.
