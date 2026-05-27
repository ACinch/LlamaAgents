"""Security-architect review of a repository, using subagents to fan out.

Usage:
    uv run python examples/security_architect_code_review.py \\
        --repo D:/repos/LLM/llama-agents \\
        --output D:/repos/LLM/llama-agents/docs/examples/_security-review.md

The target repo and the output path must be inside sandbox.allowed_dirs in
config.toml. Bump [agent].max_iterations to ~30 for medium repos.

See docs/examples/security-architect-code-review.md for the walkthrough.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

for _s in (sys.stdout, sys.stderr):
    reconfigure = getattr(_s, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

from llama_agents.agent import AgentRunOptions
from llama_agents.config import load_config
from llama_agents.events import (
    AssistantChunk,
    Done,
    LoopError,
    PlanAccepted,
    PlanProposed,
    PlanReviewed,
    ToolCallResult,
    ToolCallStart,
)
from llama_agents.runtime import Runtime


PROMPT_TEMPLATE = """Act as a senior security architect reviewing the repository at {repo}.

Plan:
1. Survey: call shell_run EXACTLY ONCE with
       command=['git', 'ls-files', '*.py']
       cwd='{repo}'
   Use the stdout (newline-separated paths relative to the repo root) as
   the authoritative file list. This honors .gitignore automatically and
   excludes .venv, build artifacts, and untracked junk that would blow the
   context window. Group the returned files into review domains:
   (a) sandbox + tools, (b) llama client + server lifecycle, (c) MCP
   bridge, (d) agent loop + subagent spawning, (e) CLI + HTTP surfaces.
2. For each domain, spawn a subagent. Give it the domain name, the relevant
   file paths, and the instruction: 'Read these files and report security
   findings in the format
   [severity: HIGH|MED|LOW] <file:line>: <issue> — <impact> — <fix>.
   Limit to findings you can defend by quoting the code.'
3. Wait for all subagents, collate the findings, deduplicate, and rank by
   severity.
4. Write the final report to {output} using fs_write_file. The report
   should have sections: Summary, High-severity findings, Medium, Low,
   Notes / future work.
5. Reply with the path and a 3-bullet executive summary.
"""


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-iterations", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rt = await Runtime.create(cfg)
    try:
        agent = rt.new_agent()
        prompt = PROMPT_TEMPLATE.format(repo=args.repo, output=args.output)
        async for ev in agent.run(prompt, AgentRunOptions(max_iterations=args.max_iterations)):
            if isinstance(ev, PlanProposed):
                print(f"\n[plan attempt {ev.attempt}]\n{ev.plan}\n")
            elif isinstance(ev, PlanReviewed):
                tag = "ACCEPT" if ev.accepted else "REJECT"
                print(f"[review {ev.attempt}: {tag}] {ev.feedback}")
            elif isinstance(ev, PlanAccepted):
                print(f"[plan ACCEPTED after {ev.attempts} attempt(s)]")
            elif isinstance(ev, ToolCallStart):
                print(f"-> {ev.name}({_brief(ev.arguments)})")
            elif isinstance(ev, ToolCallResult):
                marker = "ok" if ev.ok else "ERR"
                preview = str(ev.content)[:240].replace("\n", " ")
                print(f"   [{marker}] {preview}")
            elif isinstance(ev, AssistantChunk):
                print("\n--- assistant ---")
                print(ev.text)
            elif isinstance(ev, LoopError):
                print(f"!! {ev.error_type}: {ev.message}", file=sys.stderr)
                return 1
            elif isinstance(ev, Done):
                print(f"(done: {ev.reason})")
                return 0
    finally:
        await rt.aclose()
    return 0


def _brief(args: dict) -> str:
    """Trim long subagent task strings for terminal display."""
    out = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 120:
            out[k] = v[:120] + "…"
        else:
            out[k] = v
    return str(out)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
