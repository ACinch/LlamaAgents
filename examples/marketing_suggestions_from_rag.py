"""Generate marketing suggestions for a topic, grounded in RAG content.

Usage:
    uv run python examples/marketing_suggestions_from_rag.py --topic ACinch

Optionally save the deck to a file under sandbox.allowed_dirs:
    uv run python examples/marketing_suggestions_from_rag.py \\
        --topic ACinch \\
        --output D:/repos/LLM/llama-agents/docs/examples/_marketing-deck.md

See docs/examples/marketing-suggestions-from-rag.md for the walkthrough.
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
from llama_agents.events import AssistantChunk, Done, LoopError, ToolCallResult, ToolCallStart
from llama_agents.runtime import Runtime


PROMPT_TEMPLATE = """You are a marketing strategist. Your task: produce a deck of
marketing suggestions for {topic}.

Process:
1. Use rag__rag_query at least three times with DIFFERENT angles to gather
   grounded material. Suggested angles: '{topic} product features',
   '{topic} target users and use cases', '{topic} differentiation or
   competitors'. Vary the queries based on what you find.
2. Synthesize the results into FIVE marketing suggestions. Each must
   include: a positioning headline, the audience it targets, the supporting
   claim (cite which RAG snippet justifies it), and a one-sentence call to
   action.
3. Do NOT invent capabilities. If the RAG returns no material for a claim,
   drop the suggestion or rephrase to fit what's there.
4. Reply with the deck in plain markdown. No preamble, just the five
   suggestions numbered 1-5.{save_instruction}
"""


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--output", default=None, help="optional path to save the deck")
    parser.add_argument("--max-iterations", type=int, default=15)
    args = parser.parse_args()

    save = (
        f"\n5. Additionally, write the deck to {args.output} using "
        f"fs_write_file before replying."
        if args.output
        else ""
    )
    prompt = PROMPT_TEMPLATE.format(topic=args.topic, save_instruction=save)

    cfg = load_config(args.config)
    rt = await Runtime.create(cfg)
    try:
        agent = rt.new_agent()
        async for ev in agent.run(prompt, AgentRunOptions(max_iterations=args.max_iterations)):
            if isinstance(ev, ToolCallStart):
                print(f"-> {ev.name}({ev.arguments})")
            elif isinstance(ev, ToolCallResult):
                marker = "ok" if ev.ok else "ERR"
                preview = str(ev.content)[:200].replace("\n", " ")
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


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
