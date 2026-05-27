"""Write a long-form story in the style of a folder of .txt files.

Pipeline:
    1. Style analysis — read all .txt inputs and derive a style profile.
    2. Outline — break the story into 6-8 sections of 1500-2000 words each.
    3. Drafting — spawn one subagent per section to write its prose.
       Each subagent only sees the style profile and its own section's
       brief, so the parent's context stays compact.
    4. Editorial review — a subagent critiques the assembled draft for
       consistency (character/tense/POV), tone drift vs. the style profile,
       and language quality (clichés, repetition, unclear prose).
    5. Revision — a subagent applies the critique to produce draft 2.
    6. Polish — a final pass smooths transitions and fixes residual issues.
    7. Save — fs_write_file the final story and emit a report.

Usage:
    uv run python examples/story_from_txt_files.py \\
        --inspiration-dir D:/writing/inspiration \\
        --output D:/writing/inspiration/_generated_story.md \\
        --target-words 10000

The inspiration directory and the output path must be inside
sandbox.allowed_dirs in config.toml. The output directory will also be
used for intermediate artifacts (.outline.md, .draft1.md, .draft2.md).

Bump [agent].max_iterations to ~60 in config.toml for this workload —
the parent fans out heavily.

See docs/examples/story-from-txt-files.md for the walkthrough.
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


PROMPT_TEMPLATE = """You are a writer producing an original long-form short story in the exact
voice of a set of reference texts. Total target length: at least
{target_words} words.

Inputs directory: {inspiration}
Final output path: {output}
Intermediate artifact prefix: {output}  (you will write .outline.md,
.draft1.md, .draft2.md alongside the final file)

PIPELINE — follow it in order. Do not skip steps. Do not "summarize" or
"explain" — just execute and save artifacts.

STEP 1 — STYLE ANALYSIS
  a. fs_list_files(pattern='*.txt', base='{inspiration}').
  b. For each file, fs_read_file. (If more than 5 files, you may
     subagent_spawn one analyzer per file with task 'Read this file and
     return 5-10 bullet observations on voice, POV, sentence rhythm,
     vocabulary register, imagery, dialogue habits.')
  c. Internally compose a STYLE PROFILE — 8-15 bullets describing voice,
     POV, sentence rhythm, vocabulary register, imagery, dialogue,
     pacing, recurring motifs. You will paste this profile verbatim into
     every subagent brief below. Do NOT print it to the user yet.

STEP 2 — OUTLINE
  Spawn a subagent. Task:
     'Using this style profile: <paste full profile>
      Outline an original short story of at least {target_words} words
      across 6-8 sections. For each section provide: a title, a 2-3
      sentence summary, and a target word count (sum >= {target_words}).
      Return as a markdown numbered list. Do NOT write prose.'
  Save the returned outline to {output}.outline.md via fs_write_file.

STEP 3 — DRAFTING (parallelizable but cap at max_concurrent_agents)
  For EACH section in the outline, spawn a subagent. Task:
     'STYLE PROFILE: <paste profile>
      SECTION TO WRITE: <section title and summary>
      TARGET LENGTH: <section word count> words.
      Write only the prose of this section. No headings, no preamble, no
      author commentary. Match the voice exactly. Return the prose
      verbatim.'
  Collect the returned prose strings in section order. Concatenate them
  with a blank line between sections. Save the assembled draft to
  {output}.draft1.md via fs_write_file.

STEP 4 — EDITORIAL REVIEW
  Spawn a subagent. Task:
     'STYLE PROFILE: <paste profile>
      Read the draft at {output}.draft1.md via fs_read_file.
      Produce an editorial review. Return findings as bullets, each in
      the form [section: <n>] <issue> — <suggested fix>. Cover:
        - inconsistencies (character names, tense, POV)
        - tone drift from the style profile
        - language issues (clichés, repetition, unclear sentences,
          telling vs showing, weak verbs)
      Be specific. Quote short snippets. Limit to ~20 findings.'

STEP 5 — REVISION
  Spawn a subagent. Task:
     'STYLE PROFILE: <paste profile>
      DRAFT (read via fs_read_file from {output}.draft1.md).
      EDITORIAL FINDINGS: <paste the bullets from STEP 4>.
      Apply every finding. Preserve the voice. Do not shorten the work.
      Return the FULL revised story as prose, no commentary.'
  Save returned prose to {output}.draft2.md via fs_write_file.

STEP 6 — POLISH PASS
  Spawn a subagent. Task:
     'Read {output}.draft2.md via fs_read_file. Smooth transitions
      between sections, fix residual typos, ensure a satisfying close.
      Make NO structural changes. Return the polished story.'
  Save to {output} (the final path) via fs_write_file.

STEP 7 — REPORT
  Reply with exactly:
     - Final word count (approximate, based on draft2 length).
     - The five most impactful editorial findings that were addressed.
     - The path: {output}
  Nothing else.
"""


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--inspiration-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-words", type=int, default=10000)
    parser.add_argument("--max-iterations", type=int, default=60)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rt = await Runtime.create(cfg)
    try:
        agent = rt.new_agent()
        prompt = PROMPT_TEMPLATE.format(
            inspiration=args.inspiration_dir,
            output=args.output,
            target_words=args.target_words,
        )
        async for ev in agent.run(prompt, AgentRunOptions(max_iterations=args.max_iterations)):
            if isinstance(ev, ToolCallStart):
                print(f"-> {ev.name}({_brief(ev.arguments)})")
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


def _brief(args: dict) -> str:
    out = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 140:
            out[k] = v[:140] + "…"
        else:
            out[k] = v
    return str(out)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
