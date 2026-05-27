# Example 1 — Write a story in the style of a folder of `.txt` files

**Goal:** Point the agent at a directory of plain-text excerpts (e.g. short
stories, sample chapters, a writer's published pieces). The agent reads them,
forms a sense of the style — sentence rhythm, vocabulary, point of view — and
produces a new short story in that voice.

## Tools exercised

- `fs_list_files` — discover the `.txt` files under the inspiration folder.
- `fs_read_file` — load each one.
- `fs_write_file` — save the final story alongside the inputs.
- (Optional) `subagent_spawn` — fan out style-analysis across multiple files
  in parallel, then have the parent author the story from the collated
  observations.

## Config

In `config.toml`, add the inspiration folder to `sandbox.allowed_dirs` so the
fs tools can reach it:

```toml
[sandbox]
allowed_dirs = [
    "D:/repos/LLM/llama-agents",
    "D:/writing/inspiration",   # contains the .txt files and where we'll save the story
]
```

## Prompt

```text
Read every .txt file in D:/writing/inspiration. For each file, note the
author's distinctive habits — sentence length, vocabulary, point of view,
imagery, pacing. Synthesize those observations into a brief style profile.
Then write a new 600-800 word short story in that exact voice. The story
should be original (not a continuation of any input) and stand on its own.
When finished, save it to D:/writing/inspiration/_generated_story.md using
fs_write_file, and reply with the first paragraph plus the saved path.
```

## Invocation

```powershell
$env:PYTHONIOENCODING = "utf-8"
uv run llamactl chat --max-iterations 25 @'
Read every .txt file in D:/writing/inspiration. ...
'@
```

(Use a here-string for multi-line prompts on Windows — see the script form
in `examples/story_from_txt_files.py` for a cleaner programmatic version.)

## Expected event stream

1. `→ fs_list_files({...})` returns the list of `.txt` paths.
2. One `→ fs_read_file` per file (or a `subagent_spawn` per file if you ask
   the agent to parallelize — useful with more than ~5 inputs).
3. The model emits a style profile internally, then the story body.
4. `→ fs_write_file(_generated_story.md, ...)`.
5. Final `AssistantChunk` with the opening paragraph + path.
6. `Done(reason=finished)`.

## Notes

- Bump `--max-iterations` if you have more than ~10 input files; each
  `fs_read_file` consumes one iteration of the loop.
- The model will absorb whatever's in those files, including any explicit or
  off-topic material. Curate the inputs before running.
