# Example 3 — Marketing suggestions from RAG content

**Goal:** Mine the RAG index for everything it knows about a product or
topic, then produce a set of marketing suggestions grounded in that material.
Useful for brainstorming campaign angles, copy lines, audience segments, or
positioning statements without inventing claims the product can't support.

## Tools exercised

- `rag__rag_query` — the only tool that actually touches the knowledge base.
  The agent will typically call it several times with different angles
  (features, pain points, audience, competitors mentioned) before composing.
- (Optional) `fs_write_file` — save the suggestion deck to disk.

No filesystem access needed for the lookup itself, so this example is the
safest to run against arbitrary topics.

## Config

The RAG MCP server is the only requirement:

```toml
[[mcp_servers]]
name = "rag"
command = "node"
args = ["D:/repos/LLM/rag/dist/index.js"]
```

If you want the agent to save the deck, also ensure the output folder is
under `sandbox.allowed_dirs`.

## Prompt

```text
You are a marketing strategist. Your task: produce a deck of marketing
suggestions for ACinch.

Process:
1. Use rag__rag_query at least three times with DIFFERENT angles to gather
   grounded material. Suggested angles: 'ACinch product features',
   'ACinch target users and use cases', 'ACinch differentiation or
   competitors'. Vary the queries based on what you find.
2. Synthesize the results into FIVE marketing suggestions. Each must
   include: a positioning headline, the audience it targets, the supporting
   claim (cite which RAG snippet justifies it), and a one-sentence call to
   action.
3. Do NOT invent capabilities. If the RAG returns no material for a claim,
   drop the suggestion or rephrase to fit what's there.
4. Reply with the deck in plain markdown. No preamble, just the five
   suggestions numbered 1-5.
```

## Invocation

```powershell
$env:PYTHONIOENCODING = "utf-8"
uv run llamactl chat --max-iterations 15 @'
You are a marketing strategist ...
'@
```

Or the script form: `examples/marketing_suggestions_from_rag.py`.

## Expected event stream

1. Three (or more) `→ rag__rag_query(...)` calls with different queries.
2. The model emits the deck as a single `AssistantChunk`.
3. `Done(reason=finished)`.

## Notes

- Step 3 ("don't invent capabilities") is the load-bearing instruction.
  Without it the model will pad with confident-sounding generic claims.
- Quality of suggestions is bounded by what's in the RAG index — if the
  index is thin, ask the agent to flag gaps rather than fill them.
- For a different topic, change "ACinch" to whatever the index actually
  contains. Run `rag_list_projects` (via the RAG CLI, not the agent) first
  if you're unsure.
