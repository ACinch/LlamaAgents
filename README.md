# llama-agents

Local orchestration layer around llama.cpp. Turns a running `llama-server.exe`
into a tool-using agent with filesystem access, allowlisted shell execution,
subagent spawning, and MCP-bridged external tools.

See `docs/design.md` for the full design.

## Quickstart

```bash
uv sync --extra dev
uv run llamactl chat
```
