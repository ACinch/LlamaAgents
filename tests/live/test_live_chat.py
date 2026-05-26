import os
from pathlib import Path

import pytest

from llama_agents.config import load_config
from llama_agents.agent import AgentRunOptions
from llama_agents.runtime import Runtime
from llama_agents.events import AssistantChunk, Done


@pytest.mark.live
async def test_round_trip_simple_prompt():
    cfg_path = os.environ.get("LLAMA_AGENTS_CONFIG", "config.toml")
    cfg = load_config(cfg_path)
    rt = await Runtime.create(cfg)
    try:
        agent = rt.new_agent()
        text = ""
        async for ev in agent.run(
            "Say 'pong' and nothing else.",
            AgentRunOptions(max_iterations=2),
        ):
            if isinstance(ev, AssistantChunk):
                text = ev.text
            if isinstance(ev, Done):
                break
        assert "pong" in text.lower()
    finally:
        await rt.aclose()
