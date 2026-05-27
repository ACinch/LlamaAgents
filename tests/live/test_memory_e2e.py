import pytest

from llama_agents.agent import AgentRunOptions
from llama_agents.config import (
    AgentConfig,
    Config,
    LlamaConfig,
    MemoryConfig,
    SandboxConfig,
)
from llama_agents.runtime import Runtime


@pytest.mark.live
@pytest.mark.asyncio
async def test_memory_end_to_end(tmp_path):
    cfg = Config(
        llama=LlamaConfig(auto_spawn=False),
        sandbox=SandboxConfig(allowed_dirs=[tmp_path]),
        memory=MemoryConfig(
            root=tmp_path / "mem",
            scratch_retention_hours=-1,  # keep so we can inspect
        ),
        agent=AgentConfig(max_iterations=5),
    )
    rt = await Runtime.create(cfg)
    try:
        agent = rt.new_agent()
        events = []
        async for ev in agent.run(
            "Say hello in one sentence.",
            AgentRunOptions(max_iterations=3, skip_planning=True),
        ):
            events.append(ev)
        assert (tmp_path / "mem" / "index.sqlite").exists()
    finally:
        await rt.aclose()
