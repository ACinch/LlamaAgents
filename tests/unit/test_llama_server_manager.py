import asyncio
from pathlib import Path

import pytest

from llama_agents.config import LlamaConfig
from llama_agents.llama_client import LlamaServerManager


class _FakeClient:
    def __init__(self, reachable_after: int = 0):
        self.calls = 0
        self._reachable_after = reachable_after

    async def health(self) -> bool:
        self.calls += 1
        return self.calls > self._reachable_after


async def test_ensure_running_no_spawn_when_already_up():
    cfg = LlamaConfig(auto_spawn=True)
    client = _FakeClient(reachable_after=0)
    mgr = LlamaServerManager(cfg, client)
    await mgr.ensure_running()
    assert mgr.spawned is False


async def test_ensure_running_raises_when_unreachable_and_no_spawn():
    cfg = LlamaConfig(auto_spawn=False)
    client = _FakeClient(reachable_after=999)
    mgr = LlamaServerManager(cfg, client)
    from llama_agents.errors import LlamaUnreachable
    with pytest.raises(LlamaUnreachable):
        await mgr.ensure_running()


async def test_shutdown_only_kills_what_we_spawned():
    cfg = LlamaConfig(auto_spawn=False)
    client = _FakeClient(reachable_after=0)
    mgr = LlamaServerManager(cfg, client)
    await mgr.ensure_running()  # no spawn
    await mgr.shutdown()  # must not raise
    assert mgr.spawned is False
