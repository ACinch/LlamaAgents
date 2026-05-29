import asyncio
from pathlib import Path

import pytest

from llama_agents.config import Config, LlamaConfig, AgentConfig, SandboxConfig, MemoryConfig
from llama_agents.llama_client import ChatResponse
from llama_agents.runtime import Runtime


class FakeClient:
    async def chat(self, *, messages, tools, temperature=0.2, reasoning_budget_tokens=None):
        return ChatResponse(content="done")

    async def health(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


async def test_runtime_builds_registry_with_builtins(tmp_path: Path):
    cfg = Config(
        llama=LlamaConfig(auto_spawn=False),
        agent=AgentConfig(max_concurrent_agents=3),
        sandbox=SandboxConfig(allowed_dirs=[tmp_path], shell_allowlist=["python"]),
    )
    rt = await Runtime.create(cfg, client_factory=lambda url: FakeClient())
    names = rt.registry.names()
    assert "fs_read_file" in names
    assert "fs_write_file" in names
    assert "fs_edit_file" in names
    assert "fs_list_files" in names
    assert "shell_run" in names
    assert "subagent_spawn" in names
    await rt.aclose()


async def test_runtime_creates_subagent_with_isolated_registry(tmp_path: Path):
    cfg = Config(
        llama=LlamaConfig(auto_spawn=False),
        sandbox=SandboxConfig(allowed_dirs=[tmp_path], shell_allowlist=["python"]),
    )
    rt = await Runtime.create(cfg, client_factory=lambda url: FakeClient())
    sub = rt.new_agent()
    parent = rt.new_agent()
    assert sub is not parent
    await rt.aclose()


async def test_subagent_does_not_mutate_parent_registry(tmp_path: Path):
    from llama_agents.config import McpServerConfig  # unused, just to keep imports tidy if needed
    cfg = Config(
        llama=LlamaConfig(auto_spawn=False),
        sandbox=SandboxConfig(allowed_dirs=[tmp_path], shell_allowlist=["python"]),
    )
    rt = await Runtime.create(cfg, client_factory=lambda url: FakeClient())
    parent_names_before = set(rt.registry.names())
    # Simulate what spawn does: clone registry and unregister a tool on the clone.
    clone = rt.registry.clone()
    clone.unregister("subagent_spawn")
    assert "subagent_spawn" not in clone.names()
    assert "subagent_spawn" in rt.registry.names()
    assert set(rt.registry.names()) == parent_names_before
    await rt.aclose()


@pytest.mark.asyncio
async def test_runtime_registers_memory_recall_when_enabled(tmp_path, monkeypatch):
    cfg = Config(
        sandbox=SandboxConfig(allowed_dirs=[tmp_path]),
        memory=MemoryConfig(root=tmp_path / ".mem"),
    )
    # Avoid spawning llama-server
    monkeypatch.setattr(cfg.llama, "auto_spawn", False)

    class _FakeClient:
        async def chat(self, **_): raise NotImplementedError
        async def health(self): return True
        async def aclose(self): pass

    rt = await Runtime.create(cfg, client_factory=lambda url: _FakeClient())
    try:
        assert "memory_recall" in rt.registry.names()
    finally:
        await rt.aclose()


@pytest.mark.asyncio
async def test_runtime_uses_inert_store_when_disabled(tmp_path, monkeypatch):
    from llama_agents.memory.store import InertMemoryStore

    cfg = Config(
        sandbox=SandboxConfig(allowed_dirs=[tmp_path]),
        memory=MemoryConfig(enabled=False),
    )
    monkeypatch.setattr(cfg.llama, "auto_spawn", False)

    class _FakeClient:
        async def chat(self, **_): raise NotImplementedError
        async def health(self): return True
        async def aclose(self): pass

    rt = await Runtime.create(cfg, client_factory=lambda url: _FakeClient())
    try:
        assert isinstance(rt.memory, InertMemoryStore)
        assert "memory_recall" in rt.registry.names()  # tool registered either way
    finally:
        await rt.aclose()


@pytest.mark.asyncio
async def test_runtime_exposes_thread_store(tmp_path: Path):
    from llama_agents.thread.store import ThreadStore
    cfg = Config(
        llama=LlamaConfig(auto_spawn=False),
        sandbox=SandboxConfig(allowed_dirs=[tmp_path]),
    )
    rt = await Runtime.create(cfg, client_factory=lambda url: FakeClient())
    try:
        assert isinstance(rt.thread_store, ThreadStore)
    finally:
        await rt.aclose()
