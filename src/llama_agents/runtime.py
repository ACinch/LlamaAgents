from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Protocol

from .agent import Agent
from .config import Config
from .llama_client import LlamaClient, LlamaServerManager
from .memory.embedder import FastEmbedEmbedder
from .memory.store import InertMemoryStore, MemoryStore
from .tools.builtin.fs import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from .tools.builtin.memory import MemoryRecallTool
from .tools.builtin.shell import ShellRunTool
from .tools.builtin.subagent import SpawnSubagentTool
from .tools.mcp_bridge import McpBridge
from .tools.registry import ToolRegistry


class _ClientLike(Protocol):
    async def chat(self, *, messages, tools, temperature=0.2): ...
    async def health(self) -> bool: ...
    async def aclose(self) -> None: ...


class Runtime:
    """Holds the long-lived runtime: client, registry, bridge, semaphore."""

    def __init__(
        self,
        cfg: Config,
        client: _ClientLike,
        manager: LlamaServerManager | None,
        bridge: McpBridge | None,
        registry: ToolRegistry,
        semaphore: asyncio.Semaphore,
        memory: "MemoryStore | InertMemoryStore",
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.manager = manager
        self.bridge = bridge
        self.registry = registry
        self.semaphore = semaphore
        self.memory = memory
        self._current_run_id: str | None = None

    @classmethod
    async def create(
        cls,
        cfg: Config,
        *,
        client_factory: Callable[[str], _ClientLike] | None = None,
    ) -> "Runtime":
        client = (
            client_factory(cfg.llama.server_url)
            if client_factory
            else LlamaClient(base_url=cfg.llama.server_url)
        )
        manager = LlamaServerManager(cfg.llama, client)
        await manager.ensure_running()

        registry = ToolRegistry()
        sandbox = cfg.sandbox
        registry.register(ReadFileTool(allowed_dirs=sandbox.allowed_dirs))
        registry.register(WriteFileTool(allowed_dirs=sandbox.allowed_dirs))
        registry.register(EditFileTool(allowed_dirs=sandbox.allowed_dirs))
        registry.register(ListFilesTool(allowed_dirs=sandbox.allowed_dirs))
        registry.register(
            ShellRunTool(
                allowed_dirs=sandbox.allowed_dirs,
                allowlist=sandbox.shell_allowlist,
            )
        )

        # Build memory layer
        if cfg.memory.enabled:
            mem_root = _resolve_memory_root(cfg)
            embedder = FastEmbedEmbedder(model_name=cfg.memory.embedding_model)
            mem: MemoryStore | InertMemoryStore = MemoryStore(
                root=mem_root,
                embedder=embedder,
                chunk_size=cfg.memory.chunk_size,
                chunk_overlap=cfg.memory.chunk_overlap,
                retention_hours=cfg.memory.scratch_retention_hours,
            )
            await mem.init()
            await mem.gc_expired()
        else:
            mem = InertMemoryStore()
            await mem.init()

        sem = asyncio.Semaphore(cfg.agent.max_concurrent_agents)

        bridge: McpBridge | None = None
        if cfg.mcp_servers:
            bridge = McpBridge(cfg.mcp_servers)
            for t in await bridge.start():
                registry.register(t)

        rt = cls(cfg, client, manager, bridge, registry, sem, mem)

        # memory_recall tool (always available — InertMemoryStore returns [])
        registry.register(
            MemoryRecallTool(store=rt.memory, run_id_getter=lambda: rt._current_run_id)
        )

        # Inject the spawn tool last (it needs the runtime to make new agents).
        registry.register(
            SpawnSubagentTool(agent_factory=rt.new_agent, semaphore=sem)
        )
        return rt

    def new_agent(self) -> Agent:
        # Each agent shares the registry, but has its own conversation state.
        return Agent(client=self.client, registry=self.registry.clone())

    async def aclose(self) -> None:
        if self.bridge is not None:
            await self.bridge.aclose()
        if self.manager is not None:
            await self.manager.shutdown()
        await self.memory.close()
        await self.client.aclose()


def _resolve_memory_root(cfg: Config) -> Path:
    root_cfg = cfg.memory.root  # Path (post T1 fix)
    p = Path(root_cfg)
    if p.is_absolute():
        return p
    base = cfg.sandbox.allowed_dirs[0] if cfg.sandbox.allowed_dirs else Path.cwd()
    return base / root_cfg
