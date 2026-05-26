from __future__ import annotations

import asyncio
from typing import Callable, Protocol

from .agent import Agent
from .config import Config
from .llama_client import LlamaClient, LlamaServerManager
from .tools.builtin.fs import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
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
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.manager = manager
        self.bridge = bridge
        self.registry = registry
        self.semaphore = semaphore

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

        sem = asyncio.Semaphore(cfg.agent.max_concurrent_agents)

        bridge: McpBridge | None = None
        if cfg.mcp_servers:
            bridge = McpBridge(cfg.mcp_servers)
            for t in await bridge.start():
                registry.register(t)

        rt = cls(cfg, client, manager, bridge, registry, sem)

        # Inject the spawn tool last (it needs the runtime to make new agents).
        registry.register(
            SpawnSubagentTool(agent_factory=rt.new_agent, semaphore=sem)
        )
        return rt

    def new_agent(self) -> Agent:
        # Each agent shares the registry, but has its own conversation state.
        return Agent(client=self.client, registry=self.registry)

    async def aclose(self) -> None:
        if self.bridge is not None:
            await self.bridge.aclose()
        if self.manager is not None:
            await self.manager.shutdown()
        await self.client.aclose()
