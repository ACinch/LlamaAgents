from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any, Protocol

from ..config import McpServerConfig
from ..errors import MCPServerCrashed
from .base import Tool


class _McpClientLike(Protocol):
    async def call_tool(self, name: str, args: dict[str, Any]) -> Any: ...


class McpBridgedTool(Tool):
    """A single bridged MCP tool exposed in our registry."""

    def __init__(
        self,
        *,
        server: str,
        underlying_name: str,
        description: str,
        schema: dict[str, Any],
        client: _McpClientLike,
    ) -> None:
        self._server = server
        self._underlying = underlying_name
        self._client = client
        self.name = f"{server}__{underlying_name}"  # type: ignore[misc]
        self.description = description  # type: ignore[misc]
        self.json_schema = schema  # type: ignore[misc]

    async def invoke(self, args: dict[str, Any]) -> Any:
        try:
            return await self._client.call_tool(self._underlying, args)
        except Exception as e:  # noqa: BLE001
            raise MCPServerCrashed(self._server) from e


class McpBridge:
    """Spawns configured MCP servers and produces bridged Tools.

    Uses the official `mcp` Python SDK at runtime; tests can pass mock clients.
    """

    def __init__(self, servers: list[McpServerConfig]) -> None:
        self._servers = servers
        self._stack = AsyncExitStack()
        self._tools: list[McpBridgedTool] = []

    async def start(self) -> list[McpBridgedTool]:
        # Imported lazily so unit tests without the mcp SDK can still import this module.
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        for srv in self._servers:
            params = StdioServerParameters(command=srv.command, args=srv.args, env=srv.env or None)
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listing = await session.list_tools()
            for t in listing.tools:
                self._tools.append(
                    McpBridgedTool(
                        server=srv.name,
                        underlying_name=t.name,
                        description=t.description or "",
                        schema=t.inputSchema or {"type": "object", "properties": {}},
                        client=_SessionClient(session),
                    )
                )
        return list(self._tools)

    async def aclose(self) -> None:
        await self._stack.aclose()


class _SessionClient:
    def __init__(self, session: Any) -> None:
        self._session = session

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        result = await self._session.call_tool(name, args)
        # mcp returns a CallToolResult with .content (list of content blocks)
        return [getattr(c, "text", c) for c in (result.content or [])]
