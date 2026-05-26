from typing import Any

import pytest

from llama_agents.tools.mcp_bridge import McpBridgedTool


class _FakeMcpClient:
    def __init__(self, response: Any):
        self._response = response
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, args: dict) -> Any:
        self.calls.append((name, args))
        return self._response


async def test_bridged_tool_calls_underlying_client():
    client = _FakeMcpClient(response={"snippets": ["a", "b"]})
    tool = McpBridgedTool(
        server="rag",
        underlying_name="rag_query",
        description="search RAG",
        schema={"type": "object", "properties": {"query": {"type": "string"}}},
        client=client,
    )
    assert tool.name == "rag__rag_query"
    result = await tool.invoke({"query": "hello"})
    assert result == {"snippets": ["a", "b"]}
    assert client.calls == [("rag_query", {"query": "hello"})]
