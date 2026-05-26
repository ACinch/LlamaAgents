from __future__ import annotations

from typing import Any

from ..errors import ToolNotFound, ToolValidationError
from .base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.json_schema,
                },
            }
            for t in self._tools.values()
        ]

    async def invoke(self, name: str, args: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFound(name)
        self._validate(tool, args)
        return await tool.invoke(args)

    def clone(self) -> "ToolRegistry":
        new = ToolRegistry()
        new._tools = dict(self._tools)
        return new

    @staticmethod
    def _validate(tool: Tool, args: dict[str, Any]) -> None:
        required = tool.json_schema.get("required", [])
        missing = [r for r in required if r not in args]
        if missing:
            raise ToolValidationError(
                f"{tool.name} missing required args: {missing}"
            )
