from typing import Any

import pytest

from llama_agents.errors import ToolNotFound, ToolValidationError
from llama_agents.tools.base import Tool
from llama_agents.tools.registry import ToolRegistry


class EchoTool(Tool):
    name = "echo"
    description = "echo input back"
    json_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def invoke(self, args: dict[str, Any]) -> Any:
        return args["text"]


async def test_register_and_invoke():
    reg = ToolRegistry()
    reg.register(EchoTool())
    result = await reg.invoke("echo", {"text": "hi"})
    assert result == "hi"


async def test_invoke_unknown_raises():
    reg = ToolRegistry()
    with pytest.raises(ToolNotFound):
        await reg.invoke("nope", {})


async def test_invoke_validates_required_args():
    reg = ToolRegistry()
    reg.register(EchoTool())
    with pytest.raises(ToolValidationError):
        await reg.invoke("echo", {})


def test_schemas_emits_openai_tools_array():
    reg = ToolRegistry()
    reg.register(EchoTool())
    schemas = reg.schemas()
    assert schemas == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "echo input back",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }
    ]


def test_register_duplicate_raises():
    reg = ToolRegistry()
    reg.register(EchoTool())
    with pytest.raises(ValueError):
        reg.register(EchoTool())
