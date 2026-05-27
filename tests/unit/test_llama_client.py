import json

import httpx
import pytest

from llama_agents.errors import LlamaProtocolError, LlamaUnreachable
from llama_agents.llama_client import ChatResponse, LlamaClient, ToolCall


def _mock_transport(handler):
    return httpx.MockTransport(handler)


async def test_chat_parses_plain_assistant_message():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["messages"][0]["role"] == "user"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hi back"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    client = LlamaClient(base_url="http://x", transport=_mock_transport(handler))
    resp = await client.chat(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert isinstance(resp, ChatResponse)
    assert resp.content == "hi back"
    assert resp.tool_calls == []


async def test_chat_parses_tool_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "echo",
                                        "arguments": '{"text": "hi"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    client = LlamaClient(base_url="http://x", transport=_mock_transport(handler))
    resp = await client.chat(messages=[{"role": "user", "content": "go"}], tools=[])
    assert resp.tool_calls == [ToolCall(id="call_1", name="echo", arguments={"text": "hi"})]


async def test_chat_raises_on_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope", request=request)

    client = LlamaClient(base_url="http://x", transport=_mock_transport(handler))
    with pytest.raises(LlamaUnreachable):
        await client.chat(messages=[{"role": "user", "content": "x"}], tools=[])


async def test_chat_raises_on_bad_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    client = LlamaClient(base_url="http://x", transport=_mock_transport(handler))
    with pytest.raises(LlamaProtocolError):
        await client.chat(messages=[{"role": "user", "content": "x"}], tools=[])


async def test_chat_includes_reasoning_budget_when_set():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    client = LlamaClient(base_url="http://x", transport=_mock_transport(handler))
    await client.chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[],
        reasoning_budget_tokens=4000,
    )
    assert seen["body"]["reasoning_budget_tokens"] == 4000


async def test_chat_omits_reasoning_budget_when_unset():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    client = LlamaClient(base_url="http://x", transport=_mock_transport(handler))
    await client.chat(messages=[{"role": "user", "content": "x"}], tools=[])
    assert "reasoning_budget_tokens" not in seen["body"]


async def test_health_returns_bool():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    client = LlamaClient(base_url="http://x", transport=_mock_transport(handler))
    assert await client.health() is True
