from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from .errors import LlamaProtocolError, LlamaUnreachable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_message: dict[str, Any] | None = None


class LlamaClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=timeout, transport=transport
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        try:
            r = await self._client.get("/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools

        try:
            r = await self._client.post("/v1/chat/completions", json=payload)
        except httpx.ConnectError as e:
            raise LlamaUnreachable(str(e)) from e
        except httpx.HTTPError as e:
            raise LlamaUnreachable(str(e)) from e

        if r.status_code != 200:
            raise LlamaProtocolError(f"HTTP {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as e:
            raise LlamaProtocolError(f"unexpected response shape: {e}") from e

        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=args)
            )

        return ChatResponse(
            content=msg.get("content"),
            tool_calls=tool_calls,
            raw_message=msg,
        )
