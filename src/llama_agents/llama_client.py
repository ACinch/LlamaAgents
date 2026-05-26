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


import asyncio as _asyncio
import subprocess

from .config import LlamaConfig


class LlamaServerManager:
    """Optionally spawns llama-server.exe if not already reachable."""

    def __init__(self, cfg: LlamaConfig, client: "LlamaClient | object") -> None:
        self._cfg = cfg
        self._client = client
        self._process: subprocess.Popen | None = None

    @property
    def spawned(self) -> bool:
        return self._process is not None

    async def ensure_running(self) -> None:
        if await self._client.health():
            return
        if not self._cfg.auto_spawn:
            raise LlamaUnreachable(
                f"llama-server not reachable and auto_spawn=false"
            )
        if self._cfg.server_bin is None or self._cfg.model_path is None:
            raise LlamaUnreachable("auto_spawn requires server_bin and model_path")
        self._process = subprocess.Popen(
            [
                str(self._cfg.server_bin),
                "-m", str(self._cfg.model_path),
                "-ngl", str(self._cfg.ngl),
                "-c", str(self._cfg.ctx_size),
            ],
        )
        deadline = self._cfg.startup_timeout_seconds
        for _ in range(deadline):
            if await self._client.health():
                return
            await _asyncio.sleep(1)
        raise LlamaUnreachable(
            f"llama-server failed to become ready in {deadline}s"
        )

    async def shutdown(self) -> None:
        if self._process is None:
            return
        if not self._cfg.kill_on_exit:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None
