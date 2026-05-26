from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class Event:
    """Base marker class for all loop events."""


@dataclass
class AssistantChunk(Event):
    text: str


@dataclass
class ToolCallStart(Event):
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallResult(Event):
    call_id: str
    ok: bool
    content: Any


@dataclass
class LoopError(Event):
    error_type: str
    message: str


@dataclass
class Done(Event):
    reason: str  # "finished" | "max_iterations" | "cancelled" | "token_budget"
    final_message: str | None = None
