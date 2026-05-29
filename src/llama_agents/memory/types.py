from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

Scope = Literal["run", "plans", "all"]
WriteScope = Literal["run", "plans"]


@dataclass
class RecalledChunk:
    blob_id: str
    chunk_idx: int
    text: str
    score: float
    title: str
    kind: str


@dataclass
class BlobMeta:
    id: str
    scope: str
    thread_id: str | None     # renamed from run_id
    kind: str
    title: str
    file_path: str
    created_at: str
    metadata: dict = field(default_factory=dict)


class Embedder(Protocol):
    """Anything with an `embed` method that returns one float-list per input."""

    @property
    def dim(self) -> int: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...
