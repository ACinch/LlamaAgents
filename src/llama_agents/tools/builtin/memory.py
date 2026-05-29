from __future__ import annotations

from typing import Any, Callable

from ...memory.store import MemoryStore
from ..base import Tool


class MemoryRecallTool(Tool):
    name = "memory_recall"
    description = (
        "Retrieve previously-stored content from this run's scratch memory "
        "and past plans. Use this when you see '[evicted to memory ...]' in "
        "earlier tool results, or to look up the full text of a subagent's "
        "output via its memory_handle."
    )
    json_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "handle": {
                "type": "string",
                "description": "Optional. Restrict results to chunks from this blob_id.",
            },
            "k": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(
        self,
        *,
        store: MemoryStore,
        run_id_getter: Callable[[], str | None],
    ) -> None:
        self._store = store
        self._run_id_getter = run_id_getter

    async def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        rid = self._run_id_getter()
        chunks = await self._store.recall(
            query=args["query"],
            scope="all",
            thread_ids=[rid] if rid else None,
            handle=args.get("handle"),
            k=int(args.get("k", 5)),
        )
        return {
            "chunks": [
                {
                    "text": c.text,
                    "blob_id": c.blob_id,
                    "chunk_idx": c.chunk_idx,
                    "score": c.score,
                    "title": c.title,
                    "kind": c.kind,
                }
                for c in chunks
            ]
        }
