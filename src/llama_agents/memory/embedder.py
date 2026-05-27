from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Iterable


class HashEmbedder:
    """Deterministic, dependency-free embedder for unit tests.

    Bag-of-tokens projected into `dim` dimensions via SHA1. Crude but
    stable; equal input -> equal output; overlapping vocabularies -> high
    cosine similarity.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def init(self) -> None:
        return None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in _tokens(text):
            h = hashlib.sha1(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self._dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def _tokens(s: str) -> Iterable[str]:
    return (m.group(0).lower() for m in re.finditer(r"[A-Za-z0-9_]+", s))


class FastEmbedEmbedder:
    """fastembed-backed embedder.

    Imports fastembed lazily so the module can be imported even when
    fastembed is not installed (we then surface a clear error at .embed
    time). The model loads on first call; subsequent calls reuse it.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model = None
        self._dim_cached: int | None = None
        self._lock = asyncio.Lock()

    @property
    def dim(self) -> int:
        return self._dim_cached or 384

    async def init(self) -> None:
        await self._ensure_loaded()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        await self._ensure_loaded()
        assert self._model is not None
        model = self._model

        def _run() -> list[list[float]]:
            vecs = list(model.embed(texts))
            return [v.tolist() for v in vecs]

        return await asyncio.to_thread(_run)

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return

            def _load():
                from fastembed import TextEmbedding

                m = TextEmbedding(model_name=self._model_name)
                probe = next(m.embed(["probe"]))
                return m, int(probe.shape[0])

            model, dim = await asyncio.to_thread(_load)
            self._model = model
            self._dim_cached = dim
