import pytest

from llama_agents.memory.embedder import HashEmbedder


@pytest.mark.asyncio
async def test_hash_embedder_dim_and_determinism():
    e = HashEmbedder(dim=16)
    a = await e.embed(["hello world", "second one"])
    b = await e.embed(["hello world"])
    assert len(a) == 2
    assert len(a[0]) == 16
    assert a[0] == b[0]


@pytest.mark.asyncio
async def test_hash_embedder_similar_strings_have_high_cosine():
    import math

    e = HashEmbedder(dim=64)
    [v1, v2] = await e.embed(
        ["the quick brown fox jumps over the lazy dog",
         "the quick brown fox jumps over the lazy cat"]
    )

    def cos(a, b):
        s = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return s / (na * nb)

    assert cos(v1, v2) > 0.5
