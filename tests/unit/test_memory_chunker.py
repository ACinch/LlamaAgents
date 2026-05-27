from llama_agents.memory.chunker import chunk_markdown


def test_chunk_short_markdown_returns_one_chunk():
    chunks = chunk_markdown("# Title\n\nsmall body", chunk_size=1500, chunk_overlap=150)
    assert len(chunks) == 1
    assert chunks[0].startswith("# Title")
    assert "small body" in chunks[0]


def test_chunk_splits_by_headers():
    md = "# A\n\nbody a\n\n# B\n\nbody b\n\n# C\n\nbody c"
    chunks = chunk_markdown(md, chunk_size=20, chunk_overlap=0)
    assert len(chunks) >= 3
    assert any("body a" in c for c in chunks)
    assert any("body b" in c for c in chunks)
    assert any("body c" in c for c in chunks)


def test_chunk_oversized_section_is_split_with_overlap():
    body = "\n".join(f"line {i}" for i in range(200))
    md = f"# Long\n\n{body}"
    chunks = chunk_markdown(md, chunk_size=400, chunk_overlap=80)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.startswith("# Long") or "# Long" in c.split("\n", 1)[0]


def test_chunk_no_headers_falls_back_to_line_split():
    md = "\n".join(f"line {i}" for i in range(100))
    chunks = chunk_markdown(md, chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= 2
    assert "line 0" in chunks[0]


def test_chunk_empty_returns_empty_list():
    assert chunk_markdown("", chunk_size=1500, chunk_overlap=150) == []
    assert chunk_markdown("   \n  ", chunk_size=1500, chunk_overlap=150) == []
