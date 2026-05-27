from __future__ import annotations

import re

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def chunk_markdown(
    text: str, *, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Split markdown into chunks, preferring header boundaries.

    Mirrors the strategy of the reference RAG implementation: split on
    headers, then split oversized sections by lines with overlap, while
    keeping the current header line as context at the top of each chunk.
    """
    if not text.strip():
        return []

    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_header = ""
    current_lines: list[str] = []

    for line in lines:
        if _HEADER_RE.match(line):
            if current_lines:
                sections.append((current_header, current_lines))
            current_header = line
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_header, current_lines))

    if not sections:
        return _split_lines_with_overlap(lines, chunk_size, chunk_overlap)

    chunks: list[str] = []
    for header, section_lines in sections:
        body = "\n".join(section_lines).strip()
        if not body:
            continue
        if len(body) <= chunk_size:
            chunks.append(body)
            continue
        subs = _split_lines_with_overlap(section_lines, chunk_size, chunk_overlap)
        for sub in subs:
            if header and not sub.startswith(header):
                chunks.append(f"{header}\n\n{sub}")
            else:
                chunks.append(sub)
    return chunks


def _split_lines_with_overlap(
    lines: list[str], chunk_size: int, overlap: int
) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for line in lines:
        ls = len(line) + 1
        if size + ls > chunk_size and buf:
            chunks.append("\n".join(buf).strip())
            tail: list[str] = []
            tail_size = 0
            for prev in reversed(buf):
                if tail_size >= overlap:
                    break
                tail.insert(0, prev)
                tail_size += len(prev) + 1
            buf = tail
            size = tail_size
        buf.append(line)
        size += ls
    if buf:
        joined = "\n".join(buf).strip()
        if joined:
            chunks.append(joined)
    return chunks
