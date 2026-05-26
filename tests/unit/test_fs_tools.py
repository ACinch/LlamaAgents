from pathlib import Path

import pytest

from llama_agents.errors import SandboxViolation
from llama_agents.tools.builtin.fs import (
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    ListFilesTool,
)


async def test_read_file(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    tool = ReadFileTool(allowed_dirs=[tmp_path])
    out = await tool.invoke({"path": str(f)})
    assert out == "hello"


async def test_read_file_rejects_outside(tmp_path: Path, tmp_path_factory):
    other = tmp_path_factory.mktemp("o") / "x.txt"
    other.write_text("x")
    tool = ReadFileTool(allowed_dirs=[tmp_path])
    with pytest.raises(SandboxViolation):
        await tool.invoke({"path": str(other)})


async def test_write_file_creates(tmp_path: Path):
    f = tmp_path / "new.txt"
    tool = WriteFileTool(allowed_dirs=[tmp_path])
    out = await tool.invoke({"path": str(f), "content": "abc"})
    assert f.read_text(encoding="utf-8") == "abc"
    assert "wrote" in out.lower()


async def test_edit_file_replaces_unique_match(tmp_path: Path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\ny = 2\n", encoding="utf-8")
    tool = EditFileTool(allowed_dirs=[tmp_path])
    await tool.invoke({"path": str(f), "find": "y = 2", "replace": "y = 3"})
    assert f.read_text(encoding="utf-8") == "x = 1\ny = 3\n"


async def test_edit_file_rejects_nonunique(tmp_path: Path):
    f = tmp_path / "code.py"
    f.write_text("a\na\n", encoding="utf-8")
    tool = EditFileTool(allowed_dirs=[tmp_path])
    with pytest.raises(ValueError):
        await tool.invoke({"path": str(f), "find": "a", "replace": "b"})


async def test_edit_file_rejects_missing(tmp_path: Path):
    f = tmp_path / "code.py"
    f.write_text("hello\n", encoding="utf-8")
    tool = EditFileTool(allowed_dirs=[tmp_path])
    with pytest.raises(ValueError):
        await tool.invoke({"path": str(f), "find": "missing", "replace": "x"})


async def test_list_files_globs(tmp_path: Path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")
    tool = ListFilesTool(allowed_dirs=[tmp_path])
    out = await tool.invoke({"pattern": "*.py", "base": str(tmp_path)})
    assert sorted(out) == [str(tmp_path / "a.py"), str(tmp_path / "b.py")]
