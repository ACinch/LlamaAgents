from __future__ import annotations

from pathlib import Path
from typing import Any

from ...sandbox import check_path
from ..base import Tool


class _SandboxedFsTool(Tool):
    def __init__(self, allowed_dirs: list[Path]) -> None:
        self._allowed_dirs = [Path(d) for d in allowed_dirs]


class ReadFileTool(_SandboxedFsTool):
    name = "fs_read_file"
    description = "Read the entire contents of a UTF-8 text file."
    json_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    async def invoke(self, args: dict[str, Any]) -> str:
        p = check_path(args["path"], self._allowed_dirs)
        return p.read_text(encoding="utf-8")


class WriteFileTool(_SandboxedFsTool):
    name = "fs_write_file"
    description = "Write UTF-8 text to a file, creating parents as needed."
    json_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    async def invoke(self, args: dict[str, Any]) -> str:
        p = check_path(args["path"], self._allowed_dirs)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args["content"], encoding="utf-8")
        return f"wrote {len(args['content'])} bytes to {p}"


class EditFileTool(_SandboxedFsTool):
    name = "fs_edit_file"
    description = (
        "Replace an exact, uniquely-occurring substring in a file. "
        "Fails if the substring is missing or appears more than once."
    )
    json_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "find": {"type": "string"},
            "replace": {"type": "string"},
        },
        "required": ["path", "find", "replace"],
    }

    async def invoke(self, args: dict[str, Any]) -> str:
        p = check_path(args["path"], self._allowed_dirs)
        text = p.read_text(encoding="utf-8")
        count = text.count(args["find"])
        if count == 0:
            raise ValueError(f"find string not found in {p}")
        if count > 1:
            raise ValueError(
                f"find string occurs {count} times in {p}; must be unique"
            )
        p.write_text(text.replace(args["find"], args["replace"]), encoding="utf-8")
        return f"edited {p}"


class ListFilesTool(_SandboxedFsTool):
    name = "fs_list_files"
    description = "List files matching a glob pattern inside an allowed base dir."
    json_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "glob, e.g. **/*.py"},
            "base": {"type": "string"},
        },
        "required": ["pattern", "base"],
    }

    async def invoke(self, args: dict[str, Any]) -> list[str]:
        base = check_path(args["base"], self._allowed_dirs)
        return sorted(str(p) for p in base.glob(args["pattern"]) if p.is_file())
