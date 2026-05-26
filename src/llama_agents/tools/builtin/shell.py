from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ...sandbox import check_command, check_path
from ..base import Tool


class ShellRunTool(Tool):
    name = "shell_run"
    description = (
        "Run an allowlisted command. The command argv[0] must be in the "
        "configured allowlist. Returns {returncode, stdout, stderr}."
    )
    json_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "array",
                "items": {"type": "string"},
                "description": "argv list; argv[0] must be in allowlist",
            },
            "cwd": {"type": "string"},
            "timeout": {"type": "integer", "default": 120},
        },
        "required": ["command", "cwd"],
    }

    def __init__(self, allowed_dirs: list[Path], allowlist: list[str]) -> None:
        self._allowed_dirs = [Path(d) for d in allowed_dirs]
        self._allowlist = list(allowlist)

    async def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        cmd: list[str] = args["command"]
        check_command(cmd, self._allowlist)
        cwd = check_path(args["cwd"], self._allowed_dirs)
        if not cwd.is_dir():
            raise ValueError(f"cwd is not a directory: {cwd}")
        timeout = int(args.get("timeout", 120))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"timeout after {timeout}s",
            }
        return {
            "returncode": proc.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }
