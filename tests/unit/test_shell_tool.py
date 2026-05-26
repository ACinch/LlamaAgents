import sys
from pathlib import Path

import pytest

from llama_agents.errors import SandboxViolation
from llama_agents.tools.builtin.shell import ShellRunTool


async def test_run_allowed_command(tmp_path: Path):
    tool = ShellRunTool(
        allowed_dirs=[tmp_path], allowlist=[sys.executable.split("/")[-1].split("\\")[-1], "python"]
    )
    # Use the running interpreter as the binary; alias the basename in allowlist.
    # We test exit code + stdout capture by running python -c.
    result = await tool.invoke(
        {"command": ["python", "-c", "print('hi')"], "cwd": str(tmp_path)}
    )
    assert result["returncode"] == 0
    assert "hi" in result["stdout"]


async def test_run_rejects_unlisted(tmp_path: Path):
    tool = ShellRunTool(allowed_dirs=[tmp_path], allowlist=["git"])
    with pytest.raises(SandboxViolation):
        await tool.invoke({"command": ["rm", "-rf", "/"], "cwd": str(tmp_path)})


async def test_run_rejects_cwd_outside(tmp_path: Path, tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    tool = ShellRunTool(allowed_dirs=[tmp_path], allowlist=["python"])
    with pytest.raises(SandboxViolation):
        await tool.invoke(
            {"command": ["python", "-c", "pass"], "cwd": str(other)}
        )


async def test_run_captures_nonzero_exit(tmp_path: Path):
    tool = ShellRunTool(allowed_dirs=[tmp_path], allowlist=["python"])
    result = await tool.invoke(
        {"command": ["python", "-c", "import sys; sys.exit(7)"], "cwd": str(tmp_path)}
    )
    assert result["returncode"] == 7
