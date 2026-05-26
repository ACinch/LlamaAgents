from pathlib import Path

import pytest

from llama_agents.errors import SandboxViolation
from llama_agents.sandbox import check_path, check_command


def test_check_path_allows_file_inside_allowed_dir(tmp_path: Path):
    target = tmp_path / "foo.txt"
    target.write_text("x")
    result = check_path(target, allowed_dirs=[tmp_path])
    assert result == target.resolve()


def test_check_path_rejects_outside(tmp_path: Path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("other") / "bar.txt"
    outside.write_text("x")
    with pytest.raises(SandboxViolation):
        check_path(outside, allowed_dirs=[tmp_path])


def test_check_path_rejects_parent_traversal(tmp_path: Path):
    sneaky = tmp_path / ".." / "evil.txt"
    with pytest.raises(SandboxViolation):
        check_path(sneaky, allowed_dirs=[tmp_path / "sub"])


def test_check_command_allows_listed(tmp_path: Path):
    check_command(["git", "status"], allowlist=["git", "pytest"])


def test_check_command_rejects_unlisted():
    with pytest.raises(SandboxViolation):
        check_command(["rm", "-rf", "/"], allowlist=["git"])


def test_check_command_rejects_empty():
    with pytest.raises(SandboxViolation):
        check_command([], allowlist=["git"])
