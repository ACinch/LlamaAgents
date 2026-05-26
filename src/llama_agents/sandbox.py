from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .errors import SandboxViolation


def check_path(path: str | Path, allowed_dirs: Iterable[Path]) -> Path:
    """Resolve path and verify it lies under at least one allowed dir.

    Returns the resolved absolute Path. Raises SandboxViolation otherwise.
    """
    resolved = Path(path).resolve(strict=False)
    for d in allowed_dirs:
        base = Path(d).resolve(strict=False)
        try:
            resolved.relative_to(base)
            return resolved
        except ValueError:
            continue
    raise SandboxViolation(path=str(resolved), reason="outside allowed_dirs")


def check_command(command: list[str], allowlist: Iterable[str]) -> None:
    if not command:
        raise SandboxViolation(path="", reason="empty command")
    allowed = set(allowlist)
    if command[0] not in allowed:
        raise SandboxViolation(
            path=command[0], reason=f"command not in allowlist {sorted(allowed)}"
        )
