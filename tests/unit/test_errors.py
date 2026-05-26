import pytest
from llama_agents.errors import (
    LlamaAgentsError,
    LlamaUnreachable,
    LlamaProtocolError,
    ToolNotFound,
    ToolValidationError,
    SandboxViolation,
    MCPServerCrashed,
    MaxIterationsExceeded,
    AgentLimitExceeded,
    Cancelled,
)


def test_all_errors_inherit_from_base():
    for cls in (
        LlamaUnreachable,
        LlamaProtocolError,
        ToolNotFound,
        ToolValidationError,
        SandboxViolation,
        MCPServerCrashed,
        MaxIterationsExceeded,
        AgentLimitExceeded,
        Cancelled,
    ):
        assert issubclass(cls, LlamaAgentsError)


def test_sandbox_violation_carries_path():
    err = SandboxViolation(path="/etc/passwd", reason="outside allowed_dirs")
    assert err.path == "/etc/passwd"
    assert "outside allowed_dirs" in str(err)
