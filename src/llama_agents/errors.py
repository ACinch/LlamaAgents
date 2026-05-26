class LlamaAgentsError(Exception):
    """Base for all llama-agents errors."""


class LlamaUnreachable(LlamaAgentsError):
    """llama-server is not reachable on the configured URL."""


class LlamaProtocolError(LlamaAgentsError):
    """llama-server returned an unexpected response shape."""


class ToolNotFound(LlamaAgentsError):
    def __init__(self, name: str):
        super().__init__(f"tool not found: {name}")
        self.name = name


class ToolValidationError(LlamaAgentsError):
    """Tool arguments failed schema validation."""


class SandboxViolation(LlamaAgentsError):
    def __init__(self, path: str, reason: str):
        super().__init__(f"sandbox violation: {path} ({reason})")
        self.path = path
        self.reason = reason


class MCPServerCrashed(LlamaAgentsError):
    def __init__(self, server: str):
        super().__init__(f"MCP server crashed: {server}")
        self.server = server


class MaxIterationsExceeded(LlamaAgentsError):
    """Agent exhausted its iteration budget without finalizing."""


class AgentLimitExceeded(LlamaAgentsError):
    """Concurrent-agent cap reached."""


class Cancelled(LlamaAgentsError):
    """Loop was cancelled by a surface or signal."""
