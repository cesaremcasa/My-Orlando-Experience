from __future__ import annotations


class AgentOpsError(Exception):
    """Sanitized AgentOps failure; never include raw exception text in HTTP."""

    def __init__(self, code: str, http_status: int, detail: str) -> None:
        self.code = code
        self.http_status = http_status
        self.detail = detail
        super().__init__(detail)


class ProviderError(AgentOpsError):
    def __init__(self) -> None:
        super().__init__("provider", 502, "Response provider unavailable")


class RetrievalError(AgentOpsError):
    def __init__(self) -> None:
        super().__init__("retrieval", 500, "Retrieval failed.")


class MemoryError(AgentOpsError):
    def __init__(self) -> None:
        super().__init__("memory", 500, "Memory operation failed.")


class SchemaError(AgentOpsError):
    def __init__(self) -> None:
        super().__init__("schema", 422, "Agent output schema invalid.")


class SafetyError(AgentOpsError):
    def __init__(self) -> None:
        super().__init__("safety", 422, "Safety check failed.")


class TimeoutError_(AgentOpsError):
    def __init__(self) -> None:
        super().__init__("timeout", 504, "Agent timed out.")


class ConfigurationError(AgentOpsError):
    def __init__(self) -> None:
        super().__init__(
            "configuration",
            500,
            'AgentOps runtime requires extras. Install with: pip install -e ".[rag,agentops]"',
        )


class FeedbackError(AgentOpsError):
    def __init__(self) -> None:
        super().__init__("feedback", 500, "Feedback operation failed.")
