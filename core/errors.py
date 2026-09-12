"""Domain errors that map cleanly onto HTTP status codes."""

from __future__ import annotations


class RagError(Exception):
    """Base class for errors this system raises deliberately."""


class UpstreamServiceError(RagError):
    """A dependency (Bedrock, Qdrant, the gateway) failed.

    Carried as a typed error so the API can answer 503 with an actionable
    detail instead of a bare 500, and so the provider and operation survive
    into the trace rather than being flattened into a stack trace.
    """

    def __init__(
        self, provider: str, operation: str, message: str, *, cause: Exception | None = None
    ) -> None:
        self.provider = provider
        self.operation = operation
        self.message = message
        self.__cause__ = cause
        super().__init__(f"{provider}.{operation}: {message}")

    @property
    def detail(self) -> str:
        return str(self)
