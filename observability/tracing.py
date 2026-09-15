"""Langfuse instrumentation.

Every Bedrock and Qdrant call is wrapped in a span from the first call ever
made, not retrofitted later. Three implementations share one interface:

* ``LangfuseTracer``  - the real thing, used when credentials are configured.
* ``NoOpTracer``      - used in tests and when Langfuse is disabled; touches no network.
* ``RecordingTracer`` - captures span names so a test can assert instrumentation exists.

Langfuse v4 is itself an OpenTelemetry SDK, so nothing here creates a second
TracerProvider - doing that produces two disconnected trace trees.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal, Protocol

SpanType = Literal[
    "span", "generation", "embedding", "retriever", "tool", "chain", "guardrail"
]


class SpanHandle(Protocol):
    def update(self, **attrs: Any) -> None: ...


class _NoOpHandle:
    def update(self, **attrs: Any) -> None:
        return None


class Tracer(Protocol):
    @property
    def enabled(self) -> bool: ...

    def observe(
        self, name: str, *, as_type: SpanType = "span", **attrs: Any
    ) -> Any: ...

    async def flush(self) -> None: ...


class NoOpTracer:
    """Used when Langfuse is disabled. Must never touch the network."""

    enabled = False

    @asynccontextmanager
    async def observe(
        self, name: str, *, as_type: SpanType = "span", **attrs: Any
    ) -> AsyncIterator[_NoOpHandle]:
        yield _NoOpHandle()

    async def flush(self) -> None:
        return None


class RecordingTracer(NoOpTracer):
    """Records span names so tests can assert new steps stay instrumented."""

    enabled = False

    def __init__(self) -> None:
        self.spans: list[tuple[str, str]] = []
        self.updates: list[dict[str, Any]] = []

    @asynccontextmanager
    async def observe(
        self, name: str, *, as_type: SpanType = "span", **attrs: Any
    ) -> AsyncIterator[Any]:
        self.spans.append((name, as_type))
        recorder = self

        class _Handle:
            def update(self, **kwargs: Any) -> None:
                recorder.updates.append({"span": name, **kwargs})

        yield _Handle()

    @property
    def span_names(self) -> list[str]:
        return [name for name, _ in self.spans]


class LangfuseTracer:
    """Thin wrapper over the Langfuse v4 client."""

    enabled = True

    def __init__(self, client: Any) -> None:
        self._client = client

    @asynccontextmanager
    async def observe(
        self, name: str, *, as_type: SpanType = "span", **attrs: Any
    ) -> AsyncIterator[Any]:
        with self._client.start_as_current_observation(
            name=name, as_type=as_type, **attrs
        ) as span:
            yield span

    async def flush(self) -> None:
        # Fargate SIGTERM kills the process before the exporter drains, so this
        # is called from the FastAPI lifespan shutdown.
        self._client.flush()


_tracer: Tracer | None = None


def build_tracer(settings: Any = None) -> Tracer:
    """Real tracer when credentials exist, NoOp otherwise.

    Tests set LANGFUSE_ENABLED=false so no test can reach the network.
    """
    if os.getenv("LANGFUSE_ENABLED", "").lower() in {"false", "0", "no"}:
        return NoOpTracer()

    if settings is None:
        from core.config import get_settings

        settings = get_settings()

    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return NoOpTracer()

    try:
        from langfuse import Langfuse

        from observability.otel import build_tracer_provider

        # One provider for Langfuse and FastAPI instrumentation alike, built
        # with this service's resource. Langfuse attaches its exporter to it.
        provider = build_tracer_provider(settings.otel_service_name)
        client = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
            tracer_provider=provider,
        )
        return LangfuseTracer(client)
    except Exception:  # noqa: BLE001 - observability must never break the pipeline
        return NoOpTracer()


def get_tracer(settings: Any = None) -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = build_tracer(settings)
    return _tracer


def set_tracer(tracer: Tracer) -> None:
    """Injection point for tests."""
    global _tracer
    _tracer = tracer


def reset_tracer() -> None:
    global _tracer
    _tracer = None
