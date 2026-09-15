"""Langfuse instrumentation.

Every Bedrock and Qdrant call is wrapped in a span from the first call ever
made, not retrofitted later. Three implementations share one interface:

* ``LangfuseTracer``  - the real thing, used when credentials are configured.
* ``NoOpTracer``      - used in tests and when Langfuse is disabled; touches no network.
* ``RecordingTracer`` - captures span names so a test can assert instrumentation exists.
* ``CollectingTracer``- wraps any of the above and keeps a per-request record of
                        spans (duration, tokens, cost, provider) so the API can
                        return a trace summary with the answer. Same spans, same
                        attributes as Langfuse sees - one truth, two consumers.

Langfuse v4 is itself an OpenTelemetry SDK, so nothing here creates a second
TracerProvider - doing that produces two disconnected trace trees.
"""

from __future__ import annotations

import contextvars
import os
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Literal, Protocol

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

    @property
    def client(self) -> Any:
        return self._client

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


@dataclass
class SpanRecord:
    name: str
    type: str
    depth: int
    started_ms: float
    duration_ms: float = 0.0
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    provider: str | None = None
    model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def absorb(self, attrs: dict[str, Any]) -> None:
        usage = attrs.get("usage_details") or {}
        if "input" in usage:
            self.tokens_in = int(usage["input"])
        if "output" in usage:
            self.tokens_out = int(usage["output"])
        cost = attrs.get("cost_details") or {}
        if cost:
            self.cost_usd = float(cost.get("total", sum(v for v in cost.values() if isinstance(v, (int, float)))))
        meta = attrs.get("metadata") or {}
        self.provider = meta.get("provider", self.provider)
        self.model = attrs.get("model") or meta.get("model") or self.model
        out = attrs.get("output")
        if isinstance(out, dict):
            self.extra.update({k: v for k, v in out.items() if isinstance(v, (str, int, float, bool)) or v is None})
        for k in ("candidates", "scale", "stop_reason"):
            if k in meta:
                self.extra[k] = meta[k]


@dataclass
class RequestScope:
    spans: list[SpanRecord] = field(default_factory=list)
    trace_id: str | None = None
    started: float = field(default_factory=time.perf_counter)
    _depth: int = 0


_scope: contextvars.ContextVar[RequestScope | None] = contextvars.ContextVar("trace_scope", default=None)


@contextmanager
def request_scope() -> Iterator[RequestScope]:
    """Collect spans for one request. Context-local, so concurrent requests
    on the same service never see each other's spans."""
    scope = RequestScope()
    token = _scope.set(scope)
    try:
        yield scope
    finally:
        _scope.reset(token)


def current_spans() -> list[SpanRecord] | None:
    scope = _scope.get()
    return scope.spans if scope else None


class CollectingTracer:
    """Delegates every span to the wrapped tracer and records it for the request."""

    def __init__(self, inner: Tracer) -> None:
        self._inner = inner

    @property
    def enabled(self) -> bool:
        return self._inner.enabled

    @property
    def inner(self) -> Tracer:
        return self._inner

    @asynccontextmanager
    async def observe(
        self, name: str, *, as_type: SpanType = "span", **attrs: Any
    ) -> AsyncIterator[Any]:
        scope = _scope.get()
        async with self._inner.observe(name, as_type=as_type, **attrs) as inner_handle:
            if scope is None:
                yield inner_handle
                return
            record = SpanRecord(
                name=name, type=as_type, depth=scope._depth,
                started_ms=round((time.perf_counter() - scope.started) * 1000, 2),
                model=attrs.get("model"),
            )
            record.absorb(attrs)
            scope.spans.append(record)
            if scope.trace_id is None:
                scope.trace_id = self._trace_id()
            scope._depth += 1
            t0 = time.perf_counter()

            class _Handle:
                def update(self, **kwargs: Any) -> None:
                    record.absorb(kwargs)
                    inner_handle.update(**kwargs)

            try:
                yield _Handle()
            finally:
                scope._depth -= 1
                record.duration_ms = round((time.perf_counter() - t0) * 1000, 2)

    def _trace_id(self) -> str | None:
        client = getattr(self._inner, "client", None)
        if client is None:
            return None
        try:
            return client.get_current_trace_id()
        except Exception:  # noqa: BLE001 - observability must never break the pipeline
            return None

    def trace_url(self, trace_id: str | None) -> str | None:
        client = getattr(self._inner, "client", None)
        if client is None or not trace_id:
            return None
        try:
            return client.get_trace_url(trace_id=trace_id)
        except Exception:  # noqa: BLE001
            return None

    async def flush(self) -> None:
        await self._inner.flush()


_tracer: Tracer | None = None


def build_tracer(settings: Any = None) -> Tracer:
    """Real tracer when credentials exist, NoOp otherwise.

    Tests set LANGFUSE_ENABLED=false so no test can reach the network.
    """
    return CollectingTracer(_build_inner(settings))


def _build_inner(settings: Any = None) -> Tracer:
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
