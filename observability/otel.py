"""OpenTelemetry: one provider, shared with Langfuse.

Langfuse v4 is itself an OpenTelemetry SDK. Left to its own devices it creates
and installs a global TracerProvider; instrumenting FastAPI with a second one
produces two disconnected trace trees - the HTTP request in one place, the RAG
spans in another. So the provider is built here, once, with this service's
resource, installed globally, and handed to Langfuse, which attaches its
exporter to it. FastAPI's request spans then become the root of every trace
and the retrieval/generation spans nest underneath.

Health probes are excluded: a trace per liveness check is noise, not signal.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_provider: Any | None = None
EXCLUDED_URLS = "healthz,readyz,openapi.json,docs"


def build_tracer_provider(service_name: str, *, install_global: bool = True) -> Any:
    """Create (once) the provider Langfuse and FastAPI share."""
    global _provider
    if _provider is not None:
        return _provider

    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider

    _provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
    if install_global:
        trace.set_tracer_provider(_provider)
    return _provider


def get_tracer_provider() -> Any | None:
    return _provider


def reset_tracer_provider() -> None:
    """Test hook."""
    global _provider
    _provider = None


def instrument_app(app: Any) -> bool:
    """Wrap the FastAPI request lifecycle in spans on the shared provider.

    Safe to call when tracing is disabled: without an exporter the spans are
    no-ops. Idempotent - instrumenting twice is refused by the instrumentor,
    which is fine.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:  # pragma: no cover - optional at runtime
        logger.warning("opentelemetry-instrumentation-fastapi not installed; no HTTP spans")
        return False

    if getattr(app, "_is_instrumented_by_opentelemetry", False):
        return True
    FastAPIInstrumentor.instrument_app(
        app, tracer_provider=_provider, excluded_urls=EXCLUDED_URLS
    )
    return True
