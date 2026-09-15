"""One provider, shared: HTTP request spans must be the root of RAG traces."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from api.main import create_app
from core.service import RagService
from observability import otel
from observability.pricing import embed_cost, llm_cost, rerank_cost
from observability.tracing import RecordingTracer
from tests.fakes import FakeEmbedder, FakeLlm
from tests.test_api import build_store


# ---- pricing (pure) ---------------------------------------------------------------


def test_llm_cost_matches_list_price():
    c = llm_cost("gemini-3.5-flash", 1_000_000, 100_000)
    assert c == {"input": 1.5, "output": 0.9, "total": 2.4}


def test_llm_cost_matches_on_provider_prefixed_names():
    assert llm_cost("openai/gpt-oss-120b", 1_000_000, 0)["input"] == 0.15
    assert llm_cost("gpt-oss-120b", 1_000_000, 0)["input"] == 0.15


def test_unknown_models_cost_none_rather_than_zero():
    """None keeps the gap visible in Langfuse; 0 would silently claim it is free."""
    assert llm_cost("some-model", 10, 10) is None
    assert embed_cost("mystery", 10) is None
    assert rerank_cost(None, 1) is None


def test_embed_and_rerank_costs():
    assert embed_cost("embed-v4.0", 2_000_000) == {"input": 0.2, "total": 0.2}
    assert rerank_cost("rerank-v3.5", 500) == {"total": 1.0}


# ---- provider sharing --------------------------------------------------------------


@pytest.fixture
def fresh_provider(monkeypatch):
    otel.reset_tracer_provider()
    provider = otel.build_tracer_provider("rag-test", install_global=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter
    otel.reset_tracer_provider()


def test_provider_is_built_once():
    otel.reset_tracer_provider()
    a = otel.build_tracer_provider("rag-test", install_global=False)
    b = otel.build_tracer_provider("other-name", install_global=False)
    assert a is b
    assert a.resource.attributes["service.name"] == "rag-test"
    otel.reset_tracer_provider()


async def test_http_request_span_is_emitted_on_the_shared_provider(fresh_provider):
    provider, exporter = fresh_provider
    store = await build_store()
    service = RagService(
        embedder=FakeEmbedder(), store=store, llm=FakeLlm("Answer [1]."),
        top_k=10, rerank_top_n=3, relevance_threshold=0.0, tracer=RecordingTracer(),
    )
    app = create_app(service=service)
    assert otel.instrument_app(app) is True

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        assert (await c.post("/query", json={"question": "KYC rules"})).status_code == 200
        assert (await c.get("/healthz")).status_code == 200

    names = [s.name for s in exporter.get_finished_spans()]
    assert any("/query" in n for n in names), names
    assert not any("healthz" in n for n in names), "health probes must not produce traces"


def test_instrument_app_is_idempotent(fresh_provider):
    app = create_app(service=object())
    assert otel.instrument_app(app) is True
    assert otel.instrument_app(app) is True


def test_tracer_installs_no_second_global_provider(monkeypatch):
    """Langfuse must receive the shared provider, never build its own."""
    otel.reset_tracer_provider()
    captured = {}

    class FakeLangfuse:
        def __init__(self, **kw):
            captured.update(kw)

        def flush(self):
            pass

    import langfuse as lf_mod
    monkeypatch.setattr(lf_mod, "Langfuse", FakeLangfuse)
    monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)

    from core.config import Settings
    from observability.tracing import build_tracer

    tracer = build_tracer(Settings(langfuse_public_key="pk", langfuse_secret_key="sk", otel_service_name="rag-x"))
    assert tracer.enabled
    assert captured["tracer_provider"] is otel.get_tracer_provider()
    assert trace.get_tracer_provider() is otel.get_tracer_provider()
    otel.reset_tracer_provider()
