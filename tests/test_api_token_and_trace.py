"""Access token on the quota-spending routes; trace summary in debug responses."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from api.main import create_app
from core.config import Settings
from core.service import RagService
from observability.tracing import CollectingTracer, RecordingTracer
from tests.fakes import FakeEmbedder, FakeLlm
from tests.test_api import build_store


async def make_app(token: str | None = None, *, collecting: bool = False):
    tracer = CollectingTracer(RecordingTracer()) if collecting else RecordingTracer()
    service = RagService(
        embedder=FakeEmbedder(), store=await build_store(), llm=FakeLlm("Banks must verify identity [1]."),
        top_k=10, rerank_top_n=3, relevance_threshold=0.0, tracer=tracer,
    )
    app = create_app(service=service)
    app.state.settings = Settings(ui_access_token=SecretStr(token) if token else None)
    return app


@pytest.fixture
async def guarded() -> AsyncClient:
    app = await make_app("s3cret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_query_requires_the_token_when_configured(guarded):
    r = await guarded.post("/query", json={"question": "KYC rules"})
    assert r.status_code == 401
    r = await guarded.post("/query", json={"question": "KYC rules"}, headers={"X-Access-Token": "wrong"})
    assert r.status_code == 401
    r = await guarded.post("/query", json={"question": "KYC rules"}, headers={"X-Access-Token": "s3cret"})
    assert r.status_code == 200


async def test_token_is_accepted_as_a_query_parameter_for_event_source(guarded):
    r = await guarded.post("/query/stream?token=s3cret", json={"question": "KYC rules"})
    assert r.status_code == 200
    r = await guarded.post("/query/stream", json={"question": "KYC rules"})
    assert r.status_code == 401


async def test_health_probes_never_require_the_token(guarded):
    assert (await guarded.get("/healthz")).status_code == 200
    assert (await guarded.get("/readyz")).status_code == 200


async def test_no_token_configured_means_open():
    app = await make_app(None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        assert (await c.post("/query", json={"question": "KYC rules"})).status_code == 200


async def test_debug_response_carries_trace_and_full_chunk_text():
    app = await make_app(None, collecting=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        body = (await c.post("/query", json={"question": "KYC rules", "debug": True})).json()
    trace = body["trace"]
    names = [s["name"] for s in trace["spans"]]
    assert names, "debug responses describe their own request"
    assert any(n.startswith("rag.") or "llm" in n for n in names)
    assert trace["total_ms"] >= 0 and trace["total_cost_usd"] >= 0
    assert all("duration_ms" in s and "depth" in s for s in trace["spans"])
    chunk = body["retrieval"][0]
    assert chunk["text"].startswith(chunk["preview"]) and len(chunk["text"]) >= len(chunk["preview"])
    assert chunk["doc_id"] == "kyc1" and chunk["title"] == "Master Direction on KYC"


async def test_non_debug_response_has_no_trace():
    app = await make_app(None, collecting=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        body = (await c.post("/query", json={"question": "KYC rules"})).json()
    assert body["trace"] is None and body["retrieval"] is None
