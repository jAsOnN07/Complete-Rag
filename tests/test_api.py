"""API tests: real routing and validation, fake AWS, real in-memory Qdrant."""

from __future__ import annotations

from datetime import date

import pytest
from httpx import ASGITransport, AsyncClient
from qdrant_client import AsyncQdrantClient

from api.main import create_app
from core.models import DocumentMeta, Page, RawDocument
from core.service import RagService
from generation.prompt import NOT_FOUND_SENTINEL
from ingestion.chunker import RecursiveChunker
from observability.tracing import RecordingTracer
from retrieval.vector_store import QdrantVectorStore
from tests.fakes import ExplodingLlm, FakeEmbedder, FakeLlm

KYC_TEXT = (
    "Banks shall carry out customer due diligence and verify the identity of every "
    "customer at the time of commencement of an account based relationship. Know "
    "Your Customer records shall be updated periodically as prescribed."
)


async def build_store() -> QdrantVectorStore:
    embedder = FakeEmbedder()
    store = QdrantVectorStore(
        client=AsyncQdrantClient(location=":memory:"),
        collection="test_api",
        dim=embedder.dim,
        tracer=RecordingTracer(),
    )
    await store.ensure_collection()
    meta = DocumentMeta(
        doc_id="kyc1",
        source_path="data/raw/kyc1.pdf",
        title="Master Direction on KYC",
        regulator="RBI",
        circular_no="RBI/2026-27/201",
        issued_on=date(2026, 9, 8),
    )
    doc = RawDocument(meta=meta, pages=[Page.build(1, KYC_TEXT)])
    chunks = await RecursiveChunker(chunk_size=400, chunk_overlap=50).chunk(doc)
    vectors = await embedder.embed_documents([c.text for c in chunks])
    await store.upsert(chunks, vectors)
    return store


async def make_client(llm=None, *, threshold: float = 0.0) -> AsyncClient:
    store = await build_store()
    service = RagService(
        embedder=FakeEmbedder(),
        store=store,
        llm=llm or FakeLlm("Banks must verify identity [1]."),
        top_k=10,
        rerank_top_n=3,
        relevance_threshold=threshold,
        tracer=RecordingTracer(),
    )
    app = create_app(service=service)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture
async def client() -> AsyncClient:
    async with await make_client() as c:
        yield c


async def test_healthz_does_not_touch_the_datastore(client):
    response = await client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["config_fingerprint"]


async def test_readyz_reports_indexed_points(client):
    body = (await client.get("/readyz")).json()
    assert body["ready"] is True
    assert body["points"] > 0


async def test_query_returns_answer_and_citations(client):
    response = await client.post(
        "/query", json={"question": "What are the KYC due diligence requirements?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["citations"]
    assert body["citations"][0]["circular_no"] == "RBI/2026-27/201"
    assert body["latency_ms"] >= 0
    assert body["config_fingerprint"]


async def test_query_omits_retrieval_detail_unless_debug(client):
    body = (await client.post("/query", json={"question": "KYC rules"})).json()
    assert body["retrieval"] is None


async def test_debug_mode_exposes_retrieval(client):
    body = (
        await client.post("/query", json={"question": "KYC rules", "debug": True})
    ).json()
    assert body["retrieval"]
    assert body["retrieval"][0]["chunk_id"]
    assert body["retrieval"][0]["stage"] == "dense"


async def test_debug_mode_retrieves_only_once():
    """Debug must not double the embedding spend just to show its work."""
    store = await build_store()
    embedder = FakeEmbedder()
    service = RagService(
        embedder=embedder,
        store=store,
        llm=FakeLlm("Answer [1]."),
        top_k=10,
        rerank_top_n=3,
        relevance_threshold=0.0,
        tracer=RecordingTracer(),
    )
    app = create_app(service=service)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        await c.post("/query", json={"question": "KYC rules", "debug": True})
    assert len(embedder.calls) == 1


async def test_out_of_corpus_question_returns_200_not_found_not_an_error():
    """A valid question the corpus cannot answer is an answer, not an error."""
    async with await make_client(llm=ExplodingLlm(), threshold=1.01) as c:
        response = await c.post(
            "/query", json={"question": "What is the capital of France?"}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_found_in_context"
    assert body["citations"] == []


async def test_model_emitted_sentinel_maps_to_not_found_status():
    async with await make_client(llm=FakeLlm(NOT_FOUND_SENTINEL)) as c:
        body = (await c.post("/query", json={"question": "Unrelated"})).json()
    assert body["status"] == "not_found_in_context"


async def test_hallucinated_citations_are_surfaced_in_the_response():
    async with await make_client(llm=FakeLlm("Claim [1]. Other [42].")) as c:
        body = (await c.post("/query", json={"question": "KYC rules"})).json()
    assert body["invalid_citations"] == [42]
    assert body["invalid_citation_rate"] == pytest.approx(0.5)


async def test_usage_is_reported(client):
    body = (await client.post("/query", json={"question": "KYC rules"})).json()
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["provider"] == "fake"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": "hi"},
        {"question": "   "},
        {"question": "valid question", "top_k": 0},
        {"question": "valid question", "top_k": 999},
        {"question": "valid question", "unexpected": True},
    ],
)
async def test_invalid_requests_are_rejected(client, payload):
    assert (await client.post("/query", json=payload)).status_code == 422


async def test_top_k_is_honoured(client):
    body = (
        await client.post(
            "/query", json={"question": "KYC rules", "top_k": 1, "debug": True}
        )
    ).json()
    assert len(body["retrieval"]) == 1


async def test_openapi_schema_is_served(client):
    schema = (await client.get("/openapi.json")).json()
    assert "/query" in schema["paths"]
    assert "/healthz" in schema["paths"]


class FailingLlm:
    """Stands in for Bedrock refusing the call."""

    async def complete(self, system: str, user: str):
        from core.errors import UpstreamServiceError

        raise UpstreamServiceError(
            "bedrock", "converse", "ValidationException: Operation not allowed"
        )

    async def stream(self, system: str, user: str):
        raise NotImplementedError
        yield ""


async def test_upstream_failure_returns_503_with_actionable_detail():
    """A provider outage is not a bug in this service; 500 hides what broke."""
    async with await make_client(llm=FailingLlm()) as c:
        response = await c.post("/query", json={"question": "KYC requirements"})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "bedrock" in detail
    assert "Operation not allowed" in detail
