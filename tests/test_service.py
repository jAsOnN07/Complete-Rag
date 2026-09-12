"""End-to-end pipeline tests: real chunking, real Qdrant (in-memory), fake AWS."""

from __future__ import annotations

from datetime import date

import pytest
from qdrant_client import AsyncQdrantClient

from core.models import DocumentMeta, Page, RawDocument
from core.service import RagService
from generation.prompt import NOT_FOUND_SENTINEL
from ingestion.chunker import RecursiveChunker
from observability.tracing import RecordingTracer
from retrieval.vector_store import QdrantVectorStore
from tests.fakes import ExplodingLlm, FakeEmbedder, FakeLlm

KYC_TEXT = (
    "Banks shall carry out customer due diligence and verify the identity of "
    "every customer at the time of commencement of an account based relationship. "
    "Know Your Customer records shall be updated periodically."
)
DEPOSIT_TEXT = (
    "Interest rate on deposits shall be uniform across all branches and shall not "
    "discriminate between depositors of the same tenor. Interest on savings deposits "
    "shall be calculated on daily balances."
)


def make_doc(doc_id: str, title: str, circular_no: str, text: str) -> RawDocument:
    meta = DocumentMeta(
        doc_id=doc_id,
        source_path=f"data/raw/{doc_id}.pdf",
        title=title,
        regulator="RBI",
        circular_no=circular_no,
        issued_on=date(2026, 9, 8),
    )
    return RawDocument(meta=meta, pages=[Page.build(1, text)])


@pytest.fixture
def tracer() -> RecordingTracer:
    return RecordingTracer()


@pytest.fixture
async def store(tracer) -> QdrantVectorStore:
    embedder = FakeEmbedder()
    store = QdrantVectorStore(
        client=AsyncQdrantClient(location=":memory:"),
        collection="test_circulars",
        dim=embedder.dim,
        tracer=tracer,
    )
    await store.ensure_collection()

    chunker = RecursiveChunker(chunk_size=400, chunk_overlap=50)
    for doc in (
        make_doc("kyc1", "Master Direction on KYC", "RBI/2026-27/201", KYC_TEXT),
        make_doc("dep1", "Interest Rate on Deposits", "RBI/2026-27/202", DEPOSIT_TEXT),
    ):
        chunks = await chunker.chunk(doc)
        vectors = await embedder.embed_documents([c.text for c in chunks])
        await store.upsert(chunks, vectors)
    return store


def build_service(store, tracer, llm, *, threshold: float = 0.0) -> RagService:
    return RagService(
        embedder=FakeEmbedder(),
        store=store,
        llm=llm,
        top_k=10,
        rerank_top_n=3,
        relevance_threshold=threshold,
        tracer=tracer,
    )


async def test_indexing_persists_points(store):
    assert await store.count() > 0


async def test_answer_returns_citations(store, tracer):
    llm = FakeLlm("Banks must verify customer identity [1].")
    answer = await build_service(store, tracer, llm).answer(
        "What are the KYC due diligence requirements?"
    )
    assert answer.not_found is False
    assert answer.citations
    assert answer.citations[0].circular_no.startswith("RBI/2026-27/")
    assert answer.invalid_citations == []


async def test_retrieval_ranks_the_topically_relevant_document_first(store, tracer):
    service = build_service(store, tracer, FakeLlm())
    results = await service.retrieve("customer due diligence identity verification")
    assert results
    assert "customer due diligence" in results[0].chunk.text.lower()


async def test_citations_resolve_to_real_indexed_chunks(store, tracer):
    llm = FakeLlm("Per the direction [1].")
    service = build_service(store, tracer, llm)
    answer = await service.answer("KYC requirements")
    retrieved = await service.retrieve("KYC requirements")
    assert answer.citations[0].chunk_id in {r.chunk.chunk_id for r in retrieved}


async def test_no_llm_call_when_nothing_clears_the_threshold(store, tracer):
    """The never-hallucinate rule: not-found short-circuits before any spend."""
    service = build_service(store, tracer, ExplodingLlm(), threshold=1.01)
    answer = await service.answer("What is the capital of France?")
    assert answer.not_found is True
    assert answer.citations == []
    assert answer.answer


async def test_model_emitting_the_sentinel_is_reported_as_not_found(store, tracer):
    llm = FakeLlm(NOT_FOUND_SENTINEL)
    answer = await build_service(store, tracer, llm).answer("Unrelated question")
    assert answer.not_found is True
    assert answer.citations == []


async def test_hallucinated_citation_labels_are_dropped_and_counted(store, tracer):
    llm = FakeLlm("Claim one [1]. Claim two [99].")
    answer = await build_service(store, tracer, llm).answer("KYC requirements")
    assert 99 in answer.invalid_citations
    assert all(c.chunk_id for c in answer.citations)
    assert answer.invalid_citation_rate == pytest.approx(0.5)


async def test_token_usage_is_carried_onto_the_answer(store, tracer):
    llm = FakeLlm("Answer [1].")
    answer = await build_service(store, tracer, llm).answer("KYC requirements")
    assert answer.input_tokens > 0
    assert answer.output_tokens > 0
    assert answer.model_id == "fake-model"


async def test_every_bedrock_and_qdrant_step_is_instrumented(store, tracer):
    """Turns 'do not skip Langfuse instrumentation' into a failing test."""
    tracer.spans.clear()
    llm = FakeLlm("Answer [1].")
    await build_service(store, tracer, llm).answer("KYC requirements")
    names = tracer.span_names
    assert "rag.answer" in names
    assert "qdrant.search" in names


async def test_search_span_is_typed_as_retriever(store, tracer):
    tracer.spans.clear()
    await build_service(store, tracer, FakeLlm()).retrieve("KYC")
    assert ("qdrant.search", "retriever") in tracer.spans
