"""Cohere embed + rerank against the SDK v2 response shapes. No network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.config import Settings
from core.errors import UpstreamServiceError
from core.models import Chunk, ChunkStrategy, DocumentMeta, ScoredChunk, make_chunk_id
from observability.tracing import RecordingTracer
from retrieval.embedder import CohereEmbedder, build_embedder
from retrieval.reranker import CohereReranker, build_reranker


class FakeCohere:
    def __init__(self, dim: int = 4, scores: list[float] | None = None, error: Exception | None = None):
        self.dim = dim
        self.scores = scores
        self.error = error
        self.embed_calls: list[dict] = []
        self.rerank_calls: list[dict] = []

    async def embed(self, **kw):
        self.embed_calls.append(kw)
        if self.error:
            raise self.error
        vecs = [[float(i + 1)] * self.dim for i in range(len(kw["texts"]))]
        return SimpleNamespace(embeddings=SimpleNamespace(float_=vecs))

    async def rerank(self, **kw):
        self.rerank_calls.append(kw)
        if self.error:
            raise self.error
        scores = self.scores or [0.9, 0.05, 0.5][: len(kw["documents"])]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: kw["top_n"]]
        return SimpleNamespace(results=[SimpleNamespace(index=i, relevance_score=scores[i]) for i in order])


# ---- embed --------------------------------------------------------------------


async def test_documents_use_search_document_and_queries_use_search_query():
    fake = FakeCohere()
    e = CohereEmbedder(client=fake, model_id="embed-v4.0", dim=4, tracer=RecordingTracer())
    await e.embed_documents(["a", "b"])
    await e.embed_query("q")
    assert fake.embed_calls[0]["input_type"] == "search_document"
    assert fake.embed_calls[1]["input_type"] == "search_query"
    assert fake.embed_calls[0]["output_dimension"] == 4
    assert fake.embed_calls[0]["embedding_types"] == ["float"]


async def test_documents_are_batched_at_96_and_order_preserved():
    fake = FakeCohere()
    e = CohereEmbedder(client=fake, model_id="m", dim=4, tracer=RecordingTracer())
    vectors = await e.embed_documents([f"t{i}" for i in range(200)])
    assert [len(c["texts"]) for c in fake.embed_calls] == [96, 96, 8]
    assert len(vectors) == 200 and len(vectors[0]) == 4


async def test_embed_failure_is_typed():
    e = CohereEmbedder(client=FakeCohere(error=RuntimeError("429")), model_id="m", dim=4, tracer=RecordingTracer())
    with pytest.raises(UpstreamServiceError) as exc:
        await e.embed_query("q")
    assert exc.value.provider == "cohere"


async def test_embed_is_instrumented():
    e = CohereEmbedder(client=FakeCohere(), model_id="m", dim=4, tracer=RecordingTracer())
    await e.embed_query("q")
    assert ("embed.query", "embedding") in e._tracer.spans


# ---- rerank -------------------------------------------------------------------


def sc(doc: str, text: str, rank: int) -> ScoredChunk:
    meta = DocumentMeta(doc_id=doc, source_path="x", title=doc, regulator="RBI")
    return ScoredChunk(
        chunk=Chunk(chunk_id=make_chunk_id(doc, ChunkStrategy.FIXED, rank), doc_id=doc, text=text,
                    ordinal=rank, strategy=ChunkStrategy.FIXED, meta=meta),
        score=0.5, rank=rank, stage="fused",
    )


CANDS = [sc("a", "deposits", 0), sc("b", "ladakh districts", 1), sc("c", "fema review", 2)]


async def test_rerank_orders_by_relevance_and_reports_unit_scores():
    r = CohereReranker(client=FakeCohere(scores=[0.05, 0.9, 0.5]), model_id="rerank-v3.5", tracer=RecordingTracer())
    out = await r.rerank("ladakh", CANDS, top_n=2)
    assert [o.chunk.doc_id for o in out] == ["b", "c"]
    assert out[0].score == pytest.approx(0.9) and out[0].stage == "reranked"
    assert r.score_scale == "unit"


async def test_rerank_sends_retrieval_text_and_caps_top_n():
    fake = FakeCohere()
    await CohereReranker(client=fake, model_id="m", tracer=RecordingTracer()).rerank("q", CANDS, top_n=10)
    assert fake.rerank_calls[0]["documents"] == ["deposits", "ladakh districts", "fema review"]
    assert fake.rerank_calls[0]["top_n"] == 3


async def test_rerank_failure_is_typed():
    r = CohereReranker(client=FakeCohere(error=RuntimeError("429")), model_id="m", tracer=RecordingTracer())
    with pytest.raises(UpstreamServiceError):
        await r.rerank("q", CANDS, top_n=2)


# ---- config / factories ---------------------------------------------------------


def test_cohere_collection_is_distinct_from_bge_and_titan():
    names = {b: Settings(embedding_backend=b).collection_name() for b in ("cohere", "fastembed", "bedrock")}
    assert len(set(names.values())) == 3
    assert "embed-v4-0-1024" in names["cohere"]


def test_factories_require_a_key():
    with pytest.raises(RuntimeError):
        build_embedder(Settings(embedding_backend="cohere", cohere_api_key=None))
    with pytest.raises(RuntimeError):
        build_reranker(Settings(reranker_backend="cohere", cohere_api_key=None))


def test_cohere_threshold_is_on_the_unit_scale():
    s = Settings(reranker_backend="cohere")
    assert 0.0 < s.threshold_for("cohere") < 1.0
    assert s.final_score_scale(reranker_active=True) == "cohere"
