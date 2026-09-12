"""Hybrid retrieval: pure RRF, and server-side RRF against real in-memory Qdrant."""

from __future__ import annotations

from datetime import date

import pytest
from qdrant_client import AsyncQdrantClient

from core.models import Chunk, ChunkStrategy, DocumentMeta, ScoredChunk, make_chunk_id
from observability.tracing import RecordingTracer
from retrieval.bm25 import Bm25Encoder
from retrieval.hybrid import HybridRetriever, rrf_fuse
from retrieval.vector_store import QdrantVectorStore
from tests.fakes import FakeEmbedder


def sc(doc: str, ordinal: int, score: float, rank: int, stage: str = "dense") -> ScoredChunk:
    meta = DocumentMeta(doc_id=doc, source_path="x", title=doc, regulator="RBI")
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=make_chunk_id(doc, ChunkStrategy.RECURSIVE, ordinal), doc_id=doc,
            text=f"{doc} {ordinal}", ordinal=ordinal, strategy=ChunkStrategy.RECURSIVE, meta=meta,
        ),
        score=score, rank=rank, stage=stage,
    )


# ---- pure RRF --------------------------------------------------------------


def test_rrf_scores_follow_the_formula():
    a = [sc("d", 0, 0.9, 0)]
    assert rrf_fuse([a], k=60)[0].score == pytest.approx(1 / 61)
    assert rrf_fuse([a])[0].score == pytest.approx(1 / 2), "default k matches Qdrant (k=1)"
    assert rrf_fuse([a])[0].stage == "fused"


def test_rrf_default_reproduces_qdrant_scale():
    """Measured against Qdrant: top in both = 1.0, second in both = 0.667, lone third = 0.25."""
    top = sc("a", 0, 1.0, 0)
    second = sc("b", 0, 1.0, 1)
    third = sc("c", 0, 1.0, 2)
    fused = {f.chunk.chunk_id: f.score for f in rrf_fuse([[top, second, third], [top, second]])}
    assert fused[top.chunk.chunk_id] == pytest.approx(1.0)
    assert fused[second.chunk.chunk_id] == pytest.approx(2 / 3)
    assert fused[third.chunk.chunk_id] == pytest.approx(0.25)


def test_item_in_both_lists_outranks_item_in_one():
    both = sc("d", 0, 0.5, 1)
    only_dense = sc("d", 1, 0.9, 0)
    only_sparse = sc("d", 2, 9.0, 0)
    fused = rrf_fuse([[only_dense, both], [only_sparse, both]])
    assert fused[0].chunk.chunk_id == both.chunk.chunk_id


def test_rrf_limit_and_rank_assignment():
    fused = rrf_fuse([[sc("d", i, 1.0, i) for i in range(5)]], limit=3)
    assert [f.rank for f in fused] == [0, 1, 2]
    assert len(fused) == 3


def test_rrf_is_deterministic_on_ties():
    a = [sc("a", 0, 1.0, 0)]
    b = [sc("b", 0, 1.0, 0)]
    assert [f.chunk.chunk_id for f in rrf_fuse([a, b])] == [
        f.chunk.chunk_id for f in rrf_fuse([a, b])
    ]


# ---- real sparse + server-side fusion --------------------------------------

DOCS = {
    "uapa219": "Implementation of Section 51A of UAPA. DOR.AML.REC.219/14.06.001/2026-27 amends two entries.",
    "uapa218": "Implementation of Section 51A of UAPA. DOR.AML.REC.218/14.06.001/2026-27 amends one entry.",
    "uapa217": "Implementation of Section 51A of UAPA. DOR.AML.REC.217/14.06.001/2026-27 amends three entries.",
    "deposits": "Interest rate on deposits shall be uniform across branches for the same tenor.",
}


@pytest.fixture(scope="module")
def bm25() -> Bm25Encoder:
    return Bm25Encoder(tracer=RecordingTracer())


@pytest.fixture
async def store(bm25) -> QdrantVectorStore:
    embedder = FakeEmbedder()
    store = QdrantVectorStore(
        client=AsyncQdrantClient(location=":memory:"), collection="hybrid_test",
        dim=embedder.dim, tracer=RecordingTracer(),
    )
    await store.ensure_collection()
    chunks = []
    for doc, text in DOCS.items():
        meta = DocumentMeta(doc_id=doc, source_path="x", title=doc, regulator="RBI")
        chunks.append(Chunk(
            chunk_id=make_chunk_id(doc, ChunkStrategy.RECURSIVE, 0), doc_id=doc, text=text,
            ordinal=0, strategy=ChunkStrategy.RECURSIVE, meta=meta,
        ))
    dense = await embedder.embed_documents([c.text for c in chunks])
    sparse = await bm25.encode_documents([c.text for c in chunks])
    await store.upsert(chunks, dense, sparse)
    return store


async def test_sparse_search_discriminates_on_exact_identifier(store, bm25):
    """The q007 case: three near-identical circulars differing by one token."""
    hits = await store.search_sparse(await bm25.encode_query("DOR.AML.REC.219/14.06.001/2026-27"), k=3)
    assert hits[0].chunk.doc_id == "uapa219"
    assert hits[0].stage == "sparse"


async def test_hybrid_ranks_exact_identifier_first(store, bm25):
    retriever = HybridRetriever(embedder=FakeEmbedder(), sparse=bm25, store=store, top_k=4)
    hits = await retriever.retrieve("which circular is DOR.AML.REC.219/14.06.001/2026-27")
    assert hits[0].chunk.doc_id == "uapa219"
    assert all(h.stage == "fused" for h in hits)


async def test_server_side_rrf_matches_client_side_rrf(store, bm25):
    """Congruence: Qdrant's fusion and our pure rrf_fuse must agree on ordering."""
    embedder = FakeEmbedder()
    q = "UAPA entries amended 219"
    dense_vec = await embedder.embed_query(q)
    sparse_vec = await bm25.encode_query(q)
    dense_hits = await store.search_dense(dense_vec, k=10)
    sparse_hits = await store.search_sparse(sparse_vec, k=10)
    client_scores = {f.chunk.chunk_id: f.score for f in rrf_fuse([dense_hits, sparse_hits])}
    server_side = await store.search_hybrid(dense_vec, sparse_vec, k=4, prefetch_k=10)

    # Ties (near-identical docs) may be broken differently by the server, so
    # the invariant is "server order never contradicts client scores", not
    # "identical sequence". Top-1 must still agree exactly.
    scores_in_server_order = [client_scores[f.chunk.chunk_id] for f in server_side]
    assert scores_in_server_order == sorted(scores_in_server_order, reverse=True)
    assert server_side[0].chunk.chunk_id == max(client_scores, key=client_scores.get)
    for f in server_side:
        assert f.score == pytest.approx(client_scores[f.chunk.chunk_id], rel=1e-6)


async def test_hybrid_search_is_instrumented_as_a_retriever(store, bm25):
    tracer = store._tracer
    tracer.spans.clear()
    await store.search_hybrid(await FakeEmbedder().embed_query("x"), await bm25.encode_query("x"), k=2)
    assert ("qdrant.search_hybrid", "retriever") in tracer.spans


async def test_upsert_rejects_mismatched_sparse_count(store, bm25):
    meta = DocumentMeta(doc_id="z", source_path="x", title="z", regulator="RBI")
    chunk = Chunk(chunk_id="z::recursive::0", doc_id="z", text="t", ordinal=0,
                  strategy=ChunkStrategy.RECURSIVE, meta=meta)
    with pytest.raises(ValueError):
        await store.upsert([chunk], [[0.0] * 64], sparse=[])


async def test_count_is_zero_for_a_missing_collection():
    store = QdrantVectorStore(
        client=AsyncQdrantClient(location=":memory:"), collection="never_created",
        dim=8, tracer=RecordingTracer(),
    )
    assert await store.count() == 0
