"""The fastembed backend is local, so it is tested for real, not faked.

First run downloads ~67 MB into the HF cache; subsequent runs are offline.
"""

from __future__ import annotations

import math

import pytest

from core.config import Settings
from observability.tracing import RecordingTracer
from retrieval.embedder import FastEmbedEmbedder, build_embedder


@pytest.fixture(scope="module")
def embedder() -> FastEmbedEmbedder:
    return FastEmbedEmbedder(
        model_name="BAAI/bge-small-en-v1.5", dim=384, tracer=RecordingTracer()
    )


async def test_query_vector_has_configured_dimension(embedder):
    vector = await embedder.embed_query("customer due diligence")
    assert len(vector) == embedder.dim == 384


async def test_vectors_are_unit_normalised_for_cosine(embedder):
    vector = await embedder.embed_query("interest rate on deposits")
    assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0, abs=1e-3)


async def test_document_batch_preserves_order_and_count(embedder):
    texts = ["know your customer", "capital adequacy ratio", "priority sector lending"]
    vectors = await embedder.embed_documents(texts)
    assert len(vectors) == 3
    assert all(len(v) == 384 for v in vectors)
    single = await embedder.embed_query(texts[1])
    assert vectors[1] == pytest.approx(single, abs=1e-5)


async def test_semantically_similar_texts_are_closer(embedder):
    """A sanity check that this is an embedding model, not a hash."""
    a = await embedder.embed_query("banks must verify the identity of customers")
    b = await embedder.embed_query("KYC requires customer identity verification")
    c = await embedder.embed_query("the weather in Mumbai is humid in September")
    dot = lambda x, y: sum(p * q for p, q in zip(x, y))  # noqa: E731
    assert dot(a, b) > dot(a, c)


async def test_embedding_calls_are_instrumented(embedder):
    tracer = embedder._tracer
    tracer.spans.clear()
    await embedder.embed_query("x")
    await embedder.embed_documents(["y"])
    assert ("embed.query", "embedding") in tracer.spans
    assert ("embed.documents", "embedding") in tracer.spans


def test_factory_selects_backend_from_settings():
    local = build_embedder(Settings(embedding_backend="fastembed"))
    assert isinstance(local, FastEmbedEmbedder)
    assert local.dim == 384
