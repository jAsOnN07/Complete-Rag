"""Reranker backends. The cross-encoder is local, so it is tested for real."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.config import Settings
from core.errors import UpstreamServiceError
from core.models import Chunk, ChunkStrategy, DocumentMeta, ScoredChunk, make_chunk_id
from observability.tracing import RecordingTracer
from retrieval.reranker import (
    BedrockReranker,
    CrossEncoderReranker,
    NoopReranker,
    build_reranker,
)


def sc(doc: str, text: str, rank: int) -> ScoredChunk:
    meta = DocumentMeta(doc_id=doc, source_path="x", title=doc, regulator="RBI")
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=make_chunk_id(doc, ChunkStrategy.RECURSIVE, rank), doc_id=doc, text=text,
            ordinal=rank, strategy=ChunkStrategy.RECURSIVE, meta=meta,
        ),
        score=1.0 / (rank + 1), rank=rank, stage="fused",
    )


CANDIDATES = [
    sc("deposits", "Interest rate on deposits shall be uniform across all branches.", 0),
    sc("ladakh", "five new districts, viz., Sham, Nubra, Changthang, Zanskar and Drass in the UT of Ladakh", 1),
    sc("fema", "seven circulars as listed at Annex are being withdrawn", 2),
]


@pytest.fixture(scope="module")
def cross_encoder() -> CrossEncoderReranker:
    return CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", tracer=RecordingTracer()
    )


async def test_cross_encoder_promotes_the_relevant_candidate(cross_encoder):
    out = await cross_encoder.rerank("Which districts were formed in Ladakh?", CANDIDATES, top_n=3)
    assert out[0].chunk.doc_id == "ladakh"
    assert [o.rank for o in out] == [0, 1, 2]
    assert all(o.stage == "reranked" for o in out)


async def test_cross_encoder_scores_are_logits_that_separate_relevance(cross_encoder):
    """The whole point: in-domain-but-irrelevant lands well below zero."""
    out = await cross_encoder.rerank("Which districts were formed in Ladakh?", CANDIDATES, top_n=3)
    by_doc = {o.chunk.doc_id: o.score for o in out}
    assert by_doc["ladakh"] > 0
    assert by_doc["deposits"] < 0
    assert by_doc["ladakh"] - by_doc["deposits"] > 5


async def test_cross_encoder_truncates_to_top_n(cross_encoder):
    out = await cross_encoder.rerank("Ladakh districts", CANDIDATES, top_n=1)
    assert len(out) == 1


async def test_cross_encoder_handles_empty_candidates(cross_encoder):
    assert await cross_encoder.rerank("anything", [], top_n=5) == []


async def test_cross_encoder_is_instrumented(cross_encoder):
    tracer = cross_encoder._tracer
    tracer.spans.clear()
    await cross_encoder.rerank("q", CANDIDATES, top_n=2)
    assert ("rerank.cross_encoder", "span") in tracer.spans


def test_cross_encoder_scale_is_logit(cross_encoder):
    assert cross_encoder.backend == "cross_encoder"
    assert cross_encoder.score_scale == "logit"


# ---- noop ------------------------------------------------------------------


async def test_noop_preserves_order_and_scores():
    out = await NoopReranker().rerank("q", CANDIDATES, top_n=2)
    assert [o.chunk.doc_id for o in out] == ["deposits", "ladakh"]
    assert [o.score for o in out] == [c.score for c in CANDIDATES[:2]]


# ---- bedrock (shape recorded from the service model; no network) -----------


class FakeBedrockAgentRuntime:
    def __init__(self, results=None, error: Exception | None = None):
        self._results = results or []
        self._error = error
        self.calls: list[dict] = []

    def rerank(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return {"results": self._results}


async def test_bedrock_reranker_maps_results_by_index():
    fake = FakeBedrockAgentRuntime(
        results=[{"index": 1, "relevanceScore": 0.91}, {"index": 2, "relevanceScore": 0.4}]
    )
    rr = BedrockReranker(client=fake, model_arn="arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0", tracer=RecordingTracer())
    out = await rr.rerank("Ladakh districts", CANDIDATES, top_n=2)
    assert [o.chunk.doc_id for o in out] == ["ladakh", "fema"]
    assert out[0].score == pytest.approx(0.91)
    assert rr.score_scale == "unit"
    call = fake.calls[0]
    assert call["queries"][0]["textQuery"]["text"] == "Ladakh districts"
    assert call["rerankingConfiguration"]["bedrockRerankingConfiguration"]["numberOfResults"] == 2
    assert len(call["sources"]) == 3
    assert call["sources"][0]["inlineDocumentSource"]["textDocument"]["text"].startswith("Interest rate")


async def test_bedrock_reranker_failure_is_typed():
    fake = FakeBedrockAgentRuntime(error=RuntimeError("ValidationException: Operation not allowed"))
    rr = BedrockReranker(client=fake, model_arn="arn", tracer=RecordingTracer())
    with pytest.raises(UpstreamServiceError) as exc:
        await rr.rerank("q", CANDIDATES, top_n=2)
    assert exc.value.provider == "bedrock"


# ---- factory ---------------------------------------------------------------


def test_factory_selects_backend():
    assert isinstance(build_reranker(Settings(reranker_backend="none")), NoopReranker)
    assert isinstance(build_reranker(Settings(reranker_backend="cross_encoder")), CrossEncoderReranker)
