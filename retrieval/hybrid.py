"""Hybrid retrieval: dense + BM25 fused by Reciprocal Rank Fusion.

Fusion happens on the Qdrant server (one round trip). ``rrf_fuse`` is kept as
a pure function for two reasons: it is trivially unit-testable, and a
congruence test - server fusion and client fusion produce the same ordering on
a fixture - is a real correctness check on the server-side path.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from core.models import ScoredChunk
from core.ports import Embedder
from retrieval.bm25 import SparseVector

# Qdrant's Fusion.RRF is textbook 1/(k + rank) with 1-based ranks and k=1, not
# the usual k=60 (measured: a top hit in both lists scores 1.0, second in both
# 0.667, a lone third 0.25). The steep curve rewards agreement at the top far
# more than k=60 would. Kept in sync so the congruence test asserts exact scores.
DEFAULT_RRF_K = 1


@runtime_checkable
class SparseEncoder(Protocol):
    async def encode_query(self, text: str) -> SparseVector: ...

    async def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...


@runtime_checkable
class HybridStore(Protocol):
    async def search_dense(self, vector: Sequence[float], k: int) -> list[ScoredChunk]: ...

    async def search_hybrid(
        self, dense: Sequence[float], sparse: SparseVector, k: int, *, prefetch_k: int | None = None
    ) -> list[ScoredChunk]: ...


def rrf_fuse(
    rankings: Sequence[Sequence[ScoredChunk]], *, k: int = DEFAULT_RRF_K, limit: int | None = None
) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank)."""
    fused: dict[str, float] = {}
    keep: dict[str, ScoredChunk] = {}
    for ranking in rankings:
        for rank, scored in enumerate(ranking):
            cid = scored.chunk.chunk_id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
            keep.setdefault(cid, scored)
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    if limit is not None:
        ordered = ordered[:limit]
    return [
        ScoredChunk(chunk=keep[cid].chunk, score=score, rank=i, stage="fused")
        for i, (cid, score) in enumerate(ordered)
    ]


class DenseRetriever:
    def __init__(self, *, embedder: Embedder, store: Any, top_k: int) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k

    @property
    def mode(self) -> str:
        return "dense"

    async def retrieve(self, question: str, *, k: int | None = None) -> list[ScoredChunk]:
        vector = await self._embedder.embed_query(question)
        return await self._store.search_dense(vector, k or self._top_k)


class HybridRetriever:
    def __init__(
        self,
        *,
        embedder: Embedder,
        sparse: SparseEncoder,
        store: HybridStore,
        top_k: int,
        prefetch_k: int | None = None,
    ) -> None:
        self._embedder = embedder
        self._sparse = sparse
        self._store = store
        self._top_k = top_k
        self._prefetch_k = prefetch_k

    @property
    def mode(self) -> str:
        return "hybrid"

    async def retrieve(self, question: str, *, k: int | None = None) -> list[ScoredChunk]:
        dense = await self._embedder.embed_query(question)
        sparse = await self._sparse.encode_query(question)
        return await self._store.search_hybrid(
            dense, sparse, k or self._top_k, prefetch_k=self._prefetch_k
        )
