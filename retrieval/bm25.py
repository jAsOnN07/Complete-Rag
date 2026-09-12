"""BM25 term-frequency encoding for Qdrant sparse vectors.

There is no index here. Qdrant is the index: with
``SparseVectorParams(modifier=Modifier.IDF)`` it computes IDF server-side from
collection statistics, so the client only produces term frequencies. That
keeps "never store embeddings in memory" true, survives restarts, and stays
consistent across replicas - none of which an in-process rank_bm25 index does.

Consequence worth knowing: upserts and deletes shift the IDF statistics, so
BM25 scores are not stable across ingestion runs. Always re-run eval after
re-ingesting.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

from pydantic import BaseModel

from observability.tracing import Tracer, get_tracer

DEFAULT_BM25_MODEL = "Qdrant/bm25"


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]

    @property
    def nnz(self) -> int:
        return len(self.indices)


class Bm25Encoder:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_BM25_MODEL,
        tracer: Tracer | None = None,
        cache_dir: str | None = None,
    ) -> None:
        from fastembed import SparseTextEmbedding

        self._model_name = model_name
        self._tracer = tracer or get_tracer()
        self._model = SparseTextEmbedding(model_name=model_name, cache_dir=cache_dir)

    @property
    def model_id(self) -> str:
        return self._model_name

    def _encode_docs_sync(self, texts: Sequence[str]) -> list[SparseVector]:
        return [
            SparseVector(indices=v.indices.tolist(), values=v.values.tolist())
            for v in self._model.embed(list(texts))
        ]

    def _encode_query_sync(self, text: str) -> SparseVector:
        v = next(self._model.query_embed(text))
        return SparseVector(indices=v.indices.tolist(), values=v.values.tolist())

    async def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        async with self._tracer.observe(
            "bm25.encode_documents", input={"count": len(texts)}
        ) as span:
            vectors = await asyncio.to_thread(self._encode_docs_sync, texts)
            span.update(output={"count": len(vectors)}, metadata={"model": self._model_name})
            return vectors

    async def encode_query(self, text: str) -> SparseVector:
        async with self._tracer.observe(
            "bm25.encode_query", input={"chars": len(text)}
        ) as span:
            vector = await asyncio.to_thread(self._encode_query_sync, text)
            span.update(output={"terms": vector.nnz})
            return vector
