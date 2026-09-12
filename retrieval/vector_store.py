"""Qdrant collection and index operations.

The whole Chunk goes into the point payload, so retrieval reconstructs it with
`Chunk.model_validate(payload)` and needs no sidecar document store. At 45
documents the duplication costs nothing and removes an entire class of join bug.

Dense and sparse (BM25) vectors are named vectors on one collection. Sparse
uses ``Modifier.IDF`` so Qdrant computes IDF server-side; hybrid search is a
single Query API call with two prefetches fused by RRF on the server.
"""

from __future__ import annotations

import uuid
from typing import Any, Sequence

from qdrant_client import AsyncQdrantClient, models

from core.models import Chunk, ScoredChunk
from observability.tracing import Tracer, get_tracer
from retrieval.bm25 import SparseVector

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
QDRANT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def point_id_for(chunk_id: str) -> str:
    """Qdrant point IDs must be UUID or int, and chunk_id is neither."""
    return str(uuid.uuid5(QDRANT_NAMESPACE, chunk_id))


class QdrantVectorStore:
    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        collection: str,
        dim: int,
        distance: models.Distance = models.Distance.COSINE,
        tracer: Tracer | None = None,
    ) -> None:
        self._client = client
        self._collection = collection
        self._dim = dim
        self._distance = distance
        self._tracer = tracer or get_tracer()
        self._is_local = getattr(client, "_client", None).__class__.__name__ == "AsyncQdrantLocal"

    @property
    def collection(self) -> str:
        return self._collection

    async def ensure_collection(self, *, recreate: bool = False) -> None:
        async with self._tracer.observe(
            "qdrant.ensure_collection", input={"collection": self._collection}
        ) as span:
            exists = await self._client.collection_exists(self._collection)
            if exists and recreate:
                await self._client.delete_collection(self._collection)
                exists = False
            if not exists:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(
                            size=self._dim, distance=self._distance
                        )
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: models.SparseVectorParams(
                            modifier=models.Modifier.IDF
                        )
                    },
                )
                # Payload indexes are a no-op in Qdrant local mode, which warns
                # on every call; skip them there rather than spam the test run.
                if not self._is_local:
                    for field in ("doc_id", "meta.regulator", "meta.circular_no"):
                        await self._client.create_payload_index(
                            collection_name=self._collection,
                            field_name=field,
                            field_schema=models.PayloadSchemaType.KEYWORD,
                        )
            span.update(output={"created": not exists})

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        sparse: Sequence[SparseVector] | None = None,
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} vs {len(vectors)}"
            )
        if sparse is not None and len(sparse) != len(chunks):
            raise ValueError(
                f"chunk/sparse count mismatch: {len(chunks)} vs {len(sparse)}"
            )
        async with self._tracer.observe(
            "qdrant.upsert",
            input={"collection": self._collection, "points": len(chunks)},
        ) as span:
            points = []
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                named: dict[str, Any] = {DENSE_VECTOR_NAME: list(vector)}
                if sparse is not None:
                    named[SPARSE_VECTOR_NAME] = models.SparseVector(
                        indices=sparse[i].indices, values=sparse[i].values
                    )
                points.append(
                    models.PointStruct(
                        id=point_id_for(chunk.chunk_id),
                        vector=named,
                        payload=chunk.model_dump(mode="json"),
                    )
                )
            await self._client.upsert(
                collection_name=self._collection, points=points, wait=True
            )
            span.update(output={"upserted": len(points)})
            return len(points)

    async def search_dense(
        self, vector: Sequence[float], k: int
    ) -> list[ScoredChunk]:
        async with self._tracer.observe(
            "qdrant.search",
            as_type="retriever",
            input={"collection": self._collection, "k": k},
        ) as span:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=list(vector),
                using=DENSE_VECTOR_NAME,
                limit=k,
                with_payload=True,
            )
            scored = [
                ScoredChunk(
                    chunk=Chunk.model_validate(point.payload),
                    score=point.score,
                    rank=rank,
                    stage="dense",
                )
                for rank, point in enumerate(response.points)
            ]
            span.update(
                output={
                    "returned": len(scored),
                    "top_score": scored[0].score if scored else None,
                }
            )
            return scored

    async def search_hybrid(
        self,
        dense: Sequence[float],
        sparse: SparseVector,
        k: int,
        *,
        prefetch_k: int | None = None,
    ) -> list[ScoredChunk]:
        """One round trip: dense and sparse prefetches fused by RRF on the server.

        RRF, not a weighted sum - there is no magic constant to tune, and the
        two score scales (cosine vs BM25) never have to be reconciled.
        """
        prefetch_k = prefetch_k or max(k * 4, 20)
        async with self._tracer.observe(
            "qdrant.search_hybrid",
            as_type="retriever",
            input={"collection": self._collection, "k": k, "prefetch_k": prefetch_k},
        ) as span:
            response = await self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    models.Prefetch(
                        query=list(dense), using=DENSE_VECTOR_NAME, limit=prefetch_k
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse.indices, values=sparse.values
                        ),
                        using=SPARSE_VECTOR_NAME,
                        limit=prefetch_k,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k,
                with_payload=True,
            )
            scored = [
                ScoredChunk(
                    chunk=Chunk.model_validate(point.payload),
                    score=point.score,
                    rank=rank,
                    stage="fused",
                )
                for rank, point in enumerate(response.points)
            ]
            span.update(
                output={
                    "returned": len(scored),
                    "top_score": scored[0].score if scored else None,
                }
            )
            return scored

    async def search_sparse(self, sparse: SparseVector, k: int) -> list[ScoredChunk]:
        """Sparse-only search; exists for the client/server RRF congruence test."""
        response = await self._client.query_points(
            collection_name=self._collection,
            query=models.SparseVector(indices=sparse.indices, values=sparse.values),
            using=SPARSE_VECTOR_NAME,
            limit=k,
            with_payload=True,
        )
        return [
            ScoredChunk(
                chunk=Chunk.model_validate(p.payload), score=p.score, rank=i, stage="sparse"
            )
            for i, p in enumerate(response.points)
        ]

    async def count(self) -> int:
        """0 for a collection that does not exist yet, not a 404 - /readyz and
        the comparison runner both treat "nothing indexed" as a state."""
        if not await self._client.collection_exists(self._collection):
            return 0
        result = await self._client.count(
            collection_name=self._collection, exact=True
        )
        return result.count

    async def close(self) -> None:
        await self._client.close()


def build_qdrant_store(
    settings: Any = None, *, in_memory: bool = False
) -> QdrantVectorStore:
    if settings is None:
        from core.config import get_settings

        settings = get_settings()

    if in_memory:
        client = AsyncQdrantClient(location=":memory:")
    else:
        client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=(
                settings.qdrant_api_key.get_secret_value()
                if settings.qdrant_api_key
                else None
            ),
        )
    return QdrantVectorStore(
        client=client,
        collection=settings.collection_name(),
        dim=settings.embed_dim,
    )
