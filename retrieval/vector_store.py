"""Qdrant collection and index operations.

The whole Chunk goes into the point payload, so retrieval reconstructs it with
`Chunk.model_validate(payload)` and needs no sidecar document store. At 45
documents the duplication costs nothing and removes an entire class of join bug.

The collection is created with a named dense vector so the sparse BM25 vector
can be added alongside it at M6 without recreating the collection.
"""

from __future__ import annotations

import uuid
from typing import Any, Sequence

from qdrant_client import AsyncQdrantClient, models

from core.models import Chunk, ScoredChunk
from observability.tracing import Tracer, get_tracer

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
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} vs {len(vectors)}"
            )
        async with self._tracer.observe(
            "qdrant.upsert",
            input={"collection": self._collection, "points": len(chunks)},
        ) as span:
            points = [
                models.PointStruct(
                    id=point_id_for(chunk.chunk_id),
                    vector={DENSE_VECTOR_NAME: list(vector)},
                    payload=chunk.model_dump(mode="json"),
                )
                for chunk, vector in zip(chunks, vectors)
            ]
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

    async def count(self) -> int:
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
        dim=settings.bedrock_embed_dim,
    )
