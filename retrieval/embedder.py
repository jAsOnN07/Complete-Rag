"""Bedrock Titan Embeddings V2.

Titan V2 has no batch endpoint - one HTTP call per text - so documents are
embedded concurrently under a semaphore. boto3 is synchronous and its client is
thread-safe, so every call goes through asyncio.to_thread; calling it directly
would serialise the whole "async" API under load without ever showing up in
single-request testing.

Embeddings deliberately do NOT go through Portkey: there is no valid fallback
for an embedding model, because a different model means a different vector
space, so failing over would corrupt the index.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence

from core.errors import UpstreamServiceError
from observability.tracing import Tracer, get_tracer


class BedrockTitanEmbedder:
    def __init__(
        self,
        *,
        client: Any,
        model_id: str,
        dim: int,
        normalize: bool = True,
        max_concurrency: int = 8,
        tracer: Tracer | None = None,
    ) -> None:
        self._client = client
        self._model_id = model_id
        self._dim = dim
        self._normalize = normalize
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._tracer = tracer or get_tracer()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def _invoke_sync(self, text: str) -> list[float]:
        body = json.dumps(
            {
                "inputText": text,
                "dimensions": self._dim,
                "normalize": self._normalize,
            }
        )
        response = self._client.invoke_model(
            modelId=self._model_id,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(response["body"].read())
        return payload["embedding"]

    async def _embed_one(self, text: str) -> list[float]:
        async with self._semaphore:
            try:
                return await asyncio.to_thread(self._invoke_sync, text)
            except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
                raise UpstreamServiceError(
                    "bedrock", "embed", f"{type(exc).__name__}: {exc}", cause=exc
                ) from exc

    async def embed_query(self, text: str) -> list[float]:
        async with self._tracer.observe(
            "embed.query", as_type="embedding", input={"chars": len(text)}
        ) as span:
            vector = await self._embed_one(text)
            span.update(
                output={"dim": len(vector)},
                metadata={"model_id": self._model_id},
            )
            return vector

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        async with self._tracer.observe(
            "embed.documents", as_type="embedding", input={"count": len(texts)}
        ) as span:
            vectors = await asyncio.gather(*(self._embed_one(t) for t in texts))
            span.update(
                output={"count": len(vectors)},
                metadata={"model_id": self._model_id, "dim": self._dim},
            )
            return list(vectors)


def build_bedrock_embedder(settings: Any = None) -> BedrockTitanEmbedder:
    import boto3
    from botocore.config import Config

    if settings is None:
        from core.config import get_settings

        settings = get_settings()

    # Transport-level throttle handling only. This is not application retry
    # logic for the LLM path - Portkey owns that from M7.
    client = boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region,
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
        aws_access_key_id=(
            settings.aws_access_key_id.get_secret_value()
            if settings.aws_access_key_id
            else None
        ),
        aws_secret_access_key=(
            settings.aws_secret_access_key.get_secret_value()
            if settings.aws_secret_access_key
            else None
        ),
    )
    return BedrockTitanEmbedder(
        client=client,
        model_id=settings.bedrock_embed_model_id,
        dim=settings.bedrock_embed_dim,
    )


class FastEmbedEmbedder:
    """Local ONNX embedder (fastembed). No network after the first model download.

    A development backend that keeps retrieval, eval and tracing real while
    Bedrock is unavailable. It is a different vector space from Titan, so it
    always writes to its own collection - see Settings.collection_name.
    """

    def __init__(
        self,
        *,
        model_name: str,
        dim: int,
        tracer: Tracer | None = None,
        cache_dir: str | None = None,
        threads: int | None = 4,
    ) -> None:
        from fastembed import TextEmbedding

        self._model_name = model_name
        self._dim = dim
        self._tracer = tracer or get_tracer()
        # Measured on an 8-thread laptop: 4 threads beat 8 (hyperthread
        # oversubscription) and the default by ~40%. ONNX CPU inference here
        # runs ~300 ms per 1k-char chunk, so ingest is minutes, not seconds.
        self._model = TextEmbedding(
            model_name=model_name, cache_dir=cache_dir, threads=threads
        )

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_sync(self, texts: Sequence[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(list(texts))]

    async def embed_query(self, text: str) -> list[float]:
        async with self._tracer.observe(
            "embed.query", as_type="embedding", input={"chars": len(text)}
        ) as span:
            vector = (await asyncio.to_thread(self._embed_sync, [text]))[0]
            span.update(
                output={"dim": len(vector)},
                metadata={"model_id": self._model_name, "backend": "fastembed"},
            )
            return vector

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        async with self._tracer.observe(
            "embed.documents", as_type="embedding", input={"count": len(texts)}
        ) as span:
            vectors = await asyncio.to_thread(self._embed_sync, texts)
            span.update(
                output={"count": len(vectors)},
                metadata={"model_id": self._model_name, "dim": self._dim},
            )
            return vectors


def build_embedder(settings: Any = None) -> BedrockTitanEmbedder | FastEmbedEmbedder:
    """Select the embedding backend from config. Titan unless told otherwise."""
    if settings is None:
        from core.config import get_settings

        settings = get_settings()

    if settings.embedding_backend == "fastembed":
        return FastEmbedEmbedder(
            model_name=settings.fastembed_model, dim=settings.fastembed_dim
        )
    return build_bedrock_embedder(settings)
