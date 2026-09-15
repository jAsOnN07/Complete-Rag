"""Rerankers behind one interface, swappable by config.

Score scales differ per backend and this is not a detail: the not-found gate
reads the final score, so the threshold is resolved per scale (see
Settings.threshold_for). Cross-encoder emits unbounded logits centred on zero;
Bedrock Rerank emits 0-1; noop passes the retrieval score through untouched.

The cross-encoder is a ~90 MB model loaded once per process and run under
asyncio.to_thread - never on the event loop.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal, Sequence

from core.errors import UpstreamServiceError
from core.models import ScoredChunk
from observability.pricing import rerank_cost
from observability.tracing import Tracer, get_tracer

ScoreScale = Literal["logit", "unit", "passthrough"]
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _rescored(
    ordered: Sequence[tuple[ScoredChunk, float]], top_n: int
) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=s.chunk, score=float(score), rank=i, stage="reranked")
        for i, (s, score) in enumerate(ordered[:top_n])
    ]


class NoopReranker:
    backend = "none"
    score_scale: ScoreScale = "passthrough"

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        return list(candidates[:top_n])


class CrossEncoderReranker:
    backend = "cross_encoder"
    score_scale: ScoreScale = "logit"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_CROSS_ENCODER,
        max_length: int = 512,
        tracer: Tracer | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._model_name = model_name
        self._model = CrossEncoder(model_name, max_length=max_length)
        self._tracer = tracer or get_tracer()

    @property
    def model_id(self) -> str:
        return self._model_name

    def _predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(x) for x in self._model.predict(pairs)]

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        async with self._tracer.observe(
            "rerank.cross_encoder",
            input={"candidates": len(candidates), "top_n": top_n},
        ) as span:
            pairs = [(query, c.chunk.text_for_retrieval) for c in candidates]
            scores = await asyncio.to_thread(self._predict_sync, pairs)
            ordered = sorted(zip(candidates, scores), key=lambda t: -t[1])
            out = _rescored(ordered, top_n)
            span.update(
                output={"top_score": out[0].score if out else None, "returned": len(out)},
                metadata={"model": self._model_name, "scale": self.score_scale},
            )
            return out


class CohereReranker:
    """Cohere Rerank via the SDK. Scores are 0-1 relevance probabilities."""

    backend = "cohere"
    score_scale: ScoreScale = "unit"

    def __init__(self, *, client: Any, model_id: str, tracer: Tracer | None = None) -> None:
        self._client = client
        self._model_id = model_id
        self._tracer = tracer or get_tracer()

    @property
    def model_id(self) -> str:
        return self._model_id

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        async with self._tracer.observe(
            "rerank.cohere", input={"candidates": len(candidates), "top_n": top_n}
        ) as span:
            try:
                response = await self._client.rerank(
                    model=self._model_id,
                    query=query,
                    documents=[c.chunk.text_for_retrieval for c in candidates],
                    top_n=min(top_n, len(candidates)),
                )
            except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
                raise UpstreamServiceError(
                    "cohere", "rerank", f"{type(exc).__name__}: {exc}", cause=exc
                ) from exc
            ordered = [(candidates[r.index], float(r.relevance_score)) for r in response.results]
            out = _rescored(ordered, top_n)
            meta = getattr(response, "meta", None)
            units = getattr(meta, "billed_units", None) if meta else None
            searches = int(getattr(units, "search_units", 0) or 0) or 1
            span.update(
                output={"top_score": out[0].score if out else None, "returned": len(out)},
                usage_details={"input": searches},
                cost_details=rerank_cost(self._model_id, searches),
                metadata={"model": self._model_id, "scale": self.score_scale, "search_units": searches},
            )
            return out


class BedrockReranker:
    """Amazon Bedrock Rerank via bedrock-agent-runtime.

    Regional availability is narrower than Bedrock's; the model ARN and region
    are config. Request/response shape verified against the botocore service
    model rather than recalled.
    """

    backend = "bedrock"
    score_scale: ScoreScale = "unit"

    def __init__(
        self, *, client: Any, model_arn: str, tracer: Tracer | None = None
    ) -> None:
        self._client = client
        self._model_arn = model_arn
        self._tracer = tracer or get_tracer()

    def _rerank_sync(
        self, query: str, texts: Sequence[str], top_n: int
    ) -> list[dict[str, Any]]:
        response = self._client.rerank(
            queries=[{"type": "TEXT", "textQuery": {"text": query}}],
            sources=[
                {
                    "type": "INLINE",
                    "inlineDocumentSource": {
                        "type": "TEXT",
                        "textDocument": {"text": t},
                    },
                }
                for t in texts
            ],
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {"modelArn": self._model_arn},
                    "numberOfResults": top_n,
                },
            },
        )
        return response["results"]

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        async with self._tracer.observe(
            "rerank.bedrock", input={"candidates": len(candidates), "top_n": top_n}
        ) as span:
            texts = [c.chunk.text_for_retrieval for c in candidates]
            try:
                results = await asyncio.to_thread(
                    self._rerank_sync, query, texts, min(top_n, len(candidates))
                )
            except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
                raise UpstreamServiceError(
                    "bedrock", "rerank", f"{type(exc).__name__}: {exc}", cause=exc
                ) from exc
            ordered = [
                (candidates[r["index"]], float(r["relevanceScore"])) for r in results
            ]
            out = _rescored(ordered, top_n)
            span.update(
                output={"top_score": out[0].score if out else None, "returned": len(out)},
                metadata={"model_arn": self._model_arn, "scale": self.score_scale},
            )
            return out


def build_reranker(
    settings: Any = None,
) -> NoopReranker | CrossEncoderReranker | CohereReranker | BedrockReranker:
    if settings is None:
        from core.config import get_settings

        settings = get_settings()

    if settings.reranker_backend == "none":
        return NoopReranker()
    if settings.reranker_backend == "cross_encoder":
        return CrossEncoderReranker(model_name=settings.cross_encoder_model)
    if settings.reranker_backend == "cohere":
        from retrieval.embedder import build_cohere_client

        return CohereReranker(client=build_cohere_client(settings), model_id=settings.cohere_rerank_model)

    import boto3

    if not settings.bedrock_rerank_model_arn:
        raise RuntimeError("BEDROCK_RERANK_MODEL_ARN is required for the bedrock reranker")
    client = boto3.client(
        "bedrock-agent-runtime",
        region_name=settings.bedrock_rerank_region or settings.aws_region,
        aws_access_key_id=(
            settings.aws_access_key_id.get_secret_value() if settings.aws_access_key_id else None
        ),
        aws_secret_access_key=(
            settings.aws_secret_access_key.get_secret_value()
            if settings.aws_secret_access_key
            else None
        ),
    )
    return BedrockReranker(client=client, model_arn=settings.bedrock_rerank_model_arn)
