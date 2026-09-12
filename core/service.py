"""RagService - the single query orchestrator.

Lives here rather than in api/routes.py so the eval runner can call it
in-process, with no HTTP hop and no server to spin up.

The "never hallucinate" rule is enforced in exactly one place: if no retrieved
chunk clears the relevance threshold, the request returns a structured
not-found answer and **no LLM call is made at all**. That is cheaper, faster,
and structurally impossible to hallucinate through.
"""

from __future__ import annotations

from typing import Any, Sequence

from core.models import Answer, ScoredChunk
from core.ports import Embedder, LlmClient, Reranker, VectorStore
from generation.prompt import PromptBuilder, is_not_found_response, parse_citations
from observability.tracing import Tracer, get_tracer


class RagService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        llm: LlmClient,
        prompt_builder: PromptBuilder | None = None,
        reranker: Reranker | None = None,
        top_k: int = 20,
        rerank_top_n: int = 5,
        relevance_threshold: float = 0.0,
        tracer: Tracer | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._prompt = prompt_builder or PromptBuilder()
        self._reranker = reranker
        self._top_k = top_k
        self._rerank_top_n = rerank_top_n
        self._relevance_threshold = relevance_threshold
        self._tracer = tracer or get_tracer()

    async def count_points(self) -> int:
        return await self._store.count()

    async def retrieve(
        self, question: str, *, top_n: int | None = None
    ) -> list[ScoredChunk]:
        limit = top_n or self._rerank_top_n
        vector = await self._embedder.embed_query(question)
        candidates = await self._store.search_dense(vector, self._top_k)
        if self._reranker is not None:
            return await self._reranker.rerank(question, candidates, limit)
        return candidates[:limit]

    async def answer(self, question: str, *, top_n: int | None = None) -> Answer:
        candidates = await self.retrieve(question, top_n=top_n)
        return await self.answer_from(question, candidates)

    async def answer_from(
        self, question: str, candidates: Sequence[ScoredChunk]
    ) -> Answer:
        """Generation half, split out so debug mode does not retrieve twice."""
        async with self._tracer.observe(
            "rag.answer", input={"question": question}
        ) as span:
            relevant = [
                c for c in candidates if c.score >= self._relevance_threshold
            ]

            if not relevant:
                result = Answer.not_found_response(provider="none")
                result.chunks_considered = len(candidates)
                span.update(
                    output={"status": "not_found", "llm_called": False},
                    metadata={"candidates": len(candidates)},
                )
                return result

            payload = self._prompt.build(question, relevant)
            completion = await self._llm.complete(payload.system, payload.user)

            if is_not_found_response(completion.text):
                result = Answer.not_found_response(provider=completion.provider)
                result.chunks_considered = len(relevant)
                result.model_id = completion.model_id
                result.input_tokens = completion.input_tokens
                result.output_tokens = completion.output_tokens
                span.update(output={"status": "not_found_by_model"})
                return result

            citations, invalid = parse_citations(completion.text, payload.label_map)
            answer = Answer(
                answer=completion.text,
                citations=citations,
                # Real grounding verification arrives with Bedrock Guardrails at M8.
                grounded=bool(citations),
                not_found=False,
                provider=completion.provider,
                invalid_citations=invalid,
                chunks_considered=len(relevant),
                model_id=completion.model_id,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
            )
            span.update(
                output={
                    "status": "answered",
                    "citations": len(citations),
                    "invalid_citations": invalid,
                },
                metadata={"invalid_citation_rate": answer.invalid_citation_rate},
            )
            return answer


def build_service(settings: Any = None) -> RagService:
    from core.config import get_settings
    from generation.llm import build_bedrock_llm
    from retrieval.embedder import build_embedder
    from retrieval.vector_store import build_qdrant_store

    settings = settings or get_settings()
    # No reranker is wired until M6, so the final score is raw Qdrant cosine and
    # the threshold must be read on the dense scale, not the reranker's.
    return RagService(
        embedder=build_embedder(settings),
        store=build_qdrant_store(settings),
        llm=build_bedrock_llm(settings),
        top_k=settings.top_k,
        rerank_top_n=settings.rerank_top_n,
        relevance_threshold=settings.threshold_for("dense"),
    )
