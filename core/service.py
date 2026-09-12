"""RagService - the single query orchestrator.

Lives here rather than in api/routes.py so the eval runner can call it
in-process, with no HTTP hop and no server to spin up.

The "never hallucinate" rule is enforced in exactly one place: if no retrieved
chunk clears the relevance threshold, the request returns a structured
not-found answer and **no LLM call is made at all**. That is cheaper, faster,
and structurally impossible to hallucinate through.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

from pydantic import BaseModel, Field

from core.models import Answer, AnswerStatus, Citation, GuardReport, ScoredChunk
from core.ports import Embedder, LlmClient, Reranker, VectorStore
from retrieval.hybrid import DenseRetriever
from generation.prompt import PromptBuilder, is_not_found_response, parse_citations
from observability.tracing import Tracer, get_tracer


class StreamEvent(BaseModel):
    """One server-sent event. `meta` first, `token`s, then exactly one `final`."""

    event: str
    data: dict[str, Any] = Field(default_factory=dict)


class RagService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        llm: LlmClient,
        prompt_builder: PromptBuilder | None = None,
        reranker: Reranker | None = None,
        retriever: Any | None = None,
        input_guard: Any | None = None,
        output_guard: Any | None = None,
        top_k: int = 20,
        rerank_top_n: int = 5,
        relevance_threshold: float = 0.0,
        tracer: Tracer | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._retriever = retriever or DenseRetriever(
            embedder=embedder, store=store, top_k=top_k
        )
        self._input_guard = input_guard
        self._output_guard = output_guard
        self._prompt = prompt_builder or PromptBuilder()
        self._reranker = reranker
        self._top_k = top_k
        self._rerank_top_n = rerank_top_n
        self._relevance_threshold = relevance_threshold
        self._tracer = tracer or get_tracer()

    @property
    def reranker_active(self) -> bool:
        return self._reranker is not None

    async def count_points(self) -> int:
        return await self._store.count()

    async def retrieve(
        self, question: str, *, top_n: int | None = None
    ) -> list[ScoredChunk]:
        limit = top_n or self._rerank_top_n
        candidates = await self._retriever.retrieve(question, k=self._top_k)
        if self._reranker is not None:
            return await self._reranker.rerank(question, candidates, limit)
        return candidates[:limit]

    async def guard_input(self, question: str) -> GuardReport | None:
        """Run the input guard. Returns a report only when the question is blocked.

        Runs before retrieval, not merely before the gateway: a blocked question
        should cost nothing - no embedding, no search, no rerank.
        """
        if self._input_guard is None:
            return None
        result = await self._input_guard.check(question)
        if result.allowed:
            return None
        return GuardReport(
            input_allowed=False,
            input_violations=[f"{v.kind}: {v.detail}" for v in result.violations],
            input_degraded=result.degraded,
            injection_score=result.injection_score,
        )

    async def answer(self, question: str, *, top_n: int | None = None) -> Answer:
        blocked = await self.guard_input(question)
        if blocked is not None:
            return Answer.blocked_input(blocked)
        candidates = await self.retrieve(question, top_n=top_n)
        return await self.answer_from(question, candidates, input_checked=True)

    async def answer_from(
        self, question: str, candidates: Sequence[ScoredChunk], *, input_checked: bool = False
    ) -> Answer:
        """Generation half, split out so debug mode does not retrieve twice."""
        if not input_checked:
            blocked = await self.guard_input(question)
            if blocked is not None:
                return Answer.blocked_input(blocked)

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
            text = completion.text
            grounded = bool(citations)
            report: GuardReport | None = None

            if self._output_guard is not None:
                verdict = await self._output_guard.check(
                    question=question,
                    answer=text,
                    grounding_sources=self._prompt.grounding_sources(relevant),
                )
                report = GuardReport(
                    output_action=verdict.action,
                    output_reasons=verdict.reasons,
                    grounding_score=verdict.grounding_score,
                    grounding_threshold=verdict.grounding_threshold,
                    pii_redacted=verdict.pii_redacted,
                )
                if verdict.blocked:
                    span.update(output={"status": "blocked_output", "reasons": verdict.reasons})
                    return Answer.blocked_output(
                        report, provider=completion.provider, model_id=completion.model_id,
                        input_tokens=completion.input_tokens, output_tokens=completion.output_tokens,
                        chunks_considered=len(relevant),
                    )
                text = verdict.text
                if verdict.grounded is not None:
                    grounded = verdict.grounded

            answer = Answer(
                answer=text,
                citations=citations,
                grounded=grounded,
                not_found=False,
                provider=completion.provider,
                invalid_citations=invalid,
                chunks_considered=len(relevant),
                model_id=completion.model_id,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                guard=report,
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


    async def answer_stream(
        self, question: str, *, window_chars: int = 300, top_n: int | None = None
    ) -> AsyncIterator[StreamEvent]:
        """Streaming variant. Text is held in a window and checked before it is
        emitted, so nothing unchecked ever reaches the client; the full
        grounding check runs once on the assembled answer and is reported in
        the final event. A grounding failure cannot retract streamed tokens -
        that is why /query (non-streaming) is what the eval measures.
        """
        blocked = await self.guard_input(question)
        if blocked is not None:
            yield StreamEvent(event="final", data=Answer.blocked_input(blocked).model_dump(mode="json"))
            return

        candidates = await self.retrieve(question, top_n=top_n)
        relevant = [c for c in candidates if c.score >= self._relevance_threshold]
        if not relevant:
            result = Answer.not_found_response(provider="none")
            result.chunks_considered = len(candidates)
            yield StreamEvent(event="final", data=result.model_dump(mode="json"))
            return

        payload = self._prompt.build(question, relevant)
        yield StreamEvent(
            event="meta",
            data={"chunks_considered": len(relevant),
                  "sources": [c.chunk.to_citation().model_dump(mode="json") for c in relevant]},
        )

        assembled: list[str] = []
        window = ""
        final_delta = None
        async for delta in self._llm.stream(payload.system, payload.user):
            if delta.done:
                final_delta = delta
                break
            window += delta.text
            assembled.append(delta.text)
            if len(window) >= window_chars:
                if self._output_guard is not None:
                    verdict = await self._output_guard.check_window(window)
                    if verdict.blocked:
                        report = GuardReport(output_action=verdict.action, output_reasons=verdict.reasons)
                        yield StreamEvent(
                            event="final",
                            data=Answer.blocked_output(
                                report, provider=delta.provider or "unknown", model_id=None,
                                input_tokens=0, output_tokens=0, chunks_considered=len(relevant),
                            ).model_dump(mode="json"),
                        )
                        return
                    window = verdict.text
                yield StreamEvent(event="token", data={"text": window})
                window = ""
        if window:
            if self._output_guard is not None:
                verdict = await self._output_guard.check_window(window)
                if verdict.blocked:
                    report = GuardReport(output_action=verdict.action, output_reasons=verdict.reasons)
                    yield StreamEvent(
                        event="final",
                        data=Answer.blocked_output(
                            report, provider="unknown", model_id=None,
                            input_tokens=0, output_tokens=0, chunks_considered=len(relevant),
                        ).model_dump(mode="json"),
                    )
                    return
                window = verdict.text
            yield StreamEvent(event="token", data={"text": window})

        text = "".join(assembled)
        provider = (final_delta.provider if final_delta else None) or "unknown"
        model_id = final_delta.model_id if final_delta else None
        in_tok = final_delta.input_tokens if final_delta else 0
        out_tok = final_delta.output_tokens if final_delta else 0

        if is_not_found_response(text):
            result = Answer.not_found_response(provider=provider)
            result.chunks_considered = len(relevant)
            result.model_id, result.input_tokens, result.output_tokens = model_id, in_tok, out_tok
            yield StreamEvent(event="final", data=result.model_dump(mode="json"))
            return

        citations, invalid = parse_citations(text, payload.label_map)
        grounded = bool(citations)
        report: GuardReport | None = None
        if self._output_guard is not None:
            verdict = await self._output_guard.check(
                question=question, answer=text, grounding_sources=self._prompt.grounding_sources(relevant),
            )
            report = GuardReport(
                output_action=verdict.action, output_reasons=verdict.reasons,
                grounding_score=verdict.grounding_score, grounding_threshold=verdict.grounding_threshold,
                pii_redacted=verdict.pii_redacted,
            )
            if verdict.grounded is not None:
                grounded = verdict.grounded
            if verdict.blocked:
                yield StreamEvent(
                    event="final",
                    data=Answer.blocked_output(
                        report, provider=provider, model_id=model_id,
                        input_tokens=in_tok, output_tokens=out_tok, chunks_considered=len(relevant),
                    ).model_dump(mode="json"),
                )
                return

        answer = Answer(
            answer=text, citations=citations, grounded=grounded, not_found=False, provider=provider,
            invalid_citations=invalid, chunks_considered=len(relevant), model_id=model_id,
            input_tokens=in_tok, output_tokens=out_tok, guard=report,
        )
        yield StreamEvent(event="final", data=answer.model_dump(mode="json"))


def build_service(settings: Any = None) -> RagService:
    from core.config import get_settings
    from generation.llm import build_llm
    from retrieval.embedder import build_embedder
    from retrieval.vector_store import build_qdrant_store

    settings = settings or get_settings()
    embedder = build_embedder(settings)
    store = build_qdrant_store(settings)

    retriever: Any
    if settings.retrieval_mode == "hybrid":
        from retrieval.bm25 import Bm25Encoder
        from retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever(
            embedder=embedder, sparse=Bm25Encoder(), store=store,
            top_k=settings.top_k, prefetch_k=settings.prefetch_k,
        )
    else:
        retriever = DenseRetriever(embedder=embedder, store=store, top_k=settings.top_k)

    from guards.input_guards import build_input_guard
    from guards.output_guards import build_output_guard
    from retrieval.reranker import build_reranker

    reranker = build_reranker(settings)
    reranker_active = reranker.backend != "none"

    # The not-found gate reads whatever produces the final score - reranker if
    # wired, else RRF (vacuous) or cosine - and thresholds are per scale.
    return RagService(
        embedder=embedder,
        store=store,
        llm=build_llm(settings),
        retriever=retriever,
        reranker=reranker if reranker_active else None,
        input_guard=build_input_guard(settings),
        output_guard=build_output_guard(settings),
        top_k=settings.top_k,
        rerank_top_n=settings.rerank_top_n,
        relevance_threshold=settings.threshold_for(
            settings.final_score_scale(reranker_active=reranker_active)
        ),
    )
