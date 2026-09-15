"""Pydantic v2 request/response models for the HTTP surface.

`status` is a field, not an exception. "Not found in context" returns HTTP 200
with a well-formed body, so the eval runner, the API client and the streaming
endpoint at M8 all consume one shape, and there is exactly one place where the
never-hallucinate rule is expressed.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.models import Answer, AnswerStatus, Citation, GuardReport, ScoredChunk


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    debug: bool = False

    @model_validator(mode="after")
    def _question_is_not_only_whitespace(self) -> Self:
        if not self.question.strip():
            raise ValueError("question must not be blank")
        return self


class RetrievedChunkView(BaseModel):
    """Only returned when debug=true; never part of the normal contract."""

    chunk_id: str
    score: float
    rank: int
    stage: str
    page: int | None
    circular_no: str | None
    title: str | None = None
    doc_id: str
    preview: str
    text: str  # full chunk text: the UI highlights evidence quotes in it

    @classmethod
    def from_scored(cls, scored: ScoredChunk) -> Self:
        return cls(
            chunk_id=scored.chunk.chunk_id,
            score=scored.score,
            rank=scored.rank,
            stage=scored.stage,
            page=scored.chunk.page,
            circular_no=scored.chunk.meta.circular_no,
            title=scored.chunk.meta.title,
            doc_id=scored.chunk.doc_id,
            preview=scored.chunk.text[:200],
            text=scored.chunk.text,
        )


class SpanView(BaseModel):
    name: str
    type: str
    depth: int
    started_ms: float
    duration_ms: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    provider: str | None = None
    model: str | None = None
    extra: dict[str, object] = Field(default_factory=dict)


class TraceView(BaseModel):
    """What this request did, from the same spans Langfuse receives."""

    trace_id: str | None = None
    langfuse_url: str | None = None
    total_ms: float
    total_cost_usd: float
    spans: list[SpanView]

    @classmethod
    def from_scope(cls, scope: object, *, tracer: object, total_ms: float) -> Self:
        spans = [SpanView(**vars(s)) for s in getattr(scope, "spans", [])]
        trace_id = getattr(scope, "trace_id", None)
        url = tracer.trace_url(trace_id) if hasattr(tracer, "trace_url") else None
        return cls(
            trace_id=trace_id,
            langfuse_url=url,
            total_ms=total_ms,
            total_cost_usd=round(sum(s.cost_usd or 0.0 for s in spans), 6),
            spans=spans,
        )


class UsageView(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    model_id: str | None = None
    provider: str | None = None


class QueryResponse(BaseModel):
    status: AnswerStatus
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool
    invalid_citations: list[int] = Field(default_factory=list)
    invalid_citation_rate: float = 0.0
    chunks_considered: int = 0
    usage: UsageView = Field(default_factory=UsageView)
    latency_ms: float = 0.0
    config_fingerprint: str
    guard: GuardReport | None = None
    retrieval: list[RetrievedChunkView] | None = None
    trace: TraceView | None = None

    @classmethod
    def from_answer(
        cls,
        answer: Answer,
        *,
        fingerprint: str,
        latency_ms: float,
        retrieval: list[ScoredChunk] | None = None,
        trace: TraceView | None = None,
    ) -> Self:
        return cls(
            status=answer.status or AnswerStatus.ANSWERED,
            answer=answer.answer,
            citations=answer.citations,
            grounded=answer.grounded,
            invalid_citations=answer.invalid_citations,
            invalid_citation_rate=answer.invalid_citation_rate,
            chunks_considered=answer.chunks_considered,
            usage=UsageView(
                input_tokens=answer.input_tokens,
                output_tokens=answer.output_tokens,
                model_id=answer.model_id,
                provider=answer.provider,
            ),
            latency_ms=latency_ms,
            config_fingerprint=fingerprint,
            guard=answer.guard,
            retrieval=(
                [RetrievedChunkView.from_scored(s) for s in retrieval]
                if retrieval is not None
                else None
            ),
            trace=trace,
        )


class HealthResponse(BaseModel):
    status: str
    collection: str
    config_fingerprint: str


class ReadyResponse(BaseModel):
    ready: bool
    collection: str
    points: int | None = None
    detail: str | None = None


class ErrorResponse(BaseModel):
    detail: str


__all__ = [
    "AnswerStatus",
    "ErrorResponse",
    "HealthResponse",
    "QueryRequest",
    "QueryResponse",
    "ReadyResponse",
    "RetrievedChunkView",
    "SpanView",
    "TraceView",
    "UsageView",
]
