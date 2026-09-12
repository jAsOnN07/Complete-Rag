from __future__ import annotations

import hashlib
from datetime import date
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Regulator = Literal["RBI", "SEBI"]
RetrievalStage = Literal["dense", "sparse", "fused", "reranked"]

NOT_FOUND_MESSAGE = (
    "I could not find an answer to that in the indexed circulars."
)


class ChunkStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


def make_doc_id(source_path: str) -> str:
    normalised = source_path.replace("\\", "/").strip().lower()
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


def make_chunk_id(doc_id: str, strategy: ChunkStrategy, ordinal: int) -> str:
    return f"{doc_id}::{strategy.value}::{ordinal}"


class DocumentMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    source_path: str
    title: str
    regulator: Regulator
    circular_no: str | None = None
    issued_on: date | None = None
    source_url: str | None = None


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    title: str
    circular_no: str | None = None
    page: int | None = None


def build_context_header(meta: "DocumentMeta") -> str:
    """One line of document identity to prepend to a chunk's retrieval text.

    Measured at M5: questions that key on a short span plus the document's
    identity are only answered when the chunk carries its header context.
    Fixed windows get this by accident of overlap; other strategies lose it.
    """
    parts = [meta.title]
    if meta.circular_no:
        parts.append(meta.circular_no)
    if meta.issued_on:
        parts.append(meta.issued_on.strftime("%d %b %Y"))
    return " | ".join(parts)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    ordinal: int = Field(ge=0)
    strategy: ChunkStrategy
    section: str | None = None
    page: int | None = None
    meta: DocumentMeta
    # What retrieval sees (embedding, BM25, reranker). None means `text`.
    # Set at ingest and stored in the payload so search and rerank agree.
    retrieval_text: str | None = None

    @property
    def text_for_retrieval(self) -> str:
        return self.retrieval_text or self.text

    def with_context_header(self) -> "Chunk":
        return self.model_copy(
            update={"retrieval_text": f"{build_context_header(self.meta)}\n{self.text}"}
        )

    @field_validator("text")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk text must not be blank")
        return value

    def to_citation(self) -> Citation:
        return Citation(
            chunk_id=self.chunk_id,
            doc_id=self.doc_id,
            title=self.meta.title,
            circular_no=self.meta.circular_no,
            page=self.page,
        )


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int = Field(ge=0)
    stage: RetrievalStage

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    NOT_FOUND_IN_CONTEXT = "not_found_in_context"
    BLOCKED_INPUT = "blocked_input"
    BLOCKED_OUTPUT = "blocked_output"


class GuardReport(BaseModel):
    """What the guardrails decided. Recorded on every answer so a refusal is
    explainable and the false-positive rate on the gold set is measurable."""

    input_allowed: bool = True
    input_violations: list[str] = Field(default_factory=list)
    input_degraded: bool = False
    injection_score: float | None = None
    output_action: str | None = None
    output_reasons: list[str] = Field(default_factory=list)
    grounding_score: float | None = None
    grounding_threshold: float | None = None
    pii_redacted: list[str] = Field(default_factory=list)


BLOCKED_INPUT_MESSAGE = "Your question could not be processed."
BLOCKED_OUTPUT_MESSAGE = "The generated answer did not pass output checks and was withheld."


class Answer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool
    not_found: bool = False
    provider: str
    status: AnswerStatus | None = None
    guard: GuardReport | None = None
    # Labels the model emitted that were outside 1..k. A first-class quality
    # metric, not an error: it measures citation hallucination directly.
    invalid_citations: list[int] = Field(default_factory=list)
    chunks_considered: int = 0
    model_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def invalid_citation_rate(self) -> float:
        emitted = len(self.citations) + len(self.invalid_citations)
        return len(self.invalid_citations) / emitted if emitted else 0.0

    @model_validator(mode="after")
    def _not_found_implies_no_citations(self) -> Self:
        if self.not_found and self.citations:
            raise ValueError("a not_found answer must not carry citations")
        if self.status is None:
            self.status = (
                AnswerStatus.NOT_FOUND_IN_CONTEXT if self.not_found else AnswerStatus.ANSWERED
            )
        return self

    @classmethod
    def blocked_input(cls, report: GuardReport) -> Self:
        return cls(
            answer=BLOCKED_INPUT_MESSAGE, citations=[], grounded=True, not_found=False,
            provider="none", status=AnswerStatus.BLOCKED_INPUT, guard=report,
        )

    @classmethod
    def blocked_output(cls, report: GuardReport, *, provider: str, model_id: str | None,
                       input_tokens: int, output_tokens: int, chunks_considered: int) -> Self:
        return cls(
            answer=BLOCKED_OUTPUT_MESSAGE, citations=[], grounded=False, not_found=False,
            provider=provider, status=AnswerStatus.BLOCKED_OUTPUT, guard=report,
            model_id=model_id, input_tokens=input_tokens, output_tokens=output_tokens,
            chunks_considered=chunks_considered,
        )

    @classmethod
    def not_found_response(cls, provider: str) -> Self:
        return cls(
            answer=NOT_FOUND_MESSAGE,
            citations=[],
            grounded=True,
            not_found=True,
            provider=provider,
        )


class Page(BaseModel):
    page_number: int = Field(ge=1)
    text: str
    char_count: int = Field(ge=0)

    @classmethod
    def build(cls, page_number: int, text: str) -> Self:
        return cls(page_number=page_number, text=text, char_count=len(text.strip()))


class RawDocument(BaseModel):
    meta: DocumentMeta
    pages: list[Page]

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.pages)

    @property
    def char_count(self) -> int:
        return sum(p.char_count for p in self.pages)

    def page_for_offset(self, offset: int) -> int:
        """Map a character offset in `text` back to its 1-indexed page."""
        cursor = 0
        for page in self.pages:
            cursor += len(page.text) + 1
            if offset < cursor:
                return page.page_number
        return self.pages[-1].page_number if self.pages else 1


class ExtractionStats(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chars: int
    empty_pages: int
    stripped_lines: int

    @property
    def empty_page_ratio(self) -> float:
        return self.empty_pages / self.pages if self.pages else 0.0
