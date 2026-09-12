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


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    ordinal: int = Field(ge=0)
    strategy: ChunkStrategy
    section: str | None = None
    page: int | None = None
    meta: DocumentMeta

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


class Answer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool
    not_found: bool = False
    provider: str
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
        return self

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
