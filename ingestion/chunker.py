"""Chunking strategies.

Recursive is the default. Fixed and semantic land at M5 alongside their eval
comparison - `get_chunker` raises for them rather than silently substituting
recursive, because a silent substitution would invalidate an eval run without
anything in the results JSON showing it.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.models import Chunk, ChunkStrategy, RawDocument, make_chunk_id

# Circulars are numbered clauses ("2.", "3.1") separated by newlines, so line
# breaks are a better split point than sentence punctuation, which appears
# inside statutory references like "Act, 1999." and "June 01, 2000."
CIRCULAR_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "; ", " ", "")


class BaseChunker:
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE

    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def params_hash(self) -> str:
        payload = f"{self.strategy.value}:{self.chunk_size}:{self.chunk_overlap}"
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def _split(self, text: str) -> list[str]:  # pragma: no cover - overridden
        raise NotImplementedError

    async def chunk(self, doc: RawDocument) -> list[Chunk]:
        return await asyncio.to_thread(self.chunk_sync, doc)

    def chunk_sync(self, doc: RawDocument) -> list[Chunk]:
        text = doc.text
        if not text.strip():
            return []
        return self._finalise(doc, self._split(text))

    def _finalise(self, doc: RawDocument, pieces: Sequence[str]) -> list[Chunk]:
        """Attach identity, page and metadata to raw split text.

        Page numbers come from scanning forward through the document text, so a
        chunk that repeats earlier wording is still attributed to where it
        actually appears rather than to its first occurrence.
        """
        text = doc.text
        chunks: list[Chunk] = []
        cursor = 0
        ordinal = 0

        for piece in pieces:
            body = piece.strip()
            if not body:
                continue
            found = text.find(body, cursor)
            if found == -1:
                found = text.find(body)
            offset = found if found != -1 else cursor
            cursor = max(cursor, offset + max(len(body) - self.chunk_overlap, 1))

            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.meta.doc_id, self.strategy, ordinal),
                    doc_id=doc.meta.doc_id,
                    text=body,
                    ordinal=ordinal,
                    strategy=self.strategy,
                    page=doc.page_for_offset(offset),
                    meta=doc.meta,
                )
            )
            ordinal += 1
        return chunks


class RecursiveChunker(BaseChunker):
    strategy = ChunkStrategy.RECURSIVE

    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        separators: Sequence[str] = CIRCULAR_SEPARATORS,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=list(separators),
            length_function=len,
        )

    def _split(self, text: str) -> list[str]:
        return self._splitter.split_text(text)


def get_chunker(
    strategy: ChunkStrategy, *, chunk_size: int, chunk_overlap: int
) -> BaseChunker:
    if strategy is ChunkStrategy.RECURSIVE:
        return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raise NotImplementedError(
        f"chunking strategy {strategy.value!r} lands at M5 with its eval comparison"
    )
