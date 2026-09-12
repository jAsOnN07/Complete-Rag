"""Chunking strategies.

All three are kept side by side so the eval comparison is reproducible:

* ``recursive`` (default) - LangChain's RecursiveCharacterTextSplitter with
  separators tuned for numbered regulatory clauses.
* ``fixed``     - character windows with overlap, no separator awareness. The
  naive baseline the other two are measured against.
* ``semantic``  - sentence-level embeddings; a chunk boundary is placed where
  the cosine distance between adjacent sentences jumps above a percentile.
  Needs an embedder, which is why every chunker exposes an async ``chunk``.

Shared rules live in ``BaseChunker._finalise``: deterministic ids that encode
the strategy, page attribution from character offsets, full document metadata
on every chunk so citations need no second lookup.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Any, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.models import Chunk, ChunkStrategy, RawDocument, make_chunk_id

# Circulars are numbered clauses ("2.", "3.1") separated by newlines, so line
# breaks are a better split point than sentence punctuation, which appears
# inside statutory references like "Act, 1999." and "June 01, 2000."
CIRCULAR_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "; ", " ", "")

# Sentence boundary: terminal punctuation followed by whitespace and an
# uppercase letter, digit-clause marker or quote. "No." and "vs." style
# abbreviations are protected. Newlines always break.
_SENTENCE_RE = re.compile(
    r"(?<!\bNo)(?<!\bvs)(?<!\bviz)(?<!\bi\.e)(?<!\be\.g)(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+"
)


# Anything shorter than this is a clause marker ("2."), a stray token or a
# heading fragment, not a sentence; it is glued onto the sentence that follows.
MIN_SENTENCE_CHARS = 15
# A semantic group smaller than this is merged into its predecessor so the
# corpus never carries near-empty chunks.
MIN_CHUNK_CHARS = 60


def split_sentences(text: str) -> list[str]:
    raw = [s.strip() for s in _SENTENCE_RE.split(text) if s and s.strip()]
    out: list[str] = []
    carry = ""
    for s in raw:
        if carry:
            s = f"{carry} {s}"
            carry = ""
        if len(s) < MIN_SENTENCE_CHARS:
            carry = s
            continue
        out.append(s)
    if carry:
        if out:
            out[-1] = f"{out[-1]} {carry}"
        else:
            out.append(carry)
    return out


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


class FixedChunker(BaseChunker):
    """Sliding character windows. Splits mid-word by design - it is the baseline."""

    strategy = ChunkStrategy.FIXED

    def _split(self, text: str) -> list[str]:
        step = self.chunk_size - self.chunk_overlap
        return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[idx]


class SemanticChunker(BaseChunker):
    """Breaks where adjacent-sentence embedding distance jumps.

    Sentences are embedded once per document; a boundary is placed wherever the
    cosine distance between neighbours exceeds ``breakpoint_percentile`` of all
    such distances in the document. Groups that still exceed ``chunk_size`` are
    packed greedily so no chunk grows unbounded.
    """

    strategy = ChunkStrategy.SEMANTIC

    def __init__(
        self,
        *,
        embedder: Any,
        chunk_size: int,
        chunk_overlap: int = 0,
        breakpoint_percentile: float = 90.0,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile

    @property
    def params_hash(self) -> str:
        payload = (
            f"{self.strategy.value}:{self.chunk_size}:{self.chunk_overlap}:"
            f"{self.breakpoint_percentile}:{getattr(self._embedder, 'model_id', '?')}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    async def chunk(self, doc: RawDocument) -> list[Chunk]:
        text = doc.text
        if not text.strip():
            return []
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return self._finalise(doc, [text.strip()])

        vectors = await self._embedder.embed_documents(sentences)
        distances = [
            1.0 - _cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)
        ]
        cutoff = _percentile(distances, self.breakpoint_percentile)
        breaks = {i for i, d in enumerate(distances) if d > cutoff}

        groups: list[list[str]] = [[]]
        for i, sentence in enumerate(sentences):
            groups[-1].append(sentence)
            if i in breaks:
                groups.append([])

        pieces: list[str] = []
        for group in groups:
            pieces.extend(self._pack(group))
        return self._finalise(doc, self._merge_fragments(pieces))

    def _merge_fragments(self, pieces: Sequence[str]) -> list[str]:
        merged: list[str] = []
        for piece in pieces:
            if merged and len(piece) < MIN_CHUNK_CHARS and len(merged[-1]) + len(piece) + 1 <= self.chunk_size + MIN_CHUNK_CHARS:
                merged[-1] = f"{merged[-1]} {piece}"
            else:
                merged.append(piece)
        if len(merged) > 1 and len(merged[0]) < MIN_CHUNK_CHARS:
            merged[1] = f"{merged[0]} {merged[1]}"
            merged.pop(0)
        return merged

    def _pack(self, sentences: Sequence[str]) -> list[str]:
        """Greedy pack so a semantic group never exceeds chunk_size."""
        out: list[str] = []
        current: list[str] = []
        size = 0
        for s in sentences:
            if current and size + len(s) + 1 > self.chunk_size:
                out.append(" ".join(current))
                current, size = [], 0
            if len(s) > self.chunk_size:
                # A single oversized sentence is split at the size limit.
                for i in range(0, len(s), self.chunk_size):
                    out.append(s[i : i + self.chunk_size])
                continue
            current.append(s)
            size += len(s) + 1
        if current:
            out.append(" ".join(current))
        return out


def get_chunker(
    strategy: ChunkStrategy,
    *,
    chunk_size: int,
    chunk_overlap: int,
    embedder: Any | None = None,
    breakpoint_percentile: float = 90.0,
) -> BaseChunker:
    if strategy is ChunkStrategy.RECURSIVE:
        return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if strategy is ChunkStrategy.FIXED:
        return FixedChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if embedder is None:
        raise ValueError("semantic chunking needs an embedder")
    return SemanticChunker(
        embedder=embedder,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        breakpoint_percentile=breakpoint_percentile,
    )
