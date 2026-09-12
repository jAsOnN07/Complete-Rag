"""Protocols for every external collaborator.

These exist so the pipeline can be exercised end to end with fakes - no AWS, no
Qdrant server, no spend - and so the reranker and LLM backends stay genuinely
swappable rather than swappable in principle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator, Protocol, Sequence, runtime_checkable

from core.models import Answer, Chunk, RawDocument, ScoredChunk


@runtime_checkable
class Loader(Protocol):
    async def load(
        self, path: Path, manifest_row: dict[str, Any] | None = None
    ) -> RawDocument: ...


@runtime_checkable
class Chunker(Protocol):
    @property
    def strategy(self) -> Any: ...

    @property
    def params_hash(self) -> str: ...

    async def chunk(self, doc: RawDocument) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def dim(self) -> int: ...

    async def embed_query(self, text: str) -> list[float]: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    async def ensure_collection(self) -> None: ...

    async def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> int: ...

    async def search_dense(
        self, vector: Sequence[float], k: int
    ) -> list[ScoredChunk]: ...

    async def count(self) -> int: ...


@runtime_checkable
class Reranker(Protocol):
    @property
    def backend(self) -> str: ...

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class LlmClient(Protocol):
    async def complete(self, system: str, user: str) -> "LlmResult": ...

    def stream(self, system: str, user: str) -> AsyncIterator["LlmDelta"]: ...


@runtime_checkable
class InputGuard(Protocol):
    async def check(self, question: str) -> Any: ...


@runtime_checkable
class OutputGuard(Protocol):
    async def check(
        self, *, question: str, answer: str, grounding_sources: Sequence[str]
    ) -> Any: ...


class LlmDelta:
    """One streamed increment. The final delta carries usage and done=True."""

    def __init__(
        self,
        text: str = "",
        *,
        done: bool = False,
        model_id: str | None = None,
        provider: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        stop_reason: str | None = None,
    ) -> None:
        self.text = text
        self.done = done
        self.model_id = model_id
        self.provider = provider
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.stop_reason = stop_reason


class LlmResult:
    """Plain container so ports.py stays import-light."""

    def __init__(
        self,
        text: str,
        *,
        model_id: str,
        provider: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        stop_reason: str | None = None,
    ) -> None:
        self.text = text
        self.model_id = model_id
        self.provider = provider
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.stop_reason = stop_reason

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"LlmResult(provider={self.provider!r}, model_id={self.model_id!r}, "
            f"tokens={self.input_tokens}/{self.output_tokens})"
        )


__all__ = [
    "Answer",
    "Chunker",
    "Embedder",
    "InputGuard",
    "LlmClient",
    "LlmDelta",
    "LlmResult",
    "Loader",
    "OutputGuard",
    "Reranker",
    "VectorStore",
]
