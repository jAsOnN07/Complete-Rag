"""Fakes for offline, zero-cost end-to-end testing.

Fakes rather than mock.patch: patching boto3 internals couples tests to library
internals and breaks on every SDK bump, whereas these implement the same
Protocols the real components do.
"""

from __future__ import annotations

import hashlib
import math
from typing import AsyncIterator, Sequence

from core.ports import LlmResult


class FakeEmbedder:
    """Deterministic vectors seeded from token hashes.

    Not random: texts sharing tokens land near each other, so similarity is
    meaningful enough to exercise ranking and thresholds.
    """

    def __init__(self, dim: int = 64, model_id: str = "fake-embed") -> None:
        self._dim = dim
        self._model_id = model_id
        self.calls: list[str] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return self._vector(text)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return [self._vector(t) for t in texts]


class FakeLlm:
    """Scripted LLM. Set `response` to whatever the test needs to observe."""

    def __init__(
        self,
        response: str = "Banks must verify identity [1].",
        *,
        model_id: str = "fake-model",
        provider: str = "fake",
    ) -> None:
        self.response = response
        self.model_id = model_id
        self.provider = provider
        self.calls: list[tuple[str, str]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def complete(self, system: str, user: str) -> LlmResult:
        self.calls.append((system, user))
        return LlmResult(
            text=self.response,
            model_id=self.model_id,
            provider=self.provider,
            input_tokens=len(user) // 4,
            output_tokens=len(self.response) // 4,
            stop_reason="end_turn",
        )

    async def stream(self, system: str, user: str) -> AsyncIterator[str]:
        result = await self.complete(system, user)
        yield result.text


class ExplodingLlm:
    """Fails if called. Proves the not-found path never reaches the model."""

    async def complete(self, system: str, user: str) -> LlmResult:
        raise AssertionError("LLM must not be called when nothing clears the threshold")

    async def stream(self, system: str, user: str) -> AsyncIterator[str]:
        raise AssertionError("LLM must not be called")
        yield ""  # pragma: no cover
