"""Fakes for offline, zero-cost end-to-end testing.

Fakes rather than mock.patch: patching boto3 internals couples tests to library
internals and breaks on every SDK bump, whereas these implement the same
Protocols the real components do.
"""

from __future__ import annotations

import hashlib
import math
from typing import AsyncIterator, Sequence

from core.ports import LlmDelta, LlmResult


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

    async def stream(self, system: str, user: str) -> AsyncIterator[LlmDelta]:
        """Word-by-word deltas, then a final usage delta - the shape real streams have."""
        result = await self.complete(system, user)
        words = result.text.split(" ")
        for i, w in enumerate(words):
            yield LlmDelta(w if i == len(words) - 1 else w + " ")
        yield LlmDelta(
            done=True, model_id=result.model_id, provider=result.provider,
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            stop_reason=result.stop_reason,
        )


class ExplodingLlm:
    """Fails if called. Proves the not-found path never reaches the model."""

    async def complete(self, system: str, user: str) -> LlmResult:
        raise AssertionError("LLM must not be called when nothing clears the threshold")

    async def stream(self, system: str, user: str) -> AsyncIterator[LlmDelta]:
        raise AssertionError("LLM must not be called")
        yield LlmDelta()  # pragma: no cover


class FakeInputGuard:
    """Blocks when the question contains `trigger`."""

    def __init__(self, trigger: str = "IGNORE PREVIOUS", kind: str = "prompt_injection") -> None:
        self.trigger = trigger
        self.kind = kind
        self.calls: list[str] = []

    async def check(self, question: str):
        from guards.input_guards import InputGuardResult, Violation

        self.calls.append(question)
        if self.trigger in question:
            return InputGuardResult(
                allowed=False, violations=[Violation(kind=self.kind, detail="fake")],
                injection_score=0.99,
            )
        return InputGuardResult(allowed=True, injection_score=0.01)


class FakeOutputGuard:
    """Scriptable verdict for the output path."""

    def __init__(self, *, blocked: bool = False, grounded: bool | None = True,
                 redacted_text: str | None = None, reasons: list[str] | None = None) -> None:
        self.blocked = blocked
        self.grounded = grounded
        self.redacted_text = redacted_text
        self.reasons = reasons or (["grounding 0.10 below threshold 0.70"] if blocked else [])
        self.calls: list[dict] = []
        self.window_calls: list[str] = []

    async def check(self, *, question: str, answer: str, grounding_sources):
        from guards.output_guards import OutputGuardResult

        self.calls.append({"question": question, "answer": answer, "sources": list(grounding_sources)})
        return OutputGuardResult(
            action="GUARDRAIL_INTERVENED" if (self.blocked or self.redacted_text) else "NONE",
            text=self.redacted_text or answer, blocked=self.blocked, reasons=self.reasons,
            grounded=self.grounded, grounding_score=0.9 if self.grounded else 0.1,
            grounding_threshold=0.7, pii_redacted=["EMAIL"] if self.redacted_text else [],
        )

    async def check_window(self, window: str):
        from guards.output_guards import OutputGuardResult

        self.window_calls.append(window)
        return OutputGuardResult(text=window, blocked=self.blocked, reasons=self.reasons)
