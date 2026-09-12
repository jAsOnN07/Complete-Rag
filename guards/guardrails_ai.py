"""Guardrails AI composition of the input checks.

The checks themselves live in `input_guards.py` as plain functions; this
module wraps them as Guardrails AI validators and composes them with a
`Guard`. Two things about the framework that are not obvious:

* **It phones home by default.** A `Guard` registers an OpenTelemetry exporter
  to Guardrails' own endpoint. `configure(allow_metrics_collection=False)` is
  called on every Guard here and verified to stop the export; this is also a
  second TracerProvider, which would otherwise split traces from Langfuse.
* **A Guard runs every validator before it reports.** `OnFailAction.EXCEPTION`
  does not short-circuit the rest, so one Guard cannot express "do not call
  the classifier if length already failed". Two Guards can: the cheap checks
  run first, and the injection Guard runs only if they all pass - the same
  cheap-first order as the native guard, and a rejected input never costs a
  network call.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from guardrails import Guard, OnFailAction
from guardrails.validator_base import FailResult, PassResult, Validator, register_validator

from guards.input_guards import (
    InputGuardResult,
    Violation,
    check_language,
    check_length,
    find_pii,
)
from observability.tracing import Tracer, get_tracer

_EXC_PREFIX = "Validation failed for field with errors: "


@register_validator(name="rag/question_length", data_type="string")
class QuestionLength(Validator):
    def __init__(self, max_chars: int, on_fail: Any = None, **kwargs: Any) -> None:
        super().__init__(on_fail=on_fail, max_chars=max_chars, **kwargs)
        self.max_chars = max_chars

    def validate(self, value: Any, metadata: dict[str, Any]) -> Any:
        msg = check_length(str(value), max_chars=self.max_chars)
        return FailResult(error_message=f"length: {msg}") if msg else PassResult()


@register_validator(name="rag/english_only", data_type="string")
class EnglishOnly(Validator):
    def validate(self, value: Any, metadata: dict[str, Any]) -> Any:
        msg = check_language(str(value))
        return FailResult(error_message=f"language: {msg}") if msg else PassResult()


@register_validator(name="rag/no_pii", data_type="string")
class NoPii(Validator):
    def validate(self, value: Any, metadata: dict[str, Any]) -> Any:
        hits = find_pii(str(value))
        if not hits:
            return PassResult()
        kinds = ", ".join(sorted({h.kind for h in hits}))
        return FailResult(error_message=f"pii: remove personal data before asking ({kinds})")


@register_validator(name="rag/no_prompt_injection", data_type="string")
class NoPromptInjection(Validator):
    """Wraps a synchronous scorer. The async classifier is adapted in the guard."""

    def __init__(self, scorer: Any, threshold: float, on_fail: Any = None, **kwargs: Any) -> None:
        super().__init__(on_fail=on_fail, threshold=threshold, **kwargs)
        self._scorer = scorer
        self.threshold = threshold

    def validate(self, value: Any, metadata: dict[str, Any]) -> Any:
        score = float(self._scorer(str(value)))
        if score >= self.threshold:
            return FailResult(
                error_message=f"prompt_injection: classifier score {score:.2f} >= {self.threshold}"
            )
        return PassResult()


class GuardrailsAiInputGuard:
    """Same contract as `InputGuard.check`, composed through Guardrails AI."""

    def __init__(
        self,
        *,
        injection: Any | None,
        max_chars: int,
        injection_threshold: float = 0.5,
        fail_open: bool = False,
        tracer: Tracer | None = None,
    ) -> None:
        self._injection = injection
        self._fail_open = fail_open
        self._tracer = tracer or get_tracer()
        # Guardrails copies the metadata dict, so the score cannot travel back
        # through it; the scorer records it here instead, keyed by question.
        self._scores: dict[str, float] = {}
        self._lock = threading.Lock()

        self._cheap_guard = Guard().use(
            QuestionLength(max_chars=max_chars, on_fail=OnFailAction.NOOP),
            EnglishOnly(on_fail=OnFailAction.NOOP),
            NoPii(on_fail=OnFailAction.NOOP),
        )
        self._cheap_guard.configure(allow_metrics_collection=False)

        self._injection_guard: Guard | None = None
        if injection is not None:
            self._injection_guard = Guard().use(
                NoPromptInjection(
                    scorer=self._score_sync, threshold=injection_threshold, on_fail=OnFailAction.NOOP
                )
            )
            self._injection_guard.configure(allow_metrics_collection=False)

    @property
    def guards(self) -> list[Guard]:
        return [g for g in (self._cheap_guard, self._injection_guard) if g is not None]

    def _score_sync(self, text: str) -> float:
        """Validators are synchronous and run inside a Guardrails OTel span, so
        this must not start an event loop (OTel context tokens would detach in
        the wrong context). Classifiers expose a real sync path for this."""
        score = float(self._injection.score_sync(text))
        with self._lock:
            self._scores[text] = score
        return score

    def _take_score(self, text: str) -> float | None:
        with self._lock:
            return self._scores.pop(text, None)

    @staticmethod
    def _violations(outcome: Any) -> list[Violation]:
        if outcome.validation_passed:
            return []
        out: list[Violation] = []
        for summary in outcome.validation_summaries or []:
            reason = (summary.failure_reason or "validation failed").removeprefix(_EXC_PREFIX)
            kind, _, detail = reason.partition(": ")
            out.append(Violation(kind=kind or "input", detail=detail or reason))
        return out

    def _validate_sync(self, question: str) -> tuple[list[Violation], float | None, bool]:
        violations = self._violations(self._cheap_guard.validate(question))
        if violations or self._injection_guard is None:
            return violations, None, False

        try:
            outcome = self._injection_guard.validate(question)
        except Exception as exc:  # noqa: BLE001 - classifier transport failure
            if self._fail_open:
                return [], None, True
            return (
                [Violation(kind="guard_unavailable", detail=f"injection classifier unavailable: {type(exc).__name__}")],
                None,
                True,
            )
        return self._violations(outcome), self._take_score(question), False

    async def check(self, question: str) -> InputGuardResult:
        async with self._tracer.observe(
            "guardrails.input", as_type="guardrail", input={"chars": len(question), "engine": "guardrails_ai"}
        ) as span:
            violations, score, degraded = await asyncio.to_thread(self._validate_sync, question)
            if self._injection is None:
                degraded = True
            result = InputGuardResult(
                allowed=not violations, violations=violations, injection_score=score, degraded=degraded
            )
            span.update(
                output={"allowed": result.allowed, "violations": [v.kind for v in violations]},
                metadata={"injection_score": score, "degraded": degraded},
            )
            return result
