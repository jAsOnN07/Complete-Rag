"""Input guardrails: length, language, PII, prompt injection.

Runs before every gateway call regardless of provider. Ordered cheap-to-
expensive: the first three checks are pure and cost nothing, so a rejected
input never reaches the network; the injection classifier is Groq's hosted
prompt-guard model, called directly (not through Portkey) so the guard never
depends on the gateway it protects.

Each check is a plain function so it can be tested in isolation and composed
by Guardrails AI (see `guardrails_ai.py`); this module has no framework
dependency of its own.

Failure policy: if the classifier cannot be reached the guard fails closed by
default. A guard that silently waves requests through when it is down is not a
guard.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from pydantic import BaseModel, Field

from observability.tracing import Tracer, get_tracer

# ---- pure checks --------------------------------------------------------------

_NON_LATIN_LETTER = re.compile(r"[^\W\d_a-zA-ZÀ-ɏ]")
_LATIN_LETTER = re.compile(r"[a-zA-ZÀ-ɏ]")
# Markers that are common in the languages a regulatory Q&A is most likely to
# receive by mistake and are essentially absent from English.
_FOREIGN_MARKERS = re.compile(r"[¿¡ñçßœ]|\b(?:los|las|les|des|der|die|das|und|für|qué|cuáles|est|sont|une|avec|para|como|sobre)\b", re.IGNORECASE)
_ENGLISH_STOPWORDS = frozenset(
    "the a an of for to in on at by is are was were be been what which who whom "
    "when where how why does do did can could should would shall must may might "
    "under with from that this these those it its and or not any all".split()
)


def check_length(text: str, *, max_chars: int, min_chars: int = 3) -> str | None:
    stripped = text.strip()
    if len(stripped) < min_chars:
        return f"question is too short (min {min_chars} chars)"
    if len(stripped) > max_chars:
        return f"question is too long ({len(stripped)} > {max_chars} chars)"
    return None


def check_language(text: str) -> str | None:
    """Reject text that shows positive evidence of not being English.

    Deliberately heuristic and transparent: statistical detectors are flaky on
    short, identifier-heavy queries ("RBI/2026-27/253 amendments?"), which are
    exactly what this system receives. Absence of English stopwords is never
    grounds for rejection on its own.
    """
    letters = _LATIN_LETTER.findall(text)
    non_latin = _NON_LATIN_LETTER.findall(text)
    if non_latin and len(non_latin) / max(len(letters) + len(non_latin), 1) > 0.3:
        return "non-Latin script detected; only English questions are supported"
    words = {w.lower() for w in re.findall(r"[a-zA-ZÀ-ɏ']+", text)}
    if _FOREIGN_MARKERS.search(text) and not (words & _ENGLISH_STOPWORDS):
        return "question does not appear to be in English"
    return None


class PiiHit(BaseModel):
    kind: str
    match: str
    masked: str


_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
    ("AADHAAR", re.compile(r"\b[2-9][0-9]{3}[\s-]?[0-9]{4}[\s-]?[0-9]{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"(?:\+91[\s-]?)?\b[6-9][0-9]{4}[\s-]?[0-9]{5}\b")),
    # Bank account numbers: 11-16 digits. Circular ids and dates never reach
    # that length as a bare digit run, so this does not fire on them.
    ("ACCOUNT_NUMBER", re.compile(r"\b[0-9]{11,16}\b")),
)


def _mask(value: str) -> str:
    compact = re.sub(r"\s", "", value)
    if len(compact) <= 4:
        return "*" * len(compact)
    return "*" * (len(compact) - 4) + compact[-4:]


def find_pii(text: str) -> list[PiiHit]:
    hits: list[PiiHit] = []
    taken: list[tuple[int, int]] = []
    for kind, pattern in _PII_PATTERNS:
        for m in pattern.finditer(text):
            span = m.span()
            if any(a <= span[0] < b or a < span[1] <= b for a, b in taken):
                continue
            taken.append(span)
            hits.append(PiiHit(kind=kind, match=m.group(0), masked=_mask(m.group(0))))
    return hits


# ---- injection classifier ---------------------------------------------------


class InjectionClassifier(Protocol):
    async def score(self, text: str) -> float: ...

    def score_sync(self, text: str) -> float: ...


class GroqPromptGuard:
    """Groq-hosted meta-llama/llama-prompt-guard-2-86m.

    Returns the injection probability as the completion text (measured:
    0.0004 on real questions, 0.98-0.9996 on injections, 0 completion tokens).
    """

    def __init__(self, *, api_key: str, model_id: str, timeout: float = 10.0) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._timeout = timeout

    _URL = "https://api.groq.com/openai/v1/chat/completions"

    def _payload(self, text: str) -> dict[str, Any]:
        return {
            "model": self._model_id,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 8,
        }

    @staticmethod
    def _parse(body: dict[str, Any]) -> float:
        return float(body["choices"][0]["message"]["content"].strip())

    async def score(self, text: str) -> float:
        import httpx

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(
                self._URL, headers={"Authorization": f"Bearer {self._api_key}"}, json=self._payload(text)
            )
            r.raise_for_status()
            return self._parse(r.json())

    def score_sync(self, text: str) -> float:
        """For the Guardrails AI validator, which is synchronous and runs under
        an OpenTelemetry span: starting an event loop there breaks OTel's
        context tokens, so this path must not touch asyncio at all."""
        import httpx

        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(
                self._URL, headers={"Authorization": f"Bearer {self._api_key}"}, json=self._payload(text)
            )
            r.raise_for_status()
            return self._parse(r.json())


# ---- guard -------------------------------------------------------------------


class Violation(BaseModel):
    kind: str
    detail: str


class InputGuardResult(BaseModel):
    allowed: bool
    violations: list[Violation] = Field(default_factory=list)
    injection_score: float | None = None
    degraded: bool = False
    pii: list[PiiHit] = Field(default_factory=list)

    @property
    def summary(self) -> str:
        return "; ".join(f"{v.kind}: {v.detail}" for v in self.violations) or "allowed"


class InputGuard:
    def __init__(
        self,
        *,
        injection: InjectionClassifier | None,
        max_chars: int,
        injection_threshold: float = 0.5,
        fail_open: bool = False,
        tracer: Tracer | None = None,
    ) -> None:
        self._injection = injection
        self._max_chars = max_chars
        self._threshold = injection_threshold
        self._fail_open = fail_open
        self._tracer = tracer or get_tracer()

    async def check(self, question: str) -> InputGuardResult:
        async with self._tracer.observe(
            "guardrails.input", as_type="guardrail", input={"chars": len(question)}
        ) as span:
            violations: list[Violation] = []
            if (msg := check_length(question, max_chars=self._max_chars)) is not None:
                violations.append(Violation(kind="length", detail=msg))
            elif (msg := check_language(question)) is not None:
                violations.append(Violation(kind="language", detail=msg))

            pii = find_pii(question) if not violations else []
            if pii:
                kinds = ", ".join(sorted({p.kind for p in pii}))
                violations.append(
                    Violation(kind="pii", detail=f"remove personal data before asking ({kinds})")
                )

            score: float | None = None
            degraded = False
            if not violations and self._injection is not None:
                try:
                    score = await self._injection.score(question)
                except Exception as exc:  # noqa: BLE001 - policy decides, not the exception
                    degraded = True
                    if not self._fail_open:
                        violations.append(
                            Violation(
                                kind="guard_unavailable",
                                detail=f"injection classifier unavailable: {type(exc).__name__}",
                            )
                        )
                else:
                    if score >= self._threshold:
                        violations.append(
                            Violation(
                                kind="prompt_injection",
                                detail=f"classifier score {score:.2f} >= {self._threshold}",
                            )
                        )
            elif not violations:
                degraded = True

            result = InputGuardResult(
                allowed=not violations,
                violations=violations,
                injection_score=score,
                degraded=degraded,
                pii=pii,
            )
            span.update(
                output={"allowed": result.allowed, "violations": [v.kind for v in violations]},
                metadata={"injection_score": score, "degraded": degraded},
            )
            return result


def build_input_guard(settings: Any = None) -> Any:
    if settings is None:
        from core.config import get_settings

        settings = get_settings()
    if not settings.input_guard_enabled:
        return None

    classifier: InjectionClassifier | None = None
    if settings.injection_backend == "groq_prompt_guard" and settings.groq_api_key:
        classifier = GroqPromptGuard(
            api_key=settings.groq_api_key.get_secret_value(),
            model_id=settings.injection_model_id,
        )
    kwargs = dict(
        injection=classifier,
        max_chars=settings.max_question_chars,
        injection_threshold=settings.injection_threshold,
        fail_open=settings.input_guard_fail_open,
    )
    if settings.input_guard_engine == "guardrails_ai":
        from guards.guardrails_ai import GuardrailsAiInputGuard

        return GuardrailsAiInputGuard(**kwargs)
    return InputGuard(**kwargs)
