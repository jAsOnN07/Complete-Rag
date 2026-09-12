"""Output guardrails via Bedrock's standalone ApplyGuardrail API.

ApplyGuardrail takes text and returns a verdict - no model, no inference. That
is what lets it wrap the Groq fallback path exactly as it wraps Bedrock: the
answer's origin is irrelevant to the check. Request and response shapes are
verified against the botocore service model rather than recalled.

Policy on the verdict: the guardrail's own `action` decides. A BLOCKED filter
of any kind (grounding, content, topic, PII-block) blocks the answer. An
ANONYMIZED PII action is not a block - the redacted text is returned instead.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from core.errors import UpstreamServiceError
from observability.tracing import Tracer, get_tracer

GuardAction = Literal["NONE", "GUARDRAIL_INTERVENED"]


class OutputGuardResult(BaseModel):
    action: GuardAction = "NONE"
    text: str
    blocked: bool = False
    reasons: list[str] = Field(default_factory=list)
    grounded: bool | None = None
    grounding_score: float | None = None
    grounding_threshold: float | None = None
    relevance_score: float | None = None
    pii_redacted: list[str] = Field(default_factory=list)
    content_flags: list[str] = Field(default_factory=list)
    denied_topics: list[str] = Field(default_factory=list)
    units: dict[str, int] = Field(default_factory=dict)


class NoopOutputGuard:
    backend = "none"

    async def check(
        self, *, question: str, answer: str, grounding_sources: Sequence[str]
    ) -> OutputGuardResult:
        return OutputGuardResult(text=answer)

    async def check_window(self, window: str) -> OutputGuardResult:
        return OutputGuardResult(text=window)


def _text_block(text: str, qualifier: str) -> dict[str, Any]:
    return {"text": {"text": text, "qualifiers": [qualifier]}}


def parse_assessment(response: dict[str, Any], fallback_text: str) -> OutputGuardResult:
    """Turn an ApplyGuardrail response into a typed result."""
    outputs = response.get("outputs") or []
    text = outputs[0].get("text", fallback_text) if outputs else fallback_text
    result = OutputGuardResult(
        action=response.get("action", "NONE"),
        text=text,
        units={k: int(v) for k, v in (response.get("usage") or {}).items() if isinstance(v, int)},
    )

    for assessment in response.get("assessments") or []:
        for f in (assessment.get("contextualGroundingPolicy") or {}).get("filters", []):
            if f.get("type") == "GROUNDING":
                result.grounding_score = f.get("score")
                result.grounding_threshold = f.get("threshold")
                result.grounded = f.get("action") != "BLOCKED" and not f.get("detected", False)
                if f.get("action") == "BLOCKED":
                    result.reasons.append(
                        f"grounding {f.get('score', 0):.2f} below threshold {f.get('threshold', 0):.2f}"
                    )
            elif f.get("type") == "RELEVANCE":
                result.relevance_score = f.get("score")
                if f.get("action") == "BLOCKED":
                    result.reasons.append("answer not relevant to the question")

        sensitive = assessment.get("sensitiveInformationPolicy") or {}
        for e in sensitive.get("piiEntities", []) + sensitive.get("regexes", []):
            label = e.get("type") or e.get("name") or "PII"
            if e.get("action") == "ANONYMIZED":
                result.pii_redacted.append(label)
            elif e.get("action") == "BLOCKED":
                result.reasons.append(f"blocked PII: {label}")

        for f in (assessment.get("contentPolicy") or {}).get("filters", []):
            if f.get("detected") or f.get("action") == "BLOCKED":
                result.content_flags.append(f.get("type", "CONTENT"))
                if f.get("action") == "BLOCKED":
                    result.reasons.append(f"content policy: {f.get('type')}")

        for t in (assessment.get("topicPolicy") or {}).get("topics", []):
            if t.get("detected") or t.get("action") == "BLOCKED":
                result.denied_topics.append(t.get("name", "topic"))
                if t.get("action") == "BLOCKED":
                    result.reasons.append(f"denied topic: {t.get('name')}")

    result.blocked = bool(result.reasons)
    return result


class BedrockOutputGuard:
    backend = "bedrock"

    def __init__(
        self,
        *,
        client: Any,
        guardrail_id: str,
        version: str = "DRAFT",
        output_scope: Literal["FULL", "INTERVENTIONS"] = "FULL",
        tracer: Tracer | None = None,
    ) -> None:
        self._client = client
        self._id = guardrail_id
        self._version = version
        self._scope = output_scope
        self._tracer = tracer or get_tracer()

    def _apply_sync(self, content: list[dict[str, Any]]) -> dict[str, Any]:
        return self._client.apply_guardrail(
            guardrailIdentifier=self._id,
            guardrailVersion=self._version,
            source="OUTPUT",
            content=content,
            outputScope=self._scope,
        )

    async def _apply(self, name: str, content: list[dict[str, Any]], fallback: str) -> OutputGuardResult:
        async with self._tracer.observe(
            name, as_type="guardrail", input={"blocks": len(content)}
        ) as span:
            try:
                response = await asyncio.to_thread(self._apply_sync, content)
            except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
                raise UpstreamServiceError(
                    "bedrock", "apply_guardrail", f"{type(exc).__name__}: {exc}", cause=exc
                ) from exc
            result = parse_assessment(response, fallback)
            span.update(
                output={
                    "action": result.action, "blocked": result.blocked,
                    "grounded": result.grounded, "grounding_score": result.grounding_score,
                    "reasons": result.reasons,
                },
                metadata={"guardrail_id": self._id, "units": result.units},
            )
            return result

    async def check(
        self, *, question: str, answer: str, grounding_sources: Sequence[str]
    ) -> OutputGuardResult:
        """Full check: grounding needs both grounding_source and query qualifiers present."""
        content = [_text_block(s, "grounding_source") for s in grounding_sources]
        content.append(_text_block(question, "query"))
        content.append(_text_block(answer, "guard_content"))
        return await self._apply("guardrails.output", content, answer)

    async def check_window(self, window: str) -> OutputGuardResult:
        """Streaming window: content and PII only. Grounding a half sentence is
        meaningless, and each window is billed per ~1k characters."""
        return await self._apply("guardrails.output.window", [_text_block(window, "guard_content")], window)


def build_output_guard(settings: Any = None) -> NoopOutputGuard | BedrockOutputGuard:
    if settings is None:
        from core.config import get_settings

        settings = get_settings()
    if settings.output_guard_backend == "none":
        return NoopOutputGuard()
    if not settings.bedrock_guardrail_id:
        raise RuntimeError("BEDROCK_GUARDRAIL_ID is required for OUTPUT_GUARD_BACKEND=bedrock")

    import boto3
    from botocore.config import Config

    client = boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region,
        config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
        aws_access_key_id=(
            settings.aws_access_key_id.get_secret_value() if settings.aws_access_key_id else None
        ),
        aws_secret_access_key=(
            settings.aws_secret_access_key.get_secret_value()
            if settings.aws_secret_access_key
            else None
        ),
    )
    return BedrockOutputGuard(
        client=client,
        guardrail_id=settings.bedrock_guardrail_id,
        version=settings.bedrock_guardrail_version,
        output_scope=settings.output_guard_scope,
    )
