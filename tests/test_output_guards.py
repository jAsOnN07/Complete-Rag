"""Bedrock ApplyGuardrail wrapper, tested against the response shape from the
botocore service model. No network."""

from __future__ import annotations

import pytest

from core.errors import UpstreamServiceError
from guards.output_guards import BedrockOutputGuard, NoopOutputGuard, build_output_guard
from observability.tracing import RecordingTracer


class FakeRuntime:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.calls: list[dict] = []

    def apply_guardrail(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._response


def response(*, action="NONE", text="answer text", grounding=None, pii=(), content=(), topics=()):
    assessment = {
        "contextualGroundingPolicy": {"filters": grounding or []},
        "sensitiveInformationPolicy": {"piiEntities": list(pii), "regexes": []},
        "contentPolicy": {"filters": list(content)},
        "topicPolicy": {"topics": list(topics)},
    }
    return {
        "action": action,
        "outputs": [{"text": text}],
        "assessments": [assessment],
        "usage": {"contextualGroundingPolicyUnits": 3, "sensitiveInformationPolicyUnits": 1,
                  "contentPolicyUnits": 1, "topicPolicyUnits": 1, "wordPolicyUnits": 0,
                  "sensitiveInformationPolicyFreeUnits": 0},
    }


def guard(fake) -> BedrockOutputGuard:
    return BedrockOutputGuard(client=fake, guardrail_id="gr-1", version="DRAFT", tracer=RecordingTracer())


async def test_request_carries_sources_query_and_answer_with_qualifiers():
    fake = FakeRuntime(response())
    await guard(fake).check(question="Q?", answer="A.", grounding_sources=["src one", "src two"])
    call = fake.calls[0]
    assert call["guardrailIdentifier"] == "gr-1" and call["guardrailVersion"] == "DRAFT"
    assert call["source"] == "OUTPUT"
    content = call["content"]
    assert content[0]["text"] == {"text": "src one", "qualifiers": ["grounding_source"]}
    assert content[1]["text"] == {"text": "src two", "qualifiers": ["grounding_source"]}
    assert content[2]["text"] == {"text": "Q?", "qualifiers": ["query"]}
    assert content[3]["text"] == {"text": "A.", "qualifiers": ["guard_content"]}


async def test_clean_answer_passes_through_grounded():
    fake = FakeRuntime(response(
        grounding=[{"type": "GROUNDING", "threshold": 0.7, "score": 0.93, "action": "NONE", "detected": False},
                   {"type": "RELEVANCE", "threshold": 0.5, "score": 0.88, "action": "NONE", "detected": False}],
    ))
    r = await guard(fake).check(question="Q", answer="A", grounding_sources=["s"])
    assert r.action == "NONE" and r.blocked is False
    assert r.grounded is True
    assert r.grounding_score == pytest.approx(0.93) and r.grounding_threshold == pytest.approx(0.7)
    assert r.relevance_score == pytest.approx(0.88)
    assert r.text == "answer text"


async def test_ungrounded_answer_is_blocked_by_the_grounding_filter():
    fake = FakeRuntime(response(
        action="GUARDRAIL_INTERVENED",
        text="Sorry, the model cannot answer this question.",
        grounding=[{"type": "GROUNDING", "threshold": 0.7, "score": 0.21, "action": "BLOCKED", "detected": True}],
    ))
    r = await guard(fake).check(question="Q", answer="Made up claim.", grounding_sources=["s"])
    assert r.blocked is True
    assert r.grounded is False
    assert "grounding" in r.reasons[0].lower()


async def test_pii_anonymised_is_not_a_block_but_text_is_redacted():
    fake = FakeRuntime(response(
        action="GUARDRAIL_INTERVENED",
        text="Contact {EMAIL} for details.",
        pii=[{"match": "a@b.com", "type": "EMAIL", "action": "ANONYMIZED", "detected": True}],
    ))
    r = await guard(fake).check(question="Q", answer="Contact a@b.com for details.", grounding_sources=["s"])
    assert r.blocked is False
    assert r.text == "Contact {EMAIL} for details."
    assert r.pii_redacted == ["EMAIL"]


async def test_content_and_topic_blocks_are_reported_with_reasons():
    fake = FakeRuntime(response(
        action="GUARDRAIL_INTERVENED", text="blocked",
        content=[{"type": "HATE", "confidence": "HIGH", "action": "BLOCKED", "detected": True}],
        topics=[{"name": "investment-advice", "type": "DENY", "action": "BLOCKED", "detected": True}],
    ))
    r = await guard(fake).check(question="Q", answer="x", grounding_sources=["s"])
    assert r.blocked is True
    assert r.content_flags == ["HATE"] and r.denied_topics == ["investment-advice"]
    assert len(r.reasons) == 2


async def test_window_check_sends_no_grounding_source():
    """Mid-stream windows are content/PII only; grounding a half sentence is meaningless and billed."""
    fake = FakeRuntime(response())
    await guard(fake).check_window("partial text so far")
    content = fake.calls[0]["content"]
    assert len(content) == 1
    assert content[0]["text"]["qualifiers"] == ["guard_content"]


async def test_units_are_recorded_for_cost_accounting():
    fake = FakeRuntime(response())
    r = await guard(fake).check(question="Q", answer="A", grounding_sources=["s"])
    assert r.units["contextualGroundingPolicyUnits"] == 3


async def test_failure_is_typed():
    fake = FakeRuntime(error=RuntimeError("ValidationException: Operation not allowed"))
    with pytest.raises(UpstreamServiceError) as exc:
        await guard(fake).check(question="Q", answer="A", grounding_sources=["s"])
    assert exc.value.provider == "bedrock" and exc.value.operation == "apply_guardrail"


async def test_guard_is_instrumented_as_guardrail():
    fake = FakeRuntime(response())
    g = guard(fake)
    await g.check(question="Q", answer="A", grounding_sources=["s"])
    assert ("guardrails.output", "guardrail") in g._tracer.spans


async def test_noop_guard_passes_everything_with_unknown_grounding():
    r = await NoopOutputGuard().check(question="Q", answer="A", grounding_sources=["s"])
    assert r.blocked is False and r.grounded is None and r.text == "A"


def test_factory():
    from core.config import Settings

    assert isinstance(build_output_guard(Settings(output_guard_backend="none")), NoopOutputGuard)
    with pytest.raises(RuntimeError):
        build_output_guard(Settings(output_guard_backend="bedrock", bedrock_guardrail_id=None))
