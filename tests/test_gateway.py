"""Gateway config is pure; the client is tested against a recorded response shape."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.config import Settings
from core.errors import UpstreamServiceError
from generation.gateway import (
    FALLBACK_STATUS_CODES,
    fallback_config,
    provider_from_model,
    target_model,
)
from generation.llm import PortkeyLlmClient, build_llm
from observability.tracing import RecordingTracer


@pytest.fixture
def settings() -> Settings:
    return Settings(
        portkey_api_key="pk-test",
        llm_primary="bedrock",
        portkey_bedrock_provider="@aws",
        portkey_anthropic_provider="@anthropic",
        portkey_groq_provider="@groq",
        bedrock_llm_model_id="us.anthropic.claude-sonnet-5",
        anthropic_model_id="claude-sonnet-5",
        groq_model_id="openai/gpt-oss-120b",
        groq_reasoning_effort="low",
    )


# ---- config -----------------------------------------------------------------


def test_target_model_uses_catalog_form():
    assert target_model("@groq", "openai/gpt-oss-120b") == "@groq/openai/gpt-oss-120b"
    assert target_model("groq", "x") == "@groq/x"


def test_fallback_config_is_bedrock_then_groq(settings):
    cfg = fallback_config(settings)
    assert cfg["strategy"]["mode"] == "fallback"
    models = [t["override_params"]["model"] for t in cfg["targets"]]
    assert models == ["@aws/us.anthropic.claude-sonnet-5", "@groq/openai/gpt-oss-120b"]


def test_fallback_fires_on_4xx_not_just_5xx(settings):
    """Bedrock's entitlement failure is a 400; Groq's retired-model error a 404."""
    codes = fallback_config(settings)["strategy"]["on_status_codes"]
    assert 400 in codes and 404 in codes and 503 in codes
    assert tuple(codes) == FALLBACK_STATUS_CODES


def test_reasoning_effort_is_scoped_to_the_groq_target_only(settings):
    targets = fallback_config(settings)["targets"]
    assert "reasoning_effort" not in targets[0]["override_params"]
    assert targets[1]["override_params"]["reasoning_effort"] == "low"


def test_reasoning_effort_omitted_when_unset(settings):
    settings = settings.model_copy(update={"groq_reasoning_effort": None})
    assert "reasoning_effort" not in fallback_config(settings)["targets"][1]["override_params"]


def test_anthropic_primary_uses_bare_claude_id(settings):
    s = settings.model_copy(update={"llm_primary": "anthropic"})
    models = [t["override_params"]["model"] for t in fallback_config(s)["targets"]]
    assert models == ["@anthropic/claude-sonnet-5", "@groq/openai/gpt-oss-120b"]


@pytest.mark.parametrize(
    "served,expected",
    [
        ("us.anthropic.claude-sonnet-5", "bedrock"),
        ("anthropic.claude-sonnet-5", "bedrock"),
        ("claude-sonnet-5", "anthropic"),
        ("claude-opus-5", "anthropic"),
        ("openai/gpt-oss-120b", "groq"),
        ("gpt-oss-120b", "groq"),
        (None, "unknown"),
        ("some-other-model", "unknown"),
    ],
)
def test_provider_is_derived_from_served_model(settings, served, expected):
    assert provider_from_model(served, settings) == expected


# ---- client -----------------------------------------------------------------


def make_response(*, model: str, text: str, finish: str = "stop", pt: int = 85, ct: int = 17):
    """Shape recorded from a real Portkey/Groq round trip."""
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=ct),
    )


class FakePortkey:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._response


async def test_complete_reports_the_provider_that_actually_served(settings):
    fake = FakePortkey(make_response(model="openai/gpt-oss-120b", text="ok [1]"))
    client = PortkeyLlmClient(client=fake, settings=settings, tracer=RecordingTracer())
    result = await client.complete("sys", "user")
    assert result.text == "ok [1]"
    assert result.provider == "groq"
    assert result.model_id == "openai/gpt-oss-120b"
    assert (result.input_tokens, result.output_tokens) == (85, 17)
    assert result.stop_reason == "stop"


async def test_complete_sends_system_and_user_messages(settings):
    fake = FakePortkey(make_response(model="x", text="y"))
    await PortkeyLlmClient(client=fake, settings=settings, tracer=RecordingTracer()).complete(
        "SYSTEM", "USER"
    )
    messages = fake.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": "SYSTEM"}
    assert messages[1] == {"role": "user", "content": "USER"}
    assert fake.calls[0]["max_tokens"] == settings.llm_max_tokens


async def test_no_model_is_passed_when_the_config_owns_routing(settings):
    """The Portkey config decides the model; passing one would override the fallback."""
    fake = FakePortkey(make_response(model="x", text="y"))
    await PortkeyLlmClient(client=fake, settings=settings, tracer=RecordingTracer()).complete("s", "u")
    assert "model" not in fake.calls[0]


async def test_empty_content_is_reported_not_crashed(settings):
    """A reasoning model that exhausts max_tokens returns '' with finish=length."""
    fake = FakePortkey(make_response(model="openai/gpt-oss-120b", text="", finish="length"))
    result = await PortkeyLlmClient(client=fake, settings=settings, tracer=RecordingTracer()).complete("s", "u")
    assert result.text == ""
    assert result.stop_reason == "length"


async def test_gateway_failure_becomes_a_typed_upstream_error(settings):
    fake = FakePortkey(error=RuntimeError("Error code: 400 - inline_config_blocked"))
    client = PortkeyLlmClient(client=fake, settings=settings, tracer=RecordingTracer())
    with pytest.raises(UpstreamServiceError) as exc:
        await client.complete("s", "u")
    assert exc.value.provider == "portkey"
    assert "inline_config_blocked" in str(exc.value)


async def test_generation_is_instrumented(settings):
    tracer = RecordingTracer()
    fake = FakePortkey(make_response(model="openai/gpt-oss-120b", text="ok"))
    await PortkeyLlmClient(client=fake, settings=settings, tracer=tracer).complete("s", "u")
    assert ("llm.answer", "generation") in tracer.spans
    assert any(u.get("usage_details") == {"input": 85, "output": 17} for u in tracer.updates)


def test_factory_honours_llm_backend(settings):
    assert isinstance(build_llm(settings.model_copy(update={"llm_backend": "portkey"})), PortkeyLlmClient)
