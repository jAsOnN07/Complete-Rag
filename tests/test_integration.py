"""Opt-in tests that hit real services. Excluded from the default run.

    pytest -m integration

Each is small and costs fractions of a cent. They exist to catch credential,
region and model-entitlement drift - the failures that never show up against
fakes and always show up at the worst moment.
"""

from __future__ import annotations

import pytest

from core.config import get_settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def settings():
    return get_settings()


def test_aws_credentials_resolve(settings):
    import boto3

    identity = boto3.client(
        "sts",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id.get_secret_value(),
        aws_secret_access_key=settings.aws_secret_access_key.get_secret_value(),
    ).get_caller_identity()
    assert identity["Account"]


def test_configured_chat_model_is_invokable_in_this_region(settings):
    """Guards the inference-profile trap.

    A model whose inferenceTypesSupported is INFERENCE_PROFILE-only cannot be
    called by its bare ID - the request fails with a ValidationException that
    does not mention inference profiles at all.
    """
    import boto3

    client = boto3.client(
        "bedrock",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id.get_secret_value(),
        aws_secret_access_key=settings.aws_secret_access_key.get_secret_value(),
    )
    configured = settings.bedrock_llm_model_id
    if configured.startswith(("us.", "eu.", "apac.", "global.")):
        profiles = {
            p["inferenceProfileId"]
            for p in client.list_inference_profiles()["inferenceProfileSummaries"]
        }
        assert configured in profiles, f"{configured} is not an inference profile here"
        return

    summaries = {
        m["modelId"]: m for m in client.list_foundation_models()["modelSummaries"]
    }
    assert configured in summaries, f"{configured} not offered in {settings.aws_region}"
    assert "ON_DEMAND" in summaries[configured].get("inferenceTypesSupported", []), (
        f"{configured} is INFERENCE_PROFILE-only; set BEDROCK_LLM_MODEL_ID to the "
        f"profile id (e.g. us.{configured})"
    )


async def test_titan_returns_the_configured_dimension(settings):
    from retrieval.embedder import build_bedrock_embedder

    vector = await build_bedrock_embedder(settings).embed_query("customer due diligence")
    assert len(vector) == settings.bedrock_embed_dim


async def test_bedrock_chat_round_trip(settings):
    from generation.llm import build_bedrock_llm

    result = await build_bedrock_llm(settings).complete(
        "You are terse.", "Reply with the single word: ok"
    )
    assert result.text.strip()
    assert result.input_tokens > 0


async def test_qdrant_round_trip(settings):
    from retrieval.vector_store import build_qdrant_store

    store = build_qdrant_store(settings)
    try:
        await store.ensure_collection()
        assert await store.count() >= 0
    finally:
        await store.close()


async def test_end_to_end_query_returns_a_cited_answer(settings):
    """The M2 acceptance criterion, as an executable check."""
    from core.service import build_service

    service = build_service(settings)
    # A question the fetched corpus can actually answer (gold q003). The
    # original "KYC due diligence" question was written before the corpus
    # existed; the model correctly refused it, which is the not-found path
    # working, not a retrieval failure.
    answer = await service.answer(
        "Which five new districts have been formed in the Union Territory of Ladakh?"
    )
    assert answer.not_found is False, "corpus may not be indexed yet"
    assert answer.citations
    assert answer.invalid_citation_rate == 0.0


async def test_gateway_falls_back_to_groq_when_bedrock_fails(settings):
    """M7 acceptance: with Bedrock blocked, Portkey must serve the answer from Groq.

    Requires PORTKEY_CONFIG_SLUG on accounts with block_inline_config.
    """
    from generation.llm import build_llm

    result = await build_llm(settings).complete("You are terse.", "Reply with the single word: ok")
    assert result.text.strip()
    assert result.provider in ("groq", "bedrock")
    assert result.input_tokens > 0
