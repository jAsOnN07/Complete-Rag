"""Portkey client configuration and routing.

Portkey owns routing, retries and the Bedrock -> Groq fallback. This is the
only module that knows what Portkey is; nothing else in the application
carries retry or fallback logic, per CLAUDE.md.

Two ways to supply the routing policy:

* ``PORTKEY_CONFIG_SLUG=pc-...`` - a config saved in the Portkey dashboard.
  Preferred: the policy is a versioned artifact rather than app code, and it
  is the only option on accounts with ``block_inline_config`` enabled.
* No slug - the same policy is sent inline. Works on accounts that allow it.

``fallback_config()`` produces the exact JSON to save in the dashboard, so the
two paths cannot drift.
"""

from __future__ import annotations

from typing import Any

# Bedrock's entitlement failure is a 400, and Groq's model-retired error is a
# 404. Portkey's default fallback codes are 5xx/429 only, so the fallback would
# never fire on either without listing them explicitly.
FALLBACK_STATUS_CODES: tuple[int, ...] = (400, 401, 403, 404, 408, 429, 500, 502, 503, 504)

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_BEDROCK = "bedrock"
PROVIDER_GOOGLE = "google"
PROVIDER_GROQ = "groq"
PROVIDER_UNKNOWN = "unknown"


def target_model(provider_slug: str, model_id: str) -> str:
    """Model Catalog form: ``@slug/model``."""
    slug = provider_slug if provider_slug.startswith("@") else f"@{provider_slug}"
    return f"{slug}/{model_id}"


def primary_target(settings: Any) -> str:
    return target_model(settings.llm_primary_provider, settings.llm_primary_model)


def fallback_config(settings: Any) -> dict[str, Any]:
    """Primary (Anthropic direct or Bedrock) then Groq, as a Portkey config document."""
    groq_overrides: dict[str, Any] = {
        "model": target_model(settings.portkey_groq_provider, settings.groq_model_id)
    }
    if settings.groq_reasoning_effort:
        # Reasoning models otherwise spend the whole budget thinking. Scoped to
        # the Groq target so Bedrock never sees a parameter it may reject.
        groq_overrides["reasoning_effort"] = settings.groq_reasoning_effort
    return {
        "strategy": {"mode": "fallback", "on_status_codes": list(FALLBACK_STATUS_CODES)},
        # Retries live here, not in application code. Groq's free tier is
        # 8k tokens/minute and answers 429 with "try again in ~10s".
        "retry": {"attempts": 3, "on_status_codes": [429, 500, 502, 503, 504]},
        "targets": [
            {"override_params": {"model": primary_target(settings)}},
            {"override_params": groq_overrides},
        ],
    }


def provider_from_model(served_model: str | None, settings: Any) -> str:
    """Which provider actually answered, derived from the model id Portkey returns.

    Portkey does not expose the served target index on the response object, so
    the model id is the reliable signal: Bedrock returns Anthropic ids, Groq
    returns its own catalogue names.
    """
    if not served_model:
        return PROVIDER_UNKNOWN
    name = served_model.lower()
    # Bedrock ids carry a vendor prefix (anthropic.claude-..., us.anthropic...);
    # the Claude API returns bare ids (claude-sonnet-5).
    if name.startswith(("anthropic.", "us.anthropic.", "eu.anthropic.", "global.anthropic.", "apac.anthropic.")):
        return PROVIDER_BEDROCK
    if name.startswith("claude"):
        return PROVIDER_ANTHROPIC
    if name.startswith("gemini"):
        return PROVIDER_GOOGLE
    groq_tail = settings.groq_model_id.lower().split("/")[-1]
    if groq_tail and groq_tail in name:
        return PROVIDER_GROQ
    return PROVIDER_UNKNOWN


def build_portkey_client(settings: Any) -> Any:
    from portkey_ai import AsyncPortkey

    if settings.portkey_api_key is None:
        raise RuntimeError("PORTKEY_API_KEY is not set")

    kwargs: dict[str, Any] = {
        "api_key": settings.portkey_api_key.get_secret_value(),
        "base_url": settings.portkey_base_url,
    }
    if settings.portkey_config_slug:
        kwargs["config"] = settings.portkey_config_slug
    else:
        kwargs["config"] = fallback_config(settings)
    return AsyncPortkey(**kwargs)


if __name__ == "__main__":  # pragma: no cover
    import json

    from core.config import get_settings

    print("Save this as a config in the Portkey dashboard and set PORTKEY_CONFIG_SLUG:\n")
    print(json.dumps(fallback_config(get_settings()), indent=2))
