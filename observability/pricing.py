"""List prices for cost-per-query accounting in Langfuse.

Langfuse prices models it recognises by name, but not everything here reaches
it under a name it knows (Portkey-routed Gemini, Cohere embed and rerank, which
are not generation spans at all). Every provider span therefore carries explicit
``cost_details`` so cost-per-query in a trace is never silently zero.

These are **list prices** in USD. On a free tier the marginal cost is $0; the
list price is still the honest number to report, because it is what the same
traffic costs the moment the tier changes. Verify against the provider's pricing
page before quoting; last checked 2026-09-15.
"""

from __future__ import annotations

from typing import Any

# USD per 1M tokens: (input, output)
LLM_PRICES_PER_M: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-3.1-flash-lite": (0.10, 0.40),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "gpt-oss-120b": (0.15, 0.60),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# USD per 1M input tokens
EMBED_PRICES_PER_M: dict[str, float] = {
    "embed-v4.0": 0.10,
    "embed-english-v3.0": 0.10,
    "amazon.titan-embed-text-v2:0": 0.02,
}

# USD per search (one query + up to 100 documents)
RERANK_PRICES_PER_SEARCH: dict[str, float] = {
    "rerank-v3.5": 0.002,
    "rerank-english-v3.0": 0.002,
}


def _match(table: dict[str, Any], model: str | None) -> Any | None:
    if not model:
        return None
    if model in table:
        return table[model]
    tail = model.split("/")[-1]
    return table.get(tail)


def llm_cost(model: str | None, input_tokens: int, output_tokens: int) -> dict[str, float] | None:
    prices = _match(LLM_PRICES_PER_M, model)
    if prices is None:
        return None
    inp = input_tokens / 1_000_000 * prices[0]
    out = output_tokens / 1_000_000 * prices[1]
    return {"input": round(inp, 8), "output": round(out, 8), "total": round(inp + out, 8)}


def embed_cost(model: str | None, input_tokens: int) -> dict[str, float] | None:
    price = _match(EMBED_PRICES_PER_M, model)
    if price is None:
        return None
    total = input_tokens / 1_000_000 * price
    return {"input": round(total, 8), "total": round(total, 8)}


def rerank_cost(model: str | None, searches: int) -> dict[str, float] | None:
    price = _match(RERANK_PRICES_PER_SEARCH, model)
    if price is None:
        return None
    total = searches * price
    return {"total": round(total, 8)}
