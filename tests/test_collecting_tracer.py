"""CollectingTracer: per-request span records so the API can describe its own
request. It wraps the real tracer; it must never change what reaches Langfuse."""

from __future__ import annotations

import asyncio

import pytest

from observability.tracing import CollectingTracer, NoOpTracer, RecordingTracer, current_spans, request_scope


async def test_records_spans_with_duration_and_merged_updates():
    inner = RecordingTracer()
    tracer = CollectingTracer(inner)
    with request_scope() as scope:
        async with tracer.observe("rag.query", as_type="chain") as root:
            async with tracer.observe("llm.complete", as_type="generation", model="m") as span:
                await asyncio.sleep(0.03)
                span.update(
                    usage_details={"input": 10, "output": 5},
                    cost_details={"total": 0.001},
                    metadata={"provider": "groq"},
                    output={"served_model": "gpt-oss"},
                )
            root.update(output={"status": "answered"})
        spans = scope.spans
    assert [s.name for s in spans] == ["rag.query", "llm.complete"]
    llm = spans[1]
    assert llm.type == "generation" and llm.depth == 1 and spans[0].depth == 0
    assert llm.duration_ms >= 5  # Windows timer granularity: sleep(0.03) may return a little early
    assert llm.tokens_in == 10 and llm.tokens_out == 5 and llm.cost_usd == pytest.approx(0.001)
    assert llm.provider == "groq" and llm.model == "m"
    assert llm.extra["served_model"] == "gpt-oss"
    # The wrapped tracer saw exactly the same spans and updates.
    assert inner.span_names == ["rag.query", "llm.complete"]
    assert any(u.get("cost_details") == {"total": 0.001} for u in inner.updates)


async def test_concurrent_requests_do_not_share_span_lists():
    tracer = CollectingTracer(NoOpTracer())

    async def one(name: str, delay: float) -> list[str]:
        with request_scope() as scope:
            async with tracer.observe(name):
                await asyncio.sleep(delay)
                async with tracer.observe(f"{name}.child"):
                    pass
            return [s.name for s in scope.spans]

    a, b = await asyncio.gather(one("a", 0.02), one("b", 0.0))
    assert a == ["a", "a.child"] and b == ["b", "b.child"]


async def test_spans_outside_a_request_scope_are_dropped_not_leaked():
    tracer = CollectingTracer(NoOpTracer())
    async with tracer.observe("orphan"):
        pass
    assert current_spans() is None
    with request_scope() as scope:
        assert scope.spans == []


def test_enabled_reflects_the_inner_tracer():
    assert CollectingTracer(NoOpTracer()).enabled is False

    class Real(NoOpTracer):
        enabled = True

    assert CollectingTracer(Real()).enabled is True


async def test_an_update_stamps_the_duration_so_far():
    """Streaming spans close late (generator finalisation); the last update
    must leave a usable duration behind for the response summary."""
    tracer = CollectingTracer(NoOpTracer())
    with request_scope() as scope:
        cm = tracer.observe("llm.answer", as_type="generation")
        handle = await cm.__aenter__()
        await asyncio.sleep(0.03)
        handle.update(usage_details={"input": 1, "output": 1})
        assert scope.spans[0].duration_ms >= 5  # before the span has exited
        await cm.__aexit__(None, None, None)
