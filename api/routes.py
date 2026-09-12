"""HTTP routes.

`/query` is non-streaming and carries the full response contract - this is the
path RAGAS evaluates at M4. `/query/stream` arrives at M8 alongside the windowed
output guard, because a grounding check cannot be applied to text already sent.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, status

from api.deps import ServiceDep, SettingsDep
from api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
)

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse, tags=["ops"])
async def healthz(settings: SettingsDep) -> HealthResponse:
    """Liveness only - deliberately does not touch Qdrant or Bedrock."""
    return HealthResponse(
        status="ok",
        collection=settings.collection_name(),
        config_fingerprint=settings.fingerprint(),
    )


@router.get("/readyz", response_model=ReadyResponse, tags=["ops"])
async def readyz(service: ServiceDep, settings: SettingsDep) -> ReadyResponse:
    """Readiness: the collection must exist and hold points."""
    try:
        points = await service.count_points()
    except Exception as exc:  # noqa: BLE001 - reported, never raised to the probe
        return ReadyResponse(
            ready=False,
            collection=settings.collection_name(),
            detail=f"{type(exc).__name__}: {exc}"[:200],
        )
    return ReadyResponse(
        ready=points > 0,
        collection=settings.collection_name(),
        points=points,
        detail=None if points else "collection is empty - run ingestion.pipeline",
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["rag"],
)
async def query(
    request: QueryRequest, service: ServiceDep, settings: SettingsDep
) -> QueryResponse:
    """Answer a question over the indexed corpus.

    A question with no supporting context returns HTTP 200 with
    ``status="not_found_in_context"`` rather than an error: the caller asked a
    valid question, and "the corpus does not cover this" is a valid answer.
    """
    started = time.perf_counter()
    retrieval = None

    if request.debug:
        retrieval = await service.retrieve(request.question, top_n=request.top_k)
        answer = await service.answer_from(request.question, retrieval)
    else:
        answer = await service.answer(request.question, top_n=request.top_k)

    return QueryResponse.from_answer(
        answer,
        fingerprint=settings.fingerprint(),
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        retrieval=retrieval,
    )
