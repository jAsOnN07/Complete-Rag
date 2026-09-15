"""Endpoints behind the showcase UI: configuration, traffic, gold set,
published results, comparison tables, and an eval job runner.

Nothing here computes a second truth. Metrics come from the same functions
the eval script uses; traffic comes from the same Langfuse traces the
service emits; the eval job runs `evaluation.run_eval.evaluate` against the
live service. All routes are token-guarded (see api.deps.require_token).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.deps import ServiceDep, SettingsDep, TokenDep
from core.config import Settings

router = APIRouter(prefix="/ui/api", tags=["ui"], dependencies=[TokenDep])

PUBLISHED_DIR = Path("evaluation/results/published")
MANIFEST_PATH = Path("data/manifest.json")
TRAFFIC_CACHE_TTL_S = 60.0


# ---- meta ---------------------------------------------------------------------------


@router.get("/meta")
async def meta(service: ServiceDep, settings: SettingsDep) -> dict[str, Any]:
    from evaluation.run_eval import safe_config

    try:
        points: int | None = await service.count_points()
    except Exception:  # noqa: BLE001 - the UI must render even when Qdrant is down
        points = None
    corpus: dict[str, Any] = {"documents": 0}
    if MANIFEST_PATH.exists():
        rows = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        dates = sorted(r["issued_on"] for r in rows if r.get("issued_on"))
        corpus = {
            "documents": len(rows),
            "regulator": sorted({r.get("regulator", "RBI") for r in rows}),
            "issued_from": dates[0] if dates else None,
            "issued_to": dates[-1] if dates else None,
        }
    scale = settings.final_score_scale(reranker_active=service.reranker_active)
    return {
        "fingerprint": settings.fingerprint(),
        "config": safe_config(settings),
        "threshold": {"scale": scale, "value": settings.threshold_for(scale)},
        "fallback_model": f"{settings.portkey_groq_provider}/{settings.groq_model_id}",
        "collection": settings.collection_name(),
        "points": points,
        "corpus": corpus,
        "token_required": settings.ui_access_token is not None,
        "langfuse": settings.langfuse_public_key is not None and settings.langfuse_secret_key is not None,
    }


# ---- traffic (Langfuse public API) ----------------------------------------------------

_traffic_cache: dict[str, Any] = {"at": 0.0, "limit": 0, "body": None}


def langfuse_client(settings: Settings) -> httpx.AsyncClient | None:
    """Server-side only: the browser never sees Langfuse credentials."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    return httpx.AsyncClient(
        base_url=settings.langfuse_host,
        auth=(settings.langfuse_public_key.get_secret_value(), settings.langfuse_secret_key.get_secret_value()),
        timeout=30,
    )


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
    return round(ordered[idx], 3)


def summarise_traffic(
    traces: list[dict[str, Any]], answers: list[dict[str, Any]], generations: list[dict[str, Any]],
    *, primary_provider: str,
) -> dict[str, Any]:
    """Join the three Langfuse listings by trace id into per-query rows and totals."""
    by_trace: dict[str, dict[str, Any]] = {}
    for t in traces:
        by_trace[t["id"]] = {
            "trace_id": t["id"], "at": t.get("timestamp"), "latency_s": t.get("latency"),
            "cost_usd": t.get("totalCost"), "url": t.get("htmlPath"), "status": None,
            "provider": None, "model": None, "citations": None,
        }
    for a in answers:
        row = by_trace.get(a.get("traceId"))
        out = a.get("output") if isinstance(a.get("output"), dict) else {}
        if row is not None:
            row["status"] = out.get("status")
            row["citations"] = out.get("citations")
    for g in generations:
        row = by_trace.get(g.get("traceId"))
        if row is None:
            continue
        meta = g.get("metadata") or {}
        out = g.get("output") if isinstance(g.get("output"), dict) else {}
        row["provider"] = meta.get("provider")
        row["model"] = out.get("served_model") or g.get("model")
    rows = sorted(by_trace.values(), key=lambda r: r["at"] or "", reverse=True)
    latencies = [r["latency_s"] for r in rows if isinstance(r.get("latency_s"), (int, float))]
    costs = [r["cost_usd"] for r in rows if isinstance(r.get("cost_usd"), (int, float))]
    providers: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for r in rows:
        if r["provider"]:
            providers[r["provider"]] = providers.get(r["provider"], 0) + 1
        statuses[r["status"] or "unknown"] = statuses.get(r["status"] or "unknown", 0) + 1
    primary = primary_provider.lstrip("@")
    llm_calls = sum(providers.values())
    failovers = sum(n for p, n in providers.items() if p != primary)
    return {
        "available": True,
        "n": len(rows),
        "latency_p50_s": _percentile(latencies, 0.5),
        "latency_p95_s": _percentile(latencies, 0.95),
        "cost_mean_usd": round(sum(costs) / len(costs), 6) if costs else None,
        "cost_total_usd": round(sum(costs), 6) if costs else 0.0,
        "providers": providers,
        "failover_rate": round(failovers / llm_calls, 3) if llm_calls else None,
        "statuses": statuses,
        "recent": rows[:25],
    }


@router.get("/traffic")
async def traffic(request: Request, settings: SettingsDep, limit: int = 50) -> dict[str, Any]:
    limit = max(1, min(limit, 100))
    now = time.time()
    if _traffic_cache["body"] and _traffic_cache["limit"] == limit and now - _traffic_cache["at"] < TRAFFIC_CACHE_TTL_S:
        return {**_traffic_cache["body"], "cache_age_s": round(now - _traffic_cache["at"])}
    factory = getattr(request.app.state, "langfuse_client_factory", langfuse_client)
    client = factory(settings)
    if client is None:
        return {"available": False, "reason": "Langfuse is not configured"}
    try:
        async with client:
            t, a, g = await asyncio.gather(
                client.get("/api/public/traces", params={"limit": limit}),
                client.get("/api/public/observations", params={"name": "rag.answer", "limit": limit}),
                client.get("/api/public/observations", params={"type": "GENERATION", "limit": limit}),
            )
            for r in (t, a, g):
                r.raise_for_status()
            body = summarise_traffic(
                t.json()["data"], a.json()["data"], g.json()["data"],
                primary_provider=settings.llm_primary_provider,
            )
    except httpx.HTTPError as exc:
        return {"available": False, "reason": f"Langfuse API: {type(exc).__name__}: {exc}"[:200]}
    body["langfuse_host"] = settings.langfuse_host
    _traffic_cache.update(at=now, limit=limit, body=body)
    return {**body, "cache_age_s": 0}


# ---- gold set, published results, comparison tables ------------------------------------


@router.get("/gold")
async def gold() -> dict[str, Any]:
    from evaluation.gold import load_gold

    g = load_gold()
    return {
        "version": g.version,
        "description": g.description,
        "questions": [q.model_dump(mode="json") for q in g.questions],
    }


def _published_runs() -> list[Path]:
    return sorted(PUBLISHED_DIR.glob("*.json"), reverse=True)


@router.get("/results")
async def results() -> dict[str, Any]:
    from evaluation.run_eval import EvalResult

    runs = []
    for path in _published_runs():
        try:
            r = EvalResult.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - one bad file must not hide the rest
            continue
        runs.append({
            "run_id": r.run_id, "file": path.name, "tier": r.tier, "k": r.k,
            "started_at": r.started_at, "questions": r.questions_evaluated,
            "unverified": r.unverified_questions, "fingerprint": r.config_fingerprint,
            "label": r.config.get("label"),
            "summary": {
                "hit@1": r.aggregates.get("retrieval", {}).get("hit@1"),
                "mrr": r.aggregates.get("retrieval", {}).get("mrr"),
                "negative_gate_accuracy": r.aggregates.get("negative_gate_accuracy"),
                "answered_rate": r.aggregates.get("generation", {}).get("answered_rate_positives"),
                "faithfulness": (r.aggregates.get("ragas") or {}).get("faithfulness"),
            },
        })
    return {"runs": runs}


@router.get("/results/{run_id}")
async def result_detail(run_id: str) -> dict[str, Any]:
    from evaluation.run_eval import EvalResult

    if not re.fullmatch(r"[0-9TZ]+", run_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such run")
    for path in _published_runs():
        if path.name.startswith(run_id):
            r = EvalResult.model_validate_json(path.read_text(encoding="utf-8"))
            body = r.model_dump(mode="json")
            for g in body.get("generation", []):
                g.pop("retrieved_texts", None)  # bulky; the UI never shows it
            return body
    raise HTTPException(status.HTTP_404_NOT_FOUND, "no such run")


def parse_markdown_table(text: str) -> dict[str, Any]:
    """The compare_*.md files: a title line, bullet metadata, one pipe table."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    title = next((ln.lstrip("# ").strip() for ln in lines if ln.startswith("#")), "")
    meta = [ln[2:].strip() for ln in lines if ln.startswith("- ")]
    table = [ln for ln in lines if ln.startswith("|")]
    if len(table) < 2:
        return {"title": title, "meta": meta, "columns": [], "rows": []}
    split = lambda ln: [c.strip() for c in ln.strip("|").split("|")]  # noqa: E731
    columns = split(table[0])
    rows = []
    for ln in table[2:]:
        cells = split(ln)
        row: dict[str, Any] = {}
        for col, cell in zip(columns, cells):
            try:
                row[col] = float(cell)
            except ValueError:
                row[col] = None if cell == "-" else cell
        rows.append(row)
    return {"title": title, "meta": meta, "columns": columns, "rows": rows}


@router.get("/compare")
async def compare() -> dict[str, Any]:
    tables = []
    for path in sorted(PUBLISHED_DIR.glob("compare_*.md")):
        tables.append({"file": path.name, **parse_markdown_table(path.read_text(encoding="utf-8"))})
    return {"tables": tables}


# ---- eval job ----------------------------------------------------------------------------


class EvalRunRequest(BaseModel):
    tier: Literal["retrieval", "generation"] = "retrieval"
    k: int = Field(default=5, ge=1, le=20)
    limit: int | None = Field(default=None, ge=1, le=50)
    pace: float = Field(default=7.0, ge=0.0, le=60.0)
    # Free-tier reality (see CLAUDE.md): a generation run over all 50 burns a
    # day of Gemini quota. The cap is a default, not a wall - it is a config.
    max_generation_questions: int = 14


class EvalJob:
    def __init__(self, job_id: str, params: EvalRunRequest) -> None:
        self.id = job_id
        self.params = params
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.events: list[dict[str, Any]] = []
        self.done = asyncio.Event()
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.file: str | None = None
        self.task: asyncio.Task[None] | None = None

    def push(self, event: str, data: dict[str, Any]) -> None:
        self.events.append({"event": event, "data": data})


def _jobs(request: Request) -> dict[str, EvalJob]:
    if not hasattr(request.app.state, "eval_jobs"):
        request.app.state.eval_jobs = {}
    return request.app.state.eval_jobs


async def _run_job(job: EvalJob, settings: Settings, service: Any) -> None:
    from evaluation.gold import load_gold
    from evaluation.run_eval import evaluate, write_result

    try:
        gold = load_gold()
        limit = job.params.limit
        if job.params.tier == "generation":
            limit = min(limit or job.params.max_generation_questions, job.params.max_generation_questions)
        result = await evaluate(
            settings, gold, tier=job.params.tier, k=job.params.k, limit=limit,
            pace=job.params.pace, quiet=True, service=service,
            on_progress=lambda p: job.push("progress", p),
        )
        path = write_result(result)
        job.file = path.name
        job.result = {
            "run_id": result.run_id, "tier": result.tier, "k": result.k,
            "questions": result.questions_evaluated, "duration_s": result.duration_s,
            "fingerprint": result.config_fingerprint, "aggregates": result.aggregates,
            "file": path.name,
        }
        job.push("done", job.result)
    except asyncio.CancelledError:
        job.error = "cancelled"
        job.push("error", {"detail": "cancelled"})
        raise
    except Exception as exc:  # noqa: BLE001 - reported to the client, never lost
        job.error = f"{type(exc).__name__}: {exc}"[:300]
        job.push("error", {"detail": job.error})
    finally:
        job.done.set()


@router.post("/eval/run", status_code=status.HTTP_202_ACCEPTED)
async def eval_run(
    params: EvalRunRequest, request: Request, service: ServiceDep, settings: SettingsDep
) -> dict[str, Any]:
    jobs = _jobs(request)
    running = [j for j in jobs.values() if not j.done.is_set()]
    if running:
        # One heavy job at a time: the same rule the dev box taught us, and
        # the hosted rerankers' per-minute limits would be shared anyway.
        raise HTTPException(status.HTTP_409_CONFLICT, detail={"running": running[0].id})
    job = EvalJob(uuid.uuid4().hex[:12], params)
    jobs[job.id] = job
    job.task = asyncio.create_task(_run_job(job, settings, service))
    return {"job_id": job.id, "tier": params.tier, "started_at": job.started_at}


@router.get("/eval/jobs")
async def eval_jobs(request: Request) -> dict[str, Any]:
    return {
        "jobs": [
            {"job_id": j.id, "tier": j.params.tier, "started_at": j.started_at,
             "done": j.done.is_set(), "error": j.error, "file": j.file, "events": len(j.events)}
            for j in _jobs(request).values()
        ]
    }


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/eval/jobs/{job_id}/events")
async def eval_job_events(job_id: str, request: Request) -> StreamingResponse:
    """Replays what happened so far, then follows the job until it finishes.
    Reconnect-safe: a client that drops mid-run gets the full history back."""
    job = _jobs(request).get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such job")

    async def gen() -> AsyncIterator[str]:
        sent = 0
        yield _sse("start", {"job_id": job.id, "tier": job.params.tier, "started_at": job.started_at})
        while True:
            while sent < len(job.events):
                ev = job.events[sent]
                sent += 1
                yield _sse(ev["event"], ev["data"])
            if job.done.is_set() and sent >= len(job.events):
                return
            try:
                await asyncio.wait_for(job.done.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
