"""UI endpoints: token gate, meta, gold, published results, comparison tables,
traffic aggregation over a fake Langfuse, and the eval job with SSE progress."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from api import ui_routes
from api.main import create_app
from api.ui_routes import parse_markdown_table, summarise_traffic
from core.config import Settings
from core.service import RagService
from observability.tracing import RecordingTracer
from tests.fakes import FakeEmbedder, FakeLlm
from tests.test_api import build_store


async def make_app(token: str | None = None, **settings_kw):
    service = RagService(
        embedder=FakeEmbedder(), store=await build_store(), llm=FakeLlm("Banks must verify identity [1]."),
        top_k=10, rerank_top_n=3, relevance_threshold=0.0, tracer=RecordingTracer(),
    )
    app = create_app(service=service)
    app.state.settings = Settings(ui_access_token=SecretStr(token) if token else None, **settings_kw)
    return app


@pytest.fixture
async def client() -> AsyncClient:
    app = await make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_ui_routes_require_the_token_when_configured():
    app = await make_app("tok")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        assert (await c.get("/ui/api/meta")).status_code == 401
        assert (await c.get("/ui/api/meta", headers={"X-Access-Token": "tok"})).status_code == 200
        assert (await c.get("/ui/api/gold?token=tok")).status_code == 200


async def test_meta_describes_the_running_configuration(client):
    body = (await client.get("/ui/api/meta")).json()
    assert body["fingerprint"] and body["collection"]
    assert body["points"] >= 1  # the in-memory store holds the KYC chunks
    assert body["threshold"]["scale"] in {"dense", "rrf", "cross_encoder", "cohere", "bedrock", "none"}
    assert "primary_model_id" in body["config"] and "aws_region" in body["config"]
    assert body["token_required"] is False
    assert not any("key" in k.lower() and "secret" in str(v).lower() for k, v in body["config"].items())


async def test_gold_endpoint_serves_the_committed_set(client):
    body = (await client.get("/ui/api/gold")).json()
    assert len(body["questions"]) == 50
    assert {"id", "question", "reference", "question_type", "evidence", "expected_doc_ids"} <= set(body["questions"][0])


async def test_results_list_only_published_runs(client, tmp_path: Path, monkeypatch):
    from evaluation.run_eval import EvalResult

    monkeypatch.setattr(ui_routes, "PUBLISHED_DIR", tmp_path)
    good = EvalResult(
        run_id="20260101T000000Z", started_at="t", tier="retrieval", k=5, config_fingerprint="abc",
        config={"label": "demo"}, gold_path="g", gold_version=2, questions_evaluated=1, unverified_questions=0,
        aggregates={"retrieval": {"hit@1": 1.0, "mrr": 1.0}, "negative_gate_accuracy": 1.0},
    )
    (tmp_path / "20260101T000000Z__abc.json").write_text(good.model_dump_json(), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    body = (await client.get("/ui/api/results")).json()
    assert [r["run_id"] for r in body["runs"]] == ["20260101T000000Z"]
    assert body["runs"][0]["label"] == "demo" and body["runs"][0]["summary"]["hit@1"] == 1.0

    detail = (await client.get("/ui/api/results/20260101T000000Z")).json()
    assert detail["config_fingerprint"] == "abc"
    assert (await client.get("/ui/api/results/nope")).status_code == 404
    assert (await client.get("/ui/api/results/..%2F..%2Fetc")).status_code == 404


def test_markdown_comparison_table_parses_to_rows():
    text = (
        "# retrieval comparison - x\n\n- k: 5\n- chunking: fixed\n\n"
        "| config | hit@1 | MRR | neg gate |\n|---|---|---|---|\n"
        "| dense | 0.73 | 0.84 | 1.00 |\n| hybrid | 0.91 | 0.95 | - |\n"
    )
    t = parse_markdown_table(text)
    assert t["title"] == "retrieval comparison - x" and t["meta"] == ["k: 5", "chunking: fixed"]
    assert t["columns"] == ["config", "hit@1", "MRR", "neg gate"]
    assert t["rows"][0] == {"config": "dense", "hit@1": 0.73, "MRR": 0.84, "neg gate": 1.0}
    assert t["rows"][1]["neg gate"] is None


async def test_compare_endpoint_reads_the_published_tables(client):
    body = (await client.get("/ui/api/compare")).json()
    names = [t["file"] for t in body["tables"]]
    assert any(n.startswith("compare_retrieval") for n in names)
    assert all(t["rows"] for t in body["tables"])


def test_traffic_summary_counts_failovers_against_the_primary():
    traces = [
        {"id": "t1", "timestamp": "2026-09-15T10:00:00Z", "latency": 8.0, "totalCost": 0.0002, "htmlPath": "/p/t1"},
        {"id": "t2", "timestamp": "2026-09-15T10:01:00Z", "latency": 2.0, "totalCost": 0.0001, "htmlPath": "/p/t2"},
        {"id": "t3", "timestamp": "2026-09-15T10:02:00Z", "latency": 0.3, "totalCost": 0.0, "htmlPath": "/p/t3"},
    ]
    answers = [
        {"traceId": "t1", "output": {"status": "answered", "citations": 2}},
        {"traceId": "t2", "output": {"status": "answered", "citations": 1}},
        {"traceId": "t3", "output": {"status": "not_found", "llm_called": False}},
    ]
    gens = [
        {"traceId": "t1", "metadata": {"provider": "groq"}, "output": {"served_model": "openai/gpt-oss-120b"}},
        {"traceId": "t2", "metadata": {"provider": "google"}, "model": "gemini-3.5-flash"},
    ]
    s = summarise_traffic(traces, answers, gens, primary_provider="@google")
    assert s["n"] == 3 and s["providers"] == {"groq": 1, "google": 1}
    assert s["failover_rate"] == 0.5
    assert s["statuses"] == {"answered": 2, "not_found": 1}
    assert s["latency_p50_s"] == 2.0 and s["latency_p95_s"] == 8.0
    assert s["recent"][0]["trace_id"] == "t3"  # newest first
    assert s["recent"][2]["model"] == "openai/gpt-oss-120b"


async def test_traffic_degrades_without_langfuse(client):
    body = (await client.get("/ui/api/traffic")).json()
    assert body["available"] is False


async def test_traffic_uses_a_server_side_langfuse_client_and_caches(monkeypatch):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/traces"):
            data = [{"id": "t1", "timestamp": "2026-09-15T10:00:00Z", "latency": 1.0, "totalCost": 0.0, "htmlPath": "/x"}]
        elif request.url.params.get("name") == "rag.answer":
            data = [{"traceId": "t1", "output": {"status": "answered"}}]
        else:
            data = [{"traceId": "t1", "metadata": {"provider": "google"}}]
        return httpx.Response(200, json={"data": data, "meta": {}})

    app = await make_app(langfuse_public_key=SecretStr("pk"), langfuse_secret_key=SecretStr("sk"))
    app.state.langfuse_client_factory = lambda settings: httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://fake"
    )
    monkeypatch.setattr(ui_routes, "_traffic_cache", {"at": 0.0, "limit": 0, "body": None})
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        first = (await c.get("/ui/api/traffic?limit=10")).json()
        second = (await c.get("/ui/api/traffic?limit=10")).json()
    assert first["available"] and first["n"] == 1 and first["failover_rate"] == 0.0
    assert len(calls) == 3, "three listings per refresh"
    assert second["cache_age_s"] >= 0 and len(calls) == 3, "second call served from cache"


async def test_eval_job_streams_progress_then_done(client, tmp_path: Path, monkeypatch):
    from evaluation import run_eval

    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path)
    r = await client.post("/ui/api/eval/run", json={"tier": "retrieval", "k": 3, "limit": 2, "pace": 0})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    events = []
    async with client.stream("GET", f"/ui/api/eval/jobs/{job_id}/events") as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if line.startswith("event: "):
                events.append([line[7:], None])
            elif line.startswith("data: ") and events:
                events[-1][1] = json.loads(line[6:])
    names = [e[0] for e in events]
    assert names[0] == "start" and names[-1] == "done"
    assert names.count("progress") == 2
    done = events[-1][1]
    assert done["questions"] == 2 and "retrieval" in done["aggregates"]
    assert (tmp_path / done["file"]).exists()
    listing = (await client.get("/ui/api/eval/jobs")).json()["jobs"]
    assert listing[0]["done"] is True and listing[0]["error"] is None


async def test_only_one_eval_job_runs_at_a_time(client, tmp_path: Path, monkeypatch):
    from evaluation import run_eval

    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path)
    first = await client.post("/ui/api/eval/run", json={"tier": "retrieval", "limit": 3, "pace": 0.2})
    assert first.status_code == 202
    second = await client.post("/ui/api/eval/run", json={"tier": "retrieval", "limit": 1, "pace": 0})
    assert second.status_code == 409 and second.json()["detail"]["running"] == first.json()["job_id"]
    # Drain so the task finishes cleanly before the loop closes.
    async with client.stream("GET", f"/ui/api/eval/jobs/{first.json()['job_id']}/events") as resp:
        async for _ in resp.aiter_lines():
            pass


async def test_generation_tier_is_capped_from_the_ui(client, tmp_path: Path, monkeypatch):
    from evaluation import run_eval

    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path)
    r = await client.post("/ui/api/eval/run", json={"tier": "generation", "limit": 50, "pace": 0,
                                                    "max_generation_questions": 2})
    job_id = r.json()["job_id"]
    n = 0
    async with client.stream("GET", f"/ui/api/eval/jobs/{job_id}/events") as resp:
        async for line in resp.aiter_lines():
            if line.startswith("event: progress"):
                n += 1
            if line.startswith("data: ") and '"run_id"' in line:
                done = json.loads(line[6:])
    assert n == 2 and done["questions"] == 2 and "generation" in done["aggregates"]


async def test_unknown_job_is_404(client):
    assert (await client.get("/ui/api/eval/jobs/nope/events")).status_code == 404
