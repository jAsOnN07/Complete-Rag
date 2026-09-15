"""RAGAS / retrieval evaluation runner. A script, not a notebook.

    python -m evaluation.run_eval                    # retrieval tier: free, no LLM
    python -m evaluation.run_eval --tier generation  # + answers, citations, not-found
    python -m evaluation.run_eval --tier ragas       # + LLM-judged quality
    python -m evaluation.run_eval --tier ragas --from-results evaluation/results/<run>.json
    python -m evaluation.run_eval --limit 5

Every run writes evaluation/results/<timestamp>__<fingerprint>.json holding
per-question rows, aggregates, and the full resolved config (secrets excluded)
so a result is self-describing and two runs are comparable at a glance.

Tiers:
  retrieval   - hit@1, recall@k, MRR, nDCG@k at chunk and doc level, plus
                negative-gate accuracy. Zero LLM spend; runs on every change.
  generation  - retrieval tier + a real answer per question: not-found accuracy
                on negatives, answered-rate on positives, invalid citation rate.
  ragas       - generation tier + faithfulness / answer relevancy / context
                precision / context recall, judged by the gateway's primary
                model. --from-results re-scores a saved generation run so
                judge spend is not doubled by regenerating.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.config import Settings, get_settings
from core.models import ScoredChunk
from evaluation import metrics as M
from evaluation.gold import GOLD_PATH, GoldQuestion, GoldSet, load_gold

RESULTS_DIR = Path("evaluation/results")
Tier = Literal["retrieval", "generation", "ragas"]


class RetrievalRow(BaseModel):
    question_id: str
    question_type: str
    difficulty: str
    retrieved_chunk_ids: list[str]
    retrieved_doc_ids: list[str]
    scores: list[float]
    chunk_rel: list[bool]
    doc_rel: list[bool]
    hit_at_1: float
    chunk_recall_at_k: float
    doc_recall_at_k: float
    mrr: float
    ndcg_at_k: float
    top_score: float | None
    negative_gate_correct: bool | None = None


class GenerationRow(BaseModel):
    question_id: str
    answer: str
    error: str | None = None
    not_found: bool
    citations: list[str]
    invalid_citations: list[int]
    invalid_citation_rate: float
    provider: str
    model_id: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    not_found_correct: bool | None = None
    # The contexts the prompt actually contained (post-threshold), kept so a
    # saved run can be RAGAS-scored later without regenerating.
    retrieved_texts: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    run_id: str
    started_at: str
    tier: Tier
    k: int
    config_fingerprint: str
    config: dict[str, Any]
    gold_path: str
    gold_version: int
    questions_evaluated: int
    unverified_questions: int
    retrieval: list[RetrievalRow] = Field(default_factory=list)
    generation: list[GenerationRow] = Field(default_factory=list)
    aggregates: dict[str, Any] = Field(default_factory=dict)
    duration_s: float = 0.0
    ragas: dict[str, Any] | None = None


def score_retrieval(
    q: GoldQuestion, retrieved: list[ScoredChunk], k: int, threshold: float
) -> RetrievalRow:
    chunk_rel = [q.chunk_hits(s) for s in retrieved]
    doc_rel = [q.doc_hits(s) for s in retrieved]
    total_docs = len(q.expected_doc_ids)
    top = retrieved[0].score if retrieved else None

    # nDCG needs an exact "total relevant", which is known at document level
    # but not at chunk level (several chunks of one circular may carry the
    # evidence). So it is scored on first-occurrence-per-doc relevance: any
    # later chunk of an already-credited document counts as not relevant.
    seen_docs: set[str] = set()
    first_hit_rel: list[bool] = []
    for s, hit in zip(retrieved, chunk_rel):
        if hit and s.chunk.doc_id not in seen_docs:
            seen_docs.add(s.chunk.doc_id)
            first_hit_rel.append(True)
        else:
            first_hit_rel.append(False)

    # Doc-level recall counts distinct expected docs found, not repeated hits
    # on the same doc - six chunks from one circular is not six-out-of-six.
    found_docs = {s.chunk.doc_id for s in retrieved[:k] if q.doc_hits(s)}
    doc_recall = len(found_docs) / total_docs if total_docs else 0.0

    return RetrievalRow(
        question_id=q.id,
        question_type=q.question_type.value,
        difficulty=q.difficulty.value,
        retrieved_chunk_ids=[s.chunk.chunk_id for s in retrieved],
        retrieved_doc_ids=[s.chunk.doc_id for s in retrieved],
        scores=[round(s.score, 4) for s in retrieved],
        chunk_rel=chunk_rel,
        doc_rel=doc_rel,
        hit_at_1=M.hit_at_k(chunk_rel, 1),
        chunk_recall_at_k=M.recall_at_k(chunk_rel, k, total_docs),
        doc_recall_at_k=doc_recall,
        mrr=M.reciprocal_rank(chunk_rel),
        ndcg_at_k=M.ndcg_at_k(first_hit_rel, k, total_docs),
        top_score=top,
        negative_gate_correct=(
            (top is None or top < threshold) if q.is_negative else None
        ),
    )


def aggregate(result: EvalResult, gold: GoldSet, threshold: float = 0.0) -> dict[str, Any]:
    by_id = {q.id: q for q in gold.questions}
    pos = [r for r in result.retrieval if not by_id[r.question_id].is_negative]
    neg = [r for r in result.retrieval if by_id[r.question_id].is_negative]

    def block(rows: list[RetrievalRow]) -> dict[str, float]:
        return {
            "n": len(rows),
            "hit@1": round(M.mean([r.hit_at_1 for r in rows]), 4),
            f"chunk_recall@{result.k}": round(M.mean([r.chunk_recall_at_k for r in rows]), 4),
            f"doc_recall@{result.k}": round(M.mean([r.doc_recall_at_k for r in rows]), 4),
            "mrr": round(M.mean([r.mrr for r in rows]), 4),
            f"ndcg@{result.k}": round(M.mean([r.ndcg_at_k for r in rows]), 4),
        }

    out: dict[str, Any] = {"retrieval": block(pos), "by_type": {}}
    for qtype, questions in gold.by_type().items():
        ids = {q.id for q in questions}
        rows = [r for r in pos if r.question_id in ids]
        if rows:
            out["by_type"][qtype] = block(rows)

    if neg:
        out["negative_gate_accuracy"] = round(
            M.mean([1.0 if r.negative_gate_correct else 0.0 for r in neg]), 4
        )
        out["negative_top_scores"] = [r.top_score for r in neg]
    if pos:
        out["positive_top_score_min"] = min(r.top_score or 0.0 for r in pos)
        # A positive whose top score sits under the threshold would be refused
        # before the LLM ever ran - a silent false negative the negative-gate
        # metric cannot see.
        misses = [r.question_id for r in pos if (r.top_score or 0.0) < threshold]
        out["positive_gate_misses"] = misses

    if result.generation:
        gen = [g for g in result.generation if g.error is None]
        gpos = [g for g in gen if not by_id[g.question_id].is_negative]
        gneg = [g for g in gen if by_id[g.question_id].is_negative]
        out["generation"] = {
            "errors": sum(1 for g in result.generation if g.error),
            "answered_rate_positives": round(
                M.mean([0.0 if g.not_found else 1.0 for g in gpos]), 4
            ),
            "not_found_accuracy_negatives": round(
                M.mean([1.0 if g.not_found else 0.0 for g in gneg]), 4
            ) if gneg else None,
            "invalid_citation_rate": round(
                M.mean([g.invalid_citation_rate for g in gpos if not g.not_found]), 4
            ),
            # Answered with at least one resolvable citation. A correct answer
            # with no citation is unverifiable, which for this system is a failure.
            "citation_compliance": round(
                M.mean([1.0 if g.citations else 0.0 for g in gpos if not g.not_found]), 4
            ),
            "mean_latency_ms": round(M.mean([g.latency_ms for g in gen]), 1),
            "total_input_tokens": sum(g.input_tokens for g in gen),
            "total_output_tokens": sum(g.output_tokens for g in gen),
        }
    if result.ragas:
        out["ragas"] = {
            "judge_model": result.ragas["judge_model"],
            "scored": result.ragas["scored"],
            "skipped": result.ragas["skipped"],
            **result.ragas["means"],
        }
    return out


def ragas_samples(result: EvalResult, gold: GoldSet) -> tuple[list[Any], list[str]]:
    """Positives that produced an answer. Negatives and refusals are already
    measured exactly (not-found accuracy); RAGAS would only add noise there."""
    from evaluation.ragas_tier import RagasSample

    by_id = {q.id: q for q in gold.questions}
    samples: list[RagasSample] = []
    skipped: list[str] = []
    for g in result.generation:
        q = by_id.get(g.question_id)
        if q is None or q.is_negative:
            continue
        if g.error or g.not_found or not g.retrieved_texts:
            skipped.append(g.question_id)
            continue
        samples.append(
            RagasSample(
                question_id=g.question_id, user_input=q.question, response=g.answer,
                retrieved_contexts=g.retrieved_texts, reference=q.reference,
            )
        )
    return samples, skipped


def score_ragas(
    result: EvalResult, gold: GoldSet, settings: Settings, *, quiet: bool = False, **kw: Any
) -> None:
    """Attach RAGAS scores to a result that already holds generation rows."""
    from evaluation.ragas_tier import judge_model, score_samples

    samples, skipped = ragas_samples(result, gold)
    if not quiet:
        print(f"\nragas: scoring {len(samples)} answered positives "
              f"(skipped {len(skipped)}) with judge {judge_model(settings)}")
    scores = score_samples(samples, settings=settings, **kw)
    scores.skipped = skipped
    result.ragas = scores.model_dump()
    if not quiet:
        for qid, per in scores.per_question.items():
            cells = "  ".join(
                f"{k[:9]}={v:.2f}" if v is not None else f"{k[:9]}=nan" for k, v in per.items()
            )
            print(f"  {qid} {cells}")


def safe_config(settings: Settings) -> dict[str, Any]:
    """Resolved config with secrets structurally excluded."""
    snapshot = settings.eval_fingerprint()
    snapshot.update(
        {
            "aws_region": settings.aws_region,
            "qdrant_url": settings.qdrant_url,
            "langfuse_host": settings.langfuse_host,
        }
    )
    return snapshot


async def evaluate(
    settings: Settings,
    gold: GoldSet,
    *,
    tier: Tier = "retrieval",
    k: int = 5,
    limit: int | None = None,
    pace: float = 0.0,
    quiet: bool = False,
    service: Any | None = None,
    **ragas_kw: Any,
) -> EvalResult:
    """Run one evaluation against an explicit Settings. Pure of CLI and cache.
    `ragas_kw` (judge, embeddings, max_workers) is forwarded to the RAGAS tier."""
    questions = gold.questions[:limit] if limit else gold.questions

    if service is None:
        from core.service import build_service

        service = build_service(settings)
    scale = settings.final_score_scale(reranker_active=service.reranker_active)
    threshold = settings.threshold_for(scale)
    started = time.perf_counter()
    result = EvalResult(
        run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        started_at=datetime.now(timezone.utc).isoformat(),
        tier=tier,
        k=k,
        config_fingerprint=settings.fingerprint(),
        config=safe_config(settings),
        gold_path=str(GOLD_PATH),
        gold_version=gold.version,
        questions_evaluated=len(questions),
        unverified_questions=sum(1 for q in questions if q.verified_by is None),
    )

    def say(msg: str) -> None:
        if not quiet:
            print(msg)

    say(f"run {result.run_id}  tier={tier}  k={k}  fingerprint={result.config_fingerprint}")
    say(f"collection={settings.collection_name()}  mode={settings.retrieval_mode}  "
        f"threshold({scale})={threshold}")
    say(f"{len(questions)} questions ({result.unverified_questions} unverified)\n")

    for i, q in enumerate(questions):
        # Pacing applies to every question: hosted rerankers have per-minute
        # limits too (Cohere trial: 10 req/min), not only the LLM.
        if pace and i:
            await asyncio.sleep(pace)
        retrieved = await service.retrieve(q.question, top_n=k)
        row = score_retrieval(q, retrieved, k, threshold)
        result.retrieval.append(row)
        flag = ""
        if q.is_negative:
            flag = "gate ok" if row.negative_gate_correct else "GATE MISS"
        say(
            f"  {q.id} {q.question_type.value:<14} hit@1={row.hit_at_1:.0f} "
            f"docR@{k}={row.doc_recall_at_k:.2f} mrr={row.mrr:.2f} "
            f"top={row.top_score:.3f} {flag}"
        )

        if tier in ("generation", "ragas"):
            t0 = time.perf_counter()
            try:
                answer = await service.answer_from(q.question, retrieved)
            except Exception as exc:  # noqa: BLE001 - one failure must not lose the run
                say(f"      generation failed: {type(exc).__name__}: {str(exc)[:120]}")
                result.generation.append(
                    GenerationRow(
                        question_id=q.id, answer="", error=f"{type(exc).__name__}: {exc}"[:500],
                        not_found=False, citations=[], invalid_citations=[],
                        invalid_citation_rate=0.0, provider="error", model_id=None,
                        input_tokens=0, output_tokens=0,
                        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                    )
                )
                continue
            result.generation.append(
                GenerationRow(
                    question_id=q.id,
                    answer=answer.answer,
                    not_found=answer.not_found,
                    citations=[c.chunk_id for c in answer.citations],
                    invalid_citations=answer.invalid_citations,
                    invalid_citation_rate=answer.invalid_citation_rate,
                    provider=answer.provider,
                    model_id=answer.model_id,
                    input_tokens=answer.input_tokens,
                    output_tokens=answer.output_tokens,
                    latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                    not_found_correct=(answer.not_found == q.is_negative),
                    retrieved_texts=[s.chunk.text for s in retrieved if s.score >= threshold],
                )
            )

    if tier == "ragas":
        score_ragas(result, gold, settings, quiet=quiet, **ragas_kw)

    result.duration_s = round(time.perf_counter() - started, 2)
    result.aggregates = aggregate(result, gold, threshold)
    return result


def rescore(path: Path, gold: GoldSet, settings: Settings, **ragas_kw: Any) -> EvalResult:
    """RAGAS-score a saved generation run. Retrieval and answers are reused
    verbatim; only the judge runs. The result keeps the original run_id so the
    file is updated in place and stays comparable with its own history."""
    result = EvalResult.model_validate_json(path.read_text(encoding="utf-8"))
    if not result.generation:
        raise SystemExit(f"{path} has no generation rows; run --tier generation first")
    if not any(g.retrieved_texts for g in result.generation):
        raise SystemExit(f"{path} predates retrieved_texts; regenerate with --tier ragas")
    t0 = time.perf_counter()
    score_ragas(result, gold, settings, **ragas_kw)
    result.tier = "ragas"
    result.duration_s = round(result.duration_s + time.perf_counter() - t0, 2)
    scale = settings.final_score_scale(reranker_active=settings.reranker_backend != "none")
    result.aggregates = aggregate(result, gold, settings.threshold_for(scale))
    return result


def write_result(result: EvalResult) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{result.run_id}__{result.config_fingerprint}.json"
    out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return out


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    gold = load_gold(Path(args.gold))
    if Path("data/manifest.json").exists():
        # A quote that has drifted from the corpus makes recall silently wrong;
        # refuse to run rather than report a number nobody should trust.
        from evaluation.gold import corpus_texts, validate_evidence

        problems = validate_evidence(gold, corpus_texts())
        if problems:
            for p in problems:
                print(f"gold {p.question_id}: {p.reason} {p.quote!r}", file=sys.stderr)
            return 2
    if args.from_results:
        result = rescore(Path(args.from_results), gold, settings, max_workers=args.judge_workers)
    else:
        result = await evaluate(
            settings, gold, tier=args.tier, k=args.k, limit=args.limit, pace=args.pace,
            max_workers=args.judge_workers,
        )
    out = write_result(result)
    print("\n== aggregates ==")
    print(json.dumps(result.aggregates, indent=2))
    print(f"\nwrote {out}")

    from observability.tracing import get_tracer

    await get_tracer().flush()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", default=str(GOLD_PATH))
    p.add_argument("--tier", choices=["retrieval", "generation", "ragas"], default="retrieval")
    p.add_argument("--k", type=int, default=5, help="retrieval depth to score at")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--pace", type=float, default=None,
        help="seconds between generation calls (default 12 for generation tier: "
             "Groq free tier is 8k tokens/min)",
    )
    p.add_argument(
        "--from-results", default=None, metavar="JSON",
        help="RAGAS-score a saved generation run instead of regenerating (implies --tier ragas)",
    )
    p.add_argument(
        "--judge-workers", type=int, default=2,
        help="RAGAS concurrency; kept low because the judge sits behind free-tier rate limits",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.from_results:
        args.tier = "ragas"
    if args.pace is None:
        args.pace = 12.0 if args.tier in ("generation", "ragas") else 0.0
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
