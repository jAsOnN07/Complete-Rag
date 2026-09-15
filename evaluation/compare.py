"""Evaluate several configurations and print one comparison table.

    python -m evaluation.compare chunking      # fixed / recursive / semantic
    python -m evaluation.compare retrieval     # dense / hybrid / hybrid+rerank
    python -m evaluation.compare --k 10 chunking

Each configuration is a set of overrides applied on top of the current
Settings; everything else (embedder, gold set, k) is held constant, so the
table is a controlled comparison. Every run is also written to
evaluation/results/ as usual, and the table lands in
evaluation/results/compare_<suite>_<timestamp>.md for the README.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Settings, get_settings
from core.models import ChunkStrategy
from evaluation.gold import GoldSet, load_gold
from evaluation.run_eval import RESULTS_DIR, evaluate, write_result

SUITES: dict[str, list[tuple[str, dict[str, Any]]]] = {
    # model_copy(update=...) skips validation, so overrides must already be
    # the final types - enums, not their string values.
    "chunking": [
        ("fixed", {"chunk_strategy": ChunkStrategy.FIXED}),
        ("recursive", {"chunk_strategy": ChunkStrategy.RECURSIVE}),
        ("semantic", {"chunk_strategy": ChunkStrategy.SEMANTIC}),
    ],
    "chunking_ctx": [
        ("fixed", {"chunk_strategy": ChunkStrategy.FIXED, "chunk_context_header": False}),
        ("fixed+ctx", {"chunk_strategy": ChunkStrategy.FIXED, "chunk_context_header": True}),
        ("recursive", {"chunk_strategy": ChunkStrategy.RECURSIVE, "chunk_context_header": False}),
        ("recursive+ctx", {"chunk_strategy": ChunkStrategy.RECURSIVE, "chunk_context_header": True}),
        ("semantic", {"chunk_strategy": ChunkStrategy.SEMANTIC, "chunk_context_header": False}),
        ("semantic+ctx", {"chunk_strategy": ChunkStrategy.SEMANTIC, "chunk_context_header": True}),
    ],
    "retrieval": [
        ("dense", {"retrieval_mode": "dense", "reranker_backend": "none"}),
        ("hybrid", {"retrieval_mode": "hybrid", "reranker_backend": "none"}),
        # Reranker comes from the base config so the suite compares whatever is deployed.
        ("hybrid+rerank", {"retrieval_mode": "hybrid"}),
    ],
}

COLUMNS = ["hit@1", "chunk_recall", "doc_recall", "mrr", "ndcg", "neg_gate", "pos_miss"]


def _row(label: str, aggregates: dict[str, Any], k: int, extra: dict[str, Any]) -> dict[str, Any]:
    r = aggregates["retrieval"]
    return {
        "config": label,
        "hit@1": r["hit@1"],
        "chunk_recall": r[f"chunk_recall@{k}"],
        "doc_recall": r[f"doc_recall@{k}"],
        "mrr": r["mrr"],
        "ndcg": r[f"ndcg@{k}"],
        "neg_gate": aggregates.get("negative_gate_accuracy"),
        "pos_miss": len(aggregates.get("positive_gate_misses", [])),
        **extra,
    }


def render_table(rows: list[dict[str, Any]], k: int, extra_cols: list[str]) -> str:
    cols = ["config", *COLUMNS, *extra_cols]
    header = {
        "config": "config", "hit@1": "hit@1", "chunk_recall": f"chunk R@{k}",
        "doc_recall": f"doc R@{k}", "mrr": "MRR", "ndcg": f"nDCG@{k}", "neg_gate": "neg gate",
        "pos_miss": "pos gate miss",
    }
    lines = ["| " + " | ".join(header.get(c, c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for row in rows:
        cells = []
        for c in cols:
            v = row.get(c)
            cells.append(f"{v:.2f}" if isinstance(v, float) else ("-" if v is None else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


async def compare(
    suite: str, *, base: Settings, gold: GoldSet, k: int, limit: int | None, pace: float = 0.0
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    extra_cols: list[str] = ["chunks"] if suite.startswith("chunking") else []
    for label, overrides in SUITES[suite]:
        settings = base.model_copy(update=overrides)
        print(f"\n=== {label}  ({settings.collection_name()}) ===")
        from core.service import build_service

        service = build_service(settings)
        extra: dict[str, Any] = {}
        if suite.startswith("chunking"):
            extra["chunks"] = await service.count_points()
            if extra["chunks"] == 0:
                print(f"  collection empty - run: CHUNK_STRATEGY={label} python -m ingestion.pipeline")
                rows.append({"config": f"{label} (not indexed)"})
                continue
        result = await evaluate(settings, gold, k=k, limit=limit, quiet=True, service=service, pace=pace)
        out = write_result(result)
        print(f"  fingerprint={result.config_fingerprint}  -> {out.name}")
        rows.append(_row(label, result.aggregates, k, extra))
    return rows, extra_cols


async def main_async(args: argparse.Namespace) -> int:
    base = get_settings()
    gold = load_gold()
    rows, extra_cols = await compare(args.suite, base=base, gold=gold, k=args.k, limit=args.limit, pace=args.pace)
    table = render_table(rows, args.k, extra_cols)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"compare_{args.suite}_{stamp}.md"
    constants = {
        "embedder": base.embed_model_id,
        "gold": f"{len(gold.questions)} questions, {len(gold.unverified())} unverified",
        "k": args.k,
    }
    if args.suite.startswith("chunking"):
        constants["retrieval"] = f"{base.retrieval_mode} + {base.reranker_backend}"
        constants["chunk_size/overlap"] = f"{base.chunk_size}/{base.chunk_overlap}"
    else:
        constants["chunking"] = f"{base.chunk_strategy.value} {base.chunk_size}/{base.chunk_overlap}"
    body = (
        f"# {args.suite} comparison - {stamp}\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in constants.items())
        + "\n\n" + table + "\n"
    )
    out.write_text(body, encoding="utf-8")
    print("\n" + body)
    print(f"wrote {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("suite", choices=sorted(SUITES))
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--pace", type=float, default=0.0, help="seconds between questions (hosted rerank rate limits)")
    return asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
