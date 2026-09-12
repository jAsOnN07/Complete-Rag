"""Ask one question against the indexed corpus.

    python -m scripts.ask "What are the KYC due diligence requirements?"
    python -m scripts.ask --retrieve-only "interest rate on deposits"

--retrieve-only exercises embedding + Qdrant without calling the chat model,
which is useful while Bedrock chat access is still being sorted out.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from core.config import get_settings
from core.service import RagService


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    from generation.llm import build_bedrock_llm
    from observability.tracing import get_tracer
    from retrieval.embedder import build_bedrock_embedder
    from retrieval.vector_store import build_qdrant_store

    store = build_qdrant_store(settings)
    embedder = build_bedrock_embedder(settings)

    total = await store.count()
    print(f"collection {settings.collection_name()}: {total} points\n")
    if total == 0:
        print("collection is empty - run: python -m ingestion.pipeline")
        await store.close()
        return 1

    service = RagService(
        embedder=embedder,
        store=store,
        llm=None if args.retrieve_only else build_bedrock_llm(settings),
        top_k=settings.top_k,
        rerank_top_n=settings.rerank_top_n,
        relevance_threshold=settings.threshold_for("dense"),
    )

    if args.retrieve_only:
        for scored in await service.retrieve(args.question):
            meta = scored.chunk.meta
            print(f"[{scored.rank + 1}] score={scored.score:.4f}  {meta.circular_no}")
            print(f"     {meta.title[:74]}")
            print(f"     p.{scored.chunk.page}  {scored.chunk.text[:150]}...\n")
        await store.close()
        return 0

    answer = await service.answer(args.question)
    print(answer.answer)
    print()
    if answer.not_found:
        print("(not found in context - no citations)")
    else:
        print("Citations:")
        for c in answer.citations:
            print(f"  - {c.circular_no or c.doc_id}  p.{c.page}  {c.title[:60]}")
        if answer.invalid_citations:
            print(f"  ! hallucinated labels: {answer.invalid_citations}")
    print(
        f"\nprovider={answer.provider} model={answer.model_id} "
        f"tokens={answer.input_tokens}/{answer.output_tokens} "
        f"chunks_considered={answer.chunks_considered}"
    )

    await get_tracer().flush()
    await store.close()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("question")
    p.add_argument(
        "--retrieve-only",
        action="store_true",
        help="show retrieved chunks without calling the chat model",
    )
    try:
        return asyncio.run(run(p.parse_args()))
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
