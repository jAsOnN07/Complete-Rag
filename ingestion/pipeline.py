"""End-to-end ingestion: manifest -> load -> chunk -> embed -> Qdrant.

    python -m ingestion.pipeline --report          # extraction stats only, no AWS
    python -m ingestion.pipeline --dry-run         # chunk but do not embed or index
    python -m ingestion.pipeline                   # full ingest
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from core.config import Settings, get_settings
from core.models import Chunk, ExtractionStats
from ingestion.chunker import get_chunker
from ingestion.loader import PdfLoader


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"{path} not found - run: python -m scripts.fetch_circulars --limit 45"
        )
    return json.loads(path.read_text(encoding="utf-8"))


async def build_chunks(
    rows: list[dict[str, Any]], raw_dir: Path, settings: Settings
) -> tuple[list[Chunk], list[ExtractionStats]]:
    loader = PdfLoader()
    chunker = get_chunker(
        settings.chunk_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks: list[Chunk] = []
    stats: list[ExtractionStats] = []

    for row in rows:
        path = raw_dir / row["filename"]
        if not path.exists():
            print(f"  missing {path}", file=sys.stderr)
            continue
        doc = await loader.load(path, row)
        if loader.last_stats:
            stats.append(loader.last_stats)
        chunks.extend(await chunker.chunk(doc))
    return chunks, stats


def print_report(
    chunks: list[Chunk], stats: list[ExtractionStats], settings: Settings
) -> None:
    if not stats:
        print("no documents processed")
        return
    sizes = [len(c.text) for c in chunks]
    per_doc = [sum(1 for c in chunks if c.doc_id == s.doc_id) for s in stats]
    degraded = [s for s in stats if s.empty_page_ratio > 0.2]

    print(f"strategy        : {settings.chunk_strategy.value}")
    print(f"chunk size/ovlp : {settings.chunk_size} / {settings.chunk_overlap}")
    print(f"collection      : {settings.collection_name()}")
    print(f"documents       : {len(stats)}")
    print(f"pages           : {sum(s.pages for s in stats)}")
    print(f"chars (cleaned) : {sum(s.chars for s in stats):,}")
    print(f"lines stripped  : {sum(s.stripped_lines for s in stats):,}")
    print(f"chunks          : {len(chunks)}")
    if chunks:
        print(
            f"chunk chars     : min {min(sizes)}  mean {statistics.mean(sizes):.0f}  "
            f"max {max(sizes)}"
        )
        print(f"chunks/doc      : median {statistics.median(per_doc):.0f}")
        print(f"unique ids      : {len({c.chunk_id for c in chunks})}")
    print(f"docs >20% empty : {len(degraded)}")
    for s in degraded:
        print(f"    {s.empty_page_ratio:.0%} empty  {s.filename}")


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    rows = load_manifest(Path(args.manifest))
    chunks, stats = await build_chunks(rows, Path(args.raw_dir), settings)
    print_report(chunks, stats, settings)

    if args.report or args.dry_run:
        print("\n(no embedding or indexing performed)")
        return 0

    if not chunks:
        print("nothing to index")
        return 1

    from retrieval.embedder import build_bedrock_embedder
    from retrieval.vector_store import build_qdrant_store

    embedder = build_bedrock_embedder(settings)
    store = build_qdrant_store(settings)

    print(f"\nembedding {len(chunks)} chunks with {embedder.model_id} ...")
    vectors = await embedder.embed_documents([c.text for c in chunks])

    await store.ensure_collection(recreate=args.recreate)
    written = 0
    for start in range(0, len(chunks), args.batch_size):
        stop = start + args.batch_size
        written += await store.upsert(chunks[start:stop], vectors[start:stop])
        print(f"  upserted {written}/{len(chunks)}")

    total = await store.count()
    await store.close()
    print(f"\ncollection {settings.collection_name()} now holds {total} points")

    from observability.tracing import get_tracer

    await get_tracer().flush()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/manifest.json")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--report", action="store_true", help="extraction stats only; no AWS calls"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="load and chunk but do not index"
    )
    p.add_argument(
        "--recreate", action="store_true", help="drop and recreate the collection"
    )
    return p


def main() -> int:
    return asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
