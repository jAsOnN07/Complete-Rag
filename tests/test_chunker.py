from datetime import date

import pytest

from core.models import Chunk, ChunkStrategy, DocumentMeta, Page, RawDocument
from ingestion.chunker import RecursiveChunker, get_chunker


def build_doc(page_texts: list[str], doc_id: str = "doc123") -> RawDocument:
    meta = DocumentMeta(
        doc_id=doc_id,
        source_path=f"data/raw/{doc_id}.pdf",
        title="Master Direction on KYC",
        regulator="RBI",
        circular_no="RBI/2026-27/254",
        issued_on=date(2026, 9, 8),
    )
    return RawDocument(
        meta=meta, pages=[Page.build(i, t) for i, t in enumerate(page_texts, start=1)]
    )


@pytest.fixture
def doc() -> RawDocument:
    para = (
        "Banks shall carry out customer due diligence at the time of commencement "
        "of an account-based relationship, and shall periodically update records. "
    )
    return build_doc([para * 6, para * 6, para * 6])


async def test_chunker_produces_chunks(doc):
    chunks = await RecursiveChunker(chunk_size=400, chunk_overlap=50).chunk(doc)
    assert chunks
    assert all(isinstance(c, Chunk) for c in chunks)


async def test_chunk_ids_are_deterministic_across_runs(doc):
    chunker = RecursiveChunker(chunk_size=400, chunk_overlap=50)
    first = await chunker.chunk(doc)
    second = await chunker.chunk(doc)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


async def test_chunk_ids_encode_strategy_so_collections_never_collide(doc):
    chunks = await RecursiveChunker(chunk_size=400, chunk_overlap=50).chunk(doc)
    assert all("::recursive::" in c.chunk_id for c in chunks)
    assert all(c.strategy is ChunkStrategy.RECURSIVE for c in chunks)


async def test_ordinals_are_contiguous_from_zero(doc):
    chunks = await RecursiveChunker(chunk_size=400, chunk_overlap=50).chunk(doc)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


async def test_every_chunk_carries_document_metadata_for_citation(doc):
    chunks = await RecursiveChunker(chunk_size=400, chunk_overlap=50).chunk(doc)
    for c in chunks:
        assert c.meta.circular_no == "RBI/2026-27/254"
        assert c.meta.title == "Master Direction on KYC"
        assert c.doc_id == doc.meta.doc_id


async def test_pages_are_assigned_and_within_document_range(doc):
    chunks = await RecursiveChunker(chunk_size=400, chunk_overlap=50).chunk(doc)
    pages = {c.page for c in chunks}
    assert None not in pages
    assert min(pages) >= 1
    assert max(pages) <= len(doc.pages)


async def test_page_numbers_are_non_decreasing(doc):
    """Chunks are emitted in document order, so pages must not go backwards."""
    chunks = await RecursiveChunker(chunk_size=400, chunk_overlap=50).chunk(doc)
    pages = [c.page for c in chunks]
    assert pages == sorted(pages)


async def test_no_blank_chunks(doc):
    chunks = await RecursiveChunker(chunk_size=400, chunk_overlap=50).chunk(doc)
    assert all(c.text.strip() for c in chunks)


async def test_smaller_chunk_size_yields_more_chunks(doc):
    small = await RecursiveChunker(chunk_size=300, chunk_overlap=0).chunk(doc)
    large = await RecursiveChunker(chunk_size=1200, chunk_overlap=0).chunk(doc)
    assert len(small) > len(large)


async def test_empty_document_yields_no_chunks():
    assert await RecursiveChunker(chunk_size=400, chunk_overlap=50).chunk(
        build_doc([""])
    ) == []


def test_params_hash_changes_with_settings():
    """Eval runs are only comparable when chunking params match."""
    a = RecursiveChunker(chunk_size=400, chunk_overlap=50).params_hash
    b = RecursiveChunker(chunk_size=800, chunk_overlap=50).params_hash
    c = RecursiveChunker(chunk_size=400, chunk_overlap=50).params_hash
    assert a == c
    assert a != b


def test_get_chunker_returns_recursive_by_default():
    chunker = get_chunker(ChunkStrategy.RECURSIVE, chunk_size=500, chunk_overlap=60)
    assert isinstance(chunker, RecursiveChunker)
    assert chunker.strategy is ChunkStrategy.RECURSIVE


def test_get_chunker_rejects_unimplemented_strategies():
    """Fixed and semantic land at M5; failing loudly beats silently using recursive."""
    with pytest.raises(NotImplementedError):
        get_chunker(ChunkStrategy.SEMANTIC, chunk_size=500, chunk_overlap=60)
