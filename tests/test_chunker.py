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


# ---- fixed --------------------------------------------------------------------

from ingestion.chunker import FixedChunker, SemanticChunker, split_sentences  # noqa: E402
from tests.fakes import FakeEmbedder  # noqa: E402


async def test_fixed_chunks_are_at_most_chunk_size(doc):
    chunks = await FixedChunker(chunk_size=300, chunk_overlap=50).chunk(doc)
    assert chunks
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.strategy is ChunkStrategy.FIXED for c in chunks)
    assert all("::fixed::" in c.chunk_id for c in chunks)


async def test_fixed_windows_overlap(doc):
    chunks = await FixedChunker(chunk_size=300, chunk_overlap=50).chunk(doc)
    assert len(chunks) >= 2
    # The tail of chunk n appears at the head of chunk n+1.
    tail = chunks[0].text[-30:]
    assert tail in chunks[1].text


async def test_fixed_ignores_separators_entirely():
    """Fixed splits mid-word by design; that is what the comparison is for."""
    text = "word " * 200
    chunks = await FixedChunker(chunk_size=97, chunk_overlap=0).chunk(build_doc([text]))
    assert any(not c.text.endswith("word") for c in chunks)


async def test_fixed_pages_assigned(doc):
    chunks = await FixedChunker(chunk_size=300, chunk_overlap=50).chunk(doc)
    assert all(c.page is not None for c in chunks)


# ---- semantic -------------------------------------------------------------------


def test_split_sentences_respects_numbered_clauses_and_abbreviations():
    text = (
        "Please refer to the Directions dated November 28, 2025. 2. On a review, it has "
        "been decided to amend the date. 3. These Directions shall come into effect from "
        "October 01, 2026.\nYours faithfully"
    )
    sents = split_sentences(text)
    assert len(sents) >= 3
    assert all(s.strip() for s in sents)
    assert "".join(s + " " for s in sents).replace("  ", " ").strip().startswith("Please refer")


def test_split_sentences_on_empty():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


async def test_semantic_chunks_carry_strategy_and_ids():
    para_a = "Banks shall verify customer identity. KYC records shall be updated. Due diligence is mandatory. "
    para_b = "Interest on savings deposits is calculated daily. Deposit rates shall be uniform. Tenor determines rate. "
    doc = build_doc([para_a * 3 + para_b * 3])
    chunker = SemanticChunker(embedder=FakeEmbedder(), chunk_size=600, chunk_overlap=0)
    chunks = await chunker.chunk(doc)
    assert chunks
    assert all(c.strategy is ChunkStrategy.SEMANTIC for c in chunks)
    assert all("::semantic::" in c.chunk_id for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


async def test_semantic_breaks_at_the_topic_shift():
    """Two clearly different topics should not end up in one chunk."""
    kyc = "Banks shall verify customer identity. Know your customer records must be updated. Customer due diligence is mandatory for banks. "
    dep = "Interest on savings deposits is computed on daily balances. Deposit interest rates shall be uniform. Savings deposit tenor determines the rate. "
    doc = build_doc([kyc * 4 + dep * 4])
    chunker = SemanticChunker(embedder=FakeEmbedder(), chunk_size=2000, chunk_overlap=0, breakpoint_percentile=80)
    chunks = await chunker.chunk(doc)
    assert len(chunks) >= 2
    first, last = chunks[0].text.lower(), chunks[-1].text.lower()
    assert "customer" in first and "customer" not in last[-200:]
    assert "deposit" in last


async def test_semantic_never_exceeds_chunk_size_by_much():
    text = ("Clause one is here. " * 40 + "\n") * 5
    chunker = SemanticChunker(embedder=FakeEmbedder(), chunk_size=400, chunk_overlap=0)
    chunks = await chunker.chunk(build_doc([text]))
    assert all(len(c.text) <= 400 + 50 for c in chunks)


async def test_semantic_single_sentence_document():
    chunks = await SemanticChunker(embedder=FakeEmbedder(), chunk_size=400, chunk_overlap=0).chunk(
        build_doc(["Just one sentence here."])
    )
    assert len(chunks) == 1


def test_get_chunker_builds_every_strategy():
    assert isinstance(get_chunker(ChunkStrategy.FIXED, chunk_size=500, chunk_overlap=60), FixedChunker)
    assert isinstance(
        get_chunker(ChunkStrategy.SEMANTIC, chunk_size=500, chunk_overlap=60, embedder=FakeEmbedder()),
        SemanticChunker,
    )


def test_get_chunker_semantic_requires_an_embedder():
    with pytest.raises(ValueError):
        get_chunker(ChunkStrategy.SEMANTIC, chunk_size=500, chunk_overlap=60)


def test_params_hash_differs_across_strategies():
    r = RecursiveChunker(chunk_size=500, chunk_overlap=60).params_hash
    f = FixedChunker(chunk_size=500, chunk_overlap=60).params_hash
    assert r != f


def test_split_sentences_glues_clause_markers_to_their_sentence():
    """'2.' is a clause number, not a sentence; it must not become its own fragment."""
    sents = split_sentences("Refer to the Directions. 2. On a review it was decided. 3. Effective October 01, 2026.")
    assert all(len(s) >= 15 for s in sents), sents
    assert any(s.startswith("2. On a review") for s in sents)


async def test_semantic_produces_no_fragment_chunks():
    text = ("Refer to the Directions dated November 28, 2025. 2. On a review it has been decided "
            "to amend the date. 3. These Directions shall come into effect from October 01, 2026. ") * 6
    chunks = await SemanticChunker(embedder=FakeEmbedder(), chunk_size=400, chunk_overlap=0).chunk(
        build_doc([text])
    )
    assert all(len(c.text) >= 60 for c in chunks), [len(c.text) for c in chunks]
