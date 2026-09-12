from datetime import date

import pytest
from pydantic import ValidationError

from core.models import (
    Answer,
    Chunk,
    ChunkStrategy,
    Citation,
    DocumentMeta,
    ScoredChunk,
    make_chunk_id,
    make_doc_id,
)


@pytest.fixture
def meta() -> DocumentMeta:
    return DocumentMeta(
        doc_id=make_doc_id("data/raw/rbi_kyc_2023.pdf"),
        source_path="data/raw/rbi_kyc_2023.pdf",
        title="Master Direction on KYC",
        regulator="RBI",
        circular_no="DOR.AML.REC.27/14.01.001/2023-24",
        issued_on=date(2023, 10, 17),
        source_url="https://rbi.org.in/example",
    )


def test_doc_id_is_deterministic_and_path_normalised():
    assert make_doc_id("data/raw/a.pdf") == make_doc_id(r"data\raw\a.pdf")
    assert make_doc_id("data/raw/a.pdf") != make_doc_id("data/raw/b.pdf")


def test_chunk_id_encodes_strategy_so_strategies_can_coexist(meta):
    recursive = make_chunk_id(meta.doc_id, ChunkStrategy.RECURSIVE, 3)
    fixed = make_chunk_id(meta.doc_id, ChunkStrategy.FIXED, 3)
    assert recursive != fixed
    assert recursive.endswith("::recursive::3")


def test_chunk_rejects_blank_text(meta):
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id=make_chunk_id(meta.doc_id, ChunkStrategy.RECURSIVE, 0),
            doc_id=meta.doc_id,
            text="   ",
            ordinal=0,
            strategy=ChunkStrategy.RECURSIVE,
            meta=meta,
        )


def test_chunk_to_citation_carries_provenance(meta):
    chunk = Chunk(
        chunk_id=make_chunk_id(meta.doc_id, ChunkStrategy.RECURSIVE, 0),
        doc_id=meta.doc_id,
        text="Banks shall carry out KYC due diligence.",
        ordinal=0,
        strategy=ChunkStrategy.RECURSIVE,
        page=4,
        meta=meta,
    )
    citation = chunk.to_citation()
    assert isinstance(citation, Citation)
    assert citation.chunk_id == chunk.chunk_id
    assert citation.circular_no == meta.circular_no
    assert citation.page == 4


def test_scored_chunk_exposes_chunk_id_passthrough(meta):
    chunk = Chunk(
        chunk_id=make_chunk_id(meta.doc_id, ChunkStrategy.RECURSIVE, 1),
        doc_id=meta.doc_id,
        text="text",
        ordinal=1,
        strategy=ChunkStrategy.RECURSIVE,
        meta=meta,
    )
    scored = ScoredChunk(chunk=chunk, score=0.82, rank=0, stage="dense")
    assert scored.chunk_id == chunk.chunk_id


def test_not_found_answer_has_no_citations_and_is_grounded():
    answer = Answer.not_found_response(provider="bedrock")
    assert answer.not_found is True
    assert answer.citations == []
    assert answer.grounded is True


def test_answer_with_citations_cannot_claim_not_found(meta):
    citation = Citation(
        chunk_id="d::recursive::0",
        doc_id=meta.doc_id,
        title=meta.title,
        circular_no=meta.circular_no,
        page=1,
    )
    with pytest.raises(ValidationError):
        Answer(
            answer="Something",
            citations=[citation],
            grounded=True,
            not_found=True,
            provider="bedrock",
        )
