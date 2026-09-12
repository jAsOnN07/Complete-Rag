from datetime import date

import pytest

from core.models import Chunk, ChunkStrategy, DocumentMeta, ScoredChunk, make_chunk_id
from generation.prompt import (
    NOT_FOUND_SENTINEL,
    PROMPT_VERSION,
    PromptBuilder,
    is_not_found_response,
    parse_citations,
)


def make_scored(n: int) -> list[ScoredChunk]:
    out = []
    for i in range(n):
        meta = DocumentMeta(
            doc_id=f"doc{i}",
            source_path=f"data/raw/doc{i}.pdf",
            title=f"Circular {i}",
            regulator="RBI",
            circular_no=f"RBI/2026-27/{200 + i}",
            issued_on=date(2026, 9, 1 + i),
        )
        chunk = Chunk(
            chunk_id=make_chunk_id(f"doc{i}", ChunkStrategy.RECURSIVE, 0),
            doc_id=f"doc{i}",
            text=f"Body text for circular {i}.",
            ordinal=0,
            strategy=ChunkStrategy.RECURSIVE,
            page=i + 1,
            meta=meta,
        )
        out.append(ScoredChunk(chunk=chunk, score=1.0 - i * 0.1, rank=i, stage="dense"))
    return out


@pytest.fixture
def chunks() -> list[ScoredChunk]:
    return make_scored(3)


def test_labels_are_one_indexed_and_contiguous(chunks):
    payload = PromptBuilder().build("What is the rule?", chunks)
    assert list(payload.label_map) == [1, 2, 3]


def test_context_shows_labels_not_chunk_ids(chunks):
    """Models emit [3] reliably and mangle 'doc0::recursive::0'."""
    payload = PromptBuilder().build("What is the rule?", chunks)
    assert "[1]" in payload.user and "[3]" in payload.user
    for scored in chunks:
        assert scored.chunk.chunk_id not in payload.user


def test_context_carries_provenance_for_each_label(chunks):
    payload = PromptBuilder().build("q", chunks)
    assert "RBI/2026-27/200" in payload.user
    assert "Circular 0" in payload.user


def test_system_prompt_defends_against_injection_from_corpus(chunks):
    """Context comes from third-party PDFs, so it must be framed as data."""
    payload = PromptBuilder().build("q", chunks)
    lowered = payload.system.lower()
    assert "instruction" in lowered
    assert "data" in lowered


def test_system_prompt_specifies_the_exact_not_found_sentinel(chunks):
    payload = PromptBuilder().build("q", chunks)
    assert NOT_FOUND_SENTINEL in payload.system


def test_prompt_version_is_recorded(chunks):
    assert PromptBuilder().build("q", chunks).prompt_version == PROMPT_VERSION


def test_parse_citations_resolves_labels_to_chunks(chunks):
    payload = PromptBuilder().build("q", chunks)
    answer = "Banks must do X [1]. Also see [3]."
    citations, invalid = parse_citations(answer, payload.label_map)
    assert invalid == []
    assert [c.chunk_id for c in citations] == [
        chunks[0].chunk.chunk_id,
        chunks[2].chunk.chunk_id,
    ]
    assert citations[0].circular_no == "RBI/2026-27/200"


def test_parse_citations_deduplicates_repeated_labels(chunks):
    payload = PromptBuilder().build("q", chunks)
    citations, invalid = parse_citations("A [1]. B [1]. C [1].", payload.label_map)
    assert len(citations) == 1
    assert invalid == []


def test_parse_citations_flags_hallucinated_labels(chunks):
    """The model citing [9] when given 3 chunks is a measurable quality signal."""
    payload = PromptBuilder().build("q", chunks)
    citations, invalid = parse_citations("Claim [9] and [2].", payload.label_map)
    assert invalid == [9]
    assert [c.chunk_id for c in citations] == [chunks[1].chunk.chunk_id]


def test_parse_citations_handles_grouped_labels(chunks):
    payload = PromptBuilder().build("q", chunks)
    citations, invalid = parse_citations("Per [1, 2] this holds.", payload.label_map)
    assert invalid == []
    assert len(citations) == 2


def test_parse_citations_on_answer_without_citations(chunks):
    payload = PromptBuilder().build("q", chunks)
    citations, invalid = parse_citations("No references here.", payload.label_map)
    assert citations == []
    assert invalid == []


def test_is_not_found_response_detects_the_sentinel():
    assert is_not_found_response(NOT_FOUND_SENTINEL) is True
    assert is_not_found_response(f"  {NOT_FOUND_SENTINEL}  ") is True
    assert is_not_found_response("The circular states that banks must...") is False


def test_grounding_sources_are_the_raw_chunk_texts(chunks):
    sources = PromptBuilder().grounding_sources(chunks)
    assert sources == [c.chunk.text for c in chunks]


def test_parse_citations_accepts_fullwidth_brackets(chunks):
    """gpt-oss emits 【5】 (CJK fullwidth) for citations; dropping them is a parser bug."""
    payload = PromptBuilder().build("q", chunks)
    citations, invalid = parse_citations("Issued for commercial banks 【2】.", payload.label_map)
    assert invalid == []
    assert [c.chunk_id for c in citations] == [chunks[1].chunk.chunk_id]


def test_parse_citations_accepts_adjacent_brackets(chunks):
    payload = PromptBuilder().build("q", chunks)
    citations, _ = parse_citations("Districts are X [1][2].", payload.label_map)
    assert len(citations) == 2
