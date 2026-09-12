from datetime import date

import pytest

from core.models import Chunk, ChunkStrategy, DocumentMeta, ScoredChunk, make_chunk_id
from evaluation.gold import (
    GoldQuestion,
    GoldSet,
    QuestionType,
    normalise,
    validate_evidence,
)
from evaluation.metrics import (
    hit_at_k,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# ---- metrics ---------------------------------------------------------------


def test_hit_at_k():
    assert hit_at_k([False, True, False], 2) == 1.0
    assert hit_at_k([False, False, True], 2) == 0.0
    assert hit_at_k([], 5) == 0.0


def test_recall_at_k_is_capped_by_total_relevant():
    assert recall_at_k([True, True, False], 3, total_relevant=2) == 1.0
    assert recall_at_k([True, False, False], 3, total_relevant=2) == 0.5
    assert recall_at_k([True, True, True], 3, total_relevant=2) == 1.0
    assert recall_at_k([True], 1, total_relevant=0) == 0.0


def test_precision_at_k():
    assert precision_at_k([True, False, True, False], 4) == 0.5
    assert precision_at_k([True], 0) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([True]) == 1.0
    assert reciprocal_rank([False, False]) == 0.0


def test_ndcg_is_one_for_perfect_ranking_and_lower_otherwise():
    assert ndcg_at_k([True, True, False], 3, total_relevant=2) == pytest.approx(1.0)
    worse = ndcg_at_k([False, True, True], 3, total_relevant=2)
    assert 0.0 < worse < 1.0
    assert ndcg_at_k([False, False, False], 3, total_relevant=2) == 0.0


def test_ndcg_rewards_relevant_items_ranked_earlier():
    early = ndcg_at_k([True, False, False, False], 4, total_relevant=1)
    late = ndcg_at_k([False, False, False, True], 4, total_relevant=1)
    assert early > late


def test_mean_of_empty_is_zero():
    assert mean([]) == 0.0
    assert mean([1.0, 0.0]) == 0.5


# ---- gold schema -----------------------------------------------------------


def make_scored(doc_id: str, text: str) -> ScoredChunk:
    meta = DocumentMeta(
        doc_id=doc_id, source_path=f"{doc_id}.pdf", title=doc_id, regulator="RBI"
    )
    chunk = Chunk(
        chunk_id=make_chunk_id(doc_id, ChunkStrategy.RECURSIVE, 0),
        doc_id=doc_id,
        text=text,
        ordinal=0,
        strategy=ChunkStrategy.RECURSIVE,
        meta=meta,
    )
    return ScoredChunk(chunk=chunk, score=0.9, rank=0, stage="dense")


def test_negative_questions_must_not_carry_expected_docs():
    with pytest.raises(ValueError):
        GoldQuestion(
            id="q", question="What is the repo rate today?", reference="n/a",
            question_type=QuestionType.NEGATIVE, expected_doc_ids=["d1"],
        )


def test_positive_questions_need_expected_docs():
    with pytest.raises(ValueError):
        GoldQuestion(
            id="q", question="What is the CRR exemption?", reference="x",
            question_type=QuestionType.SINGLE_HOP,
        )


def test_cross_circular_needs_at_least_two_docs():
    with pytest.raises(ValueError):
        GoldQuestion(
            id="q", question="Which banks got amendments?", reference="x",
            question_type=QuestionType.CROSS_CIRCULAR, expected_doc_ids=["d1"],
        )


def test_chunk_hit_requires_expected_doc_and_evidence_quote():
    q = GoldQuestion(
        id="q", question="How many circulars withdrawn?", reference="seven",
        question_type=QuestionType.SINGLE_HOP, expected_doc_ids=["fema"],
        evidence=["seven circulars as listed at Annex"],
    )
    assert q.chunk_hits(make_scored("fema", "...seven circulars as listed at Annex...")) is True
    assert q.chunk_hits(make_scored("fema", "unrelated text from the same doc")) is False
    assert q.chunk_hits(make_scored("other", "seven circulars as listed at Annex")) is False


def test_evidence_matching_is_whitespace_and_quote_insensitive():
    """PDF text carries curly quotes and odd spacing; the gold set is typed."""
    q = GoldQuestion(
        id="q", question="What date was amended?", reference="x",
        question_type=QuestionType.SINGLE_HOP, expected_doc_ids=["d"],
        evidence=["amend the date 'September 30, 2026'"],
    )
    pdf_text = "to amend   the date ‘September 30,\n2026’ to"
    assert q.chunk_hits(make_scored("d", pdf_text)) is True


def test_doc_hit_ignores_evidence():
    q = GoldQuestion(
        id="q", question="Anything about FEMA?", reference="x",
        question_type=QuestionType.SINGLE_HOP, expected_doc_ids=["fema"],
        evidence=["a quote that is not in the chunk"],
    )
    assert q.doc_hits(make_scored("fema", "some text")) is True


def test_gold_set_rejects_duplicate_ids():
    q = dict(question="What is the CRR exemption?", reference="x",
             question_type=QuestionType.SINGLE_HOP, expected_doc_ids=["d"])
    with pytest.raises(ValueError):
        GoldSet(questions=[GoldQuestion(id="q1", **q), GoldQuestion(id="q1", **q)])


def test_validate_evidence_flags_missing_docs_and_quotes():
    gold = GoldSet(
        questions=[
            GoldQuestion(
                id="ok", question="How many circulars?", reference="seven",
                question_type=QuestionType.SINGLE_HOP, expected_doc_ids=["fema"],
                evidence=["seven circulars"],
            ),
            GoldQuestion(
                id="bad_quote", question="What is the tenor?", reference="x",
                question_type=QuestionType.SINGLE_HOP, expected_doc_ids=["fema"],
                evidence=["this quote does not exist"],
            ),
            GoldQuestion(
                id="bad_doc", question="Which districts formed?", reference="x",
                question_type=QuestionType.SINGLE_HOP, expected_doc_ids=["ghost"],
            ),
        ]
    )
    problems = validate_evidence(gold, {"fema": "Seven circulars are withdrawn."})
    reasons = {(p.question_id, p.reason) for p in problems}
    assert ("bad_quote", "evidence quote not found in any expected doc") in reasons
    assert ("bad_doc", "expected_doc_id not in corpus") in reasons
    assert not any(p.question_id == "ok" for p in problems)


def test_normalise_collapses_typographic_variants():
    assert normalise("‘A’  –  “B”") == normalise("'a' - \"b\"")
