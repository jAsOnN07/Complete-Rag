"""Scoring and aggregation in run_eval are pure; test them without a service."""

from __future__ import annotations

from datetime import date

from core.models import Chunk, ChunkStrategy, DocumentMeta, ScoredChunk, make_chunk_id
from evaluation.gold import GoldQuestion, GoldSet, QuestionType
from evaluation.run_eval import EvalResult, aggregate, score_retrieval


def scored(doc_id: str, text: str, score: float, rank: int) -> ScoredChunk:
    meta = DocumentMeta(doc_id=doc_id, source_path="x", title=doc_id, regulator="RBI")
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=make_chunk_id(doc_id, ChunkStrategy.RECURSIVE, rank),
            doc_id=doc_id, text=text, ordinal=rank,
            strategy=ChunkStrategy.RECURSIVE, meta=meta,
        ),
        score=score, rank=rank, stage="dense",
    )


def positive(id: str = "p", docs: list[str] | None = None, evidence: list[str] | None = None):
    return GoldQuestion(
        id=id, question="How many circulars are withdrawn?", reference="seven",
        question_type=QuestionType.SINGLE_HOP if len(docs or ["a"]) == 1 else QuestionType.CROSS_CIRCULAR,
        expected_doc_ids=docs or ["a"], evidence=evidence or [],
    )


def negative(id: str = "n"):
    return GoldQuestion(
        id=id, question="What is the capital of France?", reference="n/a",
        question_type=QuestionType.NEGATIVE,
    )


def test_perfect_single_hop_scores_one_everywhere():
    q = positive(evidence=["seven circulars"])
    rows = [scored("a", "seven circulars withdrawn", 0.9, 0), scored("b", "noise", 0.5, 1)]
    r = score_retrieval(q, rows, k=2, threshold=0.55)
    assert r.hit_at_1 == 1.0 and r.mrr == 1.0 and r.ndcg_at_k == 1.0
    assert r.doc_recall_at_k == 1.0 and r.chunk_recall_at_k == 1.0
    assert r.negative_gate_correct is None


def test_doc_found_but_evidence_missing_separates_chunk_and_doc_recall():
    """The classic dense failure: right document, wrong chunk."""
    q = positive(evidence=["DOR.AML.REC.219"])
    rows = [scored("a", "some other paragraph of the same circular", 0.8, 0)]
    r = score_retrieval(q, rows, k=1, threshold=0.55)
    assert r.doc_recall_at_k == 1.0
    assert r.chunk_recall_at_k == 0.0
    assert r.mrr == 0.0


def test_doc_recall_counts_distinct_docs_not_repeated_hits():
    """Six chunks from one circular is not six-out-of-six."""
    q = positive(docs=["a", "b", "c"])
    rows = [scored("a", "x", 0.9, i) for i in range(5)]
    r = score_retrieval(q, rows, k=5, threshold=0.55)
    assert r.doc_recall_at_k == 1 / 3


def test_negative_gate_is_correct_only_when_top_score_is_under_threshold():
    q = negative()
    assert score_retrieval(q, [scored("z", "x", 0.40, 0)], 1, 0.55).negative_gate_correct is True
    assert score_retrieval(q, [scored("z", "x", 0.70, 0)], 1, 0.55).negative_gate_correct is False
    assert score_retrieval(q, [], 1, 0.55).negative_gate_correct is True


def test_aggregate_splits_positives_and_negatives():
    gold = GoldSet(questions=[positive("p1", evidence=["seven circulars"]), negative("n1")])
    result = EvalResult(
        run_id="r", started_at="t", tier="retrieval", k=1, config_fingerprint="f",
        config={}, gold_path="g", gold_version=1, questions_evaluated=2,
        unverified_questions=2,
    )
    result.retrieval = [
        score_retrieval(gold.questions[0], [scored("a", "seven circulars", 0.9, 0)], 1, 0.55),
        score_retrieval(gold.questions[1], [scored("z", "x", 0.4, 0)], 1, 0.55),
    ]
    agg = aggregate(result, gold)
    assert agg["retrieval"]["n"] == 1
    assert agg["retrieval"]["hit@1"] == 1.0
    assert agg["negative_gate_accuracy"] == 1.0
    assert "single_hop" in agg["by_type"]


def test_ndcg_never_exceeds_one_when_a_doc_yields_several_relevant_chunks():
    """Several evidence-bearing chunks from one circular is one relevant document."""
    q = positive(evidence=["seven circulars"])
    rows = [scored("a", "seven circulars withdrawn", 0.9, i) for i in range(4)]
    r = score_retrieval(q, rows, k=4, threshold=0.55)
    assert r.ndcg_at_k <= 1.0
    assert r.ndcg_at_k == 1.0
