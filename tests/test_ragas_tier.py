"""RAGAS tier: sample selection, score mapping and re-scoring are testable
without a judge. The judge itself is an integration concern."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import Settings
from evaluation.gold import GoldQuestion, GoldSet, QuestionType
from evaluation.run_eval import EvalResult, GenerationRow, aggregate, ragas_samples, rescore
from evaluation import ragas_tier


def gold() -> GoldSet:
    return GoldSet(
        questions=[
            GoldQuestion(
                id="p1", question="How many circulars?", reference="seven",
                question_type=QuestionType.SINGLE_HOP, expected_doc_ids=["a"],
            ),
            GoldQuestion(
                id="p2", question="When is it effective?", reference="1 Oct 2026",
                question_type=QuestionType.SINGLE_HOP, expected_doc_ids=["b"],
            ),
            GoldQuestion(id="n1", question="Capital of France?", reference="n/a",
                         question_type=QuestionType.NEGATIVE),
        ]
    )


def gen(qid: str, answer: str = "Seven [1].", *, not_found: bool = False,
        error: str | None = None, texts: list[str] | None = None) -> GenerationRow:
    return GenerationRow(
        question_id=qid, answer=answer, error=error, not_found=not_found,
        citations=["a::fixed::0"] if not not_found else [], invalid_citations=[],
        invalid_citation_rate=0.0, provider="google", model_id="gemini", input_tokens=10,
        output_tokens=5, latency_ms=100.0,
        retrieved_texts=["seven circulars withdrawn"] if texts is None else texts,
    )


def result(rows: list[GenerationRow]) -> EvalResult:
    return EvalResult(
        run_id="r", started_at="t", tier="generation", k=5, config_fingerprint="f",
        config={}, gold_path="g", gold_version=1, questions_evaluated=3,
        unverified_questions=0, generation=rows,
    )


def test_samples_are_answered_positives_only():
    r = result([gen("p1"), gen("p2", not_found=True), gen("n1", not_found=True)])
    samples, skipped = ragas_samples(r, gold())
    assert [s.question_id for s in samples] == ["p1"]
    assert skipped == ["p2"]  # a refused positive is skipped and named; negatives never enter
    assert samples[0].reference == "seven" and samples[0].retrieved_contexts == ["seven circulars withdrawn"]


def test_errored_and_contextless_rows_are_skipped():
    r = result([gen("p1", error="boom"), gen("p2", texts=[])])
    samples, skipped = ragas_samples(r, gold())
    assert samples == [] and skipped == ["p1", "p2"]


def test_score_samples_maps_columns_and_handles_nan(monkeypatch):
    import pandas as pd
    import ragas

    frame = pd.DataFrame(
        {
            "faithfulness": [1.0, 0.5],
            "answer_relevancy": [0.9, float("nan")],
            "llm_context_precision_with_reference": [1.0, 1.0],
            "context_recall": [1.0, 0.0],
        }
    )

    class Result:
        def to_pandas(self):
            return frame

    seen: dict = {}

    def fake_evaluate(dataset, metrics, llm, embeddings, run_config, show_progress):
        seen["n"] = len(dataset)
        seen["metrics"] = sorted(m.name for m in metrics)
        seen["workers"] = run_config.max_workers
        return Result()

    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)
    samples = [
        ragas_tier.RagasSample(question_id="p1", user_input="q", response="a",
                               retrieved_contexts=["c"], reference="r"),
        ragas_tier.RagasSample(question_id="p2", user_input="q", response="a",
                               retrieved_contexts=["c"], reference="r"),
    ]
    settings = Settings(llm_primary_model="judge-x")
    scores = ragas_tier.score_samples(
        samples, settings=settings, judge=object(), embeddings=object(), max_workers=3
    )
    assert seen == {
        "n": 2, "workers": 3,
        "metrics": ["answer_relevancy", "context_recall", "faithfulness",
                    "llm_context_precision_with_reference"],
    }
    assert scores.judge_model == "judge-x"
    assert scores.per_question["p2"]["answer_relevancy"] is None
    assert scores.means == {
        "faithfulness": 0.75, "answer_relevancy": 0.9, "context_precision": 1.0, "context_recall": 0.5,
    }


def test_aggregate_surfaces_ragas_means_with_judge():
    r = result([gen("p1")])
    r.ragas = {
        "judge_model": "judge-x", "embedding_model": "e", "scored": 1, "skipped": ["p2"],
        "per_question": {}, "means": {"faithfulness": 1.0},
    }
    out = aggregate(r, gold())
    assert out["ragas"] == {"judge_model": "judge-x", "scored": 1, "skipped": ["p2"], "faithfulness": 1.0}


def test_rescore_reuses_saved_answers_and_only_runs_the_judge(tmp_path: Path, monkeypatch):
    saved = result([gen("p1"), gen("n1", not_found=True)])
    path = tmp_path / "run.json"
    path.write_text(saved.model_dump_json(), encoding="utf-8")
    calls: list[list[str]] = []

    def fake_score_samples(samples, *, settings, **kw):
        calls.append([s.question_id for s in samples])
        return ragas_tier.RagasScores(
            judge_model="judge-x", embedding_model="e", scored=len(samples),
            per_question={s.question_id: {"faithfulness": 1.0} for s in samples},
            means={"faithfulness": 1.0},
        )

    monkeypatch.setattr(ragas_tier, "score_samples", fake_score_samples)
    out = rescore(path, gold(), Settings(), quiet=True)
    assert calls == [["p1"]]
    assert out.tier == "ragas" and out.run_id == "r"  # same run, so the file updates in place
    assert out.aggregates["ragas"]["faithfulness"] == 1.0
    assert [g.answer for g in out.generation] == [g.answer for g in saved.generation]


def test_rescore_refuses_runs_without_contexts(tmp_path: Path):
    saved = result([gen("p1", texts=[])])
    path = tmp_path / "old.json"
    path.write_text(saved.model_dump_json(), encoding="utf-8")
    with pytest.raises(SystemExit, match="retrieved_texts"):
        rescore(path, gold(), Settings(), quiet=True)


def test_old_result_files_still_load():
    """Results written before retrieved_texts existed must still parse."""
    raw = json.loads(result([gen("p1")]).model_dump_json())
    del raw["generation"][0]["retrieved_texts"]
    del raw["ragas"]
    loaded = EvalResult.model_validate(raw)
    assert loaded.generation[0].retrieved_texts == [] and loaded.ragas is None


def test_judge_defaults_to_the_gateway_config_and_can_be_dedicated(monkeypatch):
    from pydantic import SecretStr

    built: dict = {}

    class FakeChat:
        def __init__(self, **kw):
            built.update(kw)

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChat)
    monkeypatch.setattr("ragas.llms.LangchainLLMWrapper", lambda chat: chat)

    base = dict(portkey_api_key=SecretStr("pk"), portkey_config_slug="pc-x", llm_primary_model="gen-model")
    ragas_tier.build_judge(Settings(**base))
    assert built["model"] == "gen-model"
    assert built["default_headers"]["x-portkey-config"] == "pc-x"
    assert "x-portkey-provider" not in built["default_headers"]

    ragas_tier.build_judge(Settings(**base, ragas_judge_model="judge-model", ragas_judge_provider="@other"))
    assert built["model"] == "judge-model"
    assert built["default_headers"]["x-portkey-provider"] == "@other"
    assert "x-portkey-config" not in built["default_headers"]  # no fallback chain for a dedicated judge
    assert ragas_tier.judge_model(Settings(**base, ragas_judge_model="")) == "gen-model"  # blank = unset
