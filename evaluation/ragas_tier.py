"""RAGAS scoring: faithfulness, answer relevancy, context precision, context recall.

The judge is whatever the gateway serves (Gemini primary, Groq fallback) so
judging goes through the same routing, retries and cost accounting as
generation - and the judge model is recorded in every result, because a
score from a different judge is a different number. Embeddings for answer
relevancy are Cohere, via a thin adapter around the SDK so no extra
LangChain integration package is needed.

Run on top of a generation-tier result rather than regenerating: the results
JSON stores the retrieved contexts, so re-scoring is judge-only spend.
"""

from __future__ import annotations

from typing import Any, Sequence

from pydantic import BaseModel, Field

from core.config import Settings

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


class RagasSample(BaseModel):
    question_id: str
    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str


class RagasScores(BaseModel):
    judge_model: str
    embedding_model: str
    per_question: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    means: dict[str, float | None] = Field(default_factory=dict)
    scored: int = 0
    skipped: list[str] = Field(default_factory=list)


def judge_model(settings: Settings) -> str:
    return settings.ragas_judge_model or settings.llm_primary_model


def build_judge(settings: Settings) -> Any:
    """The gateway as a LangChain chat model, wrapped for RAGAS."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    if settings.portkey_api_key is None:
        raise RuntimeError("PORTKEY_API_KEY is not set")
    headers = {"x-portkey-api-key": settings.portkey_api_key.get_secret_value()}
    if settings.ragas_judge_model:
        # A dedicated judge: direct provider call, no fallback chain (RAGAS
        # retries with backoff; a judge silently swapped mid-run is worse
        # than a NaN).
        headers["x-portkey-provider"] = settings.ragas_judge_provider or settings.llm_primary_provider
    elif settings.portkey_config_slug:
        headers["x-portkey-config"] = settings.portkey_config_slug
    chat = ChatOpenAI(
        base_url=settings.portkey_base_url,
        api_key="portkey",  # Portkey authenticates via x-portkey-api-key
        default_headers=headers,
        model=judge_model(settings),
        temperature=0.0,
        # Judging is extraction, not reasoning: a thinking model at default
        # effort spent >120 s on one faithfulness prompt and timed out.
        reasoning_effort="low",
        max_retries=0,  # retries live in the Portkey config, not here
        timeout=180,
    )
    return LangchainLLMWrapper(chat)


def build_embeddings(settings: Settings) -> Any:
    """RAGAS embeddings over the Cohere SDK (sync client: RAGAS drives its own
    executor, and a sync client avoids event-loop juggling inside it)."""
    import cohere
    from ragas.embeddings import BaseRagasEmbeddings

    if settings.cohere_api_key is None:
        raise RuntimeError("COHERE_API_KEY is not set")
    client = cohere.ClientV2(api_key=settings.cohere_api_key.get_secret_value())
    model, dim = settings.cohere_embed_model, settings.cohere_embed_dim

    class CohereRagasEmbeddings(BaseRagasEmbeddings):
        def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
            out: list[list[float]] = []
            for start in range(0, len(texts), 96):
                r = client.embed(
                    model=model, input_type=input_type, texts=list(texts[start : start + 96]),
                    embedding_types=["float"], output_dimension=dim,
                )
                out.extend([list(v) for v in r.embeddings.float_])
            return out

        def embed_query(self, text: str) -> list[float]:
            return self._embed([text], "search_query")[0]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._embed(texts, "search_document")

        async def aembed_query(self, text: str) -> list[float]:
            return self.embed_query(text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embed_documents(texts)

    return CohereRagasEmbeddings()


def score_samples(
    samples: Sequence[RagasSample],
    *,
    settings: Settings,
    max_workers: int = 2,
    judge: Any | None = None,
    embeddings: Any | None = None,
) -> RagasScores:
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.metrics import (
        AnswerRelevancy,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    judge = judge or build_judge(settings)
    embeddings = embeddings or build_embeddings(settings)
    metrics = [
        Faithfulness(llm=judge),
        # strictness = number of questions generated per answer, sent as
        # OpenAI `n`. The Groq fallback rejects n > 1, so one question, which
        # also keeps judge spend at one call per metric.
        AnswerRelevancy(llm=judge, embeddings=embeddings, strictness=1),
        LLMContextPrecisionWithReference(llm=judge),
        LLMContextRecall(llm=judge),
    ]
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=s.user_input, response=s.response,
                retrieved_contexts=s.retrieved_contexts, reference=s.reference,
            )
            for s in samples
        ]
    )
    # max_workers is low on purpose: the judge sits behind free-tier rate
    # limits (Gemini per-minute, and Groq's 8k TPM once the gateway fails
    # over). The gateway retries first; RAGAS then backs off up to two
    # minutes before a metric is given up as NaN.
    result = evaluate(
        dataset=dataset, metrics=metrics, llm=judge, embeddings=embeddings,
        run_config=RunConfig(max_workers=max_workers, timeout=240, max_retries=6, max_wait=120),
        show_progress=False,
    )
    frame = result.to_pandas()

    scores = RagasScores(judge_model=judge_model(settings), embedding_model=settings.cohere_embed_model)
    columns = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "context_precision": "llm_context_precision_with_reference",
        "context_recall": "context_recall",
    }
    for s, (_, row) in zip(samples, frame.iterrows()):
        per: dict[str, float | None] = {}
        for name, col in columns.items():
            value = row.get(col)
            per[name] = None if value is None or value != value else float(value)  # NaN guard
        scores.per_question[s.question_id] = per
    for name in columns:
        values = [v[name] for v in scores.per_question.values() if v.get(name) is not None]
        scores.means[name] = round(sum(values) / len(values), 4) if values else None
    scores.scored = len(samples)
    return scores
