from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.models import ChunkStrategy

RerankerBackend = Literal["cross_encoder", "cohere", "bedrock", "none"]
EmbeddingBackend = Literal["cohere", "bedrock", "fastembed"]
LlmBackend = Literal["portkey", "bedrock"]
RetrievalMode = Literal["dense", "hybrid"]
OutputGuardBackend = Literal["bedrock", "none"]
InjectionBackend = Literal["groq_prompt_guard", "none"]
InputGuardEngine = Literal["guardrails_ai", "native"]

# The relevance threshold is only meaningful against the scale of whatever
# produced the FINAL score, which is not always the reranker: with no reranker
# wired the final score is raw Qdrant cosine. Keying this off the reranker
# backend alone silently makes every query answerable (or none of them).
_DEFAULT_THRESHOLDS: dict[str, float] = {
    # Calibrated on bge-small-en-v1.5 with the seed gold set: positives bottom
    # out at 0.66, an out-of-domain negative scores 0.46. In-domain-but-
    # unanswerable negatives score 0.67-0.74 and CANNOT be separated by cosine
    # alone - that decision belongs to the reranker (M6) and the LLM sentinel.
    # NOTE: calibrated on bge-small. Cohere embed-v4 cosine runs lower (positives
    # min 0.47, negatives max 0.34 on gold), so dense-only mode with Cohere
    # needs RELEVANCE_THRESHOLD=0.40. The deployed config (hybrid + rerank) does
    # not read this scale; the eval reports positive gate misses to catch it.
    "dense": 0.55,
    # Qdrant RRF (k=1) is rank-based: an irrelevant query still scores 0.5 for
    # its top hit, so no threshold here carries relevance meaning. Set to 0 so
    # hybrid mode never gates pre-LLM; the reranker and sentinel own not-found.
    "rrf": 0.0,
    "cross_encoder": 0.0,   # ms-marco logit; >0 means "more relevant than not"
    # Calibrated on the gold set with embed-v4 + rerank-v3.5: positives bottom
    # out at 0.86, negatives top out at 0.09. 0.30 is the log-midpoint.
    "cohere": 0.30,
    "bedrock": 0.35,        # Bedrock Rerank returns 0-1
    "none": 0.55,           # no reranker => the dense score survives
}


class Settings(BaseSettings):
    """All tunable knobs. No model ID or collection name is hardcoded elsewhere."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # AWS / Bedrock
    aws_region: str = "us-east-1"
    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None
    # Inference-profile id: the bare model id is INFERENCE_PROFILE-only on Bedrock.
    bedrock_llm_model_id: str = "us.anthropic.claude-sonnet-5"
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"
    bedrock_embed_dim: int = 1024

    # Embedding backend. Cohere is the production default (hosted, called
    # directly - not through the gateway, because Cohere's input_type matters
    # for quality and must not be dropped by a translation layer). Titan is
    # available for accounts with Bedrock runtime; fastembed is the local
    # zero-cost dev backend. Different vector spaces => the collection name
    # encodes which one produced the index.
    embedding_backend: EmbeddingBackend = "cohere"
    cohere_api_key: SecretStr | None = None
    cohere_embed_model: str = "embed-v4.0"
    cohere_embed_dim: int = 1024
    cohere_rerank_model: str = "rerank-v3.5"
    # Embed token budget per minute; 100k is the trial-key cap. Raise for a
    # production key, or set 0 to disable pacing.
    cohere_embed_tpm: int = Field(default=100_000, ge=0)
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    fastembed_dim: int = 384

    # Bedrock Guardrails (output path, via standalone ApplyGuardrail). Wraps
    # the Groq fallback path exactly as it wraps Bedrock. "none" while the
    # account cannot invoke bedrock-runtime.
    output_guard_backend: OutputGuardBackend = "none"
    bedrock_guardrail_id: str | None = None
    bedrock_guardrail_version: str = "DRAFT"
    output_guard_scope: Literal["FULL", "INTERVENTIONS"] = "FULL"

    # Input guardrails (Guardrails AI). The injection classifier is Groq's
    # hosted prompt-guard, reached directly so the guard never depends on the
    # gateway it protects.
    input_guard_enabled: bool = True
    # guardrails_ai composes the checks through the Guardrails AI framework
    # (telemetry forced off); native runs the same checks without it.
    input_guard_engine: InputGuardEngine = "guardrails_ai"
    injection_backend: InjectionBackend = "groq_prompt_guard"
    injection_model_id: str = "meta-llama/llama-prompt-guard-2-86m"
    injection_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # A guard that cannot run fails closed unless explicitly told otherwise.
    input_guard_fail_open: bool = False
    stream_window_chars: int = Field(default=300, gt=0)

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_prefix: str = "rbi_circulars"

    # LLM gateway. Portkey is the production path (routing + Bedrock->Groq
    # fallback); "bedrock" calls Bedrock directly and exists for the M2
    # integration check only.
    llm_backend: LlmBackend = "portkey"
    # Primary target in the Portkey fallback chain: any Model Catalog provider
    # slug plus that provider's model id. The gateway exists precisely so this
    # is config - the primary has been Bedrock, Anthropic and Google without an
    # application change. Groq is the fallback.
    llm_primary_provider: str = "@google"
    llm_primary_model: str = "gemini-3.5-flash"
    portkey_api_key: SecretStr | None = None
    portkey_base_url: str = "https://api.portkey.ai/v1"
    portkey_config_slug: str | None = None
    portkey_bedrock_provider: str = "@aws"
    portkey_groq_provider: str = "@groq"
    groq_api_key: SecretStr | None = None
    # Groq retired its Llama 3 chat models; gpt-oss-120b is the current
    # strongest open-weights option there. Config-driven, so swap freely.
    groq_model_id: str = "openai/gpt-oss-120b"
    groq_reasoning_effort: str | None = "low"
    llm_max_tokens: int = Field(default=2048, gt=0)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # RAGAS judge. Unset = the gateway config (primary + fallback), which is
    # the production path but also the generator: a judge that is not the
    # generator avoids self-preference and, on free tiers, a shared daily
    # quota. When set, the judge calls this provider/model directly.
    ragas_judge_provider: str | None = None
    ragas_judge_model: str | None = None

    # Observability
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    otel_service_name: str = "rag-system"

    # Retrieval / generation tuning
    # fixed won the M5 comparison on every metric (hit@1 1.00 vs recursive
    # 0.82, nDCG 0.99 vs 0.92) and contextual headers did not close the gap.
    # All three remain available for reproducible comparison.
    chunk_strategy: ChunkStrategy = ChunkStrategy.FIXED
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    semantic_breakpoint_percentile: float = Field(default=90.0, gt=0, lt=100)
    # Prepend document identity to each chunk's retrieval text. Off by default
    # so the M5 baseline stays reproducible; measured separately.
    chunk_context_header: bool = False
    top_k: int = Field(default=20, gt=0)
    # dense: cosine only. hybrid: dense + BM25 sparse fused by RRF on the server.
    retrieval_mode: RetrievalMode = "dense"
    prefetch_k: int = Field(default=40, gt=0)
    rerank_top_n: int = Field(default=5, gt=0)
    reranker_backend: RerankerBackend = "cross_encoder"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Bedrock Rerank has a narrower regional footprint than Bedrock itself.
    bedrock_rerank_model_arn: str | None = None
    bedrock_rerank_region: str | None = None
    relevance_threshold: float | None = None
    max_question_chars: int = Field(default=1000, gt=0)
    # Shared token for /query and the UI endpoints. Unset = open (local dev,
    # tests). The deployed task has a public IP and free-tier LLM quotas.
    ui_access_token: SecretStr | None = None

    @field_validator(
        "portkey_config_slug", "groq_reasoning_effort",
        "ragas_judge_provider", "ragas_judge_model", "ui_access_token",
        "bedrock_rerank_model_arn", "bedrock_rerank_region",
        "bedrock_guardrail_id", mode="before",
    )
    @classmethod
    def _blank_string_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("relevance_threshold", mode="before")
    @classmethod
    def _blank_means_unset(cls, value: object) -> object:
        """An empty env var is "" not None, which pydantic cannot coerce to float."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def threshold_for(self, scale: str) -> float:
        """Resolve the not-found threshold against a named score scale.

        Cross-encoder emits unbounded logits, Bedrock Rerank emits 0-1, RRF
        clusters near 1/rrf_k, and raw dense search emits cosine similarity.
        An explicit RELEVANCE_THRESHOLD overrides all of them.
        """
        if self.relevance_threshold is not None:
            return self.relevance_threshold
        if scale not in _DEFAULT_THRESHOLDS:
            raise KeyError(f"unknown score scale {scale!r}")
        return _DEFAULT_THRESHOLDS[scale]

    @property
    def resolved_relevance_threshold(self) -> float:
        return self.threshold_for(self.reranker_backend)

    def final_score_scale(self, *, reranker_active: bool) -> str:
        """The scale of whatever produces the score the not-found gate reads.

        Takes an explicit flag rather than reading reranker_backend: config can
        name a reranker that is not wired (it was, for one milestone), and the
        gate would then silently read the wrong scale.
        """
        if reranker_active and self.reranker_backend != "none":
            return self.reranker_backend
        return "rrf" if self.retrieval_mode == "hybrid" else "dense"

    @property
    def primary_model_id(self) -> str:
        """The model the gateway tries first, in that provider's own id form."""
        if self.llm_backend == "bedrock":
            return self.bedrock_llm_model_id
        return self.llm_primary_model

    @property
    def embed_model_id(self) -> str:
        if self.embedding_backend == "fastembed":
            return self.fastembed_model
        if self.embedding_backend == "cohere":
            return self.cohere_embed_model
        return self.bedrock_embed_model_id

    @property
    def embed_dim(self) -> int:
        if self.embedding_backend == "fastembed":
            return self.fastembed_dim
        if self.embedding_backend == "cohere":
            return self.cohere_embed_dim
        return self.bedrock_embed_dim

    @property
    def embed_short(self) -> str:
        """Collection-safe tag for the embedding space, e.g. titan-embed-text-v2-1024."""
        tail = self.embed_model_id.split("/")[-1].split(":")[0]
        slug = re.sub(r"[^a-z0-9]+", "-", tail.lower()).strip("-")
        return f"{slug}-{self.embed_dim}"

    def collection_name(self, strategy: ChunkStrategy | None = None) -> str:
        """One collection per (chunking strategy, embedding space).

        Both are config flips for A/B eval, and neither can ever be served
        from the wrong index because the name encodes both.
        """
        strat = (strategy or self.chunk_strategy).value
        if self.chunk_context_header:
            strat = f"{strat}-ctx"
        return f"{self.qdrant_collection_prefix}_{strat}_{self.embed_short}"

    def eval_fingerprint(self) -> dict[str, object]:
        """Resolved config recorded in every eval result, so runs reproduce."""
        return {
            "chunk_strategy": self.chunk_strategy.value,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunk_context_header": self.chunk_context_header,
            "semantic_breakpoint_percentile": (
                self.semantic_breakpoint_percentile
                if self.chunk_strategy is ChunkStrategy.SEMANTIC else None
            ),
            "top_k": self.top_k,
            "retrieval_mode": self.retrieval_mode,
            "prefetch_k": self.prefetch_k,
            "rerank_top_n": self.rerank_top_n,
            "reranker_backend": self.reranker_backend,
            "reranker_model": (
                self.cohere_rerank_model if self.reranker_backend == "cohere"
                else self.cross_encoder_model if self.reranker_backend == "cross_encoder"
                else self.bedrock_rerank_model_arn
            ),
            "relevance_threshold": self.resolved_relevance_threshold,
            "relevance_threshold_dense": self.threshold_for("dense"),
            "llm_primary_provider": self.llm_primary_provider,
            "primary_model_id": self.primary_model_id,
            "llm_backend": self.llm_backend,
            "groq_model_id": self.groq_model_id,
            "embedding_backend": self.embedding_backend,
            "embed_model_id": self.embed_model_id,
            "embed_dim": self.embed_dim,
            "collection": self.collection_name(),
        }


    def fingerprint(self) -> str:
        """Short hash over retrieval-relevant config only.

        Two runs sharing a fingerprint are directly comparable; two that differ
        are not, and you can see which at a glance instead of reconstructing it
        from memory weeks later. Appears on every response and eval result.
        """
        payload = json.dumps(self.eval_fingerprint(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


@lru_cache
def get_settings() -> Settings:
    return Settings()
