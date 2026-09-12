from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.models import ChunkStrategy

RerankerBackend = Literal["cross_encoder", "bedrock", "none"]
EmbeddingBackend = Literal["bedrock", "fastembed"]
LlmBackend = Literal["portkey", "bedrock"]
RetrievalMode = Literal["dense", "hybrid"]

# The relevance threshold is only meaningful against the scale of whatever
# produced the FINAL score, which is not always the reranker: with no reranker
# wired the final score is raw Qdrant cosine. Keying this off the reranker
# backend alone silently makes every query answerable (or none of them).
_DEFAULT_THRESHOLDS: dict[str, float] = {
    # Calibrated on bge-small-en-v1.5 with the seed gold set: positives bottom
    # out at 0.66, an out-of-domain negative scores 0.46. In-domain-but-
    # unanswerable negatives score 0.67-0.74 and CANNOT be separated by cosine
    # alone - that decision belongs to the reranker (M6) and the LLM sentinel.
    "dense": 0.55,
    # Qdrant RRF (k=1) is rank-based: an irrelevant query still scores 0.5 for
    # its top hit, so no threshold here carries relevance meaning. Set to 0 so
    # hybrid mode never gates pre-LLM; the reranker and sentinel own not-found.
    "rrf": 0.0,
    "cross_encoder": 0.0,   # ms-marco logit; >0 means "more relevant than not"
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

    # Embedding backend. Titan is the production default; fastembed is a local
    # ONNX backend that keeps the whole dev loop real while Bedrock is
    # unavailable. They are different vector spaces, so the collection name
    # encodes which one produced the index.
    embedding_backend: EmbeddingBackend = "bedrock"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    fastembed_dim: int = 384

    # Bedrock Guardrails (output path, via standalone ApplyGuardrail)
    bedrock_guardrail_id: str | None = None
    bedrock_guardrail_version: str = "DRAFT"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_prefix: str = "rbi_circulars"

    # LLM gateway. Portkey is the production path (routing + Bedrock->Groq
    # fallback); "bedrock" calls Bedrock directly and exists for the M2
    # integration check only.
    llm_backend: LlmBackend = "portkey"
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

    # Observability
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    otel_service_name: str = "rag-system"

    # Retrieval / generation tuning
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
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

    @field_validator(
        "portkey_config_slug", "groq_reasoning_effort",
        "bedrock_rerank_model_arn", "bedrock_rerank_region", mode="before",
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
    def embed_model_id(self) -> str:
        if self.embedding_backend == "fastembed":
            return self.fastembed_model
        return self.bedrock_embed_model_id

    @property
    def embed_dim(self) -> int:
        if self.embedding_backend == "fastembed":
            return self.fastembed_dim
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
        return f"{self.qdrant_collection_prefix}_{strat}_{self.embed_short}"

    def eval_fingerprint(self) -> dict[str, object]:
        """Resolved config recorded in every eval result, so runs reproduce."""
        return {
            "chunk_strategy": self.chunk_strategy.value,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "retrieval_mode": self.retrieval_mode,
            "prefetch_k": self.prefetch_k,
            "rerank_top_n": self.rerank_top_n,
            "reranker_backend": self.reranker_backend,
            "cross_encoder_model": self.cross_encoder_model,
            "relevance_threshold": self.resolved_relevance_threshold,
            "relevance_threshold_dense": self.threshold_for("dense"),
            "bedrock_llm_model_id": self.bedrock_llm_model_id,
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
