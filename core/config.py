from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.models import ChunkStrategy

RerankerBackend = Literal["cross_encoder", "bedrock", "none"]

# The relevance threshold is only meaningful against the scale of whatever
# produced the FINAL score, which is not always the reranker: with no reranker
# wired the final score is raw Qdrant cosine. Keying this off the reranker
# backend alone silently makes every query answerable (or none of them).
_DEFAULT_THRESHOLDS: dict[str, float] = {
    "dense": 0.35,          # Qdrant cosine similarity, 0-1
    "rrf": 0.015,           # raw RRF, just under 1/(rrf_k=60)
    "cross_encoder": 0.0,   # ms-marco logit; >0 means "more relevant than not"
    "bedrock": 0.35,        # Bedrock Rerank returns 0-1
    "none": 0.35,           # no reranker => the dense score survives
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
    bedrock_llm_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"
    bedrock_embed_dim: int = 1024

    # Bedrock Guardrails (output path, via standalone ApplyGuardrail)
    bedrock_guardrail_id: str | None = None
    bedrock_guardrail_version: str = "DRAFT"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_prefix: str = "rbi_circulars"

    # LLM gateway
    portkey_api_key: SecretStr | None = None
    portkey_base_url: str = "https://api.portkey.ai/v1"
    portkey_bedrock_provider: str = "@bedrock-prod"
    portkey_groq_provider: str = "@groq-prod"
    groq_api_key: SecretStr | None = None
    groq_model_id: str = "llama-3.3-70b-versatile"

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
    rerank_top_n: int = Field(default=5, gt=0)
    reranker_backend: RerankerBackend = "cross_encoder"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    relevance_threshold: float | None = None
    max_question_chars: int = Field(default=1000, gt=0)

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

    def collection_name(self, strategy: ChunkStrategy | None = None) -> str:
        """One collection per chunking strategy, so A/B eval is a config flip."""
        return f"{self.qdrant_collection_prefix}_{(strategy or self.chunk_strategy).value}"

    def eval_fingerprint(self) -> dict[str, object]:
        """Resolved config recorded in every eval result, so runs reproduce."""
        return {
            "chunk_strategy": self.chunk_strategy.value,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "rerank_top_n": self.rerank_top_n,
            "reranker_backend": self.reranker_backend,
            "cross_encoder_model": self.cross_encoder_model,
            "relevance_threshold": self.resolved_relevance_threshold,
            "relevance_threshold_dense": self.threshold_for("dense"),
            "bedrock_llm_model_id": self.bedrock_llm_model_id,
            "bedrock_embed_model_id": self.bedrock_embed_model_id,
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
