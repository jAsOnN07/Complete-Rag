import pytest

from core.config import Settings
from core.models import ChunkStrategy


def test_collection_name_is_derived_per_strategy():
    settings = Settings(qdrant_collection_prefix="circulars")
    assert settings.collection_name(ChunkStrategy.FIXED).startswith("circulars_fixed_")
    assert settings.collection_name(ChunkStrategy.SEMANTIC).startswith("circulars_semantic_")


def test_collection_name_defaults_to_active_strategy():
    settings = Settings(
        qdrant_collection_prefix="circulars", chunk_strategy=ChunkStrategy.SEMANTIC
    )
    assert settings.collection_name().startswith("circulars_semantic_")


def test_eval_fingerprint_records_what_a_rerun_needs():
    fingerprint = Settings(chunk_strategy=ChunkStrategy.FIXED).eval_fingerprint()
    for key in ("chunk_strategy", "top_k", "reranker_backend", "collection"):
        assert key in fingerprint
    assert fingerprint["chunk_strategy"] == "fixed"


def test_secrets_are_not_leaked_by_repr():
    settings = Settings(qdrant_api_key="super-secret-value")
    assert "super-secret-value" not in repr(settings)


def test_relevance_threshold_defaults_per_reranker_backend():
    """A shared threshold across backends is a silent not-found bug."""
    assert Settings(reranker_backend="cross_encoder").resolved_relevance_threshold == 0.0
    assert Settings(reranker_backend="bedrock").resolved_relevance_threshold == 0.35


def test_threshold_is_resolved_against_the_scale_that_produced_the_score():
    """With no reranker the final score is raw cosine, not a reranker score."""
    settings = Settings(reranker_backend="cross_encoder")
    assert settings.threshold_for("cross_encoder") == 0.0
    assert settings.threshold_for("dense") == 0.55
    assert settings.threshold_for("rrf") == 0.0


def test_unknown_score_scale_fails_loudly():
    with pytest.raises(KeyError):
        Settings().threshold_for("nonsense")


def test_explicit_relevance_threshold_overrides_backend_default():
    settings = Settings(reranker_backend="bedrock", relevance_threshold=0.7)
    assert settings.resolved_relevance_threshold == 0.7


def test_blank_relevance_threshold_env_var_is_treated_as_unset():
    """RELEVANCE_THRESHOLD= in .env arrives as "", which is not a float."""
    settings = Settings(relevance_threshold="", reranker_backend="bedrock")
    assert settings.relevance_threshold is None
    assert settings.resolved_relevance_threshold == 0.35


def test_fingerprint_is_stable_for_identical_config():
    assert Settings(top_k=7).fingerprint() == Settings(top_k=7).fingerprint()


def test_fingerprint_changes_when_retrieval_config_changes():
    base = Settings(top_k=7)
    assert base.fingerprint() != Settings(top_k=8).fingerprint()
    assert base.fingerprint() != Settings(
        top_k=7, chunk_strategy=ChunkStrategy.FIXED
    ).fingerprint()


def test_fingerprint_ignores_secrets():
    """Rotating a key must not invalidate comparability of eval runs."""
    a = Settings(top_k=7, qdrant_api_key="key-one").fingerprint()
    b = Settings(top_k=7, qdrant_api_key="key-two").fingerprint()
    assert a == b


def test_embedding_backend_defaults_to_bedrock_titan():
    """CLAUDE.md names Titan V2 as the production embedder; local is opt-in."""
    settings = Settings()
    assert settings.embedding_backend == "bedrock"
    assert settings.embed_model_id == settings.bedrock_embed_model_id
    assert settings.embed_dim == settings.bedrock_embed_dim


def test_fastembed_backend_resolves_its_own_model_and_dim():
    settings = Settings(embedding_backend="fastembed")
    assert settings.embed_model_id == "BAAI/bge-small-en-v1.5"
    assert settings.embed_dim == 384


def test_collection_name_encodes_embedder_so_vector_spaces_never_collide():
    """A Titan index and a bge index must never share a collection."""
    titan = Settings(qdrant_collection_prefix="c", embedding_backend="bedrock")
    bge = Settings(qdrant_collection_prefix="c", embedding_backend="fastembed")
    assert titan.collection_name() != bge.collection_name()
    assert titan.collection_name().startswith("c_fixed_")
    assert "384" in bge.collection_name()


def test_fingerprint_changes_with_embedding_backend():
    assert (
        Settings(embedding_backend="bedrock").fingerprint()
        != Settings(embedding_backend="fastembed").fingerprint()
    )


def test_final_score_scale_follows_what_is_actually_wired():
    """Config may name a reranker that is not wired; the gate must not trust it."""
    hybrid_ce = Settings(retrieval_mode="hybrid", reranker_backend="cross_encoder")
    assert hybrid_ce.final_score_scale(reranker_active=True) == "cross_encoder"
    assert hybrid_ce.final_score_scale(reranker_active=False) == "rrf"
    dense_ce = Settings(retrieval_mode="dense", reranker_backend="cross_encoder")
    assert dense_ce.final_score_scale(reranker_active=False) == "dense"
    assert Settings(retrieval_mode="dense", reranker_backend="none").final_score_scale(reranker_active=True) == "dense"


def test_context_header_gets_its_own_collection_and_fingerprint():
    """Header changes the vectors, so it can never share a collection with plain chunks."""
    plain = Settings(chunk_context_header=False)
    ctx = Settings(chunk_context_header=True)
    assert plain.collection_name() != ctx.collection_name()
    assert "-ctx_" in ctx.collection_name()
    assert plain.fingerprint() != ctx.fingerprint()
