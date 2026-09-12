"""Session-wide test isolation.

Settings() reads .env by default. Tests that assert defaults must not depend
on whatever the developer happens to have in their .env, so the env file is
disabled for the whole test session. Integration tests opt back in explicitly.
"""

from __future__ import annotations

import os

import pytest

from core.config import Settings

os.environ.setdefault("LANGFUSE_ENABLED", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


@pytest.fixture(autouse=True)
def _isolate_settings_from_dotenv(request, monkeypatch):
    if "integration" in request.keywords:
        return
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    # Env vars from the developer's shell must not leak in either.
    for key in list(os.environ):
        if key.startswith(("EMBEDDING_", "LLM_", "PORTKEY_", "GROQ_", "BEDROCK_", "CHUNK_", "RERANK")):
            monkeypatch.delenv(key, raising=False)
