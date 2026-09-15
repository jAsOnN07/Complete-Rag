# syntax=docker/dockerfile:1.7
#
# Production image: FastAPI + the hosted retrieval/generation stack. No ML
# model weights are baked or downloaded except the ~10 MB BM25 sparse encoder,
# which is fetched at build time so the container never touches Hugging Face
# at startup. Ingestion is a CLI run outside the container; the API reads the
# index Qdrant Cloud already holds.

# ---- build stage: compile wheels once ------------------------------------------
FROM python:3.12-slim AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /wheels
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# ---- runtime stage ---------------------------------------------------------------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    HF_HUB_DISABLE_TELEMETRY=1 \
    LANGFUSE_ENABLED=true

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app
COPY --from=build /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt && rm -rf /wheels

# Bake the BM25 sparse model so the first request does not download it.
RUN mkdir -p /opt/fastembed && python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25', cache_dir='/opt/fastembed')" \
    && chown -R app:app /opt/fastembed

COPY --chown=app:app api ./api
COPY --chown=app:app core ./core
COPY --chown=app:app generation ./generation
COPY --chown=app:app guards ./guards
COPY --chown=app:app ingestion ./ingestion
COPY --chown=app:app observability ./observability
COPY --chown=app:app retrieval ./retrieval
COPY --chown=app:app evaluation ./evaluation
COPY --chown=app:app scripts ./scripts
# The showcase UI and what it reads: the corpus manifest (titles, dates)
# and the published eval runs; the corpus itself never ships.
COPY --chown=app:app ui ./ui
COPY --chown=app:app data/manifest.json ./data/manifest.json

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "75"]
