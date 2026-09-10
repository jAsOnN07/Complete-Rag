# Production-Grade RAG System with Evaluation & Observability

A domain-specific document Q&A system built on AWS Bedrock, Qdrant, and LangChain — with hybrid retrieval, RAGAS evaluation, and full Langfuse + OpenTelemetry observability.

## Architecture

```
[Source Docs] → [Ingestion & Chunking] → [Embeddings] → [Qdrant Vector DB]
                                                                │
[User Query] → [FastAPI] → [Hybrid Retrieval + Re-rank] ←──────┘
                    │
             [Bedrock LLM + Prompt Template]
                    │
             [Response + Citations]
                    │
       ┌────────────┴────────────┐
  [Langfuse/OTEL trace]     [RAGAS eval harness]
```

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | AWS Bedrock (Claude / Titan) |
| Embeddings | Bedrock Titan Embeddings |
| Vector DB | Qdrant |
| Orchestration | LangChain |
| API | FastAPI (async, streaming) |
| Evaluation | RAGAS |
| Observability | Langfuse + OpenTelemetry |
| Deployment | Docker → AWS Fargate |

## Project Structure

```
rag-system/
├── ingestion/          # Document loading, chunking, embedding pipeline
├── retrieval/          # Hybrid retrieval (vector + BM25) and re-ranking
├── generation/         # Bedrock LLM integration and prompt templates
├── api/                # FastAPI app with /query endpoint
├── evaluation/         # RAGAS eval harness and gold eval set
├── observability/      # Langfuse and OpenTelemetry instrumentation
├── docker/             # Dockerfile and docker-compose
├── tests/
└── README.md
```

## Setup

### Prerequisites
- Python 3.11+
- Docker
- AWS account with Bedrock access
- Qdrant instance (local or cloud)
- Langfuse account

### Environment Variables
```bash
cp .env.example .env
```

```env
AWS_ACCESS_KEY_ID=<your_access_key>
AWS_SECRET_ACCESS_KEY=<your_secret_key>
AWS_REGION=us-east-1
QDRANT_URL=<your_qdrant_url>
QDRANT_API_KEY=<your_qdrant_api_key>
LANGFUSE_PUBLIC_KEY=<your_langfuse_public_key>
LANGFUSE_SECRET_KEY=<your_langfuse_secret_key>
```

### Install & Run

```bash
pip install -r requirements.txt

# Run ingestion pipeline
python ingestion/pipeline.py

# Start API
uvicorn api.main:app --reload
```

### Docker

```bash
docker-compose up --build
```

## API

### `POST /query`
```json
{
  "question": "What are the RBI guidelines on KYC compliance?",
  "top_k": 5
}
```

Response (streamed):
```json
{
  "answer": "...",
  "citations": ["doc_id_1", "doc_id_2"]
}
```

## Evaluation

Runs RAGAS metrics against a 50-question gold eval set:

```bash
python evaluation/run_eval.py
```

| Metric | Score |
|---|---|
| Faithfulness | TBD |
| Answer Relevancy | TBD |
| Context Precision | TBD |
| Context Recall | TBD |

> Scores will be updated as the system is built and tuned.

## Observability

- Langfuse traces every retrieval, re-rank, and generation step — latency, token count, cost per query
- OpenTelemetry spans cover the full FastAPI request lifecycle
- Langfuse dashboard tracks cost/latency trends and eval scores over time

## Build Phases

- [x] Phase 1 — Ingestion & chunking
- [ ] Phase 2 — Indexing & hybrid retrieval
- [ ] Phase 3 — Generation & FastAPI
- [ ] Phase 4 — RAGAS evaluation harness
- [ ] Phase 5 — Observability
- [ ] Phase 6 — Docker & Fargate deployment

## License

MIT
