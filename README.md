# Production-Grade RAG System with Evaluation & Observability

A domain-specific document Q&A system built on AWS Bedrock, Qdrant, and LangChain — with hybrid retrieval, RAGAS evaluation, full Langfuse + OpenTelemetry observability, Portkey LLM gateway with Groq fallback, and layered guardrails.

## Architecture

```
[Source Docs] → [Ingestion & Chunking] → [Embeddings] → [Qdrant Vector DB]
                                                                │
[User Query] → [Guardrails AI] → [FastAPI] → [Hybrid Retrieval + Re-rank] ←──┘
  (input validation)                  │
                             [Portkey LLM Gateway]
                            /                    \
                   [Bedrock Claude]          [Groq fallback]
                            \                    /
                             [Bedrock Guardrails]
                          (grounding, PII, content)
                                     │
                          [Response + Citations]
                                     │
                ┌────────────────────┴────────────────────┐
           [Langfuse/OTEL trace]                  [RAGAS eval harness]
```

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | AWS Bedrock (Claude 3 Sonnet) — primary |
| LLM Fallback | Groq (Llama 3) |
| LLM Gateway | Portkey (routing, retries, unified logging) |
| Embeddings | Bedrock Titan Embeddings V2 |
| Vector DB | Qdrant |
| Orchestration | LangChain |
| API | FastAPI (async, streaming) |
| Guardrails (input) | Guardrails AI — prompt injection, PII, length, language |
| Guardrails (output) | AWS Bedrock Guardrails — grounding, PII redaction, content filtering, denied topics |
| Evaluation | RAGAS |
| Observability | Langfuse + OpenTelemetry |
| Deployment | Docker → AWS Fargate |

## Project Structure

```
rag-system/
├── ingestion/          # Document loading, chunking, embedding pipeline
├── retrieval/          # Hybrid retrieval (vector + BM25) and re-ranking
├── generation/         # Bedrock LLM integration, Portkey gateway, prompt templates
├── guardrails/         # Guardrails AI (input) + Bedrock Guardrails (output)
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
- AWS account with Bedrock access (and Bedrock Guardrails configured)
- Qdrant instance (local or cloud)
- Langfuse account
- Portkey account (free tier sufficient)

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
PORTKEY_API_KEY=<your_portkey_api_key>
GROQ_API_KEY=<your_groq_api_key>
BEDROCK_GUARDRAIL_ID=<your_guardrail_id>
BEDROCK_GUARDRAIL_VERSION=DRAFT
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

## Guardrails

### Input — Guardrails AI (runs before LLM gateway)
- Prompt injection detection
- PII detection (names, emails, account numbers)
- Input length enforcement
- Language detection (reject non-English queries)

### Output — AWS Bedrock Guardrails (runs after LLM response)
- Grounding check — verifies answer is grounded in retrieved context
- PII redaction — strips any PII that leaked into the response
- Content filtering — hate, violence, sexual content
- Denied topics — rejects responses outside the configured domain

## LLM Gateway — Portkey

- Primary: AWS Bedrock Claude 3 Sonnet
- Fallback: Groq Llama 3 (automatic on Bedrock timeout or error)
- Unified request logging across both providers via Portkey dashboard
- Retry logic and fallback handled at the gateway level — no custom retry code

## Build Phases

- [x] Phase 1 — Ingestion & chunking
- [ ] Phase 2 — Indexing & hybrid retrieval
- [ ] Phase 3 — Generation & FastAPI
- [ ] Phase 4 — LLM Gateway (Portkey + Groq fallback)
- [ ] Phase 5 — Guardrails (Guardrails AI + Bedrock Guardrails)
- [ ] Phase 6 — RAGAS evaluation harness
- [ ] Phase 7 — Observability
- [ ] Phase 8 — Docker & Fargate deployment

## License

MIT
