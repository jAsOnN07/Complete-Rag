# RBI Circular Q&A — a measured, production-shaped RAG system

Cited question answering over Reserve Bank of India circulars, built the way a
retrieval system should be run in production: **every design choice was measured
against a verified gold set before it became the default**, every provider call
is traced with cost, the primary LLM is a config value behind a gateway with
automatic failover, and the input and output of the model are guarded.

The interesting parts are the findings, not the feature list. Several
"obvious" choices lost to measurement, and the README says so.

## Headline numbers

Production stack (Gemini 3.5 Flash via Portkey → Groq fallback · Cohere
embed-v4 + rerank-v3.5 · Qdrant Cloud · fixed 1000/150 chunks), 14 verified
gold questions, k = 5:

| Retrieval | hit@1 | doc R@5 | MRR | nDCG@5 | not-found gate |
|---|---|---|---|---|---|
| dense | 0.73 | 0.94 | 0.84 | 0.84 | 1.00 |
| hybrid (dense + BM25, RRF) | 0.91 | 0.92 | 0.95 | 0.90 | 0.00 † |
| **hybrid + rerank** | **1.00** | **0.98** | **1.00** | **1.00** | **1.00** |

| Generation (Gemini 3.5 Flash) | |
|---|---|
| Answered rate on positives | 1.00 |
| Not-found accuracy on negatives | 1.00 (all three refused **before** any LLM call) |
| Invalid (hallucinated) citation rate | 0.00 |
| Citation compliance | 1.00 |
| Input-guard false positives on gold | 0 |
| Mean latency | 7.2 s (Gemini thinking dominates) |
| Cost per 14-question run | $0 on free tiers; ≈ $0.07 at list price (RAGAS adds ≈ 44 judge calls) |

| RAGAS — answers *and* judge gpt-oss-120b (Gemini's daily quota was gone, see *Provider notes*; self-judging is a known bias and the judge is a config value) | 11 positives |
|---|---|
| Faithfulness | 1.00 (11/11) |
| Answer relevancy | 0.71 (see *Findings* — embedding-space dependent) |
| Context precision (with reference) | 0.87 |
| Context recall | 0.88 |

† RRF scores are rank-based, so a threshold on them carries no relevance
meaning — the reranker restores the gate. See *Findings*.

RAGAS means exclude 5 of 44 cells the judge could not score inside the
free-tier retry budget (NaN, listed per question in the result file). The
two sub-1 context scores are a single question, q009, discussed below.

One answer in that run was served by **Groq**: Gemini returned an error
mid-eval, Portkey failed over, and the answer came back correct with two
citations. Nobody had to do anything.

## Architecture

```
                 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  RBI PDFs ────▶ │ loader       │─▶ │ chunker      │─▶ │ Cohere embed │─┐
  (45 circulars) │ letterhead   │   │ fixed|recur. │   │ + BM25 sparse│ │
                 │ stripping    │   │ |semantic    │   └──────────────┘ │
                 └──────────────┘   └──────────────┘                    ▼
                                                              ┌──────────────┐
                                                              │ Qdrant Cloud │
                                                              │ dense+sparse │
                                                              └──────┬───────┘
                                                                     │
  question ─▶ input guard ─▶ hybrid search (server-side RRF) ◀───────┘
              length·lang    │
              PII·injection  ▼
                          Cohere rerank ─▶ threshold gate ─▶ NOT FOUND (no LLM call)
                                                │ pass
                                                ▼
                          citation-forcing prompt ─▶ Portkey ─▶ Gemini 3.5 Flash
                                                        │        └─ fallback: Groq gpt-oss-120b
                                                        ▼
                          citation parser ─▶ output guard ─▶ answer + citations
                                             (Bedrock ApplyGuardrail, or LLM judge)

  every box above is a Langfuse span on one OpenTelemetry provider;
  the FastAPI request span is the root, with cost on every provider call.
```

Two endpoints: `POST /query` (full guard suite; what the evaluation measures)
and `POST /query/stream` (server-sent events; windows are checked *before*
they are emitted; the grounding verdict arrives in the final event because
streamed text cannot be retracted).

## Findings — what measurement changed

**Fixed-size chunking beat recursive and semantic.** Compared under the full
retrieval stack, the naive baseline won on every metric. The mechanism was
explainable: the two questions it won key on a short span *plus* the
document's identity, which lives in the header block. Fixed windows keep the
header and body together by accident of overlap; recursive splits them at the
first blank line; semantic isolates sentences.

| chunking | hit@1 | MRR | nDCG@5 | chunks |
|---|---|---|---|---|
| fixed | 1.00 | 1.00 | 0.99 | 553 |
| recursive | 0.82 | 0.91 | 0.92 | 545 |
| semantic | 0.64 | 0.73 | 0.76 | 813 |

**The principled fix didn't hold.** Prepending each chunk's document identity
(contextual chunk headers) lifted recursive's identifier questions but cost
document diversity — every chunk of a circular shares the header and its
chunks collapse together in embedding and BM25 space — and it hurt fixed
outright (1.00 → 0.91). Headers trade identifier recall for diversity; on
this corpus a net loss. The baseline won twice.

**Dense retrieval cannot see exact identifiers.** With a dense-only index the
question about `DOR.AML.REC.219/14.06.001/2026-27` had its evidence chunk
outside the top 30: six near-identical circulars are interchangeable to an
embedder. BM25 sparse vectors with server-side RRF moved it to rank 2; the
reranker put it at rank 1.

**Cosine similarity cannot gate in-domain-unanswerable questions.** A question
about the repo rate scores 0.67–0.74 against deposit-rate circulars — above
real positives. The cross-encoder scored the same question at −10.7 against a
positive floor of +0.56; Cohere's reranker at 0.09 against 0.86. The not-found
decision belongs to the reranker and the model's sentinel, not to embedding
similarity — and it now fires before a single LLM token is spent.

**Qdrant's RRF uses k = 1, not the textbook 60.** Measured (top hit in both
lists = 1.0, second in both = 0.667, lone third = 0.25). The steeper curve
rewards top-rank agreement far more; the pure-Python fusion is kept in sync so
a congruence test asserts exact scores against the server.

**Models cite in ways parsers don't expect.** `gpt-oss` emits `【5】` (CJK
fullwidth brackets) for citations; one correct answer arrived in bold markdown
with no brackets at all. The parser accepts both bracket forms and *citation
compliance* is a tracked metric rather than a silent failure.

**RAGAS metrics need reading, not just reporting.** Three things the numbers
above do not say on their own. *Answer relevancy* embeds a question the judge
generates from the answer and compares it with the real one; the judge turned
"The five new districts are Zanskar, …" into "What are the five new
districts?", which Cohere embed-v4 scores 0.52 against the original (a true
paraphrase scores 0.96, an unrelated question 0.10). The metric is
embedding-space dependent and its absolute value is not comparable across
embedders; it is tracked for regressions, not quoted as a grade. *Context
recall and precision* are claim-level: q009's reference is one sentence
naming six circulars, five of which were retrieved and all five cited — the
system's own doc-recall says 0.83, RAGAS says 0.00 because no single context
supports the whole sentence. When two metrics disagree that sharply, the
disagreement is the finding. And the *judge is a config value* with its own
rate limits: answer relevancy's `strictness` is sent as OpenAI `n`, and the
Groq target rejects `n > 1`; a thinking model at default effort spent more
than two minutes on one faithfulness prompt. The judge runs at low reasoning
effort, one question per answer, and is recorded in every result.

**Guardrails AI phones home by default.** A `Guard` registers an OpenTelemetry
exporter to the vendor's endpoint — also a second `TracerProvider`, which would
have split traces from Langfuse. Forced off on every Guard and verified. Two
more framework facts: a Guard runs every validator before reporting (so
cheap-first ordering needs two Guards), and validators run inside the
framework's span, so starting an event loop there breaks OTel context.

**A threshold is only meaningful on the scale that produced the score.** This
bit three separate times: a single constant was silently read on the wrong
scale as the reranker, the retrieval mode, and the embedding model changed.
Thresholds are now resolved per score scale with an explicit flag for what is
actually wired, and the eval reports *positive gate misses* — questions that
would be refused before the LLM ran — which the negative-gate metric alone
cannot see.

**The primary LLM was swapped three times without an application change.**
Bedrock (blocked at the account level) → Anthropic (verification failed) →
Gemini. `LLM_PRIMARY_PROVIDER` / `LLM_PRIMARY_MODEL` name a Portkey provider
slug and model; the gateway config owns routing, retries and fallback.
There is no retry or fallback code in the application.

## Corpus and gold set

45 RBI circulars (Jul–Sep 2026, 238 pages, 12 MB) fetched from rbi.org.in
by `scripts/fetch_circulars.py`, which walks the sequential detail-page ids
past the 22-item listing and validates every download with a `%PDF`
magic-byte check — the document host serves an HTML interstitial to
non-browser user agents, which would otherwise have silently indexed 45
error pages.

The gold set (`evaluation/gold_set.json`, 50 questions: 14 verified by the
owner, 36 drafted from the corpus text and awaiting verification) is keyed by
**document id plus verbatim evidence quotes**, never by chunk id, so comparing
chunking strategies cannot invalidate it. `python -m evaluation.gold`
machine-checks every quote against the corpus and the eval refuses to run on
a drifted set. Question types: single-hop (including exact identifiers among
nine near-identical sanctions circulars), cross-circular (families of five to
eight near-duplicate amendments), multi-hop across two or three circulars,
and negatives - out-of-domain and, harder, in-domain-unanswerable.

## Running it

```bash
conda create -n rag python=3.12 -c conda-forge && conda activate rag
pip install -r requirements.txt            # production runtime
pip install -r requirements-local.txt      # + local cross-encoder / dense model (dev)
pip install -r requirements-dev.txt        # + pytest
cp .env.example .env                       # fill in keys

python -m scripts.fetch_circulars --limit 45   # corpus -> data/raw + manifest
python -m ingestion.pipeline                   # chunk, embed, index into Qdrant
uvicorn api.main:app --reload
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"Which five new districts were formed in Ladakh?"}'
```

Evaluation is a script, not a notebook. Every run writes a self-describing
JSON with the full resolved configuration and a fingerprint, so any two runs
are comparable at a glance:

```bash
python -m evaluation.run_eval                        # retrieval tier, no LLM spend
python -m evaluation.run_eval --tier generation      # + real answers
python -m evaluation.run_eval --tier ragas           # + LLM-judged faithfulness etc.
python -m evaluation.run_eval --from-results evaluation/results/<run>.json  # judge only
python -m evaluation.compare retrieval               # dense / hybrid / rerank table
python -m evaluation.compare chunking                # fixed / recursive / semantic
```

Tests: ~300 offline (fakes at every provider seam, Qdrant in-memory, real
local models where they are local) and a small opt-in `-m integration` set
that hits real services for fractions of a cent.

## Deployment

`docker/Dockerfile` is a production-only image: the deployed stack is hosted,
so nothing runs in the container except the ~10 MB BM25 encoder, baked at
build time. `scripts/deploy_fargate.py` drives the whole AWS side from boto3
— CodeBuild → ECR → SSM secrets → ECS Fargate with a public IP — and is
idempotent, so it doubles as the committed record of what the deployment is.
No Terraform by design.

```bash
python -m scripts.deploy_fargate build      # zip -> S3 -> CodeBuild -> ECR
python -m scripts.deploy_fargate secrets    # .env -> SSM SecureString
python -m scripts.deploy_fargate deploy     # roles, cluster, task, service
python -m scripts.deploy_fargate verify     # /readyz + one real /query
python -m scripts.deploy_fargate scale 0    # stop paying
```

## Provider notes

| Concern | Deployed | Also supported (config) |
|---|---|---|
| LLM | Gemini 3.5 Flash via Portkey, Groq fallback | Bedrock Claude, Anthropic API — any Portkey provider |
| Embeddings | Cohere embed-v4.0 | Bedrock Titan V2, local bge-small |
| Reranker | Cohere rerank-v3.5 | local cross-encoder, Bedrock Rerank |
| Input guard | Guardrails AI + Groq prompt-guard classifier | native (same checks) |
| Output guard | disabled pending Bedrock entitlement | Bedrock ApplyGuardrail (built, shape-verified) |

**The free primary is a demo tier, and the numbers say so.** Google's AI
Studio free tier allows **20 requests per day** per Gemini 3.x model (the
error names the metric: `generate_content_free_tier_requests, limit: 20`).
The RAGAS run below exhausted it in the first minute; every one of the
eleven answers in that run was served by the Groq fallback, correctly cited,
with no operator action — which is the failover story working, and also a
statement that the production LLM budget has to be a paid key. Which model
served each answer is recorded per row in every result file, so no run can
quietly attribute Groq's answers to Gemini.

Bedrock was the original primary for everything. The account cannot invoke
`bedrock-runtime` (control plane works; every data-plane call returns
`ValidationException: Operation not allowed`), so the Bedrock backends are
implemented against verified API shapes and fake-tested, but have no live
evidence. That is stated rather than hidden.

## What I would do differently

- **Grow the gold set before trusting the ceilings.** 14 questions means each
  one moves hit@1 by 0.09. The set is now 50; on the 36 not-yet-verified
  additions the same stack scores hit@1 0.86, MRR 0.89, nDCG@5 0.88 and the
  negative gate drops to 0.75 — two new in-domain-unanswerable questions
  (the CRR *rate*, the *amount* of the counterfeit-note penalty) pass the
  reranker at 0.42 and 0.66 and reach the model's sentinel. Exact-identifier
  questions among nine near-identical sanctions circulars and two-document
  multi-hop questions are where the misses are. The headline table stays on
  the verified 14 until the owner has checked the rest; the harness refuses
  to run on a quote that does not appear in the corpus, so what is left to
  verify is the *questions*, not the evidence.
- **Measure per-embedder thresholds instead of per-stage.** The dense threshold
  calibrated on bge would silently refuse a Cohere positive. It is caught and
  documented, not yet parameterised by embedder.
- **Don't build the semantic chunker on a 6 GB machine.** Sentence-level
  embedding of the corpus swapped and got killed repeatedly; hosted embedding
  made the whole loop fast. The dev-machine constraint shaped more decisions
  than it should have.
- **Reach for hosted providers a day earlier.** Two milestones ran on local
  ONNX at 300 ms per chunk waiting for Bedrock; the same work took seconds on
  Cohere.

## License

MIT
