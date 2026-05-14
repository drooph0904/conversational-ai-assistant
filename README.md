---
title: Health Insurance Assistant
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.57.0
app_file: ui/app.py
pinned: false
---

# Conversational AI Customer Support Assistant — Indian Health Insurance

A two-layer customer support assistant for the Indian health insurance domain.
**Layer 1** is a pure-Python RAG chatbot grounded in publicly available policy documents.
**Layer 2** wraps it with a Dialogflow CX agent that handles structured tasks deterministically
and falls back to the RAG service for open-ended questions.

**Live demo:** [Hugging Face Spaces](https://huggingface.co/spaces/drooph0904/health-insurance-assistant)

> **Status:** Layer 1 (RAG chatbot) ✅ complete. Layer 2 (Dialogflow CX + webhook) ✅ complete.

---

## Problem statement

Customers asking insurance questions get one of two bad experiences today:
either a rigid IVR/bot that only handles a fixed menu of tasks, or a free-form LLM
chatbot that confidently hallucinates policy details. This project demonstrates
the **hybrid pattern** used in real enterprise contact centres: deterministic
flows for structured intents (claim status, hospital lookup) plus a grounded RAG
layer for open FAQs — so the system stays accurate *and* useful.

---

## Architecture

### Offline ingestion pipeline (one-time)

```
PDFs in data/
    │
    ▼  pypdf — extract text per page
Page records [{source, page, text}, ...]
    │
    ▼  chunk_text() — 500-token chunks, 50-token overlap, within page boundaries
Chunk records [{id, text, metadata}, ...]
    │
    ▼  OpenAI text-embedding-3-small — batch embed (up to 100 texts/call)
Vectors (1536-dim float lists)
    │
    ▼  chromadb.PersistentClient — upsert with deterministic IDs
chroma_db/ (cosine-space HNSW index)
```

### Online query path (per user turn)

```
User query
    │
    ▼  [Stage 1 — Retrieval]
    │
    ├─ multi_query (default)
    │       ├─ GPT-4o-mini generates 3 alternative phrasings using insurance acronyms
    │       ├─ 3 × embed_query → 3 × Chroma cosine search (k=10 each)
    │       └─ Deduplicate by chunk ID → union (≤15 unique chunks)
    │
    ├─ hyde
    │       ├─ GPT-4o-mini writes a 1-2 sentence hypothetical answer in document style
    │       └─ embed(hypothetical) → 1 × Chroma cosine search (k=10)
    │
    └─ standard
            └─ embed(query) → 1 × Chroma cosine search (k=10)
    │
    ▼  Similarity threshold guard (0.4)
    │  best chunk score < 0.4 → return fallback immediately (no LLM call)
    │
    ▼  [Stage 2 — Reranking]
    │  cross-encoder/ms-marco-MiniLM-L-6-v2 scores each (query, chunk) pair
    │  → top 3 chunks selected
    │
    ▼  [Stage 3 — Generation]
    │  GPT-4o-mini with strict system prompt:
    │  "Answer ONLY from context; else say I don't know"
    │  Context = 3 chunks + last 3 conversation turns
    │
    ▼  ChatResult {answer, citations, retrieved_chunks}
```

---

## Tech stack

| Component | Choice | Why |
|-----------|--------|-----|
| Language | Python 3.11+ | Standard for ML/AI tooling |
| LLM | `gpt-4o-mini` via `openai` | $0.00042/query, 500 RPM, no daily caps |
| Embeddings | `text-embedding-3-small` (1536-dim) | $0.02/1M tokens, true batch API, no asymmetric task types |
| Query expansion | `gpt-4o-mini` (Multi-Query + HyDE) | Solves vocabulary mismatch without re-ingestion |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80 MB, ~200 ms/10 chunks on CPU, fine-tuned on 8.8M pairs |
| Vector store | Chroma (local, persistent) | Zero-setup, cosine HNSW, fine for <100K chunks |
| PDF parsing | `pypdf` | Pure Python, no system dependencies |
| API (Layer 2) | FastAPI + Uvicorn | Async, OpenAPI docs for free |
| UI | Streamlit | Fast path to a demoable frontend |
| Orchestration | Dialogflow CX | Pairs with webhook for deterministic intent handling |
| Deployment | Hugging Face Spaces (Streamlit SDK) | Free, public URL, git-based deploys via LFS |

**Deliberately avoided:** LangChain, LlamaIndex, agent frameworks. Raw primitives only — every design decision is defensible in an interview.

---

## How to run locally

```bash
# 1. Create venv and install deps
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# Open .env and paste your OPENAI_API_KEY

# 3. Drop health-insurance PDFs into data/
#    e.g. policy brochures from Star Health, HDFC Ergo, Niva Bupa

# 4. Build the vector store (one-time; re-run when PDFs change)
python -m src.ingest

# 5. Start the UI
streamlit run ui/app.py
```

The Streamlit UI calls `generate_answer()` via a direct Python import — no separate API process needed for Layer 1.

### Optional: run the FastAPI service (needed for Layer 2 / Dialogflow)

```bash
uvicorn src.api:app --reload --port 8000
```

Quick smoke test:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"session_id":"t1","message":"What is the waiting period for pre-existing diseases?"}' | jq
```

---

## Deployment (Hugging Face Spaces)

The app is deployed at `https://huggingface.co/spaces/drooph0904/health-insurance-assistant`.

HF Spaces runs the Streamlit app directly. The binary files (Chroma DB + PDFs) are tracked with
`git-lfs` and pushed to the `huggingface` remote separately from GitHub:

```bash
# Push code only to GitHub
git push origin main

# Push code + binaries to HF Spaces
git add -f chroma_db/ data/
git commit -m "deploy: bundle chroma_db + data"
git push huggingface main --force
git reset HEAD~1   # remove binary commit locally so GitHub stays clean
```

**Required HF Space secret:** `OPENAI_API_KEY` (set in Space Settings → Secrets).

**Note on uploaded PDFs:** Files uploaded via the sidebar are ingested into the in-memory
Chroma client and persist to disk during that container session, but are lost on container
restart. To make a document permanent, add it to `data/` locally, re-ingest, and redeploy.

---

## Design decisions

### Chunking — 500 tokens / 50 overlap / within page

500 tokens (≈375 words) is large enough to contain a complete insurance fact (e.g. a full
waiting-period clause with its exceptions) while still being focused enough that the embedding
represents a single topic. The 50-token (≈37-word) overlap prevents facts from being split
across chunk boundaries. Chunks never span page breaks, keeping citations clean
(`Star Health Assure, p.6` not `p.6–7`). Chunk IDs are deterministic
(`{source}-p{page}-c{idx}`) so re-running ingest upserts rather than duplicates.

### Two-stage retrieval — bi-encoder (RETRIEVE_K=10) → cross-encoder (TOP_K=3)

The bi-encoder (dense embedding) is fast and scales to the full collection, but ranks by
directional similarity alone. The cross-encoder examines each (query, chunk) pair jointly,
capturing exact term overlap and semantic coherence that the bi-encoder misses. Running the
cross-encoder on only 10 candidates keeps latency to ~200 ms on CPU.

### Multi-Query RAG (default retrieval strategy)

Dense embeddings fail when the user's vocabulary doesn't match the document's vocabulary
(e.g. user types "United India Insurance Company"; the document uses "IHIP" throughout).
Multi-Query fixes this by asking GPT-4o-mini to rewrite the query into 3 variants using
insurance acronyms and policy-document phrasing, then retrieving for all 3 and taking the
union. At least one variant will contain the document's terminology, pulling the right page
into the candidate set before the reranker scores it.

Cost: ~$0.00005 extra per query (1 extra GPT-4o-mini call + 2 extra embedding calls).

### HyDE (alternative strategy)

Instead of embedding the user's query, GPT-4o-mini writes a 1-2 sentence hypothetical answer
in the style of an insurance policy document. Embedding that hypothetical lands the search
vector closer to actual document chunks because it uses document vocabulary. Toggle via
`RETRIEVAL_STRATEGY=hyde` in `.env`.

### Similarity threshold = 0.4

Calibrated for `text-embedding-3-small`'s score distribution. If the best bi-encoder chunk
scores below 0.4, the system skips the reranker and LLM entirely and returns the fallback
message with the escalation number. This prevents expensive LLM calls on genuinely off-topic
queries and eliminates hallucination on borderline matches.

### Confidence labels in UI

The UI labels each citation by its reranker score:
- ● High confidence — score ≥ 0.75
- ● Medium confidence — score ≥ 0.60
- ● Low confidence — score < 0.60

### In-memory session history (last 3 turns)

Stored per `session_id` in a `deque(maxlen=6)`. Enough for natural follow-ups
("what about for senior citizens?") without bloating the prompt or letting stale turns
derail the model. Redis is the production upgrade path.

### Direct Python import in UI (no HTTP)

`ui/app.py` imports `generate_answer()` directly instead of calling the FastAPI endpoint.
This means Layer 1 runs as a single process on HF Spaces — no need to run two services.
FastAPI (`src/api.py`) still exists and wraps the same function for Layer 2 (Dialogflow
webhook calls it over HTTP on port 8000).

---

## Layer 2 — Dialogflow CX

The CX agent handles structured intents deterministically before falling back to RAG.

| Intent | Handler | Data source |
|--------|---------|-------------|
| `check-claim-status` | Regex extracts claim ID, dict lookup | In-memory mock |
| `find-hospital` | City name match, list lookup | In-memory mock |
| `sys.no-match-default` (everything else) | HTTP POST to `src/api.py` | RAG pipeline |

### Running Layer 2 locally

```bash
# Terminal 1 — RAG service
uvicorn src.api:app --port 8000

# Terminal 2 — Webhook server
uvicorn dialogflow_cx.webhook.main:app --port 8001

# Terminal 3 — Expose webhook publicly
ngrok http 8001
# Copy the https://xxxx.ngrok-free.app URL
```

In the Dialogflow CX console:
- Set webhook URL → `https://xxxx.ngrok-free.app/webhook` (timeout: 30 s)
- Start page routes:
  - Intent `check-claim-status` → webhook tag `check-claim-status`
  - Intent `find-hospital` → webhook tag `find-hospital`
  - Event `sys.no-match-default` → webhook tag `faq-fallback`

**Test queries in the CX simulator:**
1. `what is the status of claim 8823` → deterministic: "Under Review"
2. `find network hospital in Mumbai` → deterministic: Kokilaben, Lilavati, Hinduja
3. `what is the waiting period for pre-existing diseases` → RAG answer with citation

**Mock claim IDs:** `8823` (Under Review), `1234` (Approved), `5678` (Rejected), `9999` (Processing), `4321` (Paid Out)

**Mock cities:** Mumbai, Delhi, Pune, Bangalore, Hyderabad

---

## Configuration reference

All knobs live in `src/config.py` and can be overridden via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *required* | Fails at import if missing |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Must match between ingest and retrieve |
| `CHAT_MODEL` | `gpt-4o-mini` | Used for answers and query expansion |
| `CHUNK_SIZE` | `500` | Approximate tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap tokens between adjacent chunks |
| `RETRIEVE_K` | `10` | Bi-encoder candidate pool per query |
| `TOP_K` | `3` | Chunks passed to LLM after reranking |
| `SIMILARITY_THRESHOLD` | `0.4` | Below this → skip reranker and LLM |
| `RETRIEVAL_STRATEGY` | `multi_query` | `multi_query`, `hyde`, or `standard` |
| `MULTI_QUERY_N` | `3` | Number of alternative queries to generate |
| `MULTI_QUERY_UNION_K` | `15` | Max chunks in the union before reranking |
| `HYDE_MAX_TOKENS` | `100` | Max tokens in the hypothetical answer |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for Stage 2 |
| `MAX_HISTORY_TURNS` | `3` | User+model turn pairs kept in session |
| `RAG_API_URL` | `http://localhost:8000/chat` | Where the webhook calls the RAG service |
| `WEBHOOK_PORT` | `8001` | Port for the Dialogflow webhook server |

---

## Known limitations

- Single-user, in-memory conversation state (no Redis, no multi-instance sync).
- Citations at document + page granularity, not character span.
- PDFs must be text-extractable (no OCR for scanned documents).
- Uploaded PDFs on HF Spaces are session-scoped (lost on container restart).
- Layer 2 webhook uses mock data; a production system would call a real claims/CRM API.

---

## What I'd add in production

- **Redis session store** — replace the in-memory `deque` for multi-instance deployments.
- **Embedding model fine-tuning** — generate synthetic (query, relevant_chunk) pairs using GPT-4 over existing PDFs; fine-tune with `sentence-transformers` `MultipleNegativesRankingLoss` to teach the model domain-specific terminology equivalences (e.g. "United India" = "IHIP").
- **Hybrid search (BM25 + dense)** — add lexical fallback for exact term matches that dense search misses; merge results before reranking.
- **Eval harness** — RAGAS or a custom (question, ground-truth) dataset to measure faithfulness and answer relevance continuously.
- **OCR pipeline** — Tesseract or AWS Textract for scanned PDFs.
- **Real claims / CRM API** — replace mock dicts in the Dialogflow webhook.
- **Persistent vector DB** — Pinecone or Weaviate for serverless, multi-tenant deployments.
