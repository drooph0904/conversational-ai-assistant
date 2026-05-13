# Conversational AI Customer Support Assistant — Indian Health Insurance

A two-layer customer support assistant for the Indian health insurance domain.
**Layer 1** is a pure-Python RAG chatbot grounded in publicly available policy documents.
**Layer 2** wraps it with a Dialogflow CX agent that handles structured tasks deterministically
and falls back to the RAG service for open-ended questions.

> **Status:** Layer 1 (RAG chatbot) complete and demoable. Layer 2 (Dialogflow CX) in progress.

## Problem statement

Customers asking insurance questions get one of two bad experiences today:
either a rigid IVR/bot that only handles a fixed menu of tasks, or a free-form LLM
chatbot that confidently hallucinates policy details. This project demonstrates
the **hybrid pattern** used in real enterprise contact centres: deterministic
flows for structured intents (claim status, branch lookup) plus a grounded RAG
layer for open FAQs, so the system stays accurate *and* useful.

## Who it's for

- A portfolio / interview project for a Conversational AI Engineer role.
- Hands-on demonstration of RAG, Dialogflow CX, GCP webhooks, and prompt engineering.

## Architecture

_Diagram and detailed write-up live in [`docs/architecture.md`](docs/architecture.md)._
TL;DR — split into an **offline ingestion pipeline** (PDFs → chunks → embeddings → Chroma)
and an **online query path** (query → embed → retrieve top-K → grounded prompt → Gemini → answer + citations).

## Tech stack

| Layer | Choice | Why |
| --- | --- | --- |
| Language | Python 3.11+ | Standard for ML/AI tooling |
| LLM | Gemini 2.5 Flash via `google-genai` | Cheap, fast, in the Google ecosystem |
| Embeddings | `text-embedding-004` | Matches the Google stack; strong quality |
| Vector store | Chroma (local, persistent) | Zero-setup, fine for < 100K chunks |
| PDF parsing | `pypdf` | Pure Python, no system deps |
| API | FastAPI + Uvicorn | Async, OpenAPI for free |
| UI | Streamlit | Fastest path to a demoable frontend |
| Orchestration | Dialogflow CX | Required by the role; pairs with webhooks |
| Deploy | Cloud Run (fallback: ngrok) | Serverless, scales to zero |

**Deliberately avoided:** LangChain, LlamaIndex, agent frameworks. Raw primitives only — interview-defensibility over convenience.

## How to run locally

```bash
# 1. Create venv and install deps
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Configure secrets (get a free key at https://aistudio.google.com/apikey)
cp .env.example .env
# Open .env and paste your real GOOGLE_API_KEY

# 3. Drop 5-10 health-insurance PDFs into data/
#    e.g. policy brochures from Star Health, HDFC Ergo, Niva Bupa, ICICI Lombard.

# 4. Build the vector store (one-time, re-run when PDFs change)
python -m src.ingest

# 5. Start the API (terminal 1)
uvicorn src.api:app --reload --port 8000

# 6. Start the UI (terminal 2)
streamlit run ui/app.py
```

### Quick API smoke test

```bash
curl -s -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"session_id":"t1","message":"What is the waiting period for pre-existing diseases?"}' | jq
```

## Design decisions

_Each entry below has a longer explanation in `docs/architecture.md`._

- **Chunk size 500 tokens / 50 overlap (within page boundaries)** — large enough to contain whole facts, small enough to keep embeddings focused. Page-bounded chunks keep citations clean (`[Star Health, p.12]`, not `[p.12-13]`).
- **Top-K = 3** — ~1.5K tokens of retrieved context. Leaves room for system instruction and the last 3 turns of history without bloating the prompt.
- **Cosine similarity in Chroma + similarity threshold = 0.6** — if the best chunk scores below 0.6, the service **skips the LLM call entirely** and returns "I don't know". Cheap, fast, hallucination-proof.
- **Asymmetric embeddings** — `RETRIEVAL_DOCUMENT` task type at ingest, `RETRIEVAL_QUERY` at query time. Google's recommended pattern for retrieval quality.
- **Deterministic chunk IDs (`source-pX-cY`)** — re-running `ingest.py` upserts rather than duplicates. Idempotent re-ingestion.
- **In-memory session store (last 3 turns, per `session_id`)** — simplest thing that works for a demo. Redis is the prod upgrade.
- **No reranker, no query rewriting, no eval harness in Layer 1** — listed as Layer 3 improvements.

## Known limitations

- Single-user, in-memory conversation state.
- No re-ranker, no query rewriting, no eval harness.
- Citations are at document+page granularity, not character-span.
- PDFs assumed to be text-extractable (no OCR).

## What I'd add in production

_Filled in during Layer 3._

## What I learned

_Will be filled in by the author after the build is complete._
