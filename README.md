# Conversational AI Customer Support Assistant — Indian Health Insurance

A two-layer customer support assistant for the Indian health insurance domain.
**Layer 1** is a pure-Python RAG chatbot grounded in publicly available policy documents.
**Layer 2** wraps it with a Dialogflow CX agent that handles structured tasks deterministically
and falls back to the RAG service for open-ended questions.

> **Status:** under construction. This README is filled in as the project grows.

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

> _Filled in once `src/api.py` and `ui/app.py` are wired up._

```bash
# 1. Create venv and install deps
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# Edit .env and paste your GOOGLE_API_KEY

# 3. Drop 5–10 health-insurance PDFs into data/

# 4. Ingest (one-time)
python -m src.ingest

# 5. Run API + UI in two terminals
uvicorn src.api:app --reload --port 8000
streamlit run ui/app.py
```

## Design decisions

_Each entry below has a longer explanation in `docs/architecture.md`._

- **Chunk size 500 tokens / 50 overlap** — large enough to contain whole facts, small enough to keep embeddings focused.
- **Top-K = 4** — ~2K tokens of context, leaves room for instructions + chat history.
- **Similarity threshold for fallback** — short-circuits to "I don't know" before calling the LLM when retrieval is weak.
- **In-memory session store (last 3 turns)** — simplest thing that works for a demo. Redis is the prod upgrade.
- **No reranker in Layer 1** — discussed in Layer 3 as a quality improvement.

## Known limitations

- Single-user, in-memory conversation state.
- No re-ranker, no query rewriting, no eval harness.
- Citations are at document+page granularity, not character-span.
- PDFs assumed to be text-extractable (no OCR).

## What I'd add in production

_Filled in during Layer 3._

## What I learned

_Will be filled in by the author after the build is complete._
