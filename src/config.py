"""Centralized configuration.

Single source of truth for every tunable knob and required secret. Every other
module imports constants from this file - no magic numbers, no scattered
os.getenv() calls.

Resolution order for any value:
    1. Real environment variable (set by the shell, Cloud Run, CI, etc.)
    2. Value from a local `.env` file (loaded by python-dotenv at import time)
    3. The default literal in this file
Real env vars always win; `.env` is a local-dev convenience and a no-op in
production.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Populate os.environ from a local `.env` if one exists. Silent no-op otherwise,
# which is exactly what we want on Cloud Run where env vars are injected directly.
load_dotenv()


# ---------- Required secrets (fail fast at import time) ----------
# Using os.environ[...] (not os.getenv) so a missing key raises KeyError
# immediately on `import src.config`, instead of silently breaking the
# first /chat request hours later.
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]


# ---------- Model selection ----------
# Both sides of the pipeline (ingest and retrieve) must use the SAME embedding
# model - otherwise the query vectors and document vectors live in different
# spaces and similarity search returns garbage.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")


# ---------- Paths ----------
# Resolve project root from this file's location so paths work regardless of
# where the script is invoked from (e.g. `python -m src.ingest` vs an IDE).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CHROMA_PATH: str = os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "chroma_db"))
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "health_insurance")


# ---------- Chunking ----------
# 500 tokens: large enough to hold a complete fact, focused enough that the
# embedding represents one topic. 50-token overlap (~10%) prevents losing
# facts that straddle a chunk boundary.
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))


# ---------- Retrieval ----------
# Two-stage retrieval: bi-encoder fetches RETRIEVE_K candidates from Chroma,
# then the cross-encoder reranker cuts them down to TOP_K for the LLM prompt.
# Wider first-stage net (10) catches relevant chunks that dense similarity
# would rank lower; reranker reorders them more accurately.
RETRIEVE_K: int = int(os.getenv("RETRIEVE_K", "10"))
TOP_K: int = int(os.getenv("TOP_K", "3"))

# If the best bi-encoder chunk scores below this, skip both reranker and LLM.
# Cheap short-circuit: if nothing is close in embedding space, reranking won't
# save it either.
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.4"))

# ---------- Reranking ----------
# cross-encoder/ms-marco-MiniLM-L-6-v2: fine-tuned on 8.8M (query, passage)
# pairs from MS MARCO. ~80 MB, runs in ~200 ms on CPU for 10 chunks.
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------- Webhook (Layer 2) ----------
RAG_API_URL: str = os.getenv("RAG_API_URL", "http://localhost:8000/chat")
WEBHOOK_PORT: int = int(os.getenv("WEBHOOK_PORT", "8001"))


# ---------- Conversation memory ----------
# Keep the last N user+assistant pairs in the prompt. Enough for follow-ups
# ("what about for senior citizens?") without bloating cost or letting stale
# turns derail the model's focus.
MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "3"))
