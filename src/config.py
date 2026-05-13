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
GOOGLE_API_KEY: str = os.environ["GOOGLE_API_KEY"]


# ---------- Model selection ----------
# Both sides of the pipeline (ingest and retrieve) must use the SAME embedding
# model - otherwise the query vectors and document vectors live in different
# spaces and similarity search returns garbage.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gemini-2.5-flash")


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
# Top-K=4 gives ~2K tokens of context, leaving room for system instructions
# and the last few turns of conversation without blowing the context window.
TOP_K: int = int(os.getenv("TOP_K", "4"))

# If the best chunk's similarity is below this, we short-circuit to
# "I don't know" without even calling the LLM. Cheap, fast, hallucination-safe.
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))


# ---------- Conversation memory ----------
# Keep the last N user+assistant pairs in the prompt. Enough for follow-ups
# ("what about for senior citizens?") without bloating cost or letting stale
# turns derail the model's focus.
MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "3"))
