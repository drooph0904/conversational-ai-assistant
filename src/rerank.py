"""Cross-encoder reranker: second-stage scoring over bi-encoder candidates.

Why two stages?
  Bi-encoder (retrieve.py): embeds query and chunks INDEPENDENTLY, then
  compares vectors with cosine similarity. Fast — scales to millions of docs,
  one embedding lookup per query. But it misses deep query-chunk interactions
  because the two sides never see each other during encoding.

  Cross-encoder (this file): concatenates [query, chunk] and runs them through
  a single transformer forward pass. Captures exact term overlap and semantic
  coherence that bi-encoders miss. Too slow to score every chunk in the corpus,
  so it runs only on the small candidate set from stage 1.

Pipeline:
  query → Chroma (bi-encoder, top RETRIEVE_K=10) → CrossEncoder (top TOP_K=3) → LLM

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  Fine-tuned on 8.8M (query, passage) pairs from MS MARCO passage ranking.
  ~80 MB, ~200 ms on CPU for 10 chunks. Downloaded once and cached by
  sentence-transformers in ~/.cache/huggingface/hub.
"""

from sentence_transformers import CrossEncoder

from src.config import RERANK_MODEL, TOP_K
from src.retrieve import Chunk

# Lazy-loaded on first query — avoids penalising startup time.
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL, max_length=512)
    return _model


def rerank(query: str, chunks: list[Chunk], top_n: int = TOP_K) -> list[Chunk]:
    """Re-order chunks by cross-encoder relevance and return the top_n.

    The original bi-encoder cosine scores are preserved on the returned Chunk
    objects — the reranker only changes the ordering, not the displayed score.
    """
    if not chunks:
        return []
    model = _get_model()
    pairs = [(query, chunk.text) for chunk in chunks]
    scores: list[float] = model.predict(pairs).tolist()
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_n]]
