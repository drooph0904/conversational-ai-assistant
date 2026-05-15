"""Online retrieval: embed a query, return top-K similar chunks from Chroma.

Uses RETRIEVAL_QUERY task type (asymmetric embeddings: ingest.py uses
RETRIEVAL_DOCUMENT). Scores are cosine similarity (1 = identical, 0 = unrelated).
"""

from dataclasses import dataclass

import chromadb
from openai import OpenAI

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    EMBEDDING_MODEL,
    MULTI_QUERY_UNION_K,
    OPENAI_API_KEY,
    TOP_K,
)


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    score: float  # cosine similarity in [0, 1]


# Module-level clients so we pay connection setup once per process.
_chroma = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _chroma.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)
_openai = OpenAI(api_key=OPENAI_API_KEY)


def get_indexed_sources() -> str:
    """Return a comma-separated list of unique document names in the collection."""
    results = _collection.get(limit=10000, include=["metadatas"])
    sources = sorted(set(m["source"] for m in results["metadatas"]))
    return ", ".join(sources) if sources else "no documents indexed yet"


def embed_query(query: str) -> list[float]:
    result = _openai.embeddings.create(input=query, model=EMBEDDING_MODEL)
    return result.data[0].embedding


def _query_collection(vector: list[float], k: int) -> list[Chunk]:
    res = _collection.query(
        query_embeddings=[vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    ids   = res.get("ids",       [[]])[0]
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    chunks: list[Chunk] = []
    for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
        score = max(0.0, min(1.0, 1.0 - float(dist)))
        chunks.append((chunk_id, Chunk(
            text=doc,
            source=str(meta.get("source", "?")),
            page=int(meta.get("page", 0)),
            score=score,
        )))
    return chunks


def retrieve(query: str, k: int = TOP_K) -> list[Chunk]:
    """Top-K chunks most similar to the query."""
    vector = embed_query(query)
    return [chunk for _, chunk in _query_collection(vector, k)]


def retrieve_multi_query(
    queries: list[str],
    k: int = TOP_K,
    union_k: int = MULTI_QUERY_UNION_K,
) -> list[Chunk]:
    """Retrieve for each query variant, deduplicate by chunk ID, return union.

    Solves vocabulary mismatch: one variant will use document terminology
    (e.g. "IHIP") even if the original query uses a full name.
    """
    seen_ids: set[str] = set()
    all_chunks: list[Chunk] = []
    for q in queries:
        vector = embed_query(q)
        for chunk_id, chunk in _query_collection(vector, k):
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            all_chunks.append(chunk)
    all_chunks.sort(key=lambda c: c.score, reverse=True)
    return all_chunks[:union_k]


def retrieve_hyde(hypothetical_answer: str, k: int = TOP_K) -> list[Chunk]:
    """Embed a hypothetical answer and search with that vector.

    The hypothetical uses document vocabulary, so the embedding lands closer
    to actual document chunks than the raw user query would.
    """
    return retrieve(hypothetical_answer, k=k)


_KEYWORD_STOPWORDS = {
    "the", "is", "are", "there", "any", "in", "or", "of", "to", "and",
    "for", "a", "an", "with", "this", "that", "which", "has", "does",
    "what", "say", "about", "other", "policy", "insurance", "policies",
    "health", "will", "not", "be", "have", "under", "if", "it", "as",
    "from", "by", "at", "on", "their", "your", "our", "also", "than",
    "how", "many", "much", "when", "where", "who", "why", "general",
    "related", "regarding", "tell", "me", "covered", "cover", "claim",
}


def keyword_retrieve(terms: list[str], k: int = 3) -> list[Chunk]:
    """Exact-text search via Chroma's $contains filter.

    Guarantees retrieval of chunks that contain the search term verbatim,
    regardless of embedding similarity. Used as a safety net for topics
    that appear only in exclusion sections (different embedding space from
    coverage-phrased queries).
    """
    seen_ids: set[str] = set()
    chunks: list[Chunk] = []
    for term in terms:
        term = term.strip().lower()
        if len(term) < 4 or term in _KEYWORD_STOPWORDS:
            continue
        try:
            res = _collection.get(
                where_document={"$contains": term},
                include=["documents", "metadatas"],
                limit=k,
            )
        except Exception:
            continue
        ids   = res.get("ids",       [])
        docs  = res.get("documents", [])
        metas = res.get("metadatas", [])
        for chunk_id, doc, meta in zip(ids, docs, metas):
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            chunks.append(Chunk(
                text=doc,
                source=str(meta.get("source", "?")),
                page=int(meta.get("page", 0)),
                score=0.25,  # floor score; cross-encoder reranker will re-order
            ))
    return chunks
