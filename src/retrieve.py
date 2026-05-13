"""Online retrieval: embed a query, return top-K similar chunks from Chroma.

Uses RETRIEVAL_QUERY task type (asymmetric embeddings: ingest.py uses
RETRIEVAL_DOCUMENT). Scores are cosine similarity (1 = identical, 0 = unrelated).
"""

from dataclasses import dataclass

import chromadb
from google import genai
from google.genai import types

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
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
_genai = genai.Client(api_key=GOOGLE_API_KEY)


def embed_query(query: str) -> list[float]:
    result = _genai.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


def retrieve(query: str, k: int = TOP_K) -> list[Chunk]:
    """Top-K chunks most similar to the query."""
    vector = embed_query(query)
    res = _collection.query(query_embeddings=[vector], n_results=k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    chunks: list[Chunk] = []
    for doc, meta, dist in zip(docs, metas, dists):
        # Chroma cosine *distance* = 1 - cosine similarity.
        score = max(0.0, min(1.0, 1.0 - float(dist)))
        chunks.append(Chunk(
            text=doc,
            source=str(meta.get("source", "?")),
            page=int(meta.get("page", 0)),
            score=score,
        ))
    return chunks
