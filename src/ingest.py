"""Offline ingestion: PDFs -> chunks -> embeddings -> Chroma.

Run: python -m src.ingest

Idempotent: deterministic chunk IDs (source-pX-cY) so re-running upserts
instead of duplicating. Chunks within a single page so citations stay clean.
"""

import re
import time
from pathlib import Path

import chromadb
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pypdf import PdfReader

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
)

# Approximate token count via word count. ~0.75 words per token for English.
WORDS_PER_TOKEN = 0.75

# Free-tier embedding limit is ~30K tokens/min; small batches with pacing
# keep us comfortably under that. Bump these on a paid plan.
EMBED_BATCH_SIZE = 5
BATCH_SLEEP_SECONDS = 12.0
RATE_LIMIT_RETRY_WAIT_SECONDS = 60.0


def load_pages(data_dir: Path) -> list[dict]:
    """One record per page across all PDFs in data_dir."""
    pages: list[dict] = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            pages.append({"source": pdf_path.name, "page": page_num, "text": text})
    return pages


def chunk_text(text: str, size_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text into overlapping ~size_tokens chunks (token-count approximated)."""
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        return []
    chunk_words = max(1, int(size_tokens * WORDS_PER_TOKEN))
    overlap_words = max(0, int(overlap_tokens * WORDS_PER_TOKEN))
    step = max(1, chunk_words - overlap_words)
    chunks: list[str] = []
    for i in range(0, len(words), step):
        piece = words[i : i + chunk_words]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if i + chunk_words >= len(words):
            break
    return chunks


def _embed_once(client: genai.Client, texts: list[str]) -> list[list[float]]:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return [e.values for e in result.embeddings]


def embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Embed a batch as RETRIEVAL_DOCUMENT vectors; retry once after a 429."""
    try:
        return _embed_once(client, texts)
    except genai_errors.ClientError as e:
        if "429" not in str(e):
            raise
        print(f"  rate-limited; sleeping {RATE_LIMIT_RETRY_WAIT_SECONDS:.0f}s and retrying once ...")
        time.sleep(RATE_LIMIT_RETRY_WAIT_SECONDS)
        return _embed_once(client, texts)


def main() -> None:
    print(f"Loading PDFs from {DATA_DIR} ...")
    pages = load_pages(DATA_DIR)
    if not pages:
        raise SystemExit(
            f"No PDFs with extractable text in {DATA_DIR}. "
            "Drop some health-insurance PDFs and re-run."
        )
    print(f"  {len(pages)} pages from {len({p['source'] for p in pages})} PDFs.")

    print("Chunking ...")
    records: list[dict] = []
    for page in pages:
        for c_idx, chunk in enumerate(chunk_text(page["text"], CHUNK_SIZE, CHUNK_OVERLAP)):
            records.append({
                "id": f"{page['source']}-p{page['page']}-c{c_idx}",
                "text": chunk,
                "metadata": {"source": page["source"], "page": page["page"]},
            })
    print(f"  {len(records)} chunks.")

    print(f"Embedding with {EMBEDDING_MODEL} (batch={EMBED_BATCH_SIZE}, pace={BATCH_SLEEP_SECONDS}s) ...")
    client = genai.Client(api_key=GOOGLE_API_KEY)
    vectors: list[list[float]] = []
    total = len(records)
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch_texts = [r["text"] for r in records[i : i + EMBED_BATCH_SIZE]]
        vectors.extend(embed_batch(client, batch_texts))
        done = min(i + EMBED_BATCH_SIZE, total)
        print(f"  {done}/{total}")
        if done < total:
            time.sleep(BATCH_SLEEP_SECONDS)

    print(f"Upserting to Chroma at {CHROMA_PATH} (collection={CHROMA_COLLECTION}) ...")
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(
        ids=[r["id"] for r in records],
        documents=[r["text"] for r in records],
        embeddings=vectors,
        metadatas=[r["metadata"] for r in records],
    )
    print(f"Done. Collection size: {collection.count()} chunks.")


if __name__ == "__main__":
    main()
