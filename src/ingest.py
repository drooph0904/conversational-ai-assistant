"""Offline ingestion: PDFs -> chunks -> embeddings -> Chroma.

Run: python -m src.ingest

Idempotent: deterministic chunk IDs (source-pX-cY) so re-running upserts
instead of duplicating. Chunks within a single page so citations stay clean.
"""

import re
import time
from io import BytesIO
from pathlib import Path

import chromadb
from openai import OpenAI
from openai import RateLimitError
from pypdf import PdfReader

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)

# Approximate token count via word count. ~0.75 words per token for English.
WORDS_PER_TOKEN = 0.75

# OpenAI paid tier allows 3000 RPM / 1M TPM for text-embedding-3-small.
# Large batches are fine; a short sleep avoids any burst-window issues.
EMBED_BATCH_SIZE = 100
BATCH_SLEEP_SECONDS = 0.5
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


def _embed_once(client: OpenAI, texts: list[str]) -> list[list[float]]:
    # OpenAI embeddings accept a list of texts and return one vector per text.
    result = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [item.embedding for item in result.data]


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch; retry once after a 429."""
    try:
        return _embed_once(client, texts)
    except RateLimitError:
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
    client = OpenAI(api_key=OPENAI_API_KEY)
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


def ingest_pdf_bytes(pdf_bytes: bytes, filename: str) -> int:
    """Ingest a single PDF from raw bytes into the Chroma collection.

    Returns the number of chunks upserted. Safe to call at runtime from the UI —
    chunks are persisted to disk immediately and visible to retrieve.py on the
    next query. On HF Spaces the disk resets on container restart, so uploaded
    PDFs are session-scoped.
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[dict] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        pages.append({"source": filename, "page": page_num, "text": text})

    if not pages:
        return 0

    records: list[dict] = []
    for page in pages:
        for c_idx, chunk in enumerate(chunk_text(page["text"], CHUNK_SIZE, CHUNK_OVERLAP)):
            records.append({
                "id": f"{filename}-p{page['page']}-c{c_idx}",
                "text": chunk,
                "metadata": {"source": filename, "page": page["page"]},
            })

    if not records:
        return 0

    client = OpenAI(api_key=OPENAI_API_KEY)
    vectors: list[list[float]] = []
    for i in range(0, len(records), EMBED_BATCH_SIZE):
        batch_texts = [r["text"] for r in records[i : i + EMBED_BATCH_SIZE]]
        vectors.extend(embed_batch(client, batch_texts))
        if i + EMBED_BATCH_SIZE < len(records):
            time.sleep(BATCH_SLEEP_SECONDS)

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
    return len(records)


if __name__ == "__main__":
    main()
