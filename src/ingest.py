"""Offline ingestion: PDFs -> chunks -> embeddings -> Chroma.

Run: python -m src.ingest

Extraction strategy (auto-selected):
  1. unstructured (strategy=fast) — better text quality from pdfminer.
     Section headers detected from page text via keyword regex.
  2. pypdf — fallback when unstructured is unavailable.

Section-aware chunking:
  Every chunk gets a `section` metadata field in Chroma
  (e.g. "EXCLUSIONS", "GENERAL CONDITIONS", "BENEFITS").
  Section is detected by scanning each page's text for all-caps header
  lines that match known insurance keywords. This ensures exclusion-clause
  chunks carry "EXCLUSIONS" as metadata, enabling targeted retrieval.

Idempotent: chunk IDs are source-pX-bY-cZ (page, block-within-page, chunk)
so re-running upserts instead of duplicating.
"""

import os
import re
import time
import tempfile
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

WORDS_PER_TOKEN = 0.75
EMBED_BATCH_SIZE = 100
BATCH_SLEEP_SECONDS = 0.5
RATE_LIMIT_RETRY_WAIT_SECONDS = 60.0

# Insurance section keywords used to identify headers in page text.
_SECTION_KEYWORDS = re.compile(
    r"\b(EXCLUSIONS?|BENEFITS?|DEFINITIONS?|CONDITIONS?|CLAIMS?|WAITING|COVERAGE|"
    r"HOSPITALISATION|HOSPITALIZATION|SCHEDULES?|ANNEXURE|PREMIUMS?|RENEWAL|"
    r"CANCELLATION|INDEMNITY|INCLUSIONS?|PRE.EXIST|GENERAL)\b",
    re.IGNORECASE,
)

# Lines that look like section headers: mostly uppercase, 1-7 words, keyword match.
_HEADER_LINE = re.compile(r"^[A-Z][A-Z\s/&():-]{3,70}$")


def _is_section_header(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 80 or "\n" in line:
        return False
    words = line.split()
    if not (1 <= len(words) <= 8):
        return False
    alpha = [c for c in line if c.isalpha()]
    if not alpha:
        return False
    upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
    return upper_ratio > 0.80 and bool(_SECTION_KEYWORDS.search(line))


def _detect_section(text: str, prev_section: str) -> str:
    """Scan page text line by line; return first section header found or keep prev."""
    for line in text.splitlines():
        if _is_section_header(line.strip()):
            return line.strip().upper()[:60]
    return prev_section


def _is_garbage(text: str) -> bool:
    """Return True for garbled text (rotated labels, table artifacts, etc.)."""
    if len(text) < 4:
        return True
    alphanum = sum(1 for c in text if c.isalnum())
    if alphanum / len(text) < 0.45:
        return True
    # Detect repeating watermarks: if unique words / total words < 0.4
    # e.g. "SBI General Insurance Company SBI General Insurance Company..."
    words = text.split()
    if len(words) >= 8 and len(set(w.lower() for w in words)) / len(words) < 0.40:
        return True
    return False


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_unstructured(source_name: str, pdf_input) -> list[dict] | None:
    """Extract text blocks using unstructured (strategy=fast / pdfminer).

    Returns list of {source, page, section, text} dicts, or None on failure.
    Section is detected from each block's text via _detect_section().
    """
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        return None

    tmp_path = None
    try:
        if isinstance(pdf_input, (str, Path)):
            path_arg = str(pdf_input)
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            tmp.write(pdf_input)
            tmp.close()
            tmp_path = tmp.name
            path_arg = tmp_path

        elements = partition_pdf(path_arg, strategy="fast")
    except Exception as exc:
        print(f"  unstructured failed ({exc}), falling back to pypdf")
        return None
    finally:
        if tmp_path:
            os.unlink(tmp_path)

    current_section = "GENERAL"
    current_page = 1
    # (page, section) -> accumulated clean text
    blocks: dict[tuple[int, str], list[str]] = {}

    for el in elements:
        el_type = type(el).__name__

        if el_type == "PageBreak":
            current_page += 1
            continue

        if hasattr(el, "metadata") and hasattr(el.metadata, "page_number"):
            pn = el.metadata.page_number
            if isinstance(pn, int) and pn > 0:
                current_page = pn

        text = (el.text or "").strip() if hasattr(el, "text") else ""
        if not text or _is_garbage(text):
            continue

        # Update section if this block looks like a section header
        if _is_section_header(text):
            current_section = text.upper()[:60]
            continue  # header is metadata, not content

        # Also scan multi-line blocks for embedded section changes
        detected = _detect_section(text, current_section)
        if detected != current_section:
            current_section = detected

        blocks.setdefault((current_page, current_section), []).append(text)

    if not blocks:
        return None

    result = []
    for (page_num, section), texts in sorted(blocks.items()):
        combined = " ".join(texts).strip()
        if combined:
            result.append({
                "source":  source_name,
                "page":    page_num,
                "section": section,
                "text":    combined,
            })
    return result


def _extract_pypdf(source_name: str, pdf_input) -> list[dict]:
    """Fallback extraction using pypdf with section detection from line patterns."""
    if isinstance(pdf_input, (str, Path)):
        reader = PdfReader(str(pdf_input))
    else:
        reader = PdfReader(BytesIO(pdf_input))

    current_section = "GENERAL"
    result = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        current_section = _detect_section(text, current_section)
        result.append({
            "source":  source_name,
            "page":    page_num,
            "section": current_section,
            "text":    text,
        })
    return result


def _extract_pages(source_name: str, pdf_input) -> list[dict]:
    """Try unstructured first; fall back to pypdf if unavailable or empty."""
    result = _extract_unstructured(source_name, pdf_input)
    if not result:
        result = _extract_pypdf(source_name, pdf_input)
    return result


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, size_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text into overlapping ~size_tokens chunks (word-count approximation)."""
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


def _build_records(pages: list[dict]) -> list[dict]:
    """Turn page-section blocks into Chroma-ready chunk records.

    ID format: source-pPAGE-bBLOCK-cCHUNK
    b (block index within page) ensures uniqueness when one page has multiple
    section blocks.
    """
    # Count blocks per page so b-index is scoped to each page.
    page_block_counter: dict[tuple[str, int], int] = {}
    records = []
    for page in pages:
        key = (page["source"], page["page"])
        b_idx = page_block_counter.get(key, 0)
        page_block_counter[key] = b_idx + 1

        for c_idx, chunk in enumerate(chunk_text(page["text"], CHUNK_SIZE, CHUNK_OVERLAP)):
            records.append({
                "id":   f"{page['source']}-p{page['page']}-b{b_idx}-c{c_idx}",
                "text": chunk,
                "metadata": {
                    "source":  page["source"],
                    "page":    page["page"],
                    "section": page["section"],
                },
            })
    return records


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_once(client: OpenAI, texts: list[str]) -> list[list[float]]:
    result = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [item.embedding for item in result.data]


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch; retry once after a 429."""
    try:
        return _embed_once(client, texts)
    except RateLimitError:
        print(f"  rate-limited; sleeping {RATE_LIMIT_RETRY_WAIT_SECONDS:.0f}s ...")
        time.sleep(RATE_LIMIT_RETRY_WAIT_SECONDS)
        return _embed_once(client, texts)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def load_pages(data_dir: Path) -> list[dict]:
    """Extract page-section blocks from all PDFs in data_dir."""
    pages: list[dict] = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        extracted = _extract_pages(pdf_path.name, pdf_path)
        pages.extend(extracted)
        unique_sections = sorted({p["section"] for p in extracted})
        print(f"  {pdf_path.name}: {len(extracted)} blocks")
        print(f"    sections detected: {unique_sections}")
    return pages


def main() -> None:
    print(f"Loading PDFs from {DATA_DIR} ...")
    pages = load_pages(DATA_DIR)
    if not pages:
        raise SystemExit(
            f"No PDFs with extractable text in {DATA_DIR}. "
            "Drop some health-insurance PDFs and re-run."
        )
    print(f"\n  Total: {len(pages)} page-section blocks "
          f"from {len({p['source'] for p in pages})} PDFs.\n")

    print("Chunking ...")
    records = _build_records(pages)
    print(f"  {len(records)} chunks.\n")

    print(f"Embedding with {EMBEDDING_MODEL} ...")
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

    print(f"\nUpserting to Chroma ({CHROMA_COLLECTION}) ...")
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

    Returns number of chunks upserted. Called at runtime from the UI.
    On HF Spaces the disk resets on container restart, so uploaded PDFs
    are session-scoped.
    """
    pages = _extract_pages(filename, pdf_bytes)
    if not pages:
        return 0

    records = _build_records(pages)
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
