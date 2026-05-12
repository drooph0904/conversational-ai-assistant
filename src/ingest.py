"""Offline ingestion pipeline.

Reads PDFs from data/, splits them into overlapping chunks, embeds each chunk
with Google text-embedding-004, and stores the vectors in a persistent Chroma
collection. Run manually whenever the corpus changes:

    python -m src.ingest
"""
