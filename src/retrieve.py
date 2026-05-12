"""Online retrieval.

Embeds the user query with the same model used at ingest time, then asks Chroma
for the top-K most similar chunks. Returns (chunks, metadata, scores).
"""
