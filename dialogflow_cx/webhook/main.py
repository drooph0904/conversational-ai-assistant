"""Dialogflow CX webhook (Layer 2).

Receives webhook calls from CX flows. For deterministic flows (claim status,
branch lookup) returns mock data. For the general-FAQ fallback intent, calls
into the RAG service and returns its answer to CX.
"""
