"""Build grounded prompt, call Gemini, return answer + citations.

Two anti-hallucination guardrails:
  1. Strict system prompt: answer ONLY from context, else say "I don't know".
  2. Similarity-threshold short-circuit: if the best chunk scores below
     SIMILARITY_THRESHOLD, skip the LLM call entirely.

In-memory per-session history (last MAX_HISTORY_TURNS user+model pairs).
Process-local; swap for Redis in production.
"""

from collections import defaultdict, deque
from dataclasses import dataclass

from google import genai
from google.genai import types

from src.config import (
    CHAT_MODEL,
    GOOGLE_API_KEY,
    MAX_HISTORY_TURNS,
    RETRIEVE_K,
    SIMILARITY_THRESHOLD,
)
from src.rerank import rerank
from src.retrieve import Chunk, retrieve


_genai = genai.Client(api_key=GOOGLE_API_KEY)

# session_id -> deque of (role, text); roles: "user", "model".
# Cap stores 2 entries per "turn", so maxlen = MAX_HISTORY_TURNS * 2.
_history: dict[str, deque] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY_TURNS * 2)
)


_ESCALATION = (
    "For further assistance, please speak with one of our support specialists "
    "at 1800-XXX-XXXX (toll-free, Mon–Sat 9 AM–6 PM IST)."
)

SYSTEM_PROMPT = (
    "You are a customer support assistant for Indian health insurance. "
    "Answer the user's question using ONLY the context passages provided. "
    "If the context does not contain the answer, reply exactly with: "
    f"\"I don't have information about that in my knowledge base. {_ESCALATION}\" "
    "Do not use outside knowledge. Be concise. "
    "Cite the sources you used inline as [source filename, page N]."
)

FALLBACK_ANSWER = (
    f"I don't have information about that in my knowledge base. {_ESCALATION}"
)


@dataclass
class ChatResult:
    answer: str
    citations: list[dict]         # [{source, page, score}, ...]
    retrieved_chunks: list[dict]  # [{source, page, score, text}, ...]


def _format_history(turns: deque) -> str:
    if not turns:
        return ""
    lines = ["Recent conversation (most recent last):"]
    for role, text in turns:
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _format_context(chunks: list[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] (source: {c.source}, page: {c.page})\n{c.text}"
        )
    return "\n\n".join(blocks)


def generate_answer(query: str, session_id: str) -> ChatResult:
    # Stage 1: bi-encoder retrieval — wide net for recall.
    candidates = retrieve(query, k=RETRIEVE_K)

    # Guardrail: if the best bi-encoder score is below threshold, nothing
    # relevant exists. Skip reranker and LLM — saves cost and latency.
    if not candidates or candidates[0].score < SIMILARITY_THRESHOLD:
        _history[session_id].append(("user", query))
        _history[session_id].append(("model", FALLBACK_ANSWER))
        return ChatResult(
            answer=FALLBACK_ANSWER,
            citations=[],
            retrieved_chunks=[],
        )

    # Stage 2: cross-encoder reranker — precision over the candidate set.
    chunks = rerank(query, candidates)

    retrieved_payload = [
        {
            "source": c.source,
            "page": c.page,
            "score": round(c.score, 4),
            "text": c.text,
        }
        for c in chunks
    ]

    history_block = _format_history(_history[session_id])
    context_block = _format_context(chunks)
    user_message = (
        (f"{history_block}\n\n" if history_block else "")
        + f"Context:\n{context_block}\n\nUser question: {query}"
    )

    response = _genai.models.generate_content(
        model=CHAT_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )
    answer = (response.text or "").strip() or FALLBACK_ANSWER

    _history[session_id].append(("user", query))
    _history[session_id].append(("model", answer))

    citations = [
        {"source": c.source, "page": c.page, "score": round(c.score, 4)}
        for c in chunks
    ]
    return ChatResult(
        answer=answer,
        citations=citations,
        retrieved_chunks=retrieved_payload,
    )
