"""Build grounded prompt, call OpenAI, return answer + citations.

Two anti-hallucination guardrails:
  1. Strict system prompt: answer ONLY from context, else say "I don't know".
  2. Similarity-threshold short-circuit: if the best chunk scores below
     SIMILARITY_THRESHOLD, skip the LLM call entirely.

Retrieval strategy (set via RETRIEVAL_STRATEGY env var):
  "multi_query" — LLM rewrites query into N variants, union their results.
  "hyde"        — LLM writes a hypothetical answer; embed that instead of query.
  "standard"    — single-query dense retrieval (original baseline).

In-memory per-session history (last MAX_HISTORY_TURNS user+model pairs).
Process-local; swap for Redis in production.
"""

from collections import defaultdict, deque
from dataclasses import dataclass

from openai import OpenAI

from src.config import (
    CHAT_MODEL,
    MAX_HISTORY_TURNS,
    MULTI_QUERY_N,
    MULTI_QUERY_UNION_K,
    OPENAI_API_KEY,
    RETRIEVE_K,
    RETRIEVAL_STRATEGY,
    SIMILARITY_THRESHOLD,
)
from src.query_expansion import generate_alternative_queries, generate_hypothetical_answer
from src.rerank import rerank
from src.retrieve import Chunk, retrieve, retrieve_hyde, retrieve_multi_query, keyword_retrieve, get_indexed_sources

_KEYWORD_STOPWORDS = {
    "the", "is", "are", "there", "any", "in", "or", "of", "to", "and",
    "for", "a", "an", "with", "this", "that", "which", "has", "does",
    "what", "say", "about", "other", "policy", "insurance", "policies",
    "health", "will", "not", "be", "have", "under", "if", "it", "as",
    "from", "by", "at", "on", "their", "your", "sbi", "star", "max",
    "also", "tell", "related", "regarding", "covered", "cover", "claim",
}


def _key_terms(text: str) -> list[str]:
    """Extract meaningful single words from a query for keyword search."""
    words = text.lower().replace("?", "").replace(",", "").split()
    seen: set[str] = set()
    terms: list[str] = []
    for w in words:
        if len(w) >= 5 and w not in _KEYWORD_STOPWORDS and w not in seen:
            seen.add(w)
            terms.append(w)
    return terms[:8]


_KB_NOUNS = {
    "company", "companies", "insurer", "insurers",
    "document", "documents", "data", "information",
}
# Specific insurer names — if present, query is about content, not KB inventory
_INSURER_NAMES = {
    "sbi", "star", "brochure", "max", "bajaj", "hdfc", "icici", "aditya",
    "niva", "care", "reliance", "tata", "manipal", "digit", "future",
    "oriental", "national", "united", "new india",
}
_KB_YOU_HAVE_PHRASES = [
    "you have", "do you have", "you got", "you contain",
    "you support", "you trained", "you know about",
]


def _is_kb_meta_query(query: str) -> bool:
    """Return True if the user is asking about what's in the knowledge base."""
    q = query.lower().strip()
    words = set(q.split())

    # If query names a specific insurer → content question, never meta
    if words & _INSURER_NAMES:
        return False

    # "you have / do you have" + a KB noun (company, insurer, document, data…)
    if any(phrase in q for phrase in _KB_YOU_HAVE_PHRASES):
        if words & _KB_NOUNS:
            return True

    # Intent word + company/insurer token (no specific name present, checked above)
    if words & {"which", "what", "list", "show"} and words & {
        "company", "companies", "insurer", "insurers"
    }:
        return True

    # Explicit inventory phrases
    inventory_phrases = [
        "list of compan", "list of insur",
        "available compan", "available insur",
        "what are the compan", "what are the insur",
        "which compan", "which insur",
        "what compan", "what insur",
        "what documents", "which documents",
        "tell me what you have", "what do you have",
        "what data do you", "what information do you",
    ]
    return any(phrase in q for phrase in inventory_phrases)


_openai = OpenAI(api_key=OPENAI_API_KEY)

# session_id -> deque of (role, text); roles: "user", "model".
# Cap stores 2 entries per "turn", so maxlen = MAX_HISTORY_TURNS * 2.
_history: dict[str, deque] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY_TURNS * 2)
)

_ESCALATION = (
    "For further assistance, please speak with one of our support specialists "
    "at 1800-XXX-XXXX (toll-free, Mon–Sat 9 AM–6 PM IST)."
)

def _build_system_prompt() -> str:
    # Re-reads Chroma metadata on every call so newly uploaded PDFs are
    # reflected immediately without a server restart.
    indexed_sources = get_indexed_sources()
    return (
        "You are a customer support assistant for Indian health insurance. "
        f"The following policy documents are available in the knowledge base: {indexed_sources}. "
        "Answer the user's question using ONLY the context passages provided. "
        "When asked which companies or policies are available, list the documents above. "
        "When comparing policies, use the context to compare them directly and recommend clearly. "
        "IMPORTANT rules for answering:\n"
        "1. If the context explicitly states something is EXCLUDED or NOT COVERED, clearly tell "
        "the user it is excluded and quote the exact exclusion clause. Do NOT say 'I don't have "
        "information' when an exclusion clause directly answers the question.\n"
        "2. Always extract and state exact numbers, percentages, waiting periods, and rupee amounts "
        "from the context — e.g. '10% of eligible hospitalisation expenses', 'INR 500 per day', "
        "'36 months waiting period'. Never paraphrase a specific figure.\n"
        "3. If the context genuinely does not contain any relevant information about the topic, "
        f"reply exactly with: \"I don't have information about that in my knowledge base. {_ESCALATION}\"\n"
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


def _answer_kb_meta(query: str, session_id: str) -> ChatResult:
    """Answer questions about what's in the knowledge base — no retrieval needed."""
    indexed_sources = get_indexed_sources()
    prompt = (
        f"The user asked: \"{query}\"\n\n"
        f"The following policy documents are currently loaded in the knowledge base:\n{indexed_sources}\n\n"
        "List each document clearly and tell the user what they can ask about. "
        "Be friendly and conversational."
    )
    response = _openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    answer = (response.choices[0].message.content or "").strip()
    _history[session_id].append(("user", query))
    _history[session_id].append(("model", answer))
    return ChatResult(answer=answer, citations=[], retrieved_chunks=[])


def generate_answer(query: str, session_id: str) -> ChatResult:
    # Short-circuit for meta-questions about available documents — these have
    # no matching chunks in the vector store so retrieval always fails them.
    if _is_kb_meta_query(query):
        return _answer_kb_meta(query, session_id)

    # Stage 1: bi-encoder retrieval — strategy selects retrieval mode.
    if RETRIEVAL_STRATEGY == "multi_query":
        alt_queries = generate_alternative_queries(query, n=MULTI_QUERY_N)
        candidates = retrieve_multi_query(alt_queries, k=RETRIEVE_K, union_k=MULTI_QUERY_UNION_K)
    elif RETRIEVAL_STRATEGY == "hyde":
        hypo = generate_hypothetical_answer(query)
        candidates = retrieve_hyde(hypo, k=RETRIEVE_K)
    else:
        candidates = retrieve(query, k=RETRIEVE_K)

    # Keyword safety net: exact term matches for topics that only appear in
    # exclusion sections and land far from the query in embedding space.
    # These bypass the cross-encoder — exact-match evidence should always
    # reach the LLM regardless of reranker scores.
    kw_chunks = keyword_retrieve(_key_terms(query))
    existing_texts = {c.text for c in candidates}
    for kchunk in kw_chunks:
        if kchunk.text not in existing_texts:
            candidates.append(kchunk)
            existing_texts.add(kchunk.text)

    # Guardrail: skip LLM if nothing relevant was found.
    # Keyword hits (exact substring match) are definitive proof the topic
    # exists in the docs, so bypass the score threshold when we have any.
    # Without keyword hits, fall back if the best dense score is too low.
    best_score = max((c.score for c in candidates), default=0.0)
    if not candidates or (not kw_chunks and best_score < SIMILARITY_THRESHOLD):
        _history[session_id].append(("user", query))
        _history[session_id].append(("model", FALLBACK_ANSWER))
        return ChatResult(
            answer=FALLBACK_ANSWER,
            citations=[],
            retrieved_chunks=[],
        )

    # Stage 2: cross-encoder reranker — precision over the candidate set.
    reranked = rerank(query, candidates)

    # Guarantee keyword chunks reach the LLM even when the cross-encoder
    # (MS MARCO, general web) demotes domain-specific exclusion clauses.
    reranked_texts = {c.text for c in reranked}
    guaranteed = [kc for kc in kw_chunks if kc.text not in reranked_texts]
    chunks = reranked + guaranteed

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

    response = _openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    answer = (response.choices[0].message.content or "").strip() or FALLBACK_ANSWER

    _history[session_id].append(("user", query))
    _history[session_id].append(("model", answer))

    citations = [
        {"source": c.source, "page": c.page, "score": round(c.score, 4)}
        for         c in chunks
    ]
    return ChatResult(
        answer=answer,
        citations=citations,
        retrieved_chunks=retrieved_payload,
    )
