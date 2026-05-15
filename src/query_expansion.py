"""LLM-based query expansion: Multi-Query and HyDE strategies.

Multi-Query: rewrites the user's question into N domain-specific variants so
that vocabulary mismatches (e.g. "United India" vs "IHIP") are covered by at
least one variant.

HyDE (Hypothetical Document Embeddings): generates a short hypothetical answer
in the style of the source documents and embeds *that* instead of the raw
query. The hypothetical naturally uses document vocabulary, bridging the gap.
"""

from openai import OpenAI

from src.config import CHAT_MODEL, HYDE_MAX_TOKENS, MULTI_QUERY_N, OPENAI_API_KEY

_openai = OpenAI(api_key=OPENAI_API_KEY)

_MULTI_QUERY_SYSTEM = (
    "You are a query expansion assistant for an Indian health insurance document "
    "retrieval system. Given a user question, generate {n} alternative search "
    "queries that:\n"
    "- Use abbreviations and acronyms that insurance policy PDFs use "
    "(e.g. IHIP, OPD, ICU, PED, SI, NCB, TPA, GST, AYUSH)\n"
    "- Rephrase to match how insurance policy text is written, not how a "
    "customer speaks\n"
    "- ALWAYS include at least one variant that searches for the topic as an "
    "EXCLUSION or NOT COVERED clause — e.g. if asked about 'pregnancy coverage', "
    "generate a variant like 'pregnancy childbirth excluded not covered' because "
    "many insurance topics appear only in the exclusions section of the policy\n"
    "- Cover different aspects (duration, eligibility, exclusions, renewal)\n"
    "Output exactly {n} queries, one per line. No numbering, no explanation."
)


def generate_alternative_queries(query: str, n: int = MULTI_QUERY_N) -> list[str]:
    """Return [original_query] + up to n LLM-generated alternative phrasings."""
    system = _MULTI_QUERY_SYSTEM.format(n=n)
    response = _openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        temperature=0.7,
        max_tokens=150,
    )
    raw = (response.choices[0].message.content or "").strip()
    alternatives = [line.strip() for line in raw.splitlines() if line.strip()]
    # Original always first — if GPT returns fewer lines than asked, the
    # original query is never lost and retrieval still works.
    return [query] + alternatives[:n]


def generate_hypothetical_answer(query: str) -> str:
    """Return a short passage written in the style of an insurance policy document.

    Embedding this hypothetical instead of the raw query pulls the search vector
    into the document's vocabulary space (uses terms like 'IHIP', 'annual
    premium', 'renewable') rather than the customer's vocabulary space.
    """
    prompt = (
        "Write a 1-2 sentence passage exactly as it would appear in an Indian "
        "health insurance policy document, answering the following question. "
        "Use the document's terminology and abbreviations (e.g. IHIP, SI, PED, "
        "OPD). Do not add any preamble.\n\n"
        f"Question: {query}"
    )
    response = _openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=HYDE_MAX_TOKENS,
    )
    return (response.choices[0].message.content or "").strip()
