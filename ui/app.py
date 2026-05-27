"""Streamlit chat UI.

Calls generate_answer() directly (no HTTP) so this works both locally
and on Hugging Face Spaces without a separate FastAPI process.
"""

import uuid
import streamlit as st
from src.chat import generate_answer
from src.ingest import ingest_pdf_bytes


st.set_page_config(
    page_title="Health Insurance Assistant",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ─────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp { background: #FFFFFF; }

/* ── Sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(170deg, #FFF8F3 0%, #FFF0E0 100%);
    border-right: 1px solid #FFE0B2;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

.sb-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 18px;
    border-bottom: 1px solid #FFD9A8;
    margin-bottom: 20px;
}
.sb-brand-icon {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #FF7043, #FF9A5C);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(255, 112, 67, 0.30);
}
.sb-brand-name  { font-size: 14px; font-weight: 700; color: #BF360C; line-height: 1.2; }
.sb-brand-sub   { font-size: 11px; color: #FFAB76; }

.sb-section-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #FF7043;
    margin: 20px 0 8px;
}

/* ── Buttons ──────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #FF7043 0%, #FF9A5C 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 9px 22px !important;
    box-shadow: 0 3px 10px rgba(255, 112, 67, 0.30) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(255, 112, 67, 0.40) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: #FF7043 !important;
    border: 1.5px solid #FF7043 !important;
    box-shadow: none !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(255, 112, 67, 0.06) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Chat input ───────────────────────────────────────── */
[data-testid="stChatInput"] {
    border: 2px solid #FFD9A8 !important;
    border-radius: 14px !important;
    background: #fff !important;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #FF7043 !important;
    box-shadow: 0 0 0 4px rgba(255, 112, 67, 0.10) !important;
}

/* ── Chat messages ────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    padding: 6px 8px !important;
    margin-bottom: 10px !important;
    border: 1px solid transparent !important;
    transition: box-shadow 0.2s !important;
}
/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 60%) !important;
    border-color: #FFCC80 !important;
}
/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #FFFFFF !important;
    border-color: #F0F0F0 !important;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05) !important;
}
/* User avatar color */
[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, #FF7043, #FF9A5C) !important;
}

/* ── Sources expander ─────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #FFD9A8 !important;
    border-radius: 10px !important;
    background: #FFFAF5 !important;
}
[data-testid="stExpander"] summary {
    color: #E64A19 !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}
[data-testid="stExpander"] summary:hover {
    background: rgba(255, 112, 67, 0.05) !important;
}

/* ── File uploader ────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #FFB74D !important;
    border-radius: 10px !important;
    background: rgba(255, 183, 77, 0.04) !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #FF7043 !important;
    background: rgba(255, 112, 67, 0.05) !important;
}

/* ── Divider ──────────────────────────────────────────── */
hr { border-color: #F5F5F5 !important; }

/* ── Confidence badges ────────────────────────────────── */
.conf-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    vertical-align: middle;
}
.conf-high { background: #E8F5E9; color: #2E7D32; }
.conf-med  { background: #FFF3E0; color: #E64A19; }
.conf-low  { background: #FFEBEE; color: #C62828; }

/* ── Main header banner ───────────────────────────────── */
.main-header {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 26px 30px;
    background: linear-gradient(130deg, #FF7043 0%, #FF9A5C 60%, #FFBE76 100%);
    border-radius: 20px;
    margin-bottom: 10px;
    box-shadow: 0 8px 28px rgba(255, 112, 67, 0.24);
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50px; right: -30px;
    width: 160px; height: 160px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 50%;
}
.main-header::after {
    content: '';
    position: absolute;
    bottom: -40px; right: 100px;
    width: 100px; height: 100px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 50%;
}
.h-icon {
    width: 56px;
    height: 56px;
    background: rgba(255, 255, 255, 0.22);
    border-radius: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 30px;
    flex-shrink: 0;
    position: relative;
    z-index: 1;
    backdrop-filter: blur(4px);
}
.h-title {
    font-size: 22px;
    font-weight: 700;
    color: #fff;
    line-height: 1.25;
    position: relative;
    z-index: 1;
}
.h-sub {
    font-size: 13px;
    color: rgba(255, 255, 255, 0.82);
    margin-top: 5px;
    position: relative;
    z-index: 1;
}

/* ── Welcome screen ───────────────────────────────────── */
.welcome-wrap {
    text-align: center;
    padding: 56px 20px 40px;
}
.welcome-icon {
    width: 84px;
    height: 84px;
    background: linear-gradient(135deg, #FF7043, #FF9A5C);
    border-radius: 24px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 42px;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(255, 112, 67, 0.28);
}
.welcome-title {
    font-size: 24px;
    font-weight: 700;
    color: #1A1A1A;
    margin-bottom: 10px;
}
.welcome-sub {
    font-size: 14.5px;
    color: #757575;
    max-width: 420px;
    margin: 0 auto 30px;
    line-height: 1.7;
}
.chip-row {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
}
.chip {
    background: #FFFFFF;
    border: 1.5px solid #FFD9A8;
    color: #E64A19;
    padding: 9px 18px;
    border-radius: 24px;
    font-size: 13px;
    font-weight: 500;
    cursor: default;
    transition: border-color 0.18s, background 0.18s;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.chip:hover { background: #FFF3E0; border-color: #FF7043; }

/* ── Footer ───────────────────────────────────────────── */
.footer-note {
    text-align: center;
    font-size: 11.5px;
    color: #BDBDBD;
    line-height: 1.8;
    padding: 10px 0 4px;
}

/* ── Force readable text in chat bubbles ──────────────── */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #1A1A1A !important;
}

/* ── Custom scrollbar ─────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #FFD9A8; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #FF7043; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────

def _build_transcript(messages: list[dict]) -> str:
    lines = ["Health Insurance Assistant — Conversation Transcript", "=" * 52, ""]
    for msg in messages:
        role = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}:")
        lines.append(msg["text"])
        if msg.get("citations"):
            sources = ", ".join(
                f"{c['source']} p.{c['page']}" for c in msg["citations"]
            )
            lines.append(f"Sources: {sources}")
        lines.append("")
    return "\n".join(lines)


def _confidence_label(score: float) -> str:
    if score >= 0.75:
        return "<span class='conf-badge conf-high'>● High confidence</span>"
    if score >= 0.60:
        return "<span class='conf-badge conf-med'>● Medium confidence</span>"
    return "<span class='conf-badge conf-low'>● Low confidence</span>"


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"📄  Sources ({len(citations)})"):
        for c in citations:
            st.markdown(
                f"- **{c['source']}**, page {c['page']} &nbsp;"
                + _confidence_label(c["score"]),
                unsafe_allow_html=True,
            )


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-icon">📋</div>
        <div>
            <div class="sb-brand-name">Knowledge Base</div>
            <div class="sb-brand-sub">Policy Documents</div>
        </div>
    </div>
    <div class="sb-section-label">Upload Document</div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a policy PDF",
        type=["pdf"],
        help="Add a new insurance policy document. Available for this session only.",
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        if st.button("Add to knowledge base", use_container_width=True):
            with st.spinner(f"Processing {uploaded_file.name}…"):
                try:
                    n_chunks = ingest_pdf_bytes(uploaded_file.read(), uploaded_file.name)
                    st.success(f"Added **{n_chunks}** chunks from **{uploaded_file.name}**.")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")
            st.caption("Uploaded PDFs reset when the app restarts.")

    if st.session_state.get("messages"):
        st.divider()
        st.markdown('<div class="sb-section-label">Export</div>', unsafe_allow_html=True)
        transcript = _build_transcript(st.session_state.messages)
        st.download_button(
            "⬇  Download transcript",
            data=transcript,
            file_name="conversation.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ── Session state ─────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = f"ui-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Header ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <div class="h-icon">🏥</div>
    <div>
        <div class="h-title">Health Insurance Assistant</div>
        <div class="h-sub">Grounded answers from your policy documents · Never guesses</div>
    </div>
</div>
""", unsafe_allow_html=True)

_, btn_col = st.columns([5, 1])
with btn_col:
    if st.button("↺ New chat", use_container_width=True):
        st.session_state.session_id = f"ui-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()


# ── Welcome screen ────────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-icon">🏥</div>
        <div class="welcome-title">How can I help you today?</div>
        <div class="welcome-sub">
            Ask anything about your health insurance policies.
            I provide answers grounded in your uploaded documents
            and clearly indicate when information isn't available.
        </div>
        <div class="chip-row">
            <span class="chip">What is covered?</span>
            <span class="chip">What are the exclusions?</span>
            <span class="chip">How do I file a claim?</span>
            <span class="chip">Premiums &amp; renewals</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Chat history ──────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])
        render_citations(msg.get("citations", []))


# ── Chat input ────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about your health insurance policies…"):
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = generate_answer(prompt, st.session_state.session_id)
                answer = result.answer
                citations = [
                    {"source": c["source"], "page": c["page"], "score": c["score"]}
                    for c in result.citations
                ]
            except Exception as e:
                answer = f"Something went wrong: {e}"
                citations = []
        st.write(answer)
        render_citations(citations)

    st.session_state.messages.append({
        "role": "assistant",
        "text": answer,
        "citations": citations,
    })


# ── Footer ────────────────────────────────────────────────────────────────

st.markdown("""
<hr>
<div class="footer-note">
    Responses are AI-generated and for informational purposes only.<br>
    Refer to your policy document for binding terms.
    Regulated by the Insurance Regulatory and Development Authority of India (IRDAI).
</div>
""", unsafe_allow_html=True)
