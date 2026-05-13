"""Streamlit chat UI.

Calls generate_answer() directly (no HTTP) so this works both locally
and on Hugging Face Spaces without a separate FastAPI process.
"""

import uuid

import streamlit as st

from src.chat import generate_answer
from src.ingest import ingest_pdf_bytes


st.set_page_config(page_title="Health Insurance Assistant", page_icon=":hospital:", layout="centered")

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


with st.sidebar:
    st.header("Knowledge Base")
    uploaded_file = st.file_uploader(
        "Upload a policy PDF",
        type=["pdf"],
        help="Add a new insurance policy document. Available for this session only.",
    )
    if uploaded_file is not None:
        if st.button("Add to knowledge base", use_container_width=True):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    n_chunks = ingest_pdf_bytes(uploaded_file.read(), uploaded_file.name)
                    st.success(f"Added {n_chunks} chunks from **{uploaded_file.name}**.")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")
            st.caption("Uploaded PDFs reset when the app restarts.")

    if st.session_state.get("messages"):
        st.divider()
        transcript = _build_transcript(st.session_state.messages)
        st.download_button(
            "Download transcript",
            data=transcript,
            file_name="conversation.txt",
            mime="text/plain",
            use_container_width=True,
        )

col1, col2 = st.columns([5, 1])
with col1:
    st.title("Health Insurance Assistant")
    st.caption("Grounded answers from your policy documents. Refuses to guess when sources are insufficient.")
with col2:
    if st.button("New chat"):
        st.session_state.session_id = f"ui-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()

if "session_id" not in st.session_state:
    st.session_state.session_id = f"ui-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []


def _confidence_label(score: float) -> str:
    if score >= 0.75:
        return "<span style='color:#2ecc71'>&#9679; High confidence</span>"
    if score >= 0.60:
        return "<span style='color:#f39c12'>&#9679; Medium confidence</span>"
    return "<span style='color:#e74c3c'>&#9679; Low confidence</span>"


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for c in citations:
            st.markdown(
                f"- **{c['source']}**, page {c['page']} &nbsp;"
                + _confidence_label(c["score"]),
                unsafe_allow_html=True,
            )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])
        render_citations(msg.get("citations", []))

if prompt := st.chat_input("Ask a question about your health insurance policies..."):
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
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

st.divider()
st.caption(
    "Responses are AI-generated and for informational purposes only. "
    "Please refer to your policy document for binding terms. "
    "Regulated by the Insurance Regulatory and Development Authority of India (IRDAI)."
)
