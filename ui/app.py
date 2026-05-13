"""Streamlit chat UI.

Calls generate_answer() directly (no HTTP) so this works both locally
and on Hugging Face Spaces without a separate FastAPI process.
"""

import uuid

import streamlit as st

from src.chat import generate_answer
from src.ingest import ingest_pdf_bytes


st.set_page_config(page_title="Health Insurance Assistant", page_icon=":hospital:", layout="centered")

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


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"Citations ({len(citations)})"):
        for c in citations:
            st.markdown(
                f"- **{c['source']}**, page {c['page']} &nbsp;"
                f"<span style='color:#888'>score {c['score']:.3f}</span>",
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
