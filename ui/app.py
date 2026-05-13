"""Streamlit chat UI for the Layer 1 RAG service.

Run (with the FastAPI service already running):
    streamlit run ui/app.py
"""

import uuid

import requests
import streamlit as st


API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="Health Insurance Assistant", page_icon=":hospital:", layout="centered")

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


# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])
        render_citations(msg.get("citations", []))

# New message
if prompt := st.chat_input("Ask a question about your health insurance policies..."):
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    API_URL,
                    json={
                        "session_id": st.session_state.session_id,
                        "message": prompt,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                citations = data.get("citations", [])
            except requests.RequestException as e:
                answer = f"Could not reach the API: {e}"
                citations = []
        st.write(answer)
        render_citations(citations)

    st.session_state.messages.append({
        "role": "assistant",
        "text": answer,
        "citations": citations,
    })
