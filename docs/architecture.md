# Architecture

> _Filled in as each component lands. The skeleton below mirrors what we'll talk through in the interview._

## High-level diagram

```mermaid
flowchart LR
    subgraph Offline ["Offline (run once)"]
        PDFs[PDFs in data/] --> Loader[pypdf loader]
        Loader --> Chunker[Chunker<br/>~500 tok, 50 overlap]
        Chunker --> Embedder1[text-embedding-004]
        Embedder1 --> Chroma[(Chroma<br/>persistent)]
    end

    subgraph Online ["Online (per chat turn)"]
        User[User] --> UI[Streamlit UI]
        UI --> API[FastAPI /chat]
        API --> Embedder2[text-embedding-004]
        Embedder2 --> Chroma
        Chroma -- top-K chunks --> Prompt[Grounded prompt builder]
        History[Session history<br/>last 3 turns] --> Prompt
        Prompt --> Gemini[Gemini 2.5 Flash]
        Gemini --> API
        API --> UI
    end
```

## Layer 2 — Dialogflow CX in front

```mermaid
flowchart LR
    User --> CX[Dialogflow CX agent]
    CX -- structured intent<br/>(claim status, branch) --> Webhook1[Mock-data webhook]
    CX -- fallback / FAQ intent --> Webhook2[RAG webhook]
    Webhook2 --> RAG[FastAPI /chat on Cloud Run]
    RAG --> Webhook2
    Webhook2 --> CX
    CX --> User
```

## Design decisions

_Each gets a paragraph of justification once the corresponding code lands._

- Chunk size and overlap
- Top-K
- Embedding + chat model choice
- Vector store choice (Chroma vs. alternatives)
- Similarity threshold and "I don't know" fallback
- Session memory window
- Why no re-ranker in Layer 1
- Why deterministic flows + RAG fallback (instead of a pure LLM agent)
