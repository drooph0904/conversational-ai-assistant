"""FastAPI service exposing /chat and /health.

Run: uvicorn src.api:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.chat import generate_answer


app = FastAPI(
    title="Health Insurance RAG Assistant",
    version="0.1.0",
    description="Layer 1: grounded RAG over Indian health insurance PDFs.",
)

# Open CORS for the demo (Streamlit on localhost, Dialogflow webhook later).
# Tighten this to specific origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, examples=["sess-1"])
    message: str = Field(min_length=1, examples=["What is the waiting period for pre-existing diseases?"])


class Citation(BaseModel):
    source: str
    page: int
    score: float


class RetrievedChunk(BaseModel):
    source: str
    page: int
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    result = generate_answer(req.message, req.session_id)
    return ChatResponse(
        answer=result.answer,
        citations=[Citation(**c) for c in result.citations],
        retrieved_chunks=[RetrievedChunk(**c) for c in result.retrieved_chunks],
    )
