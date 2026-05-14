"""Dialogflow CX webhook server (Layer 2).

Single FastAPI app on port 8001. Dialogflow CX calls POST /webhook for every
user turn. This server reads the detected intent from the CX request, routes
to the right handler, and returns a response in CX's expected format.

Three handlers:
  - check-claim-status  → deterministic mock dict lookup (no LLM)
  - find-hospital       → deterministic mock list lookup (no LLM)
  - anything else       → calls the RAG service at localhost:8000/chat

Run: uvicorn dialogflow_cx.webhook.main:app --port 8001
"""

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.config import RAG_API_URL, WEBHOOK_PORT
from src.intents import handle_claim_status, handle_find_hospital


app = FastAPI(title="Dialogflow CX Webhook", version="0.1.0")


# ---------- CX response builder ----------

def cx_response(message: str) -> dict:
    """Wrap a plain-text message in the structure Dialogflow CX expects."""
    return {
        "fulfillment_response": {
            "messages": [{"text": {"text": [message]}}]
        }
    }


def handle_faq(text: str, session_id: str) -> str:
    try:
        resp = requests.post(
            RAG_API_URL,
            json={"session_id": session_id, "message": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("answer", "I don't have information about that.")
    except requests.RequestException:
        return (
            "I'm having trouble reaching the knowledge base right now. "
            "Please try again in a moment."
        )


# ---------- Webhook endpoint ----------

@app.post("/webhook")
async def webhook(request: Request) -> JSONResponse:
    body = await request.json()

    # Extract session ID — CX session path ends with /sessions/<id>
    session_path: str = body.get("sessionInfo", {}).get("session", "unknown")
    session_id = session_path.split("/")[-1]

    # Extract intent display name and user text
    intent_info = body.get("intentInfo", {})
    intent_name: str = intent_info.get("displayName", "").lower()
    text: str = body.get("text", "")

    if "check-claim-status" in intent_name or "claim" in intent_name:
        reply = handle_claim_status(text)
    elif "find-hospital" in intent_name or "hospital" in intent_name:
        reply = handle_find_hospital(text)
    else:
        # Default fallback: route everything else to the RAG service.
        reply = handle_faq(text, session_id)

    return JSONResponse(content=cx_response(reply))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "rag_url": RAG_API_URL}


if __name__ == "__main__":
    uvicorn.run("dialogflow_cx.webhook.main:app", host="0.0.0.0", port=WEBHOOK_PORT, reload=True)
