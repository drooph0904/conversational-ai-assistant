"""Intent routing — shared between the Streamlit UI and the Dialogflow CX webhook.

Three intents:
  "claim"    → deterministic claim-status lookup (no LLM)
  "hospital" → deterministic network-hospital lookup (no LLM)
  "rag"      → open-ended question, caller should invoke generate_answer()

Detection is keyword + pattern based, matching what Dialogflow CX would do via
trained intents. Both the Streamlit UI (ui/app.py) and the webhook server
(dialogflow_cx/webhook/main.py) import from here so the logic stays in one place.
"""

import re

# ---------- Mock data (replaces a real DB in production) ----------

CLAIM_STATUS: dict[str, str] = {
    "8823": "Under Review",
    "1234": "Approved",
    "5678": "Rejected",
    "9999": "Processing",
    "4321": "Paid Out",
}

NETWORK_HOSPITALS: dict[str, list[str]] = {
    "mumbai": [
        "Kokilaben Dhirubhai Ambani Hospital, Andheri West",
        "Lilavati Hospital, Bandra",
        "Hinduja Hospital, Mahim",
    ],
    "delhi": [
        "Apollo Hospital, Sarita Vihar",
        "Fortis Hospital, Vasant Kunj",
        "Max Super Speciality Hospital, Saket",
    ],
    "pune": [
        "Ruby Hall Clinic, Camp",
        "Jehangir Hospital, Sassoon Road",
        "Sahyadri Hospital, Deccan",
    ],
    "bangalore": [
        "Manipal Hospital, Old Airport Road",
        "Fortis Hospital, Bannerghatta Road",
        "Apollo Hospital, Bannerghatta Road",
    ],
    "hyderabad": [
        "Apollo Hospital, Jubilee Hills",
        "KIMS Hospital, Secunderabad",
        "Care Hospital, Banjara Hills",
    ],
}


# ---------- Handlers ----------

def handle_claim_status(text: str) -> str:
    match = re.search(r"\b(\d{4,})\b", text)
    if not match:
        return (
            "I need your claim ID to check the status. "
            "Please say something like: 'Check status for claim 8823'."
        )
    claim_id = match.group(1)
    status = CLAIM_STATUS.get(claim_id)
    if status:
        return f"Your claim {claim_id} is currently: **{status}**."
    return (
        f"I couldn't find claim ID {claim_id} in our system. "
        "Please double-check your claim number."
    )


def handle_find_hospital(text: str) -> str:
    text_lower = text.lower()
    for city, hospitals in NETWORK_HOSPITALS.items():
        if city in text_lower:
            lines = "\n".join(f"• {h}" for h in hospitals)
            return f"Network hospitals in {city.title()}:\n{lines}"
    cities = ", ".join(c.title() for c in NETWORK_HOSPITALS)
    return (
        f"I have hospital listings for: {cities}. "
        "Which city are you looking for?"
    )


# ---------- Router ----------

_CLAIM_KEYWORDS    = ("claim", "status", "my claim", "claim status")
_HOSPITAL_KEYWORDS = ("hospital", "network hospital", "empanelled", "cashless hospital")


def route(text: str) -> tuple[str, str | None]:
    """Detect intent and return (intent, reply).

    Returns:
        ("claim",    reply_str)  — deterministic claim-status answer
        ("hospital", reply_str)  — deterministic hospital-list answer
        ("rag",      None)       — caller must invoke generate_answer()
    """
    text_lower = text.lower()

    if re.search(r"\b\d{4,}\b", text) and any(k in text_lower for k in _CLAIM_KEYWORDS):
        return "claim", handle_claim_status(text)

    if any(k in text_lower for k in _HOSPITAL_KEYWORDS):
        return "hospital", handle_find_hospital(text)

    return "rag", None
