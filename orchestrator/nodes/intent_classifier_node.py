"""
orchestrator/nodes/intent_classifier_node.py
─────────────────────────────────────────────
Classifies the user's message into one of:
    write | literature | visualize | chat | unknown

Also extracts a clean instruction string and populates preferences
from the DB on the first turn.
"""

from __future__ import annotations

import json
import logging
import traceback

from langchain_core.messages import HumanMessage, SystemMessage

from ..state import GraphState
from ._llm import get_llm
from database import repository as repo

logger = logging.getLogger(__name__)

_SYSTEM = """You are the intent classifier for an AI research writing assistant.

Given the user message, output ONLY a JSON object with these fields:
{
  "intent":      "<write | literature | visualize | chat | unknown>",
  "instruction": "<clean, imperative version of what the user wants>"
}

Intent rules:
- "write"       → user wants to draft, rephrase, expand, shorten, edit any text section
- "literature"  → user wants to search papers, find references, state-of-the-art, citations
- "visualize"   → user wants a chart, table, figure, or data visualization
- "chat"        → general question, clarification, project discussion — no content generation
- "unknown"     → genuinely ambiguous or unrelated; use sparingly

Respond with raw JSON only — no markdown fences, no extra text."""


def intent_classifier_node(state: GraphState) -> dict:
    project_id = state["project_id"]
    user_msg = state["user_message"]

    # ── Load preferences if not yet in state ──
    prefs = state.get("preferences") or {}
    if not prefs:
        try:
            prefs = repo.get_preferences(project_id) or {}
        except Exception as exc:
            logger.warning("Could not load preferences: %s", exc)
            prefs = {}

    provider = prefs.get("llm_provider", "groq")

    # ── Classify ──
    try:
        llm = get_llm(provider)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        parsed = json.loads(response.content.strip())
        intent = parsed.get("intent", "unknown")
        instruction = parsed.get("instruction", user_msg)
        error_msg = None

    except Exception as exc:
        # Print the exact API crash to the terminal
        print(f"\n🚨 [INTENT CLASSIFIER ERROR]: {exc}")
        traceback.print_exc()

        logger.error("Intent classification failed: %s", exc)
        intent = "unknown"
        instruction = user_msg
        error_msg = str(exc)

    # Validate
    valid = {"write", "literature", "visualize", "chat", "unknown"}
    if intent not in valid:
        intent = "unknown"

    return {
        "intent":      intent,
        "instruction": instruction,
        "preferences": prefs,
        "error":       error_msg,
    }
