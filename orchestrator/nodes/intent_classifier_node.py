"""
orchestrator/nodes/intent_classifier_node.py
─────────────────────────────────────────────
Classifies the user's message into one or more of:
    write | literature | visualize | search | chat | unknown

Supports compound intents like ["write", "literature"] for
requests that need both drafting and reference search.
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
  "intents":     ["<intent1>", "<intent2>"],
  "instruction": "<clean, imperative version of what the user wants>"
}

Intent rules — choose ALL that apply:
- "write"       → user wants to draft, rephrase, expand, shorten, or edit any text section
- "literature"  → user wants to SYNTHESIZE / ANALYSE papers, find state-of-the-art, build a review
- "search"      → user wants to SEARCH / FIND / LIST papers or references (no synthesis needed)
- "visualize"   → user wants a chart, table, figure, or data visualization
- "chat"        → general question or clarification — no content generation needed
- "unknown"     → genuinely ambiguous or unrelated

Key distinction — "search" vs "literature":
  • "search"      = find and list papers on a topic (raw results)
  • "literature"  = analyse, synthesize, and write a literature review from papers
  • Both together = search for papers AND produce a synthesis from them

Compound intent rules:
- Use multiple intents when the request clearly needs more than one agent.
- "chat" must always appear ALONE — never combine it with others.
- "unknown" must always appear ALONE.
- Maximum 3 intents at once.

Examples:
  "Find me the latest papers on RAG"                       → ["search"]
  "Search for papers on transformers and list them"        → ["search"]
  "Write a literature review on LLMs"                     → ["literature"]
  "Find papers on RAG and write a review from them"       → ["search", "literature"]
  "Write an intro and find me citations for it"           → ["write", "literature"]
  "Draft the results section and add a bar chart"         → ["write", "visualize"]
  "Find papers on RAG and generate a comparison table"    → ["search", "visualize"]
  "Write the intro, search for citations, and plot"       → ["write", "search", "visualize"]
  "What is transfer learning?"                            → ["chat"]
  "asdfjkl"                                               → ["unknown"]

Respond with raw JSON only — no markdown fences, no extra text."""


_VALID = {"write", "literature", "visualize", "search", "chat", "unknown"}
_SOLO = {"chat", "unknown"}   # intents that must appear alone


def _sanitize(intents: list) -> list[str]:
    """Validate and clean the intents list returned by the LLM."""
    cleaned = list(dict.fromkeys(i for i in intents if i in _VALID))

    if not cleaned:
        return ["unknown"]

    for solo in _SOLO:
        if solo in cleaned:
            return [solo]

    return cleaned[:3]


def intent_classifier_node(state: GraphState) -> dict:
    project_id = state["project_id"]
    user_msg = state["user_message"]

    # ── Load preferences if not yet in state ──────────────────────────────────
    prefs = state.get("preferences") or {}
    if not prefs:
        try:
            prefs = repo.get_preferences(project_id) or {}
        except Exception as exc:
            logger.warning("Could not load preferences: %s", exc)
            prefs = {}

    provider = prefs.get("llm_provider", "groq")

    # ── Classify ──────────────────────────────────────────────────────────────
    try:
        llm = get_llm(provider)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        parsed = json.loads(response.content.strip())
        intents_raw = parsed.get("intents", ["unknown"])
        instruction = parsed.get("instruction", user_msg)

        if isinstance(intents_raw, str):
            intents_raw = [intents_raw]

        intents = _sanitize(intents_raw)
        intent = intents[0]
        error_msg = None

    except Exception as exc:
        print(f"\n [INTENT CLASSIFIER ERROR]: {exc}")
        traceback.print_exc()
        logger.error("Intent classification failed: %s", exc)
        intents = ["unknown"]
        intent = "unknown"
        instruction = user_msg
        error_msg = str(exc)

    logger.info(
        "Classified intent(s)=%s for message=%r", intents, user_msg[:60]
    )

    return {
        "intent":        intent,
        "intents":       intents,
        "instruction":   instruction,
        "preferences":   prefs,
        "agent_outputs": {},
        "search_results": [],        # reset for this turn
        "search_summary": None,      # reset for this turn
        "error":         error_msg,
    }
