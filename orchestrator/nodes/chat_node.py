"""
orchestrator/nodes/chat_node.py
────────────────────────────────
Handles general conversation — no HITL, no DB suggestion tracking.
Saves both turns to chat_messages and responds directly.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ..state import GraphState
from ._llm import get_llm
from database import repository as repo

logger = logging.getLogger(__name__)

_SYSTEM = """You are CoWriteX, an AI assistant for academic researchers.
You help researchers plan their papers, discuss ideas, answer questions about
writing, structure, methodology, and citations.
Be concise, accurate, and academically appropriate.
If the researcher asks you to draft, edit, search literature, or create charts,
tell them to use the appropriate command so the right agent can handle it."""


def chat_node(state: GraphState) -> dict:
    project_id = state["project_id"]
    section_id = state.get("section_id")
    user_msg = state["user_message"]
    prefs = state.get("preferences", {})
    provider = prefs.get("llm_provider", "groq")

    # Build message history for context
    lc_messages = [SystemMessage(content=_SYSTEM)]
    try:
        history = repo.get_recent_messages(project_id, limit=10)
        for m in history:
            cls = HumanMessage if m["role"] == "human" else AIMessage
            lc_messages.append(cls(content=m["content"]))
    except Exception as exc:
        logger.warning("Could not load chat history: %s", exc)

    lc_messages.append(HumanMessage(content=user_msg))

    try:
        llm = get_llm(provider, temperature=0.5)
        response = llm.invoke(lc_messages)
        reply = response.content
    except Exception as exc:
        logger.error("Chat LLM failed: %s", exc)
        reply = "I'm having trouble connecting right now. Please try again in a moment."

    # Persist both sides
    try:
        repo.save_message(project_id, "human", user_msg, section_id)
        repo.save_message(project_id, "ai",    reply,    section_id)
    except Exception as exc:
        logger.warning("Could not save chat messages: %s", exc)

    return {
        "agent_output": reply,
        "error":        None,
    }
