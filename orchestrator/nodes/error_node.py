"""
orchestrator/nodes/error_node.py
─────────────────────────────────
Handles system/parse failures (intent = "unknown" or error in any node).
Logs the error and returns a user-friendly message.
Does NOT handle HITL decisions — those are routing choices, not errors.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from ..state import GraphState

logger = logging.getLogger(__name__)


def error_node(state: GraphState) -> dict:
    error = state.get("error") or "Unknown error"
    intent = state.get("intent", "unknown")
    user_msg = state.get("user_message", "")

    logger.error(
        "error_node reached | intent=%r | error=%r | user_msg=%r",
        intent, error, user_msg,
    )

    if intent == "unknown":
        reply = (
            "I'm not sure what you'd like me to do. You can ask me to:\n"
            "- **Draft or edit** a section of your paper\n"
            "- **Search the literature** for relevant papers\n"
            "- **Generate a chart or table** from your data\n"
            "- **Chat** about your research\n\n"
            "Could you rephrase your request?"
        )
    else:
        reply = (
            f"Something went wrong while processing your request.\n\n"
            f"**Error:** {error}\n\n"
            "Please try again, or contact support if the problem persists."
        )

    return {
        "messages": [AIMessage(content=reply)],
        "agent_output": reply,
        "error": error,
    }
