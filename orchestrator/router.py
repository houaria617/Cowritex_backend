"""
orchestrator/router.py
───────────────────────
All conditional-edge routing functions for the CoWriteX graph.

route_intent  — intent_classifier → agents
route_hitl    — hitl_node → persist | edit | agent (regenerate/reject)

Note: route_after_search has been removed. search_node now uses a direct
edge to persist (auto-approve), so no conditional routing is needed after it.
"""

from __future__ import annotations
import logging
from .state import GraphState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  intent_classifier  →  agents
# ─────────────────────────────────────────────────────────────────────────────

def route_intent(state: GraphState) -> str:
    """
    Decides the next node after intent classification.

    • "search"     → search_node  (auto-approves, skips HITL)
    • "literature" → literature_node  (agent owns web resource handling)
    • "write"      → writing
    • "visualize"  → visualisation
    • "chat"       → chat
    • unknown/err  → error_handler
    """
    if state.get("error") and state["intent"] == "unknown":
        return "error_handler"

    intents: list[str] = state.get("intents", [state.get("intent", "unknown")])
    primary = intents[0] if intents else "unknown"

    if primary == "search":
        return "search"

    if primary == "literature":
        return "literature"

    routing = {
        "write":     "writing",
        "visualize": "visualisation",
        "chat":      "chat",
        "unknown":   "error_handler",
    }
    destination = routing.get(primary, "error_handler")
    logger.info("route_intent: intents=%s → %s", intents, destination)
    return destination


# ─────────────────────────────────────────────────────────────────────────────
# 2.  hitl_node  →  persist | edit | agent (regenerate / reject)
#     Only reached by literature / writing / visualisation — never search.
# ─────────────────────────────────────────────────────────────────────────────

def route_hitl(state: GraphState) -> str:
    """
    Routes based on the researcher's HITL decision.

    approve    → persist
    edit       → edit
    reject /
    regenerate → back to the originating agent
    """
    action = state.get("hitl_action", "approve")
    last_agent = state.get("last_agent", "writing")

    if action == "approve":
        return "persist"

    if action == "edit":
        return "edit"

    if action in ("reject", "regenerate"):
        agent_map = {
            "writing":    "writing",
            "literature": "literature",
            "visualize":  "visualisation",
        }
        destination = agent_map.get(last_agent, "writing")
        logger.info(
            "route_hitl: action=%s last_agent=%s → %s",
            action, last_agent, destination,
        )
        return destination

    logger.warning(
        "route_hitl: unhandled action %r — defaulting to persist", action)
    return "persist"
