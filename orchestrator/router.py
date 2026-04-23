"""
orchestrator/router.py
───────────────────────
All conditional-edge routing functions for the CoWriteX graph.

route_intent        — intent_classifier → agents / search
route_after_search  — search_node → literature | merge
route_hitl          — hitl_node → persist | edit | agent (regenerate/reject)
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

    Key rules:
      • "search"  intent  → always goes to search_node
      • "literature" intent + grounded_only=False → "search_first" (search_node)
      • "literature" intent + grounded_only=True  → skip search, go to literature
      • Parallel intents (e.g. ["write","literature"]) — LangGraph fan-out is
        declared in graph.py via multiple add_edge calls; here we return the
        FIRST destination.  The graph handles parallelism through the Send API
        or sequential execution depending on LangGraph version.
      • Errors / unknown → error_handler
    """
    if state.get("error") and state["intent"] == "unknown":
        return "error_handler"

    intents: list[str] = state.get("intents", [state.get("intent", "unknown")])
    prefs = state.get("preferences", {})
    grounded_only = prefs.get("grounded_only", False)

    primary = intents[0] if intents else "unknown"

    # ── Explicit search intent ───────────────────────────────────────────────
    if primary == "search":
        return "search"

    # ── Literature: decide whether web search is needed first ────────────────
    if primary == "literature":
        if not grounded_only:
            return "search_first"   # search_node → literature_node
        return "literature"         # ChromaDB only

    # ── Standard agent routing ───────────────────────────────────────────────
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
# 2.  search_node  →  literature | merge
# ─────────────────────────────────────────────────────────────────────────────

def route_after_search(state: GraphState) -> str:
    """
    After search_node completes, decide whether to:
      • hand results to literature_node for synthesis ("literature")
      • surface results directly to HITL ("merge") for a pure search request
    """
    intents: list[str] = state.get("intents", [state.get("intent", "unknown")])

    # If "literature" is part of the compound intent, continue to lit node
    if "literature" in intents:
        logger.info("route_after_search: literature in intents → literature")
        return "literature"

    # Pure "search" intent → skip synthesis, go straight to merge/HITL
    logger.info("route_after_search: pure search → merge")
    return "merge"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  hitl_node  →  persist | edit | agent (regenerate / reject)
# ─────────────────────────────────────────────────────────────────────────────

def route_hitl(state: GraphState) -> str:
    """
    Routes based on the researcher's HITL decision.

    approve    → persist
    edit       → edit
    reject /
    regenerate → back to the last agent that produced the output
    """
    action = state.get("hitl_action", "approve")
    last_agent = state.get("last_agent", "writing")

    if action == "approve":
        return "persist"

    if action == "edit":
        return "edit"

    if action in ("reject", "regenerate"):
        # Map last_agent label → graph node name
        agent_map = {
            "writing":   "writing",
            "literature": "literature",
            "visualize": "visualisation",
            "search":    "search",
        }
        destination = agent_map.get(last_agent, "writing")
        logger.info(
            "route_hitl: action=%s last_agent=%s → %s",
            action, last_agent, destination
        )
        return destination

    # Fallback
    logger.warning(
        "route_hitl: unhandled action %r — defaulting to persist", action)
    return "persist"
