"""
orchestrator/router.py
───────────────────────
All conditional-edge routing functions for the CoWriteX graph.

route_intent   — intent_classifier → agents / search
route_hitl     — hitl_node → persist | edit | agent (regenerate/reject)

Note: route_after_search is kept for pure "search" intent only.
Literature always goes directly to literature_node regardless of
grounded_only — the literature agent handles web resources internally.
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
      • "search"     intent → always goes to search_node
      • "literature" intent → always goes directly to literature_node
                              (grounded_only flag is handled INSIDE the agent:
                               grounded_only=True  → ChromaDB only
                               grounded_only=False → agent fetches web resources itself)
      • "write"      → writing
      • "visualize"  → visualisation
      • "chat"       → chat
      • unknown/error → error_handler
    """
    if state.get("error") and state["intent"] == "unknown":
        return "error_handler"

    intents: list[str] = state.get("intents", [state.get("intent", "unknown")])
    primary = intents[0] if intents else "unknown"

    # ── Explicit search intent ────────────────────────────────────────────────
    if primary == "search":
        return "search"

    # ── Literature: always direct — agent owns web resource handling ──────────
    if primary == "literature":
        return "literature"

    # ── Standard agent routing ────────────────────────────────────────────────
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
# 2.  search_node  →  merge
# ─────────────────────────────────────────────────────────────────────────────

def route_after_search(state: GraphState) -> str:
    """
    After search_node completes for a pure "search" intent,
    surface results directly to merge/HITL.

    Literature no longer routes through search_node, so this
    function always returns "merge".
    """
    logger.info("route_after_search: → merge")
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
        agent_map = {
            "writing":    "writing",
            "literature": "literature",
            "visualize":  "visualisation",
            "search":     "search",
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
