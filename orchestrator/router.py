"""
orchestrator/router.py
───────────────────────
Edge routing functions for the LangGraph state machine.
"""

from __future__ import annotations

from .state import GraphState


def route_intent(state: GraphState) -> str:
    return state.get("intent", "unknown")


def route_hitl(state: GraphState) -> str:
    """
    Called after hitl_node resolves.
    Returns the next node name.
    """
    action = state.get("hitl_action", "approve")

    if action == "approve":
        return "persist"

    if action == "edit":
        return "edit"

    # reject / regenerate → re-run the agent that produced the output
    if action in ("reject", "regenerate"):
        return _last_agent_node(state)

    return "persist"   # safe fallback


def route_to_last_agent(state: GraphState) -> str:
    """Direct router used in graph wiring for reject/regenerate edges."""
    return _last_agent_node(state)


def _last_agent_node(state: GraphState) -> str:
    agent = state.get("last_agent")
    mapping = {
        "writing":    "writing",
        "literature": "literature",
        "visualize":  "visualisation",
    }
    return mapping.get(agent, "intent_classifier")
