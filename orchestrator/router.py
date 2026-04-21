

from __future__ import annotations
from .state import GraphState


def route_intent(state: GraphState) -> list[str] | str:
    """
    Returns a list of node names for parallel fan-out,
    or a single string for single-intent routing.
    LangGraph supports list returns from conditional edges
    for Send-based parallelism.
    """
    intents = state.get("intents") or [state.get("intent", "unknown")]

    mapping = {
        "write":      "writing",
        "literature": "literature",
        "visualize":  "visualisation",
        "chat":       "chat",
        "unknown":    "error_handler",
    }

    nodes = [mapping[i] for i in intents if i in mapping]

    # Single intent — return string (existing behavior)
    if len(nodes) == 1:
        return nodes[0]

    # Multiple intents — return list (LangGraph fans out in parallel)
    return nodes


def route_hitl(state: GraphState) -> str:
    action = state.get("hitl_action", "approve")

    if action == "approve":
        return "persist"
    if action == "edit":
        return "edit"
    if action in ("reject", "regenerate"):
        return _last_agent_node(state)

    return "persist"


def _last_agent_node(state: GraphState) -> str:
    agent = state.get("last_agent")
    mapping = {
        "writing":    "writing",
        "literature": "literature",
        "visualize":  "visualisation",
    }
    return mapping.get(agent, "intent_classifier")
