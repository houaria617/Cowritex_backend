"""
orchestrator/nodes/output_node.py
──────────────────────────────────
Final node before END.
Formats the response that will be sent back to the frontend/API layer.
Puts a clean response dict into state["messages"] so the caller can read it.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from ..state import GraphState


def output_node(state: GraphState) -> dict:
    agent_output = state.get("agent_output") or ""
    intent = state.get("intent", "chat")
    hitl_action = state.get("hitl_action")

    # Build a human-readable status line
    if hitl_action == "approve":
        status_line = "✅ Output approved and saved."
    elif hitl_action == "edit":
        status_line = "✏️ Your edited version has been saved."
    elif hitl_action in ("reject", "regenerate"):
        # Shouldn't normally reach output after reject/regenerate — but just in case
        status_line = "🔄 Regenerating…"
    else:
        status_line = ""

    final_text = agent_output
    if status_line:
        final_text = f"{status_line}\n\n{agent_output}"

    return {
        "messages": [AIMessage(content=final_text)],
        "error":    None,
    }
