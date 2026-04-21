"""
orchestrator/nodes/merge_node.py
─────────────────────────────────
Waits for all parallel agents to finish and merges
their outputs into a single agent_output for HITL review.
"""

from __future__ import annotations
import logging
from ..state import GraphState

logger = logging.getLogger(__name__)


def merge_node(state: GraphState) -> dict:
    """
    Combines outputs from parallel agents into one
    consolidated agent_output for the researcher to review.
    """
    intents = state.get("intents", [state.get("intent", "unknown")])
    agent_outputs = state.get("agent_outputs", {})

    sections = []

    # Render each agent's output under a clear heading
    label_map = {
        "writing":    "📝 Drafted Text",
        "literature": "📚 Literature Results",
        "visualize":  "📊 Visualisation",
    }

    for intent in intents:
        output = agent_outputs.get(intent)
        if output:
            label = label_map.get(intent, intent.capitalize())
            sections.append(f"## {label}\n\n{output}")

    merged = "\n\n---\n\n".join(sections) if sections else None

    logger.info(
        "merge_node: combined %d agent output(s) for intents=%s",
        len(sections), intents
    )

    return {
        "agent_output": merged,
    }
