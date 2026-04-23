"""
orchestrator/nodes/merge_node.py
─────────────────────────────────
Waits for all parallel agents to finish and merges
their outputs into a single agent_output for HITL review.

Key fix vs original:
  • agent_outputs is keyed by AGENT name ("search", "literature", "writing",
    "visualisation") but `intents` contains INTENT names ("search",
    "literature", "write", "visualize").
  • We now normalise intent → agent key before looking up agent_outputs,
    so a pure "search" intent (or any compound that includes it) always
    surfaces the search summary in the merged output.
"""

from __future__ import annotations
import logging
from ..state import GraphState

logger = logging.getLogger(__name__)

# Maps intent name (from classifier) → agent_outputs key (set by each node)
_INTENT_TO_AGENT_KEY: dict[str, str] = {
    # intent "write"  → agent_outputs["writing"]
    "write":      "writing",
    "writing":    "writing",        # defensive alias
    "literature": "literature",
    "search":     "search",
    # intent "visualize" → agent_outputs["visualisation"]
    "visualize":  "visualisation",
    "visualisation": "visualisation",
}

# Human-readable section headers for the merged HITL output
_LABEL_MAP: dict[str, str] = {
    "writing":       "📝 Drafted Text",
    "literature":    "📚 Literature Review",
    "search":        "🔍 Search Results",
    "visualisation": "📊 Visualisation",
}


def merge_node(state: GraphState) -> dict:
    """
    Combines outputs from all active agents into one consolidated
    agent_output for the researcher to review at the HITL checkpoint.
    """
    intents = state.get("intents", [state.get("intent", "unknown")])
    agent_outputs = state.get("agent_outputs", {})

    sections: list[str] = []
    seen_keys: set[str] = set()

    for intent in intents:
        agent_key = _INTENT_TO_AGENT_KEY.get(intent, intent)

        # Avoid duplicates when both "write" and "writing" appear
        if agent_key in seen_keys:
            continue
        seen_keys.add(agent_key)

        output = agent_outputs.get(agent_key)
        if output:
            label = _LABEL_MAP.get(agent_key, agent_key.capitalize())
            sections.append(f"## {label}\n\n{output}")

    merged = "\n\n---\n\n".join(sections) if sections else None

    logger.info(
        "merge_node: combined %d agent output(s) | intents=%s | keys_found=%s",
        len(sections), intents, list(seen_keys),
    )

    return {
        "agent_output": merged,
    }
