"""
orchestrator/nodes/edit_node.py
────────────────────────────────
Researcher chose "edit" — they supply human_edited_text.
We accept that text as the new version (author_type = "human")
and skip the agent entirely.
"""

from __future__ import annotations

import logging

from ..state import GraphState
from database import repository as repo

logger = logging.getLogger(__name__)


def edit_node(state: GraphState) -> dict:
    section_id = state.get("section_id")
    human_edited_text = state.get(
        "human_edited_text") or state.get("agent_output", "")

    if not human_edited_text:
        return {"error": "edit_node received no edited text", "agent_output": None}

    # Save as a human-authored version
    if section_id:
        try:
            repo.save_new_version(
                section_id=section_id,
                content=human_edited_text,
                author_type="human",
                suggestion_id=None,
            )
        except Exception as exc:
            logger.warning("Could not save human-edited version: %s", exc)

    return {
        "agent_output": human_edited_text,
        "error":        None,
    }
