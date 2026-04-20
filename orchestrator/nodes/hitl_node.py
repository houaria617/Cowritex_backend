"""
orchestrator/nodes/hitl_node.py
────────────────────────────────
This node is where the graph PAUSES (interrupt_before=["hitl"]).
When the researcher resumes, their decision arrives via state update:
    hitl_action   → "approve" | "edit" | "reject" | "regenerate"
    hitl_feedback → optional text (for reject / regenerate)
    human_edited_text → optional (for edit action)

The node itself only validates the incoming action and
updates the pending ai_suggestion status in the DB.
The routing happens in router.route_hitl().
"""

from __future__ import annotations

import logging

from ..state import GraphState
from database import repository as repo

logger = logging.getLogger(__name__)

_VALID_ACTIONS = {"approve", "edit", "reject", "regenerate"}


def hitl_node(state: GraphState) -> dict:
    """
    Called AFTER the researcher has submitted their decision.
    (LangGraph resumes the graph here with the updated state.)
    """
    section_id = state.get("section_id")
    hitl_action = state.get("hitl_action", "approve")

    # Validate action
    if hitl_action not in _VALID_ACTIONS:
        logger.warning(
            "Unknown hitl_action %r — defaulting to approve", hitl_action)
        hitl_action = "approve"

    # Update the pending suggestion status
    if section_id:
        try:
            suggestion = repo.get_pending_suggestion(section_id)
            if suggestion:
                db_status = {
                    "approve":    "accepted",
                    "edit":       "edited",
                    "reject":     "rejected",
                    "regenerate": "rejected",   # treated as rejection + retry
                }.get(hitl_action, "accepted")

                repo.resolve_suggestion(
                    suggestion_id=suggestion["id"],
                    status=db_status,
                    feedback=state.get("hitl_feedback"),
                )
        except Exception as exc:
            logger.warning("Could not resolve suggestion in DB: %s", exc)

    return {"hitl_action": hitl_action}
