"""
orchestrator/nodes/persist_node.py
────────────────────────────────────
Saves the approved / edited agent_output as the new current document version.
Also saves both chat turns (user message + AI output) to chat_messages.
Updates project progress.
"""

from __future__ import annotations

import logging

from ..state import GraphState
from database import repository as repo

logger = logging.getLogger(__name__)


def persist_node(state: GraphState) -> dict:
    project_id = state["project_id"]
    section_id = state.get("section_id")
    agent_output = state.get("agent_output", "")
    hitl_action = state.get("hitl_action", "approve")
    user_msg = state["user_message"]

    # ── Save document version ──
    # edit_node already saved for "edit"; we save for "approve" here.
    if section_id and hitl_action == "approve" and agent_output:
        try:
            # Find the pending suggestion that was just approved
            suggestion = repo.get_pending_suggestion(section_id)
            suggestion_id = suggestion["id"] if suggestion else None

            repo.save_new_version(
                section_id=section_id,
                content=agent_output,
                author_type="ai",
                suggestion_id=suggestion_id,
            )
        except Exception as exc:
            logger.warning("Could not save document version: %s", exc)

    # ── Save chat history ──
    try:
        repo.save_message(project_id, "human", user_msg, section_id)
        repo.save_message(project_id, "ai", agent_output or "", section_id)
    except Exception as exc:
        logger.warning("Could not save chat messages: %s", exc)

    # ── Update project progress (simple heuristic) ──
    # Count sections with at least one current version vs total sections
    try:
        sections_res = (
            repo.db.table("sections")
            .select("id")
            .eq("project_id", project_id)
            .execute()
        )
        total = len(sections_res.data or [])
        if total > 0:
            done = 0
            for sec in (sections_res.data or []):
                content = repo.get_current_content(sec["id"])
                if content:
                    done += 1
            progress = int((done / total) * 100)
            repo.update_project_progress(project_id, progress)
    except Exception as exc:
        logger.warning("Could not update project progress: %s", exc)

    return {"error": None}
