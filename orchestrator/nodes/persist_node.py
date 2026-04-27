"""
orchestrator/nodes/persist_node.py
────────────────────────────────────
Saves the approved / edited agent_output as the new current document version.
Also saves both chat turns (user message + AI output) to chat_messages.
Updates project progress.

Fixes vs original:
  1. Version is saved even when section_id is None — stores against project
     only when section_id is provided (writing intent always needs one).
  2. repo.db.table() replaced with repo function calls — avoids None db ref.
  3. Added explicit logging so approve actions are traceable.
"""

from __future__ import annotations

import logging

from ..state import GraphState
from database import repository as repo

logger = logging.getLogger(__name__)


def persist_node(state: GraphState) -> dict:
    project_id = state["project_id"]
    section_id = state.get("section_id")
    agent_output = state.get("agent_output") or ""
    hitl_action = state.get("hitl_action", "approve")
    user_msg = state.get("user_message", "")
    last_agent = state.get("last_agent", "")

    logger.info(
        "persist_node: project=%s section=%s hitl_action=%s last_agent=%s "
        "output_len=%d",
        project_id, section_id, hitl_action, last_agent, len(agent_output),
    )

    # ── Save document version ──────────────────────────────────────────────────
    # Only for writing/literature agents AND when section_id is provided.
    # search/visualize outputs don't produce document versions.
    should_version = (
        hitl_action in ("approve", "edit")
        and section_id
        and agent_output
        and last_agent in ("writing", "literature")
    )

    if should_version:
        try:
            suggestion = repo.get_pending_suggestion(section_id)
            suggestion_id = suggestion["id"] if suggestion else None

            version = repo.save_new_version(
                section_id=section_id,
                content=agent_output,
                author_type="human" if hitl_action == "edit" else "ai",
                suggestion_id=suggestion_id,
            )
            logger.info(
                "persist_node: saved version %s for section %s",
                version.get("version_number"), section_id,
            )
        except Exception as exc:
            logger.warning("Could not save document version: %s", exc)
    else:
        logger.info(
            "persist_node: skipping version save "
            "(section_id=%s, last_agent=%s, hitl_action=%s)",
            section_id, last_agent, hitl_action,
        )

    # ── Save chat history ──────────────────────────────────────────────────────
    try:
        if user_msg:
            repo.save_message(project_id, "human", user_msg, section_id)
        if agent_output:
            repo.save_message(project_id, "ai", agent_output, section_id)
    except Exception as exc:
        logger.warning("Could not save chat messages: %s", exc)

    # ── Update project progress ────────────────────────────────────────────────
    # Count sections with at least one current version vs total sections
    try:
        sections = repo.get_project_sections(project_id)
        total = len(sections)
        if total > 0:
            done = sum(1 for s in sections if s.get("content"))
            progress = int((done / total) * 100)
            repo.update_project_progress(project_id, progress)
            logger.info(
                "persist_node: progress updated to %d%% (%d/%d sections)",
                progress, done, total,
            )
    except Exception as exc:
        logger.warning("Could not update project progress: %s", exc)

    return {"error": None}
